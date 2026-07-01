#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.evaluation import _axis_angle_error_deg, _line_distance
from rgbd_urdf_mvp.perception.reart_adapter import _read_frame_ply

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_reart_visualization import infer_joint_from_poses  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Add ReArt to the Lightwheel microwave path comparison.")
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reart-root", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    reference_path = args.reference_json.expanduser().resolve()
    reart_root = args.reart_root.expanduser().resolve()
    retry_root = args.retry_root.expanduser().resolve() if args.retry_root else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else reart_root / "_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_json(reference_path)
    optimized_rows = reference.get("optimized_tracking", [])
    feedforward_rows = _feedforward_rows(reference)
    primary_rows = [_evaluate_object(row, reart_root) for row in optimized_rows]
    retry_rows = _evaluate_retries(optimized_rows, retry_root, reart_root) if retry_root else []
    reart_rows = _select_reart_rows(primary_rows, retry_rows)
    summaries = {
        "optimized": _summary(optimized_rows),
        "feedforward": _summary(feedforward_rows),
        "reart": _summary(reart_rows),
    }
    payload = {
        "source": "lightwheel-microwave-three-path-comparison",
        "reference_json": str(reference_path),
        "reart_root": str(reart_root),
        "summary": summaries,
        "optimized": optimized_rows,
        "feedforward": feedforward_rows,
        "reart": reart_rows,
        "reart_primary_runs": primary_rows,
        "reart_retry_runs": retry_rows,
        "notes": [
            "ReArt uses an already-open canonical frame selected at 90% of the observed joint excursion.",
            "ReArt joint axes are fitted from all canonical-relative parent/child rotations.",
            "ReArt parent/child identity is selected without GT geometry scoring: the lower-motion part is the parent.",
            "Axis angle is sign-invariant. Axis position is the shortest line-to-line distance normalized by object bbox scale.",
            "ReArt segmentation purity is diagnostic only and uses nearest exported GT-labelled points.",
        ],
    }
    json_path = output_dir / "three_path_comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(output_dir / "three_path_comparison_per_object.tsv", optimized_rows, feedforward_rows, reart_rows)
    _write_per_object_plots(output_dir, optimized_rows, feedforward_rows, reart_rows)
    _write_markdown(output_dir / "three_path_comparison_summary.md", payload)
    print(json.dumps({"three_path_comparison_summary": str(json_path)}, indent=2))
    return 0


def _evaluate_object(
    gt: dict[str, Any],
    reart_root: Path,
    *,
    run_dir: Path | None = None,
    run_name: str = "primary",
) -> dict[str, Any]:
    object_id = str(gt["object_id"])
    run_dir = run_dir or reart_root / object_id
    result_path = run_dir / "reart" / object_id / "result.pkl"
    timing = _read_json(run_dir / "remote_reart_timing.json")
    base = {
        "object_id": object_id,
        "path": "reart",
        "gt_joint_type": gt.get("gt_joint_type"),
        "gt_axis": gt.get("gt_axis"),
        "gt_pivot": gt.get("gt_pivot"),
        "bbox_scale_m": gt.get("bbox_scale_m"),
        "source_path": str(result_path),
        "run_name": run_name,
        "runtime_s": _nested_float(timing, "server_timings", "reart_s")
        or _nested_float(timing, "server_timings", "reart_wrapper_total_s"),
        "peak_memory_delta_mib": _nested_float(timing, "server_resource_usage", "reart", "peak_memory_delta_mib"),
    }
    if not result_path.exists():
        return {**base, "matched": False, "reason": "missing_result"}
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    poses = np.asarray(result["pred_pose_list"], dtype=float)
    labels = np.asarray(result["pred_cano_part"], dtype=int)
    cano_pc = np.asarray(result["cano_pc"], dtype=float)
    connections = result.get("joint_connection") or []
    if not connections:
        return {**base, "matched": False, "reason": "missing_joint_connection"}
    parent, child = _select_parent_child(poses, connections)
    joint = infer_joint_from_poses(poses, parent, child)
    predicted_type = str(joint["joint_type"])
    gt_type = str(gt.get("gt_joint_type"))
    axis = [float(value) for value in joint["axis"]]
    pivot = [float(value) for value in joint["pivot"]]
    gt_axis = [float(value) for value in gt["gt_axis"]]
    gt_pivot = [float(value) for value in gt["gt_pivot"]]
    scale = float(gt["bbox_scale_m"])
    position_error_m = _line_distance(pivot, axis, gt_pivot, gt_axis)
    purity = _segmentation_purity(
        [run_dir / "sequence" / object_id, Path("outputs/reart_sequences/lightwheel_microwaves_open") / object_id],
        int(result.get("cano_idx", 0)),
        cano_pc,
        labels,
        parent,
        child,
    )
    return {
        **base,
        "matched": True,
        "predicted_joint_type": predicted_type,
        "joint_type_correct": predicted_type == gt_type,
        "axis_angle_error_deg": _axis_angle_error_deg(axis, gt_axis),
        "axis_position_error_m": position_error_m,
        "axis_position_error_normalized": position_error_m / scale,
        "axis_position_mse_normalized": (position_error_m / scale) ** 2,
        "predicted_axis": axis,
        "predicted_pivot": pivot,
        "parent_part": parent,
        "child_part": child,
        "cano_idx": int(result.get("cano_idx", 0)),
        "rotation_range_rad": joint.get("rotation_range_rad"),
        "translation_range_m": joint.get("translation_range"),
        "axis_fit_alignment": joint.get("axis_fit_alignment"),
        "segmentation_purity": purity,
    }


def _select_parent_child(poses: np.ndarray, connections: list[Any]) -> tuple[int, int]:
    parent, child = [int(value) for value in connections[0]]
    parent_motion = _part_motion_score(poses[:, parent])
    child_motion = _part_motion_score(poses[:, child])
    return (parent, child) if parent_motion <= child_motion else (child, parent)


def _part_motion_score(transforms: np.ndarray) -> float:
    translations = transforms[:, :3, 3]
    translation_range = float(np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)))
    angles = []
    for transform in transforms:
        trace = float(np.trace(transform[:3, :3]))
        angles.append(math.acos(max(-1.0, min(1.0, (trace - 1.0) * 0.5))))
    return translation_range + (max(angles) - min(angles))


def _segmentation_purity(
    sequence_dirs: list[Path],
    cano_idx: int,
    cano_pc: np.ndarray,
    labels: np.ndarray,
    parent: int,
    child: int,
) -> dict[str, Any] | None:
    source_path = next(
        (directory / f"frame_{cano_idx:04d}.ply" for directory in sequence_dirs if (directory / f"frame_{cano_idx:04d}.ply").exists()),
        sequence_dirs[-1] / f"frame_{cano_idx:04d}.ply",
    )
    if not source_path.exists():
        return None
    source = _read_frame_ply(source_path)
    source_xyz = np.asarray([point[:3] for point in source], dtype=float)
    source_ids = np.asarray([int(point[3]) for point in source], dtype=int)
    nearest = np.argmin(((cano_pc[:, None, :] - source_xyz[None, :, :]) ** 2).sum(axis=2), axis=1)
    gt_ids = source_ids[nearest]
    dominant_ids = [int(value) for value in sorted(set(source_ids), key=lambda value: -int(np.sum(source_ids == value)))]
    gt_base = dominant_ids[0] if dominant_ids else None
    gt_door = dominant_ids[1] if len(dominant_ids) > 1 else None
    return {
        "gt_base_part_id": gt_base,
        "gt_door_part_id": gt_door,
        "parent_base_fraction": _fraction(gt_ids[labels == parent], gt_base),
        "parent_door_fraction": _fraction(gt_ids[labels == parent], gt_door),
        "child_base_fraction": _fraction(gt_ids[labels == child], gt_base),
        "child_door_fraction": _fraction(gt_ids[labels == child], gt_door),
    }


def _evaluate_retries(
    optimized_rows: list[dict[str, Any]],
    retry_root: Path,
    reart_root: Path,
) -> list[dict[str, Any]]:
    gt_by_object = {str(row["object_id"]): row for row in optimized_rows}
    rows = []
    if not retry_root.exists():
        return rows
    for run_dir in sorted(path for path in retry_root.iterdir() if path.is_dir()):
        object_id = run_dir.name.split("_cano", 1)[0]
        gt = gt_by_object.get(object_id)
        if gt is None:
            continue
        row = _evaluate_object(gt, reart_root, run_dir=run_dir, run_name=run_dir.name)
        rows.append(row)
    return rows


def _select_reart_rows(
    primary_rows: list[dict[str, Any]],
    retry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retries_by_object: dict[str, list[dict[str, Any]]] = {}
    for row in retry_rows:
        retries_by_object.setdefault(str(row["object_id"]), []).append(row)
    selected = []
    for primary in primary_rows:
        candidates = [primary, *retries_by_object.get(str(primary["object_id"]), [])]
        best = max(candidates, key=_internal_run_quality)
        chosen = dict(best)
        chosen["selected_run"] = best.get("run_name", "primary")
        chosen["selection_policy"] = "joint availability, then axis-fit alignment and rotation evidence"
        selected.append(chosen)
    return selected


def _internal_run_quality(row: dict[str, Any]) -> tuple[float, float, float]:
    if row.get("matched") is not True:
        return (0.0, 0.0, 0.0)
    alignment = float(row.get("axis_fit_alignment") or 0.0)
    rotation = min(float(row.get("rotation_range_rad") or 0.0), math.pi) / math.pi
    return (1.0, alignment, rotation)


def _fraction(values: np.ndarray, target: int | None) -> float | None:
    if target is None or len(values) == 0:
        return None
    return float(np.mean(values == target))


def _feedforward_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows = reference.get("feedforward_particulate_best_joint_normalized")
    if isinstance(rows, list):
        return rows
    for key in ("feedforward_particulate_best_joint", "feedforward_particulate"):
        rows = reference.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("matched") is not False]
    correct = [row for row in matched if row.get("joint_type_correct") is True]
    return {
        "object_count": len(rows),
        "matched_count": len(matched),
        "coverage": len(matched) / len(rows) if rows else None,
        "joint_type_accuracy": len(correct) / len(matched) if matched else None,
        "axis_angle_error_deg_mean_type_correct": _aggregate(row.get("axis_angle_error_deg") for row in correct),
        "axis_position_error_normalized_mean_type_correct": _aggregate(
            row.get("axis_position_error_normalized") for row in correct
        ),
        "axis_position_rmse_normalized_type_correct": _rmse(
            row.get("axis_position_mse_normalized") for row in correct
        ),
        "runtime_s": _aggregate(row.get("runtime_s") for row in rows),
        "peak_memory_delta_mib": _aggregate(row.get("peak_memory_delta_mib") for row in rows),
    }


def _aggregate(values: Any) -> dict[str, float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {"count": len(finite), "mean": mean(finite), "median": median(finite), "min": min(finite), "max": max(finite)}


def _rmse(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return math.sqrt(mean(finite)) if finite else None


def _write_tsv(path: Path, *groups: list[dict[str, Any]]) -> None:
    rows = [row for group in groups for row in group]
    columns = [
        "path", "object_id", "matched", "gt_joint_type", "predicted_joint_type", "joint_type_correct",
        "axis_angle_error_deg", "axis_position_error_normalized", "runtime_s", "peak_memory_delta_mib",
        "cano_idx", "parent_part", "child_part", "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_per_object_plots(
    output_dir: Path,
    optimized: list[dict[str, Any]],
    feedforward: list[dict[str, Any]],
    reart: list[dict[str, Any]],
) -> None:
    rows_by_method = {
        "optimized": {str(row["object_id"]): row for row in optimized},
        "feedforward": {str(row["object_id"]): row for row in feedforward},
        "reart": {str(row["object_id"]): row for row in reart},
    }
    object_ids = [str(row["object_id"]) for row in optimized]
    _write_line_svg(
        output_dir / "per_object_axis_angle_error_deg.svg",
        "Per-object Axis Angle Error",
        "degrees",
        object_ids,
        {
            method: [rows_by_method[method].get(object_id, {}).get("axis_angle_error_deg") for object_id in object_ids]
            for method in rows_by_method
        },
    )
    _write_line_svg(
        output_dir / "per_object_axis_position_error_normalized.svg",
        "Per-object Axis Position Error",
        "bbox-normalized distance",
        object_ids,
        {
            method: [
                rows_by_method[method].get(object_id, {}).get("axis_position_error_normalized")
                for object_id in object_ids
            ]
            for method in rows_by_method
        },
    )
    _write_line_svg(
        output_dir / "reart_per_object_runtime_s.svg",
        "ReArt Runtime Per Object",
        "seconds",
        object_ids,
        {"ReArt": [rows_by_method["reart"].get(object_id, {}).get("runtime_s") for object_id in object_ids]},
    )
    _write_line_svg(
        output_dir / "reart_per_object_peak_vram_mib.svg",
        "ReArt GPU0 Peak Memory Delta Per Object",
        "MiB",
        object_ids,
        {"ReArt": [rows_by_method["reart"].get(object_id, {}).get("peak_memory_delta_mib") for object_id in object_ids]},
    )


def _write_line_svg(
    path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    series: dict[str, list[Any]],
) -> None:
    colors = {"optimized": "#2f6f9f", "feedforward": "#d9872b", "reart": "#3b9b62"}
    colors["ReArt"] = colors["reart"]
    width, height = 1320, 440
    left, right, top, bottom = 80, 30, 48, 100
    finite = [
        float(value)
        for values in series.values()
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    y_max = max(finite + [1.0]) * 1.08
    x_step = (width - left - right) / max(1, len(labels) - 1)
    y_scale = (height - top - bottom) / y_max
    elements = [
        f'<text x="{width / 2:.1f}" y="25" text-anchor="middle" font-size="17">{title}</text>',
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {height / 2:.1f})">{y_label}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = height - bottom - value * y_scale
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        elements.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{value:.3g}</text>')
    for index, label in enumerate(labels):
        x = left + index * x_step
        short = label.replace("microwave", "m")
        elements.append(
            f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="end" font-size="9" '
            f'transform="rotate(-55 {x:.1f} {height-bottom+18})">{short}</text>'
        )
    for series_index, (name, values) in enumerate(series.items()):
        color = colors.get(name, "#666")
        current: list[str] = []
        segments: list[list[str]] = []
        for index, value in enumerate(values):
            if value is None or not math.isfinite(float(value)):
                if current:
                    segments.append(current)
                    current = []
                continue
            x = left + index * x_step
            y = height - bottom - float(value) * y_scale
            current.append(f"{x:.1f},{y:.1f}")
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        if current:
            segments.append(current)
        for points in segments:
            if len(points) >= 2:
                elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + series_index * 145
        elements.append(f'<line x1="{legend_x}" y1="42" x2="{legend_x+22}" y2="42" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x+28}" y="46" font-size="11">{name}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Lightwheel Microwave Three-path Comparison",
        "",
        "| Path | Coverage | Joint type accuracy | Mean axis angle deg | Axis position RMSE norm | Mean runtime s | Peak VRAM delta MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("optimized", "feedforward", "reart"):
        item = payload["summary"][method]
        lines.append(
            f"| {method} | {_fmt(item['coverage'])} | {_fmt(item['joint_type_accuracy'])} | "
            f"{_fmt(_mean_metric(item['axis_angle_error_deg_mean_type_correct']))} | "
            f"{_fmt(item['axis_position_rmse_normalized_type_correct'])} | "
            f"{_fmt(_mean_metric(item['runtime_s']))} | {_fmt(_mean_metric(item['peak_memory_delta_mib']))} |"
        )
    lines.extend(
        [
            "",
            "![Per-object axis angle](per_object_axis_angle_error_deg.svg)",
            "",
            "![Per-object axis position](per_object_axis_position_error_normalized.svg)",
            "",
            "![ReArt runtime](reart_per_object_runtime_s.svg)",
            "",
            "![ReArt GPU memory](reart_per_object_peak_vram_mib.svg)",
            "",
            "## ReArt Per Object",
            "",
        ]
    )
    lines.extend(
        [
            "| Object | Matched | Type | Angle deg | Position norm | Canonical frame | Child door purity |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["reart"]:
        purity = row.get("segmentation_purity") or {}
        lines.append(
            f"| {row['object_id']} | {row.get('matched')} | {row.get('predicted_joint_type', '-')} | "
            f"{_fmt(row.get('axis_angle_error_deg'))} | {_fmt(row.get('axis_position_error_normalized'))} | "
            f"{row.get('cano_idx', '-')} | {_fmt(purity.get('child_door_fraction'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_metric(value: Any) -> float | None:
    return float(value["mean"]) if isinstance(value, dict) and value.get("mean") is not None else None


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4g}"


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
