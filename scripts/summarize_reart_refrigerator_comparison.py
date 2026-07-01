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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_reart_visualization import infer_joint_from_poses  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare refrigerator optimized/feedforward/ReArt outputs.")
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reart-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--type-mismatch-penalty", type=float, default=1.0)
    args = parser.parse_args()

    reference_path = args.reference_json.expanduser().resolve()
    reart_root = args.reart_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else reart_root / "_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_json(reference_path)
    optimized = reference.get("optimized_tracking", [])
    feedforward = reference.get("feedforward_particulate_set_match", [])
    gt_by_object = _group(optimized)
    reart_candidates = []
    reart_rows = []
    for object_id, gt_rows in gt_by_object.items():
        candidates = _load_reart_candidates(reart_root, object_id)
        reart_candidates.extend(candidates)
        reart_rows.extend(_match_candidates(gt_rows, candidates, float(args.type_mismatch_penalty)))

    timing_reference = reference.get("summary", {}).get("timing_and_vram", {})
    payload = {
        "source": "lightwheel-refrigerator-three-path-comparison",
        "reference_json": str(reference_path),
        "reart_root": str(reart_root),
        "summary": {
            "optimized": _summary(optimized),
            "feedforward": _summary(feedforward),
            "reart": _summary(reart_rows),
            "optimized_timing_vram": timing_reference,
            "feedforward_timing_vram": timing_reference,
            "reart_timing_vram": _reart_resource_summary(reart_root),
        },
        "optimized": optimized,
        "feedforward": feedforward,
        "reart": reart_rows,
        "reart_candidates": reart_candidates,
        "notes": [
            "ReArt candidates are derived from every predicted graph connection.",
            "Parent is selected as the lower-motion endpoint before joint fitting.",
            "GT/ReArt correspondence uses one-to-one set matching with type mismatch, axis angle, and normalized axis-line position cost.",
            "Prismatic axis position is reported for metric compatibility but remains weakly observable.",
            "Runtime and GPU0 peak memory delta are read from each remote_reart_timing.json, including objects with no matched joint.",
        ],
    }
    json_path = output_dir / "three_path_comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(output_dir / "three_path_comparison_per_joint.tsv", optimized, feedforward, reart_rows)
    _write_markdown(output_dir / "three_path_comparison_summary.md", payload)
    _write_plots(output_dir, optimized, feedforward, reart_rows, reart_root)
    print(json.dumps({"three_path_comparison_summary": str(json_path)}, indent=2))
    return 0


def _load_reart_candidates(root: Path, object_id: str) -> list[dict[str, Any]]:
    result_path = root / object_id / "reart" / object_id / "result.pkl"
    if not result_path.exists():
        return []
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    poses = np.asarray(result["pred_pose_list"], dtype=float)
    connections = result.get("joint_connection") or []
    candidates = []
    for edge_index, connection in enumerate(connections):
        endpoint_a, endpoint_b = [int(value) for value in connection]
        parent, child = _select_parent_child(poses, endpoint_a, endpoint_b)
        joint = infer_joint_from_poses(poses, parent, child)
        candidates.append(
            {
                "object_id": object_id,
                "candidate_index": edge_index,
                "parent_part": parent,
                "child_part": child,
                "predicted_joint_type": joint["joint_type"],
                "predicted_axis": joint["axis"],
                "predicted_pivot": joint["pivot"],
                "rotation_range_rad": joint.get("rotation_range_rad"),
                "translation_range_m": joint.get("translation_range"),
                "axis_fit_alignment": joint.get("axis_fit_alignment"),
                "cano_idx": int(result.get("cano_idx", 0)),
                "source_path": str(result_path),
            }
        )
    return candidates


def _match_candidates(
    gt_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    type_mismatch_penalty: float,
) -> list[dict[str, Any]]:
    pairs = []
    for gt_index, gt in enumerate(gt_rows):
        scale = float(gt["bbox_scale_m"])
        for candidate_index, candidate in enumerate(candidates):
            angle = _axis_angle_error_deg(candidate["predicted_axis"], gt["gt_axis"])
            position = _line_distance(
                candidate["predicted_pivot"],
                candidate["predicted_axis"],
                gt["gt_pivot"],
                gt["gt_axis"],
            ) / scale
            type_correct = candidate["predicted_joint_type"] == gt["gt_joint_type"]
            score = angle / 180.0 + min(position, 1.0) + (0.0 if type_correct else type_mismatch_penalty)
            pairs.append((score, gt_index, candidate_index, type_correct, angle, position))
    pairs.sort(key=lambda item: item[0])
    matched: dict[int, tuple[Any, ...]] = {}
    used_candidates = set()
    for pair in pairs:
        _, gt_index, candidate_index, *_ = pair
        if gt_index in matched or candidate_index in used_candidates:
            continue
        matched[gt_index] = pair
        used_candidates.add(candidate_index)

    rows = []
    for gt_index, gt in enumerate(gt_rows):
        base = {
            "object_id": gt["object_id"],
            "path": "reart_set_match",
            "gt_joint_name": gt.get("gt_joint_name"),
            "gt_joint_type": gt.get("gt_joint_type"),
            "gt_axis": gt.get("gt_axis"),
            "gt_pivot": gt.get("gt_pivot"),
            "gt_pivot_normalized": gt.get("gt_pivot_normalized"),
            "bbox_scale_m": gt.get("bbox_scale_m"),
            "candidate_count": len(candidates),
        }
        pair = matched.get(gt_index)
        if pair is None:
            rows.append({**base, "matched": False, "reason": "no_unmatched_candidate"})
            continue
        score, _, candidate_index, type_correct, angle, position = pair
        candidate = candidates[candidate_index]
        rows.append(
            {
                **base,
                **candidate,
                "matched": True,
                "joint_type_correct": bool(type_correct),
                "match_score": score,
                "axis_angle_error_deg": angle,
                "axis_position_error_normalized": position,
                "axis_position_mse_normalized": position * position,
            }
        )
    return rows


def _select_parent_child(poses: np.ndarray, endpoint_a: int, endpoint_b: int) -> tuple[int, int]:
    return (
        (endpoint_a, endpoint_b)
        if _motion_score(poses[:, endpoint_a]) <= _motion_score(poses[:, endpoint_b])
        else (endpoint_b, endpoint_a)
    )


def _motion_score(transforms: np.ndarray) -> float:
    translations = transforms[:, :3, 3]
    translation_range = float(np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)))
    angles = []
    for transform in transforms:
        cosine = max(-1.0, min(1.0, (float(np.trace(transform[:3, :3])) - 1.0) * 0.5))
        angles.append(math.acos(cosine))
    return translation_range + max(angles) - min(angles)


def _summary(rows: list[dict[str, Any]], *, include_by_type: bool = True) -> dict[str, Any]:
    matched = [row for row in rows if row.get("matched") is not False]
    correct = [row for row in matched if row.get("joint_type_correct") is True]
    return {
        "joint_count": len(rows),
        "matched_count": len(matched),
        "coverage": len(matched) / len(rows) if rows else None,
        "joint_type_accuracy": len(correct) / len(matched) if matched else None,
        "axis_angle_error_deg": _aggregate(row.get("axis_angle_error_deg") for row in correct),
        "axis_position_error_normalized": _aggregate(row.get("axis_position_error_normalized") for row in correct),
        "axis_position_rmse_normalized": _rmse(row.get("axis_position_mse_normalized") for row in correct),
        "by_gt_joint_type": {
            joint_type: _summary(
                [row for row in rows if row.get("gt_joint_type") == joint_type],
                include_by_type=False,
            )
            for joint_type in sorted({str(row.get("gt_joint_type")) for row in rows})
        }
        if rows and include_by_type
        else {},
    }


def _reart_resource_summary(root: Path) -> dict[str, Any]:
    rows = []
    for timing_path in sorted(root.glob("*/remote_reart_timing.json")):
        timing = _read_json(timing_path)
        rows.append(
            {
                "object_id": timing_path.parent.name,
                "runtime_s": _nested(timing, "server_timings", "reart_s"),
                "peak_memory_delta_mib": _nested(
                    timing, "server_resource_usage", "reart", "peak_memory_delta_mib"
                ),
                "peak_gpu_index": _nested(timing, "server_resource_usage", "reart", "peak_gpu_index"),
            }
        )
    return {
        "per_object": rows,
        "runtime_s": _aggregate(row["runtime_s"] for row in rows),
        "peak_memory_delta_mib": _aggregate(row["peak_memory_delta_mib"] for row in rows),
    }


def _write_tsv(path: Path, *groups: list[dict[str, Any]]) -> None:
    columns = [
        "path", "object_id", "gt_joint_name", "gt_joint_type", "predicted_joint_name",
        "predicted_joint_type", "matched", "joint_type_correct", "axis_angle_error_deg",
        "axis_position_error_normalized", "candidate_count", "candidate_index", "parent_part",
        "child_part", "cano_idx", "axis_fit_alignment", "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for group in groups:
            writer.writerows(group)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Lightwheel Refrigerator Three-path Comparison",
        "",
        "| Path | Joints | Matched | Coverage | Joint type acc | Mean angle deg | Position RMSE norm | Mean runtime s | Median peak VRAM delta MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    resources = payload["summary"]["reart_timing_vram"]
    for method in ("optimized", "feedforward", "reart"):
        item = payload["summary"][method]
        runtime = _aggregate_mean(resources.get("runtime_s")) if method == "reart" else None
        vram = _aggregate_median(resources.get("peak_memory_delta_mib")) if method == "reart" else None
        lines.append(
            f"| {method} | {item['joint_count']} | {item['matched_count']} | {_fmt(item['coverage'])} | "
            f"{_fmt(item['joint_type_accuracy'])} | {_fmt(_aggregate_mean(item['axis_angle_error_deg']))} | "
            f"{_fmt(item['axis_position_rmse_normalized'])} | {_fmt(runtime)} | {_fmt(vram)} |"
        )
    lines.extend(
        [
            "",
            "![Joint type accuracy](per_object_joint_type_accuracy.svg)",
            "",
            "![Axis angle error](per_object_axis_angle_error_deg.svg)",
            "",
            "![Runtime](reart_per_object_runtime_s.svg)",
            "",
            "![GPU memory](reart_per_object_peak_vram_mib.svg)",
            "",
            "## ReArt Per Object",
            "",
            "| Object | GT joints | ReArt candidates | Matched | Type accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    gt_groups = _group(payload["reart"])
    candidate_groups = _group(payload["reart_candidates"])
    for object_id in sorted(gt_groups):
        rows = gt_groups[object_id]
        matched = [row for row in rows if row.get("matched") is not False]
        correct = [row for row in matched if row.get("joint_type_correct") is True]
        lines.append(
            f"| {object_id} | {len(rows)} | {len(candidate_groups.get(object_id, []))} | {len(matched)} | "
            f"{_fmt(len(correct) / len(matched) if matched else None)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(
    output_dir: Path,
    optimized: list[dict[str, Any]],
    feedforward: list[dict[str, Any]],
    reart: list[dict[str, Any]],
    reart_root: Path,
) -> None:
    object_ids = sorted(_group(optimized))
    groups = {"optimized": _group(optimized), "feedforward": _group(feedforward), "reart": _group(reart)}
    _line_svg(
        output_dir / "per_object_joint_type_accuracy.svg",
        "Per-object Joint Type Accuracy",
        "accuracy",
        object_ids,
        {
            method: [
                _object_accuracy(groups[method].get(object_id, []), "joint_type_accuracy")
                for object_id in object_ids
            ]
            for method in groups
        },
    )
    _line_svg(
        output_dir / "per_object_axis_angle_error_deg.svg",
        "Per-object Axis Angle Error",
        "degrees",
        object_ids,
        {
            method: [_object_accuracy(groups[method].get(object_id, []), "axis_angle") for object_id in object_ids]
            for method in groups
        },
    )
    resources = _reart_resource_summary(reart_root)["per_object"]
    resource_by_object = {row["object_id"]: row for row in resources}
    _line_svg(
        output_dir / "reart_per_object_runtime_s.svg",
        "ReArt Runtime Per Object",
        "seconds",
        object_ids,
        {"ReArt": [resource_by_object.get(object_id, {}).get("runtime_s") for object_id in object_ids]},
    )
    _line_svg(
        output_dir / "reart_per_object_peak_vram_mib.svg",
        "ReArt GPU0 Peak Memory Delta Per Object",
        "MiB",
        object_ids,
        {"ReArt": [resource_by_object.get(object_id, {}).get("peak_memory_delta_mib") for object_id in object_ids]},
    )


def _line_svg(
    path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    series: dict[str, list[Any]],
) -> None:
    colors = {"optimized": "#2f6f9f", "feedforward": "#d9872b", "reart": "#3b9b62", "ReArt": "#3b9b62"}
    width, height = 1320, 440
    left, right, top, bottom = 80, 30, 48, 100
    finite = [float(v) for values in series.values() for v in values if v is not None and math.isfinite(float(v))]
    y_max = max(finite + [1.0]) * 1.08
    x_step = (width - left - right) / max(1, len(labels) - 1)
    y_scale = (height - top - bottom) / y_max
    elements = [
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="17">{title}</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {height/2})">{y_label}</text>',
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
        elements.append(
            f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="end" font-size="9" '
            f'transform="rotate(-55 {x:.1f} {height-bottom+18})">{label.replace("refrigerator", "r")}</text>'
        )
    for series_index, (name, values) in enumerate(series.items()):
        color = colors.get(name, "#666")
        points = []
        for index, value in enumerate(values):
            if value is None or not math.isfinite(float(value)):
                continue
            x = left + index * x_step
            y = height - bottom - float(value) * y_scale
            points.append(f"{x:.1f},{y:.1f}")
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        if len(points) >= 2:
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx = left + series_index * 145
        elements.append(f'<line x1="{lx}" y1="42" x2="{lx+22}" y2="42" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{lx+28}" y="46" font-size="11">{name}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(elements) + "</svg>\n",
        encoding="utf-8",
    )


def _object_accuracy(rows: list[dict[str, Any]], metric: str) -> float | None:
    matched = [row for row in rows if row.get("matched") is not False]
    if not matched:
        return None
    if metric == "joint_type_accuracy":
        return sum(row.get("joint_type_correct") is True for row in matched) / len(matched)
    correct = [row for row in matched if row.get("joint_type_correct") is True]
    values = [float(row["axis_angle_error_deg"]) for row in correct if row.get("axis_angle_error_deg") is not None]
    return mean(values) if values else None


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(str(row["object_id"]), []).append(row)
    return output


def _aggregate(values: Any) -> dict[str, float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {"count": len(finite), "mean": mean(finite), "median": median(finite), "min": min(finite), "max": max(finite)}


def _aggregate_mean(value: Any) -> float | None:
    return float(value["mean"]) if isinstance(value, dict) and value.get("mean") is not None else None


def _aggregate_median(value: Any) -> float | None:
    return float(value["median"]) if isinstance(value, dict) and value.get("median") is not None else None


def _rmse(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return math.sqrt(mean(finite)) if finite else None


def _nested(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4g}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


if __name__ == "__main__":
    raise SystemExit(main())
