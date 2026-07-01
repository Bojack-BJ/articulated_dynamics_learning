#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from rgbd_urdf_mvp.kinematics.evaluation import _axis_angle_error_deg, _line_distance
from rgbd_urdf_mvp.kinematics.feedforward_evaluation import _parse_urdf_joint_candidates, _rotate_z


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare optimized and feedforward articulation outputs.")
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--optimized-root", type=Path, required=True)
    parser.add_argument("--feedforward-root", type=Path, required=True)
    parser.add_argument("--timing-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--yaw-degrees", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--feedforward-type-mismatch-penalty", type=float, default=1.0)
    parser.add_argument("--optimized-joint-rotation-threshold-rad", type=float, default=None)
    parser.add_argument("--optimized-joint-translation-threshold-m", type=float, default=None)
    args = parser.parse_args()

    batch_config = args.batch_config.expanduser().resolve()
    optimized_root = args.optimized_root.expanduser().resolve()
    feedforward_root = args.feedforward_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else feedforward_root / "_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = args.timing_summary.expanduser().resolve() if args.timing_summary else output_dir / "timing_summary.json"

    optimized_rows: list[dict[str, Any]] = []
    feedforward_rows: list[dict[str, Any]] = []
    per_object: list[dict[str, Any]] = []
    for object_id in _object_ids(batch_config):
        object_optimized_rows, gt_rows, optimized_source_summary = _load_optimized_rows(optimized_root, object_id)
        object_feedforward_rows = _feedforward_rows(
            feedforward_root=feedforward_root,
            object_id=object_id,
            gt_rows=gt_rows,
            yaw_degrees=tuple(args.yaw_degrees),
            type_mismatch_penalty=float(args.feedforward_type_mismatch_penalty),
        )
        optimized_rows.extend(object_optimized_rows)
        feedforward_rows.extend(object_feedforward_rows)
        per_object.append(
            {
                "object_id": object_id,
                "optimized": _summarize_accuracy(object_optimized_rows),
                "feedforward": _summarize_accuracy(object_feedforward_rows),
                "optimized_source_summary": optimized_source_summary,
            }
        )

    timing = _read_json(timing_path)
    payload = {
        "source": "articulation-path-comparison",
        "batch_config": str(batch_config),
        "optimized_root": str(optimized_root),
        "feedforward_root": str(feedforward_root),
        "timing_summary": str(timing_path) if timing_path.exists() else None,
        "optimized_joint_thresholds": {
            "rotation_threshold_rad": args.optimized_joint_rotation_threshold_rad,
            "translation_threshold_m": args.optimized_joint_translation_threshold_m,
        },
        "summary": {
            "optimized_accuracy": _summarize_accuracy(optimized_rows),
            "feedforward_accuracy": _summarize_accuracy(feedforward_rows),
            "optimized_by_joint_type": _summarize_by_joint_type(optimized_rows),
            "feedforward_by_joint_type": _summarize_by_joint_type(feedforward_rows),
            "optimized_joint_type_errors": _joint_type_error_summary(optimized_rows),
            "timing_and_vram": _timing_and_vram(timing),
        },
        "notes": [
            "Optimized accuracy uses simulator child-part ground-truth correspondence from evaluate-kinematic-model.",
            "Feedforward accuracy uses set matching between GT joints and PARTICULATE URDF joints.",
            "Feedforward matching tests all URDF joints under the configured yaw rotations.",
            "Axis position MSE/RMSE summary fields are computed only on matched joints whose joint type is correct.",
            "For prismatic joints, axis direction and q replay are more diagnostic than pivot-like position; slide-axis location is weakly observable from motion alone.",
            "Optimized timing excludes segmentation model time.",
        ],
        "optimized_tracking": optimized_rows,
        "feedforward_particulate_set_match": feedforward_rows,
        "per_object": per_object,
    }

    (output_dir / "path_comparison_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_per_joint_tsv(output_dir / "path_comparison_per_joint.tsv", optimized_rows + feedforward_rows)
    _write_plots(output_dir, payload)
    _write_markdown(output_dir / "path_comparison_summary.md", payload)
    print(json.dumps({"path_comparison_summary": str((output_dir / "path_comparison_summary.json").resolve())}, indent=2))
    return 0


def _object_ids(batch_config: Path) -> list[str]:
    out = []
    for line in batch_config.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) >= 3:
            out.append(fields[2])
    return out


def _load_optimized_rows(optimized_root: Path, object_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pointcloud_dir = optimized_root / object_id / "pointcloud_4d_partseg"
    evaluation_path = pointcloud_dir / "kinematic_evaluation.json"
    evaluation = _read_json(evaluation_path)
    scale = _bbox_scale(pointcloud_dir / "fusion_manifest.json")
    rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    for item in evaluation.get("per_joint", []):
        if not isinstance(item, dict) or not item.get("matched"):
            continue
        gt = {
            "object_id": object_id,
            "gt_joint_name": item.get("ground_truth_joint_name"),
            "gt_joint_type": item.get("ground_truth_joint_type"),
            "gt_axis": _vec3(item.get("ground_truth_axis"), [0.0, 0.0, 1.0]),
            "gt_pivot": _vec3(item.get("ground_truth_pivot"), [0.0, 0.0, 0.0]),
            "gt_pivot_normalized": _scale_vec(_vec3(item.get("ground_truth_pivot"), [0.0, 0.0, 0.0]), 1.0 / scale),
            "bbox_scale_m": scale,
        }
        gt_rows.append(gt)
        position_error_m = _float_or_none(item.get("pivot_error_m"))
        position_error_norm = position_error_m / scale if position_error_m is not None else None
        rows.append(
            {
                **gt,
                "path": "optimized_tracking",
                "matched": True,
                "predicted_joint_name": item.get("name"),
                "predicted_joint_type": item.get("predicted_joint_type"),
                "joint_type_correct": bool(item.get("joint_type_correct")),
                "axis_angle_error_deg": _float_or_none(item.get("axis_angle_error_deg")),
                "axis_position_error_m": position_error_m,
                "axis_position_mse_m2": position_error_m * position_error_m if position_error_m is not None else None,
                "axis_position_error_normalized": position_error_norm,
                "axis_position_mse_normalized": position_error_norm * position_error_norm
                if position_error_norm is not None
                else None,
                "predicted_axis": item.get("predicted_axis"),
                "predicted_pivot": item.get("predicted_pivot"),
                "predicted_pivot_normalized": _scale_vec(_vec3(item.get("predicted_pivot"), [0.0, 0.0, 0.0]), 1.0 / scale),
                "source_path": str(evaluation_path),
            }
        )
    return rows, gt_rows, evaluation.get("summary", {})


def _feedforward_rows(
    feedforward_root: Path,
    object_id: str,
    gt_rows: list[dict[str, Any]],
    yaw_degrees: tuple[int, ...],
    type_mismatch_penalty: float,
) -> list[dict[str, Any]]:
    urdfs = sorted((feedforward_root / object_id / "particulate").glob("urdf_*/model.urdf"))
    urdf = urdfs[-1] if urdfs else None
    if urdf is None:
        return [{**gt, "path": "feedforward_particulate_set_match", "matched": False, "reason": "missing_urdf"} for gt in gt_rows]
    candidates = _parse_urdf_joint_candidates(urdf)
    candidate_pairs = []
    for gt_index, gt in enumerate(gt_rows):
        for candidate_index, candidate in enumerate(candidates):
            for yaw_deg in yaw_degrees:
                axis = _rotate_z(candidate["axis"], yaw_deg)
                pivot = _rotate_z(candidate["pivot"], yaw_deg)
                angle_error = _axis_angle_error_deg(axis, gt["gt_axis"])
                position_error = _line_distance(pivot, axis, gt["gt_pivot_normalized"], gt["gt_axis"])
                type_correct = candidate["joint_type"] == gt["gt_joint_type"]
                score = position_error + angle_error / 180.0 + (0.0 if type_correct else type_mismatch_penalty)
                candidate_pairs.append(
                    (score, gt_index, candidate_index, yaw_deg, type_correct, angle_error, position_error, axis, pivot, candidate)
                )
    candidate_pairs.sort(key=lambda item: item[0])
    matched_by_gt: dict[int, tuple[Any, ...]] = {}
    used_gt: set[int] = set()
    used_candidate: set[int] = set()
    for pair in candidate_pairs:
        _, gt_index, candidate_index, *_ = pair
        if gt_index in used_gt or candidate_index in used_candidate:
            continue
        used_gt.add(gt_index)
        used_candidate.add(candidate_index)
        matched_by_gt[gt_index] = pair
        if len(used_gt) == len(gt_rows) or len(used_candidate) == len(candidates):
            break

    rows = []
    for gt_index, gt in enumerate(gt_rows):
        base = {
            **gt,
            "path": "feedforward_particulate_set_match",
            "source_path": str(urdf),
            "candidate_count": len(candidates),
        }
        pair = matched_by_gt.get(gt_index)
        if pair is None:
            rows.append({**base, "matched": False, "reason": "no_unmatched_candidate"})
            continue
        score, _, candidate_index, yaw_deg, type_correct, angle_error, position_error, axis, pivot, candidate = pair
        rows.append(
            {
                **base,
                "matched": True,
                "predicted_joint_name": candidate.get("joint_name"),
                "predicted_joint_type": candidate.get("joint_type"),
                "joint_type_correct": bool(type_correct),
                "selected_candidate_index": candidate_index,
                "best_yaw_deg": yaw_deg,
                "match_score": score,
                "axis_angle_error_deg": angle_error,
                "axis_position_error_normalized": position_error,
                "axis_position_mse_normalized": position_error * position_error,
                "predicted_axis": axis,
                "predicted_pivot_normalized": pivot,
                "raw_predicted_axis": candidate.get("axis"),
                "raw_predicted_pivot": candidate.get("pivot"),
            }
        )
    return rows


def _summarize_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("matched") is not False]
    type_correct = [row for row in matched if row.get("joint_type_correct") is True]
    return {
        "joint_count": len(rows),
        "matched_joint_count": len(matched),
        "joint_coverage": len(matched) / len(rows) if rows else None,
        "joint_type_accuracy": len(type_correct) / len(matched) if matched else None,
        "correct_type_count": len(type_correct),
        "axis_angle_error_deg_mean_all_matched": _mean(row.get("axis_angle_error_deg") for row in matched),
        "axis_angle_error_deg_median_all_matched": _median(row.get("axis_angle_error_deg") for row in matched),
        "axis_angle_error_deg_mean_type_correct": _mean(row.get("axis_angle_error_deg") for row in type_correct),
        "axis_position_error_normalized_mean_type_correct": _mean(
            row.get("axis_position_error_normalized") for row in type_correct
        ),
        "axis_position_mse_normalized_mean_type_correct": _mean(
            row.get("axis_position_mse_normalized") for row in type_correct
        ),
        "axis_position_rmse_normalized_type_correct": _rmse_from_mse(
            row.get("axis_position_mse_normalized") for row in type_correct
        ),
        "axis_position_mse_m2_mean_type_correct": _mean(row.get("axis_position_mse_m2") for row in type_correct),
        "axis_position_rmse_m_type_correct": _rmse_from_mse(row.get("axis_position_mse_m2") for row in type_correct),
    }


def _summarize_by_joint_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for joint_type in sorted({str(row.get("gt_joint_type", "unknown")) for row in rows}):
        typed_rows = [row for row in rows if str(row.get("gt_joint_type", "unknown")) == joint_type]
        out[joint_type] = _summarize_accuracy(typed_rows)
    return out


def _joint_type_error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("matched") is not False and row.get("joint_type_correct") is False]
    by_pair: dict[str, int] = {}
    for row in errors:
        key = f"{row.get('gt_joint_type')}->{row.get('predicted_joint_type')}"
        by_pair[key] = by_pair.get(key, 0) + 1
    return {
        "count": len(errors),
        "by_pair": by_pair,
        "rows": [
            {
                "object_id": row.get("object_id"),
                "gt_joint_name": row.get("gt_joint_name"),
                "gt_joint_type": row.get("gt_joint_type"),
                "predicted_joint_type": row.get("predicted_joint_type"),
                "axis_angle_error_deg": row.get("axis_angle_error_deg"),
                "axis_position_error_normalized": row.get("axis_position_error_normalized"),
            }
            for row in errors
        ],
    }


def _timing_and_vram(timing: dict[str, Any]) -> dict[str, Any]:
    summary = timing.get("summary", {})
    keys = [
        "optimized_total_measured_s",
        "optimized_cotracker_s",
        "optimized_track_peak_current_allocated_mib",
        "optimized_track_peak_driver_or_reserved_mib",
        "feedforward_server_hunyuan3d_s",
        "feedforward_server_particulate_s",
        "feedforward_server_total_s",
        "feedforward_hunyuan3d_peak_memory_delta_mib",
        "feedforward_particulate_peak_memory_delta_mib",
    ]
    out = {key: summary.get(key, {}) for key in keys}
    out["optimized_batch_timing"] = timing.get("optimized_batch_timing", {})
    return out


def _write_per_joint_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "path",
        "object_id",
        "gt_joint_name",
        "gt_joint_type",
        "predicted_joint_name",
        "predicted_joint_type",
        "matched",
        "joint_type_correct",
        "axis_angle_error_deg",
        "axis_position_error_normalized",
        "axis_position_mse_normalized",
        "axis_position_error_m",
        "axis_position_mse_m2",
        "bbox_scale_m",
        "best_yaw_deg",
        "candidate_count",
        "source_path",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_format_tsv(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    per_object = payload["per_object"]
    _svg_grouped_line(
        output_dir / "per_object_joint_type_accuracy.svg",
        title="Per-object Joint Type Accuracy",
        y_label="accuracy (%)",
        groups=[
            (
                item["object_id"].replace("refrigerator", "r"),
                {
                    "optimized": _percent(item["optimized"].get("joint_type_accuracy")),
                    "feedforward": _percent(item["feedforward"].get("joint_type_accuracy")),
                },
            )
            for item in per_object
        ],
        y_max=105.0,
    )
    _svg_grouped_line(
        output_dir / "per_object_axis_angle_error_deg.svg",
        title="Per-object Axis Angle Error (type-correct)",
        y_label="degrees",
        groups=[
            (
                item["object_id"].replace("refrigerator", "r"),
                {
                    "optimized": item["optimized"].get("axis_angle_error_deg_mean_type_correct"),
                    "feedforward": item["feedforward"].get("axis_angle_error_deg_mean_type_correct"),
                },
            )
            for item in per_object
        ],
    )
    _svg_grouped_line(
        output_dir / "per_object_axis_position_rmse_norm.svg",
        title="Per-object Axis Position RMSE (type-correct)",
        y_label="bbox-normalized RMSE",
        groups=[
            (
                item["object_id"].replace("refrigerator", "r"),
                {
                    "optimized": item["optimized"].get("axis_position_rmse_normalized_type_correct"),
                    "feedforward": item["feedforward"].get("axis_position_rmse_normalized_type_correct"),
                },
            )
            for item in per_object
        ],
    )
    _svg_grouped_line(
        output_dir / "per_joint_axis_angle_error_deg.svg",
        title="Per-joint Axis Angle Error",
        y_label="degrees",
        groups=_per_joint_plot_groups(payload),
        width=1500,
        height=520,
    )
    _svg_grouped_line(
        output_dir / "summary_accuracy.svg",
        title="Overall Accuracy Summary",
        y_label="percent",
        groups=[
            (
                "joint type",
                {
                    "optimized": _percent(summary["optimized_accuracy"].get("joint_type_accuracy")),
                    "feedforward": _percent(summary["feedforward_accuracy"].get("joint_type_accuracy")),
                },
            ),
            (
                "coverage",
                {
                    "optimized": _percent(summary["optimized_accuracy"].get("joint_coverage")),
                    "feedforward": _percent(summary["feedforward_accuracy"].get("joint_coverage")),
                },
            ),
        ],
        y_max=105.0,
        width=720,
        height=360,
    )
    timing = summary["timing_and_vram"]
    _svg_grouped_line(
        output_dir / "timing_summary.svg",
        title="Timing Summary",
        y_label="seconds / object",
        groups=[
            ("opt total", {"seconds": _metric_mean(timing.get("optimized_total_measured_s"))}),
            ("opt CoTracker", {"seconds": _metric_mean(timing.get("optimized_cotracker_s"))}),
            ("Hunyuan3D", {"seconds": _metric_mean(timing.get("feedforward_server_hunyuan3d_s"))}),
            ("PARTICULATE", {"seconds": _metric_mean(timing.get("feedforward_server_particulate_s"))}),
            ("ff total", {"seconds": _metric_mean(timing.get("feedforward_server_total_s"))}),
        ],
        width=780,
        height=360,
    )
    _svg_grouped_line(
        output_dir / "vram_summary.svg",
        title="Peak Memory Summary",
        y_label="MiB",
        groups=[
            ("opt current", {"MiB": _metric_mean(timing.get("optimized_track_peak_current_allocated_mib"))}),
            ("opt driver", {"MiB": _metric_mean(timing.get("optimized_track_peak_driver_or_reserved_mib"))}),
            ("Hunyuan delta", {"MiB": _metric_mean(timing.get("feedforward_hunyuan3d_peak_memory_delta_mib"))}),
            ("Particulate delta", {"MiB": _metric_mean(timing.get("feedforward_particulate_peak_memory_delta_mib"))}),
        ],
        width=780,
        height=360,
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    timing = summary["timing_and_vram"]
    opt = summary["optimized_accuracy"]
    ff = summary["feedforward_accuracy"]
    lines = [
        "# Refrigerator Optimized vs Feedforward Comparison",
        "",
        f"- Optimized root: `{payload['optimized_root']}`",
        f"- Feedforward root: `{payload['feedforward_root']}`",
        "- Accuracy is per joint. Axis position MSE/RMSE below is conditioned on correct joint type.",
    ]
    threshold_text = _threshold_text(payload.get("optimized_joint_thresholds", {}))
    if threshold_text:
        lines.append(f"- Optimized joint inference threshold override: {threshold_text}.")
    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            "| Method | Joints | Matched | Coverage | Joint type acc | Mean angle deg (type-correct) | Pos RMSE norm (type-correct) | Pos RMSE m (type-correct) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            _accuracy_row("optimized tracking", opt),
            _accuracy_row("feedforward PARTICULATE set-match", ff),
            "",
            "![Accuracy summary](summary_accuracy.svg)",
            "",
            "![Per-object joint type accuracy](per_object_joint_type_accuracy.svg)",
            "",
            "![Per-object axis angle error](per_object_axis_angle_error_deg.svg)",
            "",
            "![Per-object axis position RMSE](per_object_axis_position_rmse_norm.svg)",
            "",
            "## Joint Type Breakdown",
            "",
            "| Method | GT type | Joints | Matched | Joint type acc | Mean angle deg | Pos RMSE norm | Pos RMSE m |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, key in [
        ("optimized tracking", "optimized_by_joint_type"),
        ("feedforward PARTICULATE set-match", "feedforward_by_joint_type"),
    ]:
        for joint_type, item in sorted(summary.get(key, {}).items()):
            lines.append(_type_breakdown_row(method, joint_type, item))
    lines.extend(
        [
            "",
            "## Worst Per-joint Axis Errors",
            "",
            "Full per-joint data is in `path_comparison_per_joint.tsv`.",
            "",
            "| Method | Object | GT joint | GT type | Pred type | Type ok | Angle deg | Pos error norm |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in _worst_axis_rows(payload, limit=16):
        lines.append(_worst_axis_row(row))
    lines.extend(
        [
            "",
            "![Per-joint axis angle error](per_joint_axis_angle_error_deg.svg)",
            "",
            "## Optimized Joint Type Diagnosis",
            "",
        ]
    )
    error_summary = summary.get("optimized_joint_type_errors", {})
    pair_counts = error_summary.get("by_pair", {})
    if pair_counts:
        pair_text = ", ".join(f"`{key}`: `{value}`" for key, value in sorted(pair_counts.items()))
        lines.append(f"- Optimized joint-type errors: {pair_text}.")
        lines.extend(
            [
                "- If the dominant error is `prismatic->revolute`, sparse drawer tracks can make full SE(3) pose fitting report apparent rotation.",
                "- Use denser drawer tracking and inspect `metrics.track_model_comparison` before relying on rotation-range thresholds alone.",
            ]
        )
    else:
        lines.append("- Optimized joint-type errors: none.")
        lines.append(
            "- This run uses dense refrigerator tracking plus guarded 3D-track replay residual diagnostics for revolute/prismatic selection."
        )
    lines.extend(
        [
            "",
            "## Timing And VRAM",
            "",
            "| Metric | Count | Mean | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("optimized total measured s", "optimized_total_measured_s"),
        ("optimized CoTracker s", "optimized_cotracker_s"),
        ("optimized CoTracker peak current MiB", "optimized_track_peak_current_allocated_mib"),
        ("optimized CoTracker peak driver/reserved MiB", "optimized_track_peak_driver_or_reserved_mib"),
        ("feedforward Hunyuan3D s", "feedforward_server_hunyuan3d_s"),
        ("feedforward PARTICULATE s", "feedforward_server_particulate_s"),
        ("feedforward server total s", "feedforward_server_total_s"),
        ("Hunyuan3D peak VRAM delta MiB", "feedforward_hunyuan3d_peak_memory_delta_mib"),
        ("PARTICULATE peak VRAM delta MiB", "feedforward_particulate_peak_memory_delta_mib"),
    ]:
        lines.append(_metric_row(label, timing.get(key, {})))
    batch_timing = timing.get("optimized_batch_timing", {})
    if batch_timing.get("available") and batch_timing.get("wall_clock_s") is not None:
        lines.extend(
            [
                "",
                f"- Optimized batch wall-clock: `{float(batch_timing['wall_clock_s']):.3f} s` for "
                f"`{batch_timing.get('objects_completed')}/{batch_timing.get('objects_total')}` objects, "
                f"jobs=`{batch_timing.get('jobs')}`, tracking_jobs=`{batch_timing.get('tracking_jobs')}`.",
            ]
        )
    lines.extend(
        [
            "",
            "![Timing summary](timing_summary.svg)",
            "",
            "![VRAM summary](vram_summary.svg)",
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_grouped_line(
    path: Path,
    title: str,
    y_label: str,
    groups: list[tuple[str, dict[str, float | None]]],
    y_max: float | None = None,
    width: int = 1120,
    height: int = 430,
) -> None:
    colors = {"optimized": "#2563eb", "feedforward": "#dc2626", "seconds": "#2563eb", "MiB": "#2563eb"}
    series_names = list(dict.fromkeys(name for _, values in groups for name in values))
    values = [value for _, item in groups for value in item.values() if value is not None and math.isfinite(float(value))]
    if y_max is None:
        y_max = max(values + [1.0]) * 1.15
    y_min = 0.0
    left, right, top, bottom = 72, 28, 58, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_step = plot_w / max(1, len(groups) - 1)

    def x_at(index: int) -> float:
        return left + index * x_step

    def y_at(value: float) -> float:
        return top + plot_h - (float(value) - y_min) / max(1e-9, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#111827}.title{font-size:22px;font-weight:700}.axis{font-size:12px;fill:#4b5563}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.5}.point{stroke:white;stroke-width:1.5}.legend{font-size:13px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" class="title">{html.escape(title)}</text>',
    ]
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{_short_number(value)}</text>')
    parts.append(f'<text x="{left}" y="{top-14}" class="axis">{html.escape(y_label)}</text>')

    for series_name in series_names:
        points = []
        for index, (_label, values_by_series) in enumerate(groups):
            value = values_by_series.get(series_name)
            if value is None:
                continue
            points.append((x_at(index), y_at(float(value)), float(value)))
        if len(points) >= 2:
            d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y, _v) in enumerate(points))
            parts.append(f'<path d="{d}" class="line" stroke="{colors.get(series_name, "#374151")}"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors.get(series_name, "#374151")}" class="point"/>')
    for index, (label, _values) in enumerate(groups):
        x = x_at(index)
        rotation = -35 if len(groups) > 10 else 0
        if rotation:
            parts.append(
                f'<text x="{x:.1f}" y="{height-bottom+30}" transform="rotate({rotation} {x:.1f} {height-bottom+30})" text-anchor="end" class="axis">{html.escape(label)}</text>'
            )
        else:
            parts.append(f'<text x="{x:.1f}" y="{height-bottom+30}" text-anchor="middle" class="axis">{html.escape(label)}</text>')
    legend_x = left
    legend_y = height - 24
    for series_name in series_names:
        color = colors.get(series_name, "#374151")
        parts.append(f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+24}" y2="{legend_y-4}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{legend_x+12}" cy="{legend_y-4}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/>')
        parts.append(f'<text x="{legend_x+32}" y="{legend_y}" class="legend">{html.escape(series_name)}</text>')
        legend_x += 160
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _accuracy_row(label: str, item: dict[str, Any]) -> str:
    return "| " + " | ".join(
        [
            label,
            str(item.get("joint_count", 0)),
            str(item.get("matched_joint_count", 0)),
            _format_percent(item.get("joint_coverage")),
            _format_percent(item.get("joint_type_accuracy")),
            _format_number(item.get("axis_angle_error_deg_mean_type_correct")),
            _format_number(item.get("axis_position_rmse_normalized_type_correct")),
            _format_number(item.get("axis_position_rmse_m_type_correct")),
        ]
    ) + " |"


def _type_breakdown_row(method: str, joint_type: str, item: dict[str, Any]) -> str:
    return "| " + " | ".join(
        [
            method,
            joint_type,
            str(item.get("joint_count", 0)),
            str(item.get("matched_joint_count", 0)),
            _format_percent(item.get("joint_type_accuracy")),
            _format_number(item.get("axis_angle_error_deg_mean_type_correct")),
            _format_number(item.get("axis_position_rmse_normalized_type_correct")),
            _format_number(item.get("axis_position_rmse_m_type_correct")),
        ]
    ) + " |"


def _worst_axis_rows(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in [
            *payload.get("optimized_tracking", []),
            *payload.get("feedforward_particulate_set_match", []),
        ]
        if row.get("matched") is not False and _float_or_none(row.get("axis_angle_error_deg")) is not None
    ]
    rows.sort(key=lambda row: float(row.get("axis_angle_error_deg") or 0.0), reverse=True)
    return rows[:limit]


def _worst_axis_row(row: dict[str, Any]) -> str:
    return "| " + " | ".join(
        [
            str(row.get("path", "")),
            str(row.get("object_id", "")),
            str(row.get("gt_joint_name", "")),
            str(row.get("gt_joint_type", "")),
            str(row.get("predicted_joint_type", "")),
            "yes" if row.get("joint_type_correct") else "no",
            _format_number(row.get("axis_angle_error_deg")),
            _format_number(row.get("axis_position_error_normalized")),
        ]
    ) + " |"


def _per_joint_plot_groups(payload: dict[str, Any]) -> list[tuple[str, dict[str, float | None]]]:
    grouped: dict[tuple[str, str], dict[str, float | None]] = {}
    labels: dict[tuple[str, str], str] = {}
    for path_name, series_name in [
        ("optimized_tracking", "optimized"),
        ("feedforward_particulate_set_match", "feedforward"),
    ]:
        for row in payload.get(path_name, []):
            if row.get("matched") is False:
                continue
            object_id = str(row.get("object_id", ""))
            joint_name = str(row.get("gt_joint_name", ""))
            key = (object_id, joint_name)
            labels[key] = f"{object_id.replace('refrigerator', 'r')}:{_short_joint_name(joint_name)}"
            grouped.setdefault(key, {})[series_name] = _float_or_none(row.get("axis_angle_error_deg"))
    return [(labels[key], grouped[key]) for key in sorted(grouped)]


def _short_joint_name(name: str) -> str:
    return (
        name.replace("_door_joint", "")
        .replace("_joint", "")
        .replace("freezer", "fz")
        .replace("fridge_", "fr_")
    )


def _metric_row(label: str, item: dict[str, Any]) -> str:
    return "| " + " | ".join(
        [
            label,
            str(item.get("count", 0)),
            _format_number(item.get("mean_s")),
            _format_number(item.get("median_s")),
            _format_number(item.get("min_s")),
            _format_number(item.get("max_s")),
        ]
    ) + " |"


def _threshold_text(thresholds: dict[str, Any]) -> str:
    parts = []
    rotation = thresholds.get("rotation_threshold_rad")
    translation = thresholds.get("translation_threshold_m")
    if rotation is not None:
        parts.append(f"`rotation_threshold_rad={float(rotation):.3g}`")
    if translation is not None:
        parts.append(f"`translation_threshold_m={float(translation):.3g}`")
    return ", ".join(parts)


def _bbox_scale(manifest_path: Path) -> float:
    manifest = _read_json(manifest_path)
    lower = manifest.get("bounds", {}).get("lower", [0.0, 0.0, 0.0])
    upper = manifest.get("bounds", {}).get("upper", [1.0, 1.0, 1.0])
    scale = max(abs(float(hi) - float(lo)) for lo, hi in zip(lower, upper))
    return scale if scale > 1e-9 else 1.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _vec3(raw: Any, fallback: list[float]) -> list[float]:
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    return list(fallback)


def _scale_vec(vec: list[float], scale: float) -> list[float]:
    return [float(value) * scale for value in vec]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite(values: Any) -> list[float]:
    out = []
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            out.append(numeric)
    return out


def _mean(values: Any) -> float | None:
    values = _finite(values)
    return sum(values) / len(values) if values else None


def _median(values: Any) -> float | None:
    values = _finite(values)
    return median(values) if values else None


def _rmse_from_mse(values: Any) -> float | None:
    values = _finite(values)
    return math.sqrt(sum(values) / len(values)) if values else None


def _percent(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return None if numeric is None else 100.0 * numeric


def _metric_mean(item: Any) -> float | None:
    return _float_or_none(item.get("mean_s")) if isinstance(item, dict) else None


def _format_tsv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _format_percent(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None else f"{100.0 * numeric:.1f}%"


def _format_number(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None else f"{numeric:.4g}"


def _short_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
