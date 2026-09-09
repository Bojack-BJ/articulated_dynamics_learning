#!/usr/bin/env python3
"""Summarize paper-style AiM mesh and kinematic metrics for one experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    axis_path = metrics_dir / "aim_axis_per_joint.json"
    axis_payload = json.loads(axis_path.read_text(encoding="utf-8")) if axis_path.exists() else {"joints": [], "summary": {}}
    mesh_rows: dict[str, dict[str, Any]] = {}
    for path in metrics_dir.glob("*/mesh_voxel_iou.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mesh_rows[path.parent.name] = payload
    joints_by_object: dict[str, list[dict[str, Any]]] = {}
    for row in axis_payload.get("joints", []):
        joints_by_object.setdefault(str(row["object_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for object_id in sorted(set(mesh_rows) | set(joints_by_object)):
        mesh = mesh_rows.get(object_id, {})
        joints = joints_by_object.get(object_id, [])
        matched = [row for row in joints if row.get("matched_joint") is not None]
        type_correct = [row for row in matched if row.get("type_correct")]
        axis = [float(row["axis_error_deg_type_correct"]) for row in type_correct if row.get("axis_error_deg_type_correct") is not None]
        line_m = [float(row["axis_line_error_m"]) for row in type_correct if row.get("axis_line_error_m") is not None]
        line_norm = [float(row["axis_line_error_bbox_normalized"]) for row in type_correct if row.get("axis_line_error_bbox_normalized") is not None]
        revolute_motion = [float(row["part_motion_abs_error"]) for row in type_correct if row.get("gt_type") == "revolute" and row.get("part_motion_abs_error") is not None]
        prismatic_motion = [float(row["part_motion_abs_error"]) for row in type_correct if row.get("gt_type") == "prismatic" and row.get("part_motion_abs_error") is not None]
        rows.append(
            {
                "object_id": object_id,
                "full_object_mesh_voxel_iou": mesh.get("full_object", {}).get("voxel_iou") if isinstance(mesh.get("full_object"), dict) else None,
                "mesh_voxel_iou": mesh.get("mean_matched_voxel_iou"),
                "predicted_mesh_count": mesh.get("predicted_mesh_count"),
                "gt_mesh_part_count": mesh.get("gt_part_count"),
                "predicted_motion_count": len(joints),
                "matched_joint_count": len(matched),
                "joint_type_accuracy": sum(bool(row.get("type_correct")) for row in matched) / len(matched) if matched else None,
                "axis_angle_error_deg_type_correct": _mean(axis),
                "axis_line_position_error_m": _mean(line_m),
                "axis_line_position_error_bbox_normalized": _mean(line_norm),
                "revolute_motion_error_rad": _mean(revolute_motion),
                "prismatic_motion_error_m": _mean(prismatic_motion),
            }
        )
    with (output / "per_object_paper_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["object_id"])
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "object_count": len(rows),
        "full_object_mesh_voxel_iou_mean": _mean([float(row["full_object_mesh_voxel_iou"]) for row in rows if row["full_object_mesh_voxel_iou"] is not None]),
        "mesh_voxel_iou_mean": _mean([float(row["mesh_voxel_iou"]) for row in rows if row["mesh_voxel_iou"] is not None]),
        "joint_type_accuracy": (
            sum(float(row["joint_type_accuracy"]) * int(row["matched_joint_count"]) for row in rows if row["joint_type_accuracy"] is not None)
            / sum(int(row["matched_joint_count"]) for row in rows if row["joint_type_accuracy"] is not None)
            if any(row["joint_type_accuracy"] is not None for row in rows) else None
        ),
        "axis_angle_error_deg_type_correct_mean": _mean([float(row["axis_angle_error_deg_type_correct"]) for row in rows if row["axis_angle_error_deg_type_correct"] is not None]),
        "axis_line_position_error_m_mean": _mean([float(row["axis_line_position_error_m"]) for row in rows if row["axis_line_position_error_m"] is not None]),
        "axis_line_position_error_bbox_normalized_mean": _mean([float(row["axis_line_position_error_bbox_normalized"]) for row in rows if row["axis_line_position_error_bbox_normalized"] is not None]),
        "revolute_motion_error_rad_mean": _mean([float(row["revolute_motion_error_rad"]) for row in rows if row["revolute_motion_error_rad"] is not None]),
        "prismatic_motion_error_m_mean": _mean([float(row["prismatic_motion_error_m"]) for row in rows if row["prismatic_motion_error_m"] is not None]),
        "metric_scope": "Official AiM TSDF meshes at 4 mm plus AiM motion.json against simulation GT; all axis metrics conditional on a Hungarian-matched, type-correct joint.",
    }
    (output / "paper_metrics_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# AiM Paper-Style Metrics",
        "",
        "Mesh IoU uses AiM's official TSDF-fused component meshes voxelized at 4 mm against the simulation visual meshes. Axis metrics are reported only for Hungarian-matched, type-correct joints.",
        "",
        "| Object | Full-object voxel IoU | Part voxel IoU | Pred./GT meshes | Type acc. | Axis angle (deg) | Axis-line (m) | Revolute motion (rad) | Prismatic motion (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_id']} | {_fmt(row['full_object_mesh_voxel_iou'])} | {_fmt(row['mesh_voxel_iou'])} | {row['predicted_mesh_count']}/{row['gt_mesh_part_count']} | {_fmt(row['joint_type_accuracy'])} | {_fmt(row['axis_angle_error_deg_type_correct'])} | {_fmt(row['axis_line_position_error_m'])} | {_fmt(row['revolute_motion_error_rad'])} | {_fmt(row['prismatic_motion_error_m'])} |"
        )
    lines += ["", "## Aggregate", "", "```json", json.dumps(aggregate, indent=2), "```", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
