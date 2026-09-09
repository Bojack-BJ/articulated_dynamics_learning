#!/usr/bin/env python3
"""Aggregate per-scene manual real-data evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    all_joints: list[dict[str, object]] = []
    for path in sorted(args.root.glob("*/manual_gt_metrics.json")):
        payload = json.loads(path.read_text())
        segmentation = payload["segmentation"]
        kinematics = payload["kinematics"]
        per_joint = kinematics["per_joint"]
        all_joints.extend(per_joint)
        rows.append({
            "scene": path.parent.name,
            "gt_parts": segmentation["gt_part_count"],
            "pred_parts": segmentation["predicted_part_count"],
            "part_iou": segmentation["one_to_one_mean_iou"],
            "ari": segmentation["adjusted_rand_index"],
            "gt_joints": kinematics["gt_joint_count"],
            "pred_joints": kinematics["predicted_joint_count"],
            "directed_coverage": kinematics["directed_joint_coverage"],
            "type_accuracy_matched": kinematics["joint_type_accuracy"],
            "axis_error_deg_type_correct": kinematics["type_correct_axis_error_deg_mean"],
            "axis_line_bbox_revolute": kinematics["revolute_axis_line_bbox_normalized_mean"],
        })

    matched = [joint for joint in all_joints if joint.get("matched")]
    type_correct = [joint for joint in matched if joint.get("joint_type_correct")]
    axis = [float(joint["axis_angle_error_deg"]) for joint in type_correct if joint.get("axis_angle_error_deg") is not None]
    lines = [float(joint["axis_line_distance_bbox_normalized"]) for joint in type_correct if joint.get("axis_line_distance_bbox_normalized") is not None]
    gt_joint_count = sum(int(row["gt_joints"]) for row in rows)
    predicted_joint_count = sum(int(row["pred_joints"]) for row in rows)
    edge_precision = len(matched) / max(1, predicted_joint_count)
    edge_recall = len(matched) / max(1, gt_joint_count)
    aggregate = {
        "scene_count": len(rows),
        "gt_joint_count": gt_joint_count,
        "predicted_joint_count": predicted_joint_count,
        "matched_joint_count": len(matched),
        "type_correct_joint_count": len(type_correct),
        "directed_edge_precision_micro": edge_precision,
        "directed_edge_recall_micro": edge_recall,
        "directed_edge_f1_micro": 2.0 * edge_precision * edge_recall / max(1e-12, edge_precision + edge_recall),
        "directed_joint_coverage_micro": edge_recall,
        "joint_type_accuracy_matched_micro": len(type_correct) / max(1, len(matched)),
        "type_correct_joint_recall_micro": len(type_correct) / max(1, gt_joint_count),
        "axis_error_deg_type_correct_mean": float(np.mean(axis)) if axis else None,
        "axis_error_deg_type_correct_median": float(np.median(axis)) if axis else None,
        "axis_error_deg_type_correct_p90": float(np.percentile(axis, 90)) if axis else None,
        "revolute_axis_line_bbox_normalized_mean": float(np.mean(lines)) if lines else None,
        "part_iou_scene_macro": float(np.mean([float(row["part_iou"]) for row in rows])),
        "ari_scene_macro": float(np.mean([float(row["ari"]) for row in rows])),
    }

    with (args.root / "manual_gt_metrics_by_scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "manual_gt_metrics_aggregate.json").write_text(
        json.dumps({"aggregate": aggregate, "by_scene": rows}, indent=2) + "\n"
    )
    print(json.dumps({"aggregate": aggregate, "by_scene": rows}, indent=2))


if __name__ == "__main__":
    main()
