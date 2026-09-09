#!/usr/bin/env python3
"""Import Ours relation-head joints and viewer links into the external suite."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--joint-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--setting", default="full_neural")
    parser.add_argument("--method", default="hybrid")
    args = parser.parse_args()

    joints: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.joint_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["setting"] == args.setting:
                joints[row["object_id"]].append(row)

    viewers: dict[str, str] = {}
    with args.benchmark_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == args.method:
                viewers[row["object_id"]] = row.get("viewer_html", "")

    updated = []
    for object_id, object_joints in joints.items():
        metrics_path = (
            args.suite_root / "per_object" / object_id / "ours_hybrid" / "metrics.json"
        )
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        detected = [row for row in object_joints if _bool(row["edge_detected"])]
        type_correct = [row for row in detected if _bool(row["type_correct"])]
        valid_axis = [
            row
            for row in type_correct
            if _bool(row["valid"]) and _float(row["axis_error_deg"]) is not None
        ]
        revolute_axis_line = [
            row
            for row in valid_axis
            if row["joint_type"] == "revolute"
            and _float(row["axis_line_error_bbox_normalized"]) is not None
        ]
        matches = [_joint_payload(row) for row in detected]
        data["kinematics"] = {
            "source": "full_neural_relation_head",
            "gt_joint_count": len(object_joints),
            "predicted_joint_count": len(detected),
            "matched_joint_count": len(detected),
            "joint_type_accuracy": _ratio(len(type_correct), len(detected)),
            "axis_angle_deg_type_correct": _mean(valid_axis, "axis_error_deg"),
            "revolute_axis_line_bbox": _mean(
                revolute_axis_line, "axis_line_error_bbox_normalized"
            ),
            "matches": matches,
        }
        data.setdefault("artifacts", {})
        if viewers.get(object_id):
            data["artifacts"]["viewer_html"] = viewers[object_id]
        data["artifacts"]["joint_source_csv"] = str(args.joint_csv.resolve())
        support = data.setdefault("metric_support", {})
        support["kinematics"] = (
            "full neural relation-head output; axis error is conditioned on "
            "detected edges with correct joint type"
        )
        metrics_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        updated.append(object_id)
    print(json.dumps({"updated": len(updated), "objects": sorted(updated)}, indent=2))
    return 0


def _joint_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "joint_id": row["joint_id"],
        "predicted_type": row["estimated_joint_type"],
        "gt_type": row["joint_type"],
        "type_correct": _bool(row["type_correct"]),
        "axis": _vector(row["estimated_axis"]),
        "axis_line_point": _vector(row["estimated_line_point"]),
        "gt_axis": _vector(row["gt_axis"]),
        "gt_axis_line_point": _vector(row["gt_line_point"]),
        "axis_angle_deg": _float(row["axis_error_deg"]),
        "axis_line_distance_bbox": _float(row["axis_line_error_bbox_normalized"]),
        "edge_probability": _float(row["edge_probability"]),
        "parent_slot": _integer(row["parent_slot"]),
        "child_slot": _integer(row["child_slot"]),
    }


def _vector(value: str) -> list[float]:
    return [float(item) for item in ast.literal_eval(value)]


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _integer(value: str) -> int | None:
    return int(value) if value.strip() else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_float(row[key]) for row in rows]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


if __name__ == "__main__":
    raise SystemExit(main())
