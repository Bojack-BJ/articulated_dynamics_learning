#!/usr/bin/env python3
"""Evaluate an ArtiPoint axis prediction against an Arti4D scene annotation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def undirected_axis_error_deg(predicted: list[float], target: list[float]) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    pred /= np.linalg.norm(pred)
    gt /= np.linalg.norm(gt)
    return math.degrees(math.acos(float(np.clip(abs(np.dot(pred, gt)), 0.0, 1.0))))


def evaluate(
    prediction: dict[str, object],
    scene: dict[str, dict[str, list[float]]],
    axis_name: str,
    gt_joint_type: str,
) -> dict[str, object]:
    gt = scene[axis_name]
    predicted_type = str(prediction["joint_type"])
    type_correct = predicted_type == gt_joint_type
    return {
        "status": "success",
        "method": "artipoint",
        "protocol": "native_arti4d",
        "axis_name": axis_name,
        "predicted_joint_type": predicted_type,
        "gt_joint_type": gt_joint_type,
        "joint_type_correct": type_correct,
        "predicted_axis": prediction["axis"],
        "gt_axis": gt["axis"],
        "axis_angle_error_deg_type_correct": (
            undirected_axis_error_deg(prediction["axis"], gt["axis"]) if type_correct else None
        ),
        "axis_line_error": None,
        "axis_line_error_reason": (
            "not_defined_for_prismatic" if gt_joint_type == "prismatic" else "bbox_scale_not_provided"
        ),
        "predicted_center": prediction.get("center"),
        "gt_axis_position": gt.get("position"),
        "frame_range": [prediction.get("start_frame"), prediction.get("end_frame")],
        "frame_count": int(prediction["end_frame"]) - int(prediction["start_frame"]) + 1,
        "gt_usage": {
            "interaction_interval": True,
            "part_labels_in_inference": False,
            "axis_or_joint_type_in_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("scene_annotation", type=Path)
    parser.add_argument("--axis-name", required=True)
    parser.add_argument("--gt-joint-type", choices=("revolute", "prismatic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text())
    scene = json.loads(args.scene_annotation.read_text())
    metrics = evaluate(prediction, scene, args.axis_name, args.gt_joint_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
