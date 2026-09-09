#!/usr/bin/env python3
"""Summarize original and rotated task accuracy from an SO(3) audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("axis_samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rotated-source",
        default="augmented_prediction:slot_and_relation_geometry",
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    type_correct = [as_bool(row["type_correct"]) for row in rows]
    edge_detected = [as_bool(row["edge_detected"]) for row in rows]
    errors = [
        float(row["axis_error_deg"])
        for row, correct in zip(rows, type_correct)
        if correct and row["axis_error_deg"]
    ]
    all_errors = [float(row["axis_error_deg"]) for row in rows if row["axis_error_deg"]]
    return {
        "count": len(rows),
        "joint_type_accuracy": statistics.fmean(type_correct) if rows else None,
        "edge_recall": statistics.fmean(edge_detected) if rows else None,
        "type_correct_axis_count": len(errors),
        "axis_mean_deg": statistics.fmean(errors) if errors else None,
        "axis_median_deg": statistics.median(errors) if errors else None,
        "axis_p75_deg": percentile(errors, 0.75) if errors else None,
        "axis_p90_deg": percentile(errors, 0.90) if errors else None,
        "all_axis_count": len(all_errors),
        "all_axis_mean_deg": statistics.fmean(all_errors) if all_errors else None,
        "all_axis_median_deg": statistics.median(all_errors) if all_errors else None,
        "all_axis_p75_deg": percentile(all_errors, 0.75) if all_errors else None,
        "all_axis_p90_deg": percentile(all_errors, 0.90) if all_errors else None,
    }


def axis(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(float(row[key]) for key in ("axis_x", "axis_y", "axis_z"))


def undirected_angle_deg(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    cosine = abs(dot / max(left_norm * right_norm, 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def summarize_rotated(
    rows: list[dict[str, str]], source: str
) -> dict[str, float | int | None]:
    gt = {
        (row["object_id"], row["joint_id"], row["rotation_index"]): axis(row)
        for row in rows
        if row["source"] == "augmented_gt"
    }
    predictions = [row for row in rows if row["source"] == source]
    errors = [
        undirected_angle_deg(
            axis(row), gt[(row["object_id"], row["joint_id"], row["rotation_index"])]
        )
        for row in predictions
    ]
    return {
        "count": len(predictions),
        "joint_type_accuracy": None,
        "edge_recall": None,
        "type_correct_axis_count": None,
        "axis_count": len(errors),
        "axis_mean_deg": statistics.fmean(errors) if errors else None,
        "axis_median_deg": statistics.median(errors) if errors else None,
        "axis_p75_deg": percentile(errors, 0.75) if errors else None,
        "axis_p90_deg": percentile(errors, 0.90) if errors else None,
    }


def main() -> int:
    args = parse_args()
    with args.axis_samples.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    original = summarize([row for row in rows if row["source"] == "prediction"])
    rotated = summarize_rotated(rows, args.rotated_source)
    delta = {}
    for key in (
        "joint_type_accuracy",
        "edge_recall",
        "axis_mean_deg",
        "axis_median_deg",
        "axis_p75_deg",
        "axis_p90_deg",
    ):
        left, right = original[key], rotated[key]
        delta[key] = None if left is None or right is None else float(right - left)
    all_axis_delta = {
        key: float(rotated[key] - original[f"all_{key}"])
        for key in ("axis_mean_deg", "axis_median_deg", "axis_p75_deg", "axis_p90_deg")
    }
    report = {
        "original": original,
        "rotated": rotated,
        "rotated_minus_original": delta,
        "rotated_minus_original_all_axis": all_axis_delta,
        "rotated_source": args.rotated_source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
