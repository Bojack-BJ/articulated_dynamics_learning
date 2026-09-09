#!/usr/bin/env python3
"""Build observable motion/quality risk-coverage diagnostics for joint predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    alpha = position - low
    return values[low] * (1 - alpha) + values[high] * alpha


def summarize(rows: list[dict[str, object]], total: int, score: str) -> dict[str, object]:
    axis = [float(row["axis_error_deg"]) for row in rows if row["type_correct"]]
    return {
        "score": score,
        "retained_count": len(rows),
        "coverage": len(rows) / total,
        "edge_recall": sum(bool(row["edge_detected"]) for row in rows) / len(rows),
        "type_accuracy": sum(bool(row["type_correct"]) for row in rows) / len(rows),
        "axis_count": len(axis),
        "axis_mean_deg": sum(axis) / len(axis) if axis else None,
        "axis_median_deg": percentile(axis, 0.5),
        "axis_p90_deg": percentile(axis, 0.9),
        "axis_above_60_rate": sum(value > 60 for value in axis) / len(axis) if axis else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = [row for row in csv.DictReader(args.samples.open()) if row["source"] == "prediction"]
    rows = []
    for row in raw:
        rows.append({
            **row,
            "motion_magnitude": float(row["motion_magnitude"]),
            "observability": float(row["observability"]),
            "axis_error_deg": float(row["axis_error_deg"]),
            "type_correct": row["type_correct"] == "True",
            "edge_detected": row["edge_detected"] == "True",
        })
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for score in ("observability", "motion_magnitude"):
        ordered = sorted(rows, key=lambda row: float(row[score]), reverse=True)
        for target in (1.0, 0.9, 0.8, 0.7, 0.5):
            count = max(1, round(len(rows) * target))
            reports.append(summarize(ordered[:count], len(rows), score))
    (output / "risk_coverage.json").write_text(json.dumps(reports, indent=2) + "\n")
    with (output / "risk_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print(output / "risk_coverage.csv")


if __name__ == "__main__":
    main()
