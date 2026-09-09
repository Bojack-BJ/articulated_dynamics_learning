#!/usr/bin/env python3
"""Summarize type and axis failures from a relation-head rotation audit."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path("outputs/neural_slot_so3_pilot_v1/neural_head_final_matrix_v1")
    source = root / "no_rotation_all/seed_20260831/rotation_audit/axis_direction_samples.csv"
    output = root / "hard_tail_analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in csv.DictReader(source.open()) if row["source"] == "prediction"]
    for row in rows:
        row["axis_error_deg"] = float(row["axis_error_deg"]) if row["axis_error_deg"] else None
        row["motion_magnitude"] = float(row["motion_magnitude"])
        row["observability"] = float(row["observability"])
        row["type_correct"] = row["type_correct"] == "True"
        row["edge_detected"] = row["edge_detected"] == "True"

    failures = [
        row for row in rows
        if not row["edge_detected"] or not row["type_correct"]
        or (row["axis_error_deg"] is not None and row["axis_error_deg"] > 60.0)
    ]
    failures.sort(
        key=lambda row: (
            row["axis_error_deg"] if row["axis_error_deg"] is not None else 181.0
        ), reverse=True,
    )
    fieldnames = list(failures[0])
    with (output / "per_joint_failures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)

    type_errors = [row for row in rows if not row["type_correct"]]
    axis_tail = [
        row for row in rows
        if row["type_correct"] and row["axis_error_deg"] is not None and row["axis_error_deg"] > 60
    ]
    missed_edges = [row for row in rows if not row["edge_detected"]]

    def counts(items: list[dict], key: str) -> str:
        return ", ".join(f"{name}={count}" for name, count in sorted(Counter(x[key] for x in items).items()))

    lines = [
        "# Final VN hard-tail analysis", "",
        f"- Joints: {len(rows)}",
        f"- Missed edges: {len(missed_edges)} ({len(missed_edges)/len(rows):.1%})",
        f"- Type errors: {len(type_errors)} ({len(type_errors)/len(rows):.1%})",
        f"- Type-correct axis errors >60 deg: {len(axis_tail)} ({len(axis_tail)/len(rows):.1%})",
        "", "## Axis tail (>60 deg, type correct)", "",
        f"- Motion bins: {counts(axis_tail, 'motion_bin')}",
        f"- Observability bins: {counts(axis_tail, 'observability_bin')}",
        f"- Joint types: {counts(axis_tail, 'joint_type')}",
        f"- Categories: {counts(axis_tail, 'category')}",
        "", "## Type errors", "",
        f"- Motion bins: {counts(type_errors, 'motion_bin')}",
        f"- Observability bins: {counts(type_errors, 'observability_bin')}",
        f"- Joint types: {counts(type_errors, 'joint_type')}",
        f"- Categories: {counts(type_errors, 'category')}",
        "", "## Interpretation", "",
        "The axis tail is reported only for type-correct joints. Type errors and missed edges are kept separate so confidence gating cannot hide topology or semantic failures.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output / "summary.md")


if __name__ == "__main__":
    main()
