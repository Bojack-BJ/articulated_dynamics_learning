#!/usr/bin/env python3
"""Aggregate the final neural-head matrix across seeds and evaluation domains."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("outputs/neural_slot_so3_pilot_v1/neural_head_final_matrix_v1")
VARIANTS = ("no_rotation_frozen", "no_rotation_decoder", "no_rotation_all", "haar_all")
DOMAINS = ("full_test", "aligned20")
METRICS = (
    "object_count", "joint_count", "edge_f1", "joint_type_accuracy",
    "axis_error_deg", "axis_error_median_deg", "axis_error_p75_deg",
    "axis_error_p90_deg", "axis_line_error_normalized", "revolute_type_accuracy",
    "prismatic_type_accuracy", "revolute_axis_error_deg", "prismatic_axis_error_deg",
    "raw_illegal_graph_rate", "slot_ari", "slot_ri",
)


def main() -> None:
    rows = []
    for variant in VARIANTS:
        for domain in DOMAINS:
            seed_rows = []
            for path in sorted((ROOT / variant).glob(f"seed_*/{domain}_metrics.json")):
                payload = json.loads(path.read_text())
                all_row = next(row for row in payload["rows"] if row["category"] == "all")
                seed_rows.append(all_row)
            if not seed_rows:
                continue
            row = {"variant": variant, "domain": domain, "seed_count": len(seed_rows)}
            for metric in METRICS:
                values = [float(item[metric]) for item in seed_rows]
                row[metric] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            rows.append(row)

    output_csv = ROOT / "aggregate_metrics.csv"
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Neural Head Final Matrix", "", "Mean over three seeds.", ""]
    for domain in DOMAINS:
        lines.extend([
            f"## {domain}", "",
            "| Variant | Edge F1 | Type Acc. | Axis mean | median | P90 | Axis line | Slot ARI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            if row["domain"] != domain:
                continue
            lines.append(
                f"| {row['variant']} | {row['edge_f1']:.4f} | {row['joint_type_accuracy']:.4f} "
                f"| {row['axis_error_deg']:.2f} | {row['axis_error_median_deg']:.2f} "
                f"| {row['axis_error_p90_deg']:.2f} | {row['axis_line_error_normalized']:.4f} "
                f"| {row['slot_ari']:.4f} |"
            )
        lines.append("")
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(output_csv)


if __name__ == "__main__":
    main()
