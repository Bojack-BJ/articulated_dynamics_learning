#!/usr/bin/env python3
"""Compare raw and invariant-slot Relation Heads on exactly matched joints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SEEDS = (20260831, 20260832, 20260833)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--invariant-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prediction_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["object_id"], row["joint_id"]): row
            for row in rows
            if row["source"] == "prediction"
        }


def boolean(value: str) -> bool:
    return value.lower() == "true"


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def main() -> int:
    args = parse_args()
    paired: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    coverage = []
    for seed in SEEDS:
        raw_seed_root = args.raw_root / "rotation_audit" / f"seed_{seed}"
        if not raw_seed_root.exists():
            raw_seed_root = args.raw_root / f"seed_{seed}"
        raw = prediction_rows(
            raw_seed_root / "axis_direction_samples.csv"
        )
        invariant = prediction_rows(
            args.invariant_root / f"seed_{seed}" / "equivariance_audit" / "axis_direction_samples.csv"
        )
        common = sorted(raw.keys() & invariant.keys())
        coverage.append({"seed": seed, "raw": len(raw), "invariant": len(invariant), "matched": len(common)})
        for key in common:
            old, new = raw[key], invariant[key]
            paired[key].append({
                "seed": seed,
                "raw_axis": float(old["axis_error_deg"]) if old["axis_error_deg"] else None,
                "invariant_axis": float(new["axis_error_deg"]) if new["axis_error_deg"] else None,
                "raw_type_correct": boolean(old["type_correct"]),
                "invariant_type_correct": boolean(new["type_correct"]),
                "raw_edge": boolean(old["edge_detected"]),
                "invariant_edge": boolean(new["edge_detected"]),
                "joint_type": new["joint_type"],
                "motion_magnitude": float(new["motion_magnitude"]),
                "motion_bin": new["motion_bin"],
                "observability": float(new["observability"]),
                "observability_bin": new["observability_bin"],
            })

    rows = []
    for (object_id, joint_id), samples in sorted(paired.items()):
        raw_axis = [float(row["raw_axis"]) for row in samples if row["raw_axis"] is not None]
        invariant_axis = [float(row["invariant_axis"]) for row in samples if row["invariant_axis"] is not None]
        raw_mean = mean(raw_axis)
        invariant_mean = mean(invariant_axis)
        delta = invariant_mean - raw_mean
        representative = samples[0]
        if invariant_mean > 60.0:
            status = "persistent_hard_tail"
        elif delta > 5.0:
            status = "worsened"
        elif delta < -5.0:
            status = "improved"
        else:
            status = "stable"
        rows.append({
            "object_id": object_id,
            "joint_id": joint_id,
            "joint_type": representative["joint_type"],
            "motion_magnitude": representative["motion_magnitude"],
            "motion_bin": representative["motion_bin"],
            "observability": representative["observability"],
            "observability_bin": representative["observability_bin"],
            "raw_axis_mean_deg": raw_mean,
            "invariant_axis_mean_deg": invariant_mean,
            "axis_delta_deg": delta,
            "raw_type_accuracy": mean([float(row["raw_type_correct"]) for row in samples]),
            "invariant_type_accuracy": mean([float(row["invariant_type_correct"]) for row in samples]),
            "raw_edge_recall": mean([float(row["raw_edge"]) for row in samples]),
            "invariant_edge_recall": mean([float(row["invariant_edge"]) for row in samples]),
            "status": status,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "matched_joint_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categories = {
        name: [row for row in rows if row["status"] == name]
        for name in ("persistent_hard_tail", "worsened", "improved", "stable")
    }
    report = {
        "coverage": coverage,
        "matched_joint_count": len(rows),
        "counts": {name: len(group) for name, group in categories.items()},
        "axis_delta_mean_deg": mean([float(row["axis_delta_deg"]) for row in rows]),
        "edge_regressions": [
            row for row in rows
            if float(row["invariant_edge_recall"]) < float(row["raw_edge_recall"])
        ],
        "type_regressions": [
            row for row in rows
            if float(row["invariant_type_accuracy"]) < float(row["raw_type_accuracy"])
        ],
        "groups": categories,
    }
    (args.output_dir / "hard_tail_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "matched_joint_count": len(rows),
        "counts": report["counts"],
        "edge_regression_count": len(report["edge_regressions"]),
        "type_regression_count": len(report["type_regressions"]),
        "output": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
