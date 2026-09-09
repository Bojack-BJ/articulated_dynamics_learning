#!/usr/bin/env python3
"""Compare the full SO(3) Relation Head run with previous main candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "edge_f1", "joint_type_accuracy", "revolute_type_accuracy",
    "prismatic_type_accuracy", "axis_error_deg", "axis_error_median_deg",
    "axis_error_p90_deg", "axis_line_error_normalized",
    "revolute_axis_error_deg", "prismatic_axis_error_deg",
    "illegal_graph_rate", "raw_illegal_graph_rate", "slot_ari", "slot_ri",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_run", type=Path)
    parser.add_argument("old_training_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_rows = [
        json.loads(path.read_text())["test_metrics"]
        for path in sorted(args.new_run.glob("seed_*/training_summary.json"))
    ]
    if not new_rows:
        raise FileNotFoundError("No new-run training summaries")
    rows: list[dict[str, object]] = []
    for name in (
        "cotracker_axis_trajectory_reference_frozen",
        "cotracker_axis_trajectory_reference_decoder",
        "cotracker_axis_cross_track_frozen",
        "cotracker_axis_cross_track_decoder",
    ):
        path = args.old_training_root / name / "training_summary.json"
        metrics = json.loads(path.read_text())["test_metrics"]
        rows.append({
            "model": name, "seed_count": 1,
            **{metric: metrics.get(metric) for metric in METRICS},
        })
    aggregate = {metric: _aggregate(new_rows, metric) for metric in METRICS}
    rows.append({
        "model": "vector_neuron_type20_full",
        "seed_count": len(new_rows),
        **{metric: aggregate[metric]["mean"] for metric in METRICS},
    })
    with (args.output_dir / "full_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"new_three_seed": aggregate, "comparison": rows}
    (args.output_dir / "full_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    baseline = rows[0]
    current = rows[-1]
    lines = [
        "# Full Relation Head Comparison",
        "",
        "All rows use the same PartNet-Mobility CoTracker-only manifest and test split. "
        "The new model is a three-seed mean; previous checkpoints are single runs.",
        "",
        "| Model | Edge F1 | Type Acc. | Axis Mean | Axis Median | Axis P90 | Axis-line | Illegal graph |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {_f(row['edge_f1'])} | {_f(row['joint_type_accuracy'])} | "
            f"{_f(row['axis_error_deg'])} | {_f(row['axis_error_median_deg'])} | "
            f"{_f(row['axis_error_p90_deg'])} | {_f(row['axis_line_error_normalized'])} | "
            f"{_f(row['illegal_graph_rate'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Relative to the frozen trajectory-reference checkpoint, the new model changes "
        f"Edge F1 from {_f(baseline['edge_f1'])} to {_f(current['edge_f1'])}, type accuracy "
        f"from {_f(baseline['joint_type_accuracy'])} to {_f(current['joint_type_accuracy'])}, "
        f"and median axis error from {_f(baseline['axis_error_median_deg'])} deg to "
        f"{_f(current['axis_error_median_deg'])} deg.",
        "",
        f"The tail is not solved: axis P90 changes from {_f(baseline['axis_error_p90_deg'])} "
        f"deg to {_f(current['axis_error_p90_deg'])} deg, and axis-line error changes from "
        f"{_f(baseline['axis_line_error_normalized'])} to "
        f"{_f(current['axis_line_error_normalized'])}. Report these regressions and analyze "
        "low-observability candidate failures rather than presenting only the mean/median gain.",
        "",
        "The constrained decoder makes exported graphs legal. Raw threshold graph conflicts "
        "remain separately reported and Edge F1 is computed before constrained decoding.",
    ]
    (args.output_dir / "full_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def _aggregate(rows: list[dict[str, object]], metric: str) -> dict[str, float | None]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return {
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def _f(value: object) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
