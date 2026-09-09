#!/usr/bin/env python3
"""Compare current and fixed-end AiM runs under the same early budget."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-metrics-root", type=Path, required=True)
    parser.add_argument("--fixed-metrics-root", type=Path, required=True)
    parser.add_argument("--current-runs-root", type=Path, required=True)
    parser.add_argument("--fixed-runs-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _load_metrics(root: Path, object_id: str) -> dict[str, float | int]:
    payload = json.loads((root / object_id / "aim.json").read_text(encoding="utf-8"))
    metrics = payload["primary"]["covered_only_metrics"]
    return {
        "predicted_part_count": int(payload["predicted_part_count"]),
        "gt_part_count": int(payload["gt_part_count"]),
        "point_iou": float(metrics["one_to_one_mean_iou"]),
        "ari": float(metrics["adjusted_rand_index"]),
        "ri": float(metrics["rand_index"]),
        "coverage": float(payload["primary"]["geometry_coverage"]),
    }


def _status(root: Path, object_id: str) -> dict:
    path = root / object_id / "run_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    args = parse_args()
    object_ids = sorted(
        path.parent.name
        for path in args.current_metrics_root.glob("*/aim.json")
        if (args.fixed_metrics_root / path.parent.name / "aim.json").is_file()
    )
    rows = []
    for object_id in object_ids:
        current = _load_metrics(args.current_metrics_root, object_id)
        fixed = _load_metrics(args.fixed_metrics_root, object_id)
        current_status = _status(args.current_runs_root, object_id)
        fixed_status = _status(args.fixed_runs_root, object_id)
        row = {"object_id": object_id}
        for prefix, metrics, status in (
            ("current", current, current_status),
            ("fixed_end", fixed, fixed_status),
        ):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            row[f"{prefix}_runtime_s"] = status.get("runtime_s")
        row.update(
            {
                "delta_predicted_part_count": (
                    fixed["predicted_part_count"] - current["predicted_part_count"]
                ),
                "delta_point_iou": fixed["point_iou"] - current["point_iou"],
                "delta_ari": fixed["ari"] - current["ari"],
                "delta_ri": fixed["ri"] - current["ri"],
                "delta_coverage": fixed["coverage"] - current["coverage"],
                "causal_interpretation": (
                    "diagnostic_only_end_branch_not_consumed_at_8000_motion_iterations"
                ),
            }
        )
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["object_id"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
