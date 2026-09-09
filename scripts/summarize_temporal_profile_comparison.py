#!/usr/bin/env python3
"""Summarize paired low- and high-temporal-resolution axis evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SETTINGS = ("full_neural", "detected_pred_parts_analytic", "oracle_gt_parts_analytic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_report", type=Path)
    parser.add_argument("new_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract(report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for setting in SETTINGS:
        overall = report["settings"][setting]["overall"]
        end_to_end = report.get("end_to_end", {}).get(setting, {})
        result[setting] = {
            "joint_count": overall["joint_count"],
            "edge_recall": end_to_end.get("edge_recall"),
            "type_accuracy": end_to_end.get("type_accuracy"),
            "axis_mean_deg": overall.get("axis_error_mean_deg"),
            "axis_median_deg": overall.get("axis_error_median_deg"),
            "axis_above_80_count": overall.get("axis_error_above_80_count"),
            "joint_success_at_10_deg": end_to_end.get("joint_success_at_10_deg"),
            "joint_success_at_20_deg": end_to_end.get("joint_success_at_20_deg"),
        }
    return result


def main() -> int:
    args = parse_args()
    old = extract(json.loads(args.old_report.read_text(encoding="utf-8")))
    new = extract(json.loads(args.new_report.read_text(encoding="utf-8")))
    payload = {
        "comparison_scope": "same five training objects; temporal stress test, not held-out evaluation",
        "old": {"recording_fps": 15.0, "tracking_hz": 3.75, "metrics": old},
        "new": {"recording_fps": 30.0, "tracking_hz": 15.0, "metrics": new},
        "delta_new_minus_old": {
            setting: {
                key: (new[setting][key] - old[setting][key])
                for key in new[setting]
                if isinstance(new[setting][key], (int, float))
                and isinstance(old[setting][key], (int, float))
            }
            for setting in SETTINGS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
