#!/usr/bin/env python3
"""Aggregate DTA no-GT replay selections and attach provenance to suite metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root", type=Path, default=Path("outputs/external_baseline_suite_v1")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-update-metrics", action="store_true")
    args = parser.parse_args()
    root = args.suite_root.expanduser().resolve()
    output = (args.output_dir or root / "dta_replay_selection_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    object_summaries = []
    for selection_path in sorted(
        (root / "per_object").glob(
            "partnet_*/dta/adapter/replay_model_selection.json"
        )
    ):
        object_id = selection_path.parents[2].name
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selections = payload["selections"]
        counts = Counter(item["selected_type"] for item in selections)
        object_summaries.append(
            {
                "object_id": object_id,
                "joint_count": len(selections),
                "revolute_count": counts["revolute"],
                "prismatic_count": counts["prismatic"],
                "ambiguous_count": counts["ambiguous"],
                "mean_relative_margin": statistics.fmean(
                    item["selection_relative_margin"] for item in selections
                ),
                "artifact": str(selection_path),
            }
        )
        for item in selections:
            rows.append(
                {
                    "object_id": object_id,
                    "joint_index": item["joint_index"],
                    "moving_part_index": item["moving_part_index"],
                    "selected_type": item["selected_type"],
                    "best_hypothesis": item["best_hypothesis"],
                    "ambiguous": item["ambiguous"],
                    "ambiguity_reasons": ";".join(item["ambiguity_reasons"]),
                    "prismatic_residual_bbox": item[
                        "residual_normalized_by_bbox"
                    ]["prismatic"],
                    "revolute_residual_bbox": item[
                        "residual_normalized_by_bbox"
                    ]["revolute"],
                    "selection_relative_margin": item[
                        "selection_relative_margin"
                    ],
                }
            )
        if not args.no_update_metrics:
            _update_metrics(root, object_id, selection_path, selections, counts)

    counts = Counter(row["selected_type"] for row in rows)
    summary = {
        "schema": "dta-no-gt-replay-selection-summary-v1",
        "object_count": len(object_summaries),
        "joint_count": len(rows),
        "selected_type_counts": dict(counts),
        "ambiguous_rate": counts["ambiguous"] / len(rows) if rows else None,
        "median_relative_margin": (
            statistics.median(row["selection_relative_margin"] for row in rows)
            if rows
            else None
        ),
        "official_dta_baseline_replaced": False,
        "label": "DTA + no-GT replay model selection (project adapter)",
        "objects": object_summaries,
    }
    _write_csv(output / "per_joint.csv", rows)
    _write_csv(output / "per_object.csv", object_summaries)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(
        "\n".join(
            (
                "# DTA No-GT Replay Selection",
                "",
                f"- Objects: {summary['object_count']}",
                f"- Joint hypotheses: {summary['joint_count']}",
                f"- Revolute: {counts['revolute']}",
                f"- Prismatic: {counts['prismatic']}",
                f"- Ambiguous: {counts['ambiguous']}",
                f"- Ambiguous rate: {summary['ambiguous_rate']:.3f}",
                f"- Median relative replay margin: {summary['median_relative_margin']:.3f}",
                "",
                "This converted diagnostic does not replace the official DTA dual-hypothesis result.",
                "Joint-type accuracy remains pending common predicted-part to GT-joint matching.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


def _update_metrics(
    root: Path,
    object_id: str,
    selection_path: Path,
    selections: list[dict[str, Any]],
    counts: Counter,
) -> None:
    path = root / "per_object" / object_id / "dta" / "metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    kinematics = metrics.setdefault("kinematics", {})
    kinematics.update(
        {
            "support": "converted_no_gt_replay_model_selection",
            "joint_type_selection": "constrained_target_surface_replay",
            "selection_provenance": "project_adapter_not_official_dta",
            "selected_joint_count": len(selections),
            "selected_revolute_count": counts["revolute"],
            "selected_prismatic_count": counts["prismatic"],
            "ambiguous_joint_count": counts["ambiguous"],
            "joint_type_accuracy": None,
            "joint_type_accuracy_status": "pending_common_gt_joint_matching",
        }
    )
    metrics.setdefault("artifacts", {})["dta_replay_model_selection"] = str(
        selection_path
    )
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
