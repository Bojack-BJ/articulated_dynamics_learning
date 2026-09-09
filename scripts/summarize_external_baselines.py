#!/usr/bin/env python3
"""Summarize ReArt, AiM, and Hybrid on a shared object list."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_tsv", type=Path)
    parser.add_argument("--aim-raw-root", type=Path, required=True)
    parser.add_argument("--reart-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = _load_objects(args.objects_tsv)
    reart_rows = {
        row["object_id"]: row
        for row in _read_csv(args.reart_csv)
        if row.get("status") == "success"
    }
    rows: list[dict[str, Any]] = []
    for object_id, category in objects:
        for method in ("Hybrid", "AiM"):
            path = args.aim_raw_root / object_id / f"{method.lower()}.json"
            if path.is_file():
                metrics = json.loads(path.read_text(encoding="utf-8"))["primary"]["covered_only_metrics"]
                rows.append(_metric_row(object_id, category, method, metrics))
        if object_id in reart_rows:
            source = reart_rows[object_id]
            rows.append(
                {
                    "object_id": object_id,
                    "category": category,
                    "method": "ReArt",
                    "point_iou": float(source["point_iou"]),
                    "ari": float(source["ari"]),
                    "gt_part_count": int(source["gt_part_count"]),
                    "predicted_part_count": int(source["predicted_part_count"]),
                    "part_count_error": int(source["predicted_part_count"]) - int(source["gt_part_count"]),
                    "exact_part_count": int(source["predicted_part_count"]) == int(source["gt_part_count"]),
                    "undersegmented": int(source["undersegmented_gt_part_count"]) > 0,
                    "unmatched_gt_part_count": int(source["unmatched_gt_part_count"]),
                    "largest_cluster_ratio": float(source["largest_cluster_ratio"]),
                }
            )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "pilot_comparison_per_object.csv", rows)
    summaries = _summaries(rows, len(objects))
    _write_csv(output / "pilot_comparison_summary.csv", summaries)
    common_ids = {
        object_id
        for object_id, _category in objects
        if all(any(row["object_id"] == object_id and row["method"] == method for row in rows)
               for method in ("Hybrid", "AiM", "ReArt"))
    }
    paired = _summaries([row for row in rows if row["object_id"] in common_ids], len(common_ids))
    _write_report(output / "comparison_summary.md", summaries, paired, common_ids)
    return 0


def _load_objects(path: Path) -> list[tuple[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return [(row["object_id"], row["category"]) for row in csv.DictReader(stream, delimiter="\t")]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_row(object_id: str, category: str, method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "category": category,
        "method": method,
        "point_iou": float(metrics["one_to_one_mean_iou"]),
        "ari": float(metrics["adjusted_rand_index"]),
        "gt_part_count": int(metrics["gt_part_count"]),
        "predicted_part_count": int(metrics["predicted_part_count"]),
        "part_count_error": int(metrics["predicted_part_count"]) - int(metrics["gt_part_count"]),
        "exact_part_count": int(metrics["predicted_part_count"]) == int(metrics["gt_part_count"]),
        "undersegmented": int(metrics["undersegmented_gt_part_count"]) > 0,
        "unmatched_gt_part_count": int(metrics["unmatched_gt_part_count"]),
        "largest_cluster_ratio": float(metrics["largest_cluster_ratio"]),
    }


def _summaries(rows: list[dict[str, Any]], requested_count: int) -> list[dict[str, Any]]:
    output = []
    for method in ("Hybrid", "AiM", "ReArt"):
        selected = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "success_count": len(selected),
                "requested_count": requested_count,
                "point_iou_mean": _mean(selected, "point_iou"),
                "ari_mean": _mean(selected, "ari"),
                "undersegmentation_rate": (
                    statistics.fmean(float(row["undersegmented"]) for row in selected) if selected else None
                ),
                "exact_part_count_rate": (
                    statistics.fmean(float(row["exact_part_count"]) for row in selected) if selected else None
                ),
                "mean_part_count_error": _mean(selected, "part_count_error"),
                "mean_unmatched_gt_parts": _mean(selected, "unmatched_gt_part_count"),
                "largest_cluster_ratio_mean": _mean(selected, "largest_cluster_ratio"),
            }
        )
    return output


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return statistics.fmean(float(row[key]) for row in rows) if rows else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    common_ids: set[str],
) -> None:
    lines = [
        "# External Baseline Pilot Comparison",
        "",
        "The primary table reports each method on its own successful runs. The paired table uses only",
        "objects completed by all three methods, preventing silent exclusion of ReArt failures.",
        "",
        "## All Requested Objects",
        "",
        "| Method | Success | Point IoU | ARI | Exact count | Underseg. | Unmatched GT | Largest cluster |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_summary_lines(summaries))
    lines.extend(
        [
            "",
            f"## Paired Common-Success Objects ({len(common_ids)})",
            "",
            f"Objects: {', '.join(sorted(common_ids)) or 'none'}",
            "",
            "| Method | Success | Point IoU | ARI | Exact count | Underseg. | Unmatched GT | Largest cluster |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_summary_lines(paired))
    lines.extend(
        [
            "",
            "## Metric Scope",
            "",
            "- All methods use the same observed-reference-point GT domain and source-frame convention.",
            "- ReArt receives dense world-frame point clouds fused from the same shared three RGB-D views.",
            "- ReArt joint type, axis, pivot, and directed topology are unsupported by the current adapter.",
            f"- This report covers {summaries[0]['requested_count'] if summaries else 0} requested objects.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"| {row['method']} | {row['success_count']}/{row['requested_count']} | "
        f"{_fmt(row['point_iou_mean'])} | {_fmt(row['ari_mean'])} | "
        f"{_fmt(row['exact_part_count_rate'])} | {_fmt(row['undersegmentation_rate'])} | "
        f"{_fmt(row['mean_unmatched_gt_parts'])} | {_fmt(row['largest_cluster_ratio_mean'])} |"
        for row in rows
    ]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
