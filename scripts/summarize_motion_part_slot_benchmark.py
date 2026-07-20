#!/usr/bin/env python3
"""Summarize object-disjoint motion-part slot predictions by split and category."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments


METRICS = (
    "mean_cluster_purity",
    "weighted_cluster_purity",
    "mean_gt_coverage",
    "one_to_one_mean_iou",
    "one_to_one_mean_f1",
    "pairwise_same_part_f1",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "largest_cluster_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("benchmark_dir", type=Path)
    return parser.parse_args()


def category_for(object_id: str) -> str:
    return "refrigerator" if object_id.startswith("refrigerator") else "microwave"


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {metric: mean(float(row[metric]) for row in rows) for metric in METRICS}
    result.update(
        {
            "object_count": len(rows),
            "part_count_exact_rate": mean(float(row["part_count_exact"]) for row in rows),
            "undersegmented_object_rate": mean(float(row["undersegmented_gt_part_count"] > 0) for row in rows),
            "oversegmented_object_rate": mean(float(row["oversegmented_pred_slot_count"] > 0) for row in rows),
        }
    )
    return result


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    rows = []
    for entry in entries:
        object_id = entry["object_id"]
        split = entry["split"]
        prediction_path = benchmark_dir / split / object_id / "motion_part_tracks_slots.json"
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        tracks = payload["tracks"]
        predicted = np.asarray([int(track["part_id"]) for track in tracks], dtype=np.int64)
        labels = np.asarray([int(track["original_part_id"]) for track in tracks], dtype=np.int64)
        metrics = evaluate_slot_assignments(predicted, labels)
        segmentation = payload["motion_segmentation"]
        existence = segmentation["slot_existence_probabilities"]
        rows.append(
            {
                "object_id": object_id,
                "category": category_for(object_id),
                "split": split,
                "track_count": len(tracks),
                **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                "active_slot_count": len(segmentation["active_slots"]),
                "max_existence_probability": max(existence),
                "min_active_existence_probability": min(existence[index] for index in segmentation["active_slots"]),
            }
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f'{row["split"]}/overall'].append(row)
        groups[f'{row["split"]}/{row["category"]}'].append(row)
    summary = {name: aggregate(group) for name, group in sorted(groups.items())}
    report = {"manifest": str(manifest), "benchmark_dir": str(benchmark_dir), "per_object": rows, "summary": summary}
    (benchmark_dir / "benchmark_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(rows[0])
    with (benchmark_dir / "benchmark_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Motion-Part Slot Benchmark", "", "Object-disjoint split; metrics use simulator labels only for evaluation.", ""]
    lines.append("| Group | N | Purity | GT coverage | 1:1 IoU | Pair F1 | ARI | Exact K | Underseg | Overseg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, values in summary.items():
        lines.append(
            f'| {name} | {values["object_count"]} | {values["mean_cluster_purity"]:.3f} | '
            f'{values["mean_gt_coverage"]:.3f} | {values["one_to_one_mean_iou"]:.3f} | '
            f'{values["pairwise_same_part_f1"]:.3f} | {values["adjusted_rand_index"]:.3f} | '
            f'{values["part_count_exact_rate"]:.3f} | {values["undersegmented_object_rate"]:.3f} | '
            f'{values["oversegmented_object_rate"]:.3f} |'
        )
    (benchmark_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
