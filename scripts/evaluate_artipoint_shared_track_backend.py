#!/usr/bin/env python3
"""Evaluate ArtiPoint's single-joint backend on Track2Art tracks.

This is a diagnostic adapter evaluation, not an end-to-end ArtiPoint result.
For multi-joint objects, the one predicted joint is matched to the best
type-compatible canonical GT joint and is therefore an optimistic upper bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def axis_error_deg(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr /= np.linalg.norm(a_arr)
    b_arr /= np.linalg.norm(b_arr)
    return math.degrees(math.acos(float(np.clip(abs(a_arr @ b_arr), -1.0, 1.0))))


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--canonical-gt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-csv", type=Path, required=True)
    parser.add_argument("--ours-per-object", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    gt_payload = json.loads(args.canonical_gt.read_text())
    canonical = defaultdict(dict)
    for row in gt_payload["joints"]:
        if row["setting"] != "oracle_gt_parts_analytic":
            continue
        canonical[row["object_id"]][row["joint_id"]] = {
            "joint_id": row["joint_id"],
            "joint_type": row["joint_type"],
            "gt_axis": row["gt_axis"],
        }

    with args.manifest.open(newline="") as handle:
        manifest = {row["object_id"]: row for row in csv.DictReader(handle)}
    with args.runtime_csv.open(newline="") as handle:
        runtimes = {row["object_id"]: row for row in csv.DictReader(handle)}

    rows = []
    for object_id, meta in manifest.items():
        pred = predictions.get(object_id)
        gt_joints = list(canonical.get(object_id, {}).values())
        row = {
            "object_id": object_id,
            "category": meta.get("category"),
            "gt_part_count": int(meta["gt_part_count"]),
            "complexity": meta.get("complexity_bucket") or meta.get("complexity"),
            "status": "success" if pred and gt_joints else "missing",
            "gt_joint_count": len(gt_joints),
            "predicted_joint_count": 1 if pred else 0,
            "max_joint_coverage": 1.0 / len(gt_joints) if pred and gt_joints else 0.0,
            "predicted_type": pred.get("joint_type") if pred else None,
            "matched_gt_joint": None,
            "matched_gt_type": None,
            "type_correct_best_match": False,
            "axis_error_deg_type_correct": None,
            "runtime_s": float(runtimes[object_id]["runtime_s"]) if object_id in runtimes else None,
            "tracks_input": int(runtimes[object_id]["tracks_world"]) if object_id in runtimes else None,
            "tracks_dynamic": int(runtimes[object_id]["tracks_world_filtered"]) if object_id in runtimes else None,
            "tracks_final": int(runtimes[object_id]["tracks_smoothed"]) if object_id in runtimes else None,
        }
        if pred and gt_joints:
            compatible = [joint for joint in gt_joints if joint["joint_type"] == pred["joint_type"]]
            if compatible:
                best = min(compatible, key=lambda joint: axis_error_deg(pred["axis"], joint["gt_axis"]))
                row["matched_gt_joint"] = best["joint_id"]
                row["matched_gt_type"] = best["joint_type"]
                row["type_correct_best_match"] = True
                row["axis_error_deg_type_correct"] = axis_error_deg(pred["axis"], best["gt_axis"])
            else:
                best = min(gt_joints, key=lambda joint: axis_error_deg(pred["axis"], joint["gt_axis"]))
                row["matched_gt_joint"] = best["joint_id"]
                row["matched_gt_type"] = best["joint_type"]
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_object.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def summarize(group: list[dict]) -> dict:
        successful = [row for row in group if row["status"] == "success"]
        errors = [row["axis_error_deg_type_correct"] for row in successful if row["axis_error_deg_type_correct"] is not None]
        return {
            "requested": len(group),
            "success": len(successful),
            "type_accuracy_oracle_best_match": mean([float(row["type_correct_best_match"]) for row in successful]),
            "axis_error_mean_deg_type_correct": mean(errors),
            "axis_error_median_deg_type_correct": median(errors),
            "axis_error_over_30_deg_count": sum(error > 30 for error in errors),
            "mean_max_joint_coverage": mean([row["max_joint_coverage"] for row in successful]),
            "mean_runtime_s": mean([row["runtime_s"] for row in successful if row["runtime_s"] is not None]),
        }

    summary = {
        "protocol": "Track2Art object-mask tracks -> unchanged ArtiPoint backend",
        "evaluation": "Canonical Track2Art GT joints; one predicted joint uses optimistic best-GT matching",
        "all": summarize(rows),
        "single_joint_only": summarize([row for row in rows if row["gt_joint_count"] == 1]),
        "multi_joint_only": summarize([row for row in rows if row["gt_joint_count"] > 1]),
        "by_complexity": {},
    }
    for bucket in sorted({row["complexity"] for row in rows}):
        summary["by_complexity"][bucket] = summarize([row for row in rows if row["complexity"] == bucket])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    comparison = None
    if args.ours_per_object:
        object_ids = {row["object_id"] for row in rows}
        single_ids = {row["object_id"] for row in rows if row["gt_joint_count"] == 1}
        with args.ours_per_object.open(newline="") as handle:
            ours_rows = [
                row for row in csv.DictReader(handle)
                if row["configuration"] == "hybrid + full_neural" and row["object_id"] in object_ids
            ]

        def summarize_ours(selected: list[dict]) -> dict:
            def column(name: str) -> list[float]:
                return [float(row[name]) for row in selected if row[name] not in ("", "nan")]

            return {
                "n": len(selected),
                "edge_recall_object_macro": mean(column("edge_recall")),
                "joint_type_accuracy_object_macro": mean(column("joint_type_accuracy")),
                "axis_error_mean_object_macro_deg": mean(column("type_correct_axis_error_mean_deg")),
                "axis_error_median_object_macro_deg": median(column("type_correct_axis_error_mean_deg")),
                "point_iou": mean(column("point_iou")),
                "ari": mean(column("ari")),
                "ri": mean(column("rand_index")),
            }

        comparison = {
            "warning": "ArtiPoint emits one joint and uses best-GT oracle matching on multi-joint objects; Track2Art predicts the graph. Axis aggregates are therefore diagnostic, not a strict method ranking.",
            "aligned_20": {
                "track2art_hybrid_full_neural": summarize_ours(ours_rows),
                "artipoint_backend_on_track2art_tracks": summary["all"],
            },
            "single_joint_5": {
                "track2art_hybrid_full_neural": summarize_ours([row for row in ours_rows if row["object_id"] in single_ids]),
                "artipoint_backend_on_track2art_tracks": summary["single_joint_only"],
            },
        }
        (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    lines = [
        "# Track2Art Tracks to ArtiPoint Backend",
        "",
        "This is a controlled backend diagnostic, not an end-to-end ArtiPoint result. Track2Art supplies only world-frame object-mask trajectories; slot labels and GT labels are removed. ArtiPoint then runs its unchanged filtering, reliability, DBSCAN, smoothing, and screw-fitting backend.",
        "",
        "For multi-joint objects, ArtiPoint emits one dominant joint. Matching that joint to the best canonical GT joint is an optimistic oracle upper bound; `max_joint_coverage = 1 / GT joint count` exposes this limitation.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    if comparison:
        lines.extend(["", "## Controlled Comparison", "", "```json", json.dumps(comparison, indent=2), "```"])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
