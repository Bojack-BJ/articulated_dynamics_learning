#!/usr/bin/env python3
"""Re-evaluate external baselines on common kinematic all/observable domains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_aim_pointcloud_iou,
    evaluate_labeled_points_on_reference,
    evaluate_track_pointcloud_iou,
    read_ascii_labeled_ply,
)
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    build_kinematic_evaluation_domain,
    load_kinematic_evaluation_domain,
)
from rgbd_urdf_mvp.benchmarks.reart_baseline import evaluate_reart_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--relation-root",
        type=Path,
        default=Path("/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-union-points", type=int, default=32)
    parser.add_argument("--min-visible-frames", type=int, default=2)
    return parser.parse_args()


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    primary = result["primary"]
    covered = primary.get("covered_only_metrics") or {}
    aware = primary.get("coverage_aware_metrics") or {}
    coverage_2pct = next(
        (
            row.get("geometry_coverage")
            for row in result.get("thresholds", [])
            if abs(float(row.get("distance_ratio_bbox", -1.0)) - 0.02) < 1e-12
        ),
        None,
    )
    return {
        "point_iou": aware.get("one_to_one_mean_iou"),
        "ari": covered.get("adjusted_rand_index"),
        "ri": covered.get("rand_index"),
        "geometry_coverage": primary.get("geometry_coverage"),
        "geometry_coverage_2pct": coverage_2pct,
        "predicted_part_count": result.get("predicted_part_count"),
        "gt_part_count": result.get("gt_part_count"),
        "unmatched_gt_part_count": aware.get("unmatched_gt_part_count"),
        "missing_reference_part_ids": result.get("reference_domain_missing_part_ids", []),
    }


def _evaluate_labeled_ply(
    prediction: Path,
    reference: Path,
    part_map: dict[int, int],
    selected: list[int],
) -> dict[str, Any]:
    points, labels = read_ascii_labeled_ply(prediction, label_mode="auto")
    return evaluate_labeled_points_on_reference(
        points,
        labels,
        reference,
        reference_part_id_map=part_map,
        reference_part_ids=selected,
        distance_ratios=(0.02, 0.05, 1.0),
        primary_distance_ratio=1.0,
    )


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = root / "outputs/external_baseline_suite_v1"
    rows = list(csv.DictReader((suite / "manifest.csv").open(encoding="utf-8")))
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for manifest_row in rows:
        object_id = manifest_row["object_id"]
        object_output = output / object_id
        object_output.mkdir(parents=True, exist_ok=True)
        episode = root / "outputs/aim_style_protocol_v1_recordings" / object_id / "episode.json"
        frame_dir = episode.parent / "pointcloud_4d_partseg/frames"
        references = sorted(frame_dir.glob("frame_*.ply"))
        if not episode.exists() or not references:
            failures.append({"object_id": object_id, "method": "domain", "reason": "missing_gt_artifact"})
            continue
        episode_payload = json.loads(episode.read_text(encoding="utf-8"))
        segmentation = episode_payload.get("metadata", {}).get("part_segmentation")
        if not isinstance(segmentation, dict):
            failures.append({"object_id": object_id, "method": "domain", "reason": "missing_part_segmentation"})
            continue
        frame_labels = []
        valid_references = []
        for path in references:
            try:
                labels = read_ascii_labeled_ply(path, label_mode="part_id")[1]
            except ValueError as exc:
                if "contains no vertices" in str(exc):
                    continue
                raise
            frame_labels.append(labels)
            valid_references.append(path)
        if not frame_labels:
            failures.append({"object_id": object_id, "method": "domain", "reason": "all_gt_frames_empty"})
            continue
        # All recordings share the same MuJoCo world convention. Use the first
        # non-empty labeled interaction frame rather than the two-state mesh
        # sampling artifact, whose labels may be binary foreground IDs.
        common_reference = valid_references[0]
        domain_payload = build_kinematic_evaluation_domain(
            segmentation,
            frame_labels,
            min_union_points=args.min_union_points,
            min_visible_frames=args.min_visible_frames,
        )
        domain_payload.update({
            "object_id": object_id,
            "episode_json": str(episode),
            "common_reference_ply": str(common_reference),
            "reference_frame_count": len(valid_references),
            "empty_reference_frame_count": len(references) - len(valid_references),
        })
        domain_path = object_output / "kinematic_evaluation_domain.json"
        domain_path.write_text(json.dumps(domain_payload, indent=2) + "\n", encoding="utf-8")

        predictions = {
            "ours_hybrid": (
                "track",
                args.relation_root
                / "outputs/partnet_core_v1_training/slot_benchmark_three_methods_test/hybrid/test"
                / object_id
                / "motion_part_tracks_slots.json",
            ),
            "aim": (
                "aim",
                root / "outputs/external_baselines_v1/aim_style_aligned_runs" / object_id
                / "motion_seg_final/segmented_point.ply",
            ),
            "reart": (
                "reart",
                root / "outputs/external_baselines_v1/reart_runs_aligned_4x512" / object_id
                / object_id
                / "result.pkl",
            ),
            "dta": (
                "ply",
                suite / "per_object" / object_id / "dta/adapter/start_labeled_parts.ply",
            ),
            "artgs": (
                "ply",
                suite / "per_object" / object_id / "artgs/adapter/start_labeled_gaussians.ply",
            ),
            "videoartgs": (
                "ply",
                suite / "per_object" / object_id / "videoartgs/adapter/start_labeled_parts.ply",
            ),
            "paris": (
                "ply",
                suite / "per_object" / object_id / "paris/adapter/labeled_parts.ply",
            ),
        }
        for method, (kind, prediction) in predictions.items():
            if not prediction.exists():
                failures.append({"object_id": object_id, "method": method, "reason": "missing_prediction"})
                continue
            for domain_name in ("all", "observable"):
                part_map, selected = load_kinematic_evaluation_domain(domain_path, domain_name)
                try:
                    if kind == "track":
                        result = evaluate_track_pointcloud_iou(
                            prediction,
                            common_reference,
                            source_frame_index=0,
                            reference_part_id_map=part_map,
                            reference_part_ids=selected,
                            distance_ratios=(0.02, 0.05, 1.0),
                            primary_distance_ratio=1.0,
                        )
                    elif kind == "aim":
                        result = evaluate_aim_pointcloud_iou(
                            prediction,
                            common_reference,
                            reference_part_id_map=part_map,
                            reference_part_ids=selected,
                            distance_ratios=(0.02, 0.05, 1.0),
                            primary_distance_ratio=1.0,
                        )
                    elif kind == "reart":
                        result = evaluate_reart_segmentation(
                            prediction,
                            common_reference,
                            reference_part_id_map=part_map,
                            reference_part_ids=selected,
                            distance_ratios=(0.02, 0.05, 1.0),
                            primary_distance_ratio=1.0,
                        )
                    else:
                        result = _evaluate_labeled_ply(
                            prediction, common_reference, part_map, selected
                        )
                    result.update({
                        "object_id": object_id,
                        "method": method,
                        "kinematic_domain": domain_name,
                        "kinematic_domain_manifest": str(domain_path),
                    })
                    result_path = object_output / f"{method}_{domain_name}.json"
                    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    metric_rows.append({
                        "object_id": object_id,
                        "category": manifest_row["category"],
                        "method": method,
                        "domain": domain_name,
                        **_metric_row(result),
                    })
                except Exception as exc:  # Preserve per-method failures in a batch audit.
                    failures.append({
                        "object_id": object_id,
                        "method": method,
                        "domain": domain_name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })

    fieldnames = [
        "object_id", "category", "method", "domain", "point_iou", "ari", "ri",
        "geometry_coverage", "geometry_coverage_2pct", "predicted_part_count", "gt_part_count",
        "unmatched_gt_part_count", "missing_reference_part_ids",
    ]
    with (output / "per_object_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"objects": len(rows), "metric_rows": len(metric_rows), "failures": len(failures), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
