#!/usr/bin/env python3
"""Summarize the Storage-47648 AiM camera-trajectory ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = ("current_orbit", "front_loaded_orbit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _threshold(payload: dict[str, Any], ratio: float = 0.02) -> dict[str, Any]:
    return next(
        row
        for row in payload["thresholds"]
        if abs(float(row["distance_ratio_bbox"]) - ratio) <= 1e-9
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir or args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads((args.root / "protocol_validation.json").read_text(encoding="utf-8"))
    rows = []
    part_rows = []
    for variant in VARIANTS:
        object_id = f"partnet_47648_{variant}"
        native_metric = json.loads(
            (args.root / "report/raw" / object_id / "aim.json").read_text(encoding="utf-8")
        )
        metric = json.loads(
            (args.root / "report/common_reference" / f"{variant}.json").read_text(
                encoding="utf-8"
            )
        )
        status = json.loads(
            (args.root / "runs" / object_id / "run_status.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (args.root / "audit" / variant / "audit.json").read_text(encoding="utf-8")
        )
        common = _threshold(metric)
        native = _threshold(native_metric)
        common_metrics = common["covered_only_metrics"]
        rows.append(
            {
                "variant": variant,
                "status": status.get("status"),
                "moving_gaussian_count": status.get("moving_gaussian_count"),
                "predicted_component_count": status.get("predicted_component_count"),
                "predicted_part_count": common_metrics["predicted_part_count"],
                "gt_part_count": metric["gt_part_count"],
                "covered_gt_part_count_at_2pct": common_metrics["gt_part_count"],
                "point_iou": common_metrics["one_to_one_mean_iou"],
                "ari": common_metrics["adjusted_rand_index"],
                "rand_index": common_metrics["rand_index"],
                "common_reference_geometry_coverage_at_2pct": common[
                    "geometry_coverage"
                ],
                "native_view_geometry_coverage_at_2pct": native["geometry_coverage"],
                "runtime_s": status.get("runtime_s"),
                "azimuth_sweep_at_motion_end_deg": validation["camera"][variant][
                    "azimuth_sweep_at_motion_end_deg"
                ],
                "total_azimuth_sweep_deg": validation["camera"][variant][
                    "total_azimuth_sweep_deg"
                ],
            }
        )
        for part in audit["parts"]:
            part_rows.append(
                {
                    "variant": variant,
                    "gt_part_id": part["gt_part_id"],
                    "visible_frame_ratio": part.get("visible_frame_ratio"),
                    "image_centroid_displacement_px_max": part.get(
                        "image_centroid_displacement_px_max"
                    ),
                    "dynamic_ratio_within_part": part.get("dynamic_ratio_within_part"),
                    "initial_dynamic_support_fraction": part.get(
                        "nearest_assigned_fraction_of_initial_dynamic_set"
                    ),
                    "below_global_10pct": part.get(
                        "below_official_global_10pct_threshold"
                    ),
                }
            )
    for path, values in (
        (output / "variant_summary.csv", rows),
        (output / "per_part_support.csv", part_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    control, treatment = rows
    markdown = f"""# Storage-47648 Camera-Trajectory Ablation

The two recordings have identical joint trajectories
(`max_abs_difference={validation['joint_trajectory_max_abs_difference']}`) and differ
only in camera timing. Both use one nominal orbit, 500 frames at 15 Hz, the same
elevation schedule, and unchanged official AiM segmentation/RANSAC.

| Variant | Moving components | Pred./GT parts | Dynamic Gaussians | Point IoU | ARI | RI | Common-ref. coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current orbit | {control['predicted_component_count']} | {control['predicted_part_count']}/{control['gt_part_count']} | {control['moving_gaussian_count']} | {control['point_iou']:.3f} | {control['ari']:.3f} | {control['rand_index']:.3f} | {control['common_reference_geometry_coverage_at_2pct']:.3f} |
| Front-loaded orbit | {treatment['predicted_component_count']} | {treatment['predicted_part_count']}/{treatment['gt_part_count']} | {treatment['moving_gaussian_count']} | {treatment['point_iou']:.3f} | {treatment['ari']:.3f} | {treatment['rand_index']:.3f} | {treatment['common_reference_geometry_coverage_at_2pct']:.3f} |

The front-loaded trajectory covers approximately 180 degrees by the end of articulated
motion and completes the remaining hemisphere afterward. The control preserves the
existing phase-modulated orbit implementation.

This controlled run does not support the hypothesis that front-hemisphere camera
coverage during articulation is the primary cause of under-segmentation if the
front-loaded result does not increase component count or segmentation quality.
"""
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"summary": str(output / "summary.md"), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
