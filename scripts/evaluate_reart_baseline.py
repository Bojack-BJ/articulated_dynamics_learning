#!/usr/bin/env python3
"""Evaluate an official ReArt result on the shared observed-point GT domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import read_original_part_ids
from rgbd_urdf_mvp.benchmarks.reart_baseline import evaluate_reart_segmentation
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    load_kinematic_evaluation_domain,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_pkl", type=Path)
    parser.add_argument("reference_ply", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-part-ids-from-tracks", type=Path)
    parser.add_argument("--kinematic-domain-manifest", type=Path)
    parser.add_argument("--kinematic-domain", choices=("all", "observable"), default="all")
    parser.add_argument("--distance-ratios", type=float, nargs="+", default=(0.02, 0.05, 1.0))
    parser.add_argument("--primary-distance-ratio", type=float, default=1.0)
    args = parser.parse_args()

    if args.kinematic_domain_manifest and args.reference_part_ids_from_tracks:
        raise SystemExit(
            "A kinematic domain manifest replaces method-dependent track-ID filtering"
        )
    part_id_map = None
    reference_part_ids = (
        read_original_part_ids(args.reference_part_ids_from_tracks)
        if args.reference_part_ids_from_tracks
        else None
    )
    if args.kinematic_domain_manifest:
        part_id_map, reference_part_ids = load_kinematic_evaluation_domain(
            args.kinematic_domain_manifest,
            args.kinematic_domain,
        )
    result = evaluate_reart_segmentation(
        args.result_pkl,
        args.reference_ply,
        reference_part_ids=reference_part_ids,
        reference_part_id_map=part_id_map,
        distance_ratios=args.distance_ratios,
        primary_distance_ratio=args.primary_distance_ratio,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reart_evaluation": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
