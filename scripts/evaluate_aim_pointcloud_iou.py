#!/usr/bin/env python3
"""Evaluate official AiM segmentation on a labeled observed point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_aim_multiframe_union,
    evaluate_aim_pointcloud_iou,
    evaluate_track_pointcloud_iou,
    read_original_part_ids,
)
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    load_kinematic_evaluation_domain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path, help="AiM segmented PLY or motion-part track JSON")
    parser.add_argument("reference_ply", type=Path, help="Labeled fused frame PLY with part_id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--prediction-format",
        choices=("auto", "aim-ply", "track-json"),
        default="auto",
    )
    parser.add_argument("--source-frame-index", type=int)
    parser.add_argument(
        "--reference-part-ids-from-tracks",
        type=Path,
        help="Restrict the reference PLY to original_part_id values represented by this track JSON",
    )
    parser.add_argument(
        "--reference-part-ids",
        type=int,
        nargs="+",
        help="Restrict the reference PLY to these semantic/original part IDs.",
    )
    parser.add_argument(
        "--kinematic-domain-manifest",
        type=Path,
        help="Method-independent raw-to-kinematic part mapping and domain selection.",
    )
    parser.add_argument(
        "--kinematic-domain",
        choices=("all", "observable"),
        default="all",
    )
    parser.add_argument("--distance-ratios", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--primary-distance-ratio", type=float, default=0.02)
    parser.add_argument(
        "--frame-pair",
        nargs=2,
        action="append",
        metavar=("PREDICTED_PLY", "REFERENCE_PLY"),
        help=(
            "Time-aligned AiM/GT PLY pair. Repeat for a multi-frame union "
            "evaluation; positional prediction/reference arguments are ignored."
        ),
    )
    parser.add_argument(
        "--component-label-ply",
        type=Path,
        help=(
            "AiM segmented_point.ply supplying component colors by Gaussian index "
            "when frame-pair prediction PLYs contain appearance SH colors."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference_part_ids and args.reference_part_ids_from_tracks:
        raise SystemExit(
            "Use either --reference-part-ids or --reference-part-ids-from-tracks"
        )
    if args.kinematic_domain_manifest and (
        args.reference_part_ids or args.reference_part_ids_from_tracks
    ):
        raise SystemExit(
            "A kinematic domain manifest replaces raw --reference-part-ids selection"
        )
    prediction_format = args.prediction_format
    if prediction_format == "auto":
        prediction_format = "track-json" if args.prediction.suffix.lower() == ".json" else "aim-ply"
    part_id_map = None
    selected_part_ids = (
        read_original_part_ids(args.reference_part_ids_from_tracks)
        if args.reference_part_ids_from_tracks
        else args.reference_part_ids
    )
    if args.kinematic_domain_manifest:
        part_id_map, selected_part_ids = load_kinematic_evaluation_domain(
            args.kinematic_domain_manifest,
            args.kinematic_domain,
        )
    common = {
        "distance_ratios": args.distance_ratios,
        "primary_distance_ratio": args.primary_distance_ratio,
        "reference_part_ids": selected_part_ids,
        "reference_part_id_map": part_id_map,
    }
    if args.frame_pair:
        result = evaluate_aim_multiframe_union(
            [(Path(prediction), Path(reference)) for prediction, reference in args.frame_pair],
            component_label_ply=args.component_label_ply,
            **common,
        )
    elif prediction_format == "track-json":
        result = evaluate_track_pointcloud_iou(
            args.prediction,
            args.reference_ply,
            source_frame_index=args.source_frame_index,
            **common,
        )
    else:
        result = evaluate_aim_pointcloud_iou(args.prediction, args.reference_ply, **common)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aim_pointcloud_iou": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
