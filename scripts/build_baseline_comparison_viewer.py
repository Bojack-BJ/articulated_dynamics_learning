#!/usr/bin/env python3
"""Build a unified GT/Ours/AiM/ReArt motion-decomposition viewer."""

from __future__ import annotations

import argparse
import json

from rgbd_urdf_mvp.perception.baseline_viewer import (
    BaselineComparisonViewerBuilder,
    BaselineViewerConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_ply", help="GT-labeled reference PLY")
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--aim-run")
    parser.add_argument("--aim-metrics")
    parser.add_argument("--hybrid-tracks")
    parser.add_argument("--hybrid-metrics")
    parser.add_argument("--reart-prediction")
    parser.add_argument("--reart-metrics")
    parser.add_argument("--reart-sequence-manifest")
    parser.add_argument("--reart-status")
    parser.add_argument("--gt-frames-dir")
    parser.add_argument("--gt-mesh-episode")
    parser.add_argument("--gt-mesh-opacity", type=float, default=0.28)
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--gt-geometry-max-points", type=int, default=4_000)
    parser.add_argument("--max-trajectories", type=int, default=100)
    parser.add_argument("--timeline-steps", type=int, default=31)
    parser.add_argument("--gt-mapping-distance-ratio", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--axis-remap", default="x,y,z")
    args = parser.parse_args()
    output = BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=args.output_html,
            reference_ply=args.reference_ply,
            aim_run=args.aim_run,
            aim_metrics=args.aim_metrics,
            hybrid_tracks=args.hybrid_tracks,
            hybrid_metrics=args.hybrid_metrics,
            reart_prediction=args.reart_prediction,
            reart_metrics=args.reart_metrics,
            reart_sequence_manifest=args.reart_sequence_manifest,
            reart_status=args.reart_status,
            gt_frames_dir=args.gt_frames_dir,
            gt_mesh_episode=args.gt_mesh_episode,
            gt_mesh_opacity=args.gt_mesh_opacity,
            max_points=args.max_points,
            gt_geometry_max_points=args.gt_geometry_max_points,
            max_trajectories=args.max_trajectories,
            timeline_steps=args.timeline_steps,
            gt_mapping_distance_ratio=args.gt_mapping_distance_ratio,
            random_seed=args.random_seed,
            axis_remap=args.axis_remap,
        )
    )
    print(json.dumps({"viewer_html": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
