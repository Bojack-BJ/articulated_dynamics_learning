#!/usr/bin/env python3
"""Build an all/observable kinematic-part domain from fixed GT references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import read_ascii_labeled_ply
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    build_kinematic_evaluation_domain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_json", type=Path)
    parser.add_argument("reference_ply", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-union-points", type=int, default=32)
    parser.add_argument("--min-visible-frames", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode = json.loads(args.episode_json.expanduser().resolve().read_text(encoding="utf-8"))
    segmentation = episode.get("metadata", {}).get("part_segmentation")
    if not isinstance(segmentation, dict):
        raise SystemExit(f"Episode has no MuJoCo part segmentation: {args.episode_json}")
    frame_labels = [
        read_ascii_labeled_ply(path, label_mode="part_id")[1]
        for path in args.reference_ply
    ]
    result = build_kinematic_evaluation_domain(
        segmentation,
        frame_labels,
        min_union_points=args.min_union_points,
        min_visible_frames=args.min_visible_frames,
    )
    result.update({
        "episode_json": str(args.episode_json.expanduser().resolve()),
        "reference_ply": [str(path.expanduser().resolve()) for path in args.reference_ply],
    })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kinematic_evaluation_domain": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
