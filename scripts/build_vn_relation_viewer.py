#!/usr/bin/env python3
"""Render a Track2Art viewer using slot assignments from relation inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.perception.object_mask_flow_html import (
    ObjectMaskFlowHtmlBuilder,
    ObjectMaskFlowHtmlConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("relation", type=Path)
    parser.add_argument("--output-tracks", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=1200)
    args = parser.parse_args()

    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    relation = json.loads(args.relation.read_text(encoding="utf-8"))
    assignments = relation.get("track_slot_assignments", [])
    rows = [row for row in tracks.get("tracks", []) if isinstance(row, dict)]
    if len(rows) != len(assignments):
        raise ValueError(
            f"Track/assignment mismatch: tracks={len(rows)}, assignments={len(assignments)}"
        )
    for row, slot in zip(rows, assignments):
        row.setdefault("original_part_id", row.get("part_id", -1))
        row["part_id"] = int(slot)
        row["part_name"] = f"motion_slot_{int(slot)}"
    tracks["tracks"] = rows
    tracks["prediction_source"] = "vector-neuron relation checkpoint slot assignments"
    args.output_tracks.parent.mkdir(parents=True, exist_ok=True)
    args.output_tracks.write_text(json.dumps(tracks), encoding="utf-8")

    episode = tracks.get("input_episode_path") or tracks.get("episode_path")
    ObjectMaskFlowHtmlBuilder().build(
        ObjectMaskFlowHtmlConfig(
            motion_tracks=args.output_tracks,
            output_html=args.output_html,
            joint_inference=args.relation,
            mjcf_replay_episode=episode if episode and Path(episode).exists() else None,
            max_tracks=args.max_tracks,
            frame_stride=2,
            full_timeline=True,
            trail_length=16,
            color_by="pred_cluster",
        )
    )
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
