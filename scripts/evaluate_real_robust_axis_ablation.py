#!/usr/bin/env python3
"""Compare legacy and robust analytic fitting on manually annotated real scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.kinematics.analytic_joint_axis import (
    AnalyticAxisConfig,
    axis_angle_error_deg,
    axis_line_distance,
    estimate_analytic_joint_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-jump-m", type=float, default=0.08)
    parser.add_argument("--huber-delta-m", type=float, default=0.02)
    return parser.parse_args()


def track_tensor(tracks: list[dict], part_id: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    selected = [track for track in tracks if int(track.get("original_part_id", -1)) == part_id]
    points = np.zeros((len(selected), frame_count, 3), dtype=np.float64)
    visible = np.zeros((len(selected), frame_count), dtype=bool)
    for row, track in enumerate(selected):
        for sample in track.get("samples", []):
            frame = int(sample.get("frame_index", -1))
            xyz = sample.get("xyz_world")
            if 0 <= frame < frame_count and sample.get("visible", False) and xyz is not None:
                points[row, frame] = xyz
                visible[row, frame] = True
    return points, visible


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.tracks.read_text(encoding="utf-8"))
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    tracks = artifact.get("tracks", [])
    frame_count = 1 + max(
        (int(sample.get("frame_index", -1)) for track in tracks for sample in track.get("samples", [])),
        default=-1,
    )
    configs = {
        "legacy": AnalyticAxisConfig(),
        "robust": AnalyticAxisConfig(
            robust_segment_fitting=True,
            max_segment_translation_jump=args.max_jump_m,
            huber_delta_m=args.huber_delta_m,
        ),
    }
    rows = []
    for joint in annotation.get("joints", []):
        parent_id = int(joint["parent_part_id"])
        child_id = int(joint["child_part_id"])
        parent = track_tensor(tracks, parent_id, frame_count)
        child = track_tensor(tracks, child_id, frame_count)
        row = {"name": joint.get("name"), "joint_type": joint["joint_type"]}
        for name, config in configs.items():
            estimate = estimate_analytic_joint_axis(
                parent[0], parent[1], child[0], child[1], joint["joint_type"], config
            )
            result = estimate.to_dict()
            result["axis_error_deg"] = (
                axis_angle_error_deg(estimate.axis, joint["axis"])
                if estimate.valid and estimate.axis is not None else None
            )
            result["axis_line_error_m"] = (
                axis_line_distance(estimate.line_point, estimate.axis, joint["pivot"], joint["axis"])
                if joint["joint_type"] == "revolute" and estimate.valid and estimate.line_point is not None
                else None
            )
            row[name] = result
        rows.append(row)
    payload = {"tracks": str(args.tracks.resolve()), "annotation": str(args.annotation.resolve()), "joints": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
