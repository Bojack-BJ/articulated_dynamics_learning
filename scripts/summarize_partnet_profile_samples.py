#!/usr/bin/env python3
"""Summarize recording, tracking, and kinematic checks for profile samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_motion(frames: list[dict[str, Any]]) -> dict[str, Any]:
    by_joint: dict[str, list[float]] = {}
    for frame in frames:
        positions = frame.get("action_log", {}).get("joint_positions", {})
        for name, value in positions.items():
            by_joint.setdefault(str(name), []).append(float(value))
    output = {}
    for name, values in by_joint.items():
        lower, upper = min(values), max(values)
        span = upper - lower
        threshold = max(span * 0.02, 1e-6)
        onset = next((index for index, value in enumerate(values) if abs(value - values[0]) > threshold), None)
        output[name] = {
            "q_min": lower,
            "q_max": upper,
            "observed_span": span,
            "motion_onset_frame": onset,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    catalog = _load(args.catalog)
    rows = []
    for expected in catalog.get("objects", []):
        root = args.recording_root / expected["object_id"]
        pointcloud = root / "pointcloud_4d_partseg"
        episode = _load(root / "episode.json")
        tracks = _load(pointcloud / "part_tracks.json")
        joints = _load(pointcloud / "joint_inference.json")
        evaluation = _load(pointcloud / "kinematic_evaluation.json")
        rows.append(
            {
                "profile": expected["profile"],
                "object_id": expected["object_id"],
                "source_category": expected["source_category"],
                "expected_joint_count": expected["joint_count"],
                "expected_joint_types": expected["joint_types"],
                "recorded_joint_motion": _joint_motion(episode.get("frames", [])),
                "track_count": len(tracks.get("tracks", [])),
                "part_track_counts": tracks.get("part_track_counts", {}),
                "predicted_joint_count": len(joints.get("joints", [])),
                "predicted_joint_types": [row.get("joint_type") for row in joints.get("joints", [])],
                "evaluation": evaluation.get("summary", {}),
                "viewer_exists": (pointcloud / "viewer_pose_flow.html").exists(),
                "viewer_size_mb": round((pointcloud / "viewer_pose_flow.html").stat().st_size / 1e6, 2),
            }
        )
    result = {"objects": rows}
    text = json.dumps(result, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
