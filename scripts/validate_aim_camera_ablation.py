#!/usr/bin/env python3
"""Validate that an AiM camera ablation changes only camera timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_episode", type=Path)
    parser.add_argument("treatment_episode", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_matrix(episode: dict) -> tuple[list[str], np.ndarray]:
    frames = episode["frames"]
    names = sorted(frames[0]["action_log"]["joint_positions"])
    values = np.asarray(
        [
            [frame["action_log"]["joint_positions"][name] for name in names]
            for frame in frames
        ],
        dtype=np.float64,
    )
    return names, values


def _azimuth(episode: dict) -> np.ndarray:
    lookat = np.asarray(episode["metadata"]["lookat"], dtype=np.float64)
    camera = np.asarray([frame["camera_pose"] for frame in episode["frames"]])
    relative = camera[:, :3, 3] - lookat
    return np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))


def main() -> int:
    args = parse_args()
    control = _load(args.control_episode)
    treatment = _load(args.treatment_episode)
    control_names, control_joints = _joint_matrix(control)
    treatment_names, treatment_joints = _joint_matrix(treatment)
    if control_names != treatment_names or control_joints.shape != treatment_joints.shape:
        raise RuntimeError("Joint trajectory schemas differ")
    split = float(
        treatment["metadata"]["aim_protocol"]["interaction_motion_end_fraction"]
    )
    split_index = min(round(split * (len(treatment["frames"]) - 1)), len(treatment["frames"]) - 1)
    rows = {}
    for name, episode in (("current_orbit", control), ("front_loaded_orbit", treatment)):
        azimuth = _azimuth(episode)
        rows[name] = {
            "frame_count": len(episode["frames"]),
            "trajectory": episode["metadata"]["aim_protocol"][
                "interaction_camera_trajectory"
            ],
            "azimuth_start_deg": float(np.degrees(azimuth[0])),
            "azimuth_at_motion_end_deg": float(np.degrees(azimuth[split_index])),
            "azimuth_end_deg": float(np.degrees(azimuth[-1])),
            "azimuth_sweep_at_motion_end_deg": float(
                np.degrees(azimuth[split_index] - azimuth[0])
            ),
            "total_azimuth_sweep_deg": float(np.degrees(azimuth[-1] - azimuth[0])),
        }
    payload = {
        "joint_names": control_names,
        "joint_trajectory_max_abs_difference": float(
            np.max(np.abs(control_joints - treatment_joints))
        ),
        "motion_end_fraction": split,
        "motion_end_frame": split_index,
        "camera": rows,
        "valid": bool(
            np.max(np.abs(control_joints - treatment_joints)) <= 1e-9
            and abs(rows["front_loaded_orbit"]["azimuth_sweep_at_motion_end_deg"] - 120.0)
            <= 2.0
            and abs(rows["front_loaded_orbit"]["total_azimuth_sweep_deg"] - 360.0)
            <= 2.0
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
