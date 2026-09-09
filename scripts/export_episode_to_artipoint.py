#!/usr/bin/env python3
"""Export an RGB-D episode into ArtiPoint's Arti4D-compatible layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def artipoint_camera_to_world(
    camera_to_world: Any,
    *,
    convention: str,
) -> np.ndarray:
    """Convert recorder poses from right/up/forward to OpenCV camera axes."""
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {matrix.shape}")
    if convention == "mujoco-gl-forward":
        # Recorder camera columns are right, up, forward (an improper basis).
        # ArtiPoint backprojects pixels as right, down, forward, so only the
        # image-y camera axis must be flipped.
        matrix = matrix @ np.diag([1.0, -1.0, 1.0, 1.0])
    elif convention not in {"camera-to-world", "right-handed-camera-to-world"}:
        raise ValueError(f"Unsupported camera pose convention: {convention}")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not np.isclose(determinant, 1.0, atol=1e-5):
        raise ValueError(f"Camera rotation is not proper after conversion: det={determinant}")
    return matrix


def camera_info(width: int, height: int, intrinsics: dict[str, float]) -> str:
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    k = (fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
    p = (fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0)
    return "\n".join(
        [
            f"width: {width}",
            f"height: {height}",
            "distortion_model: none",
            "D: (0.0, 0.0, 0.0, 0.0, 0.0)",
            f"K: ({', '.join(map(str, k))})",
            "R: (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)",
            f"P: ({', '.join(map(str, p))})",
            "binning_x: 0",
            "binning_y: 0",
            "do_rectify: False",
            "",
        ]
    )


def export_episode(episode_path: Path, output_dir: Path) -> dict[str, object]:
    episode = json.loads(episode_path.read_text())
    source_root = episode_path.parent
    frames = episode["frames"]
    source_pose_convention = str(
        (episode.get("metadata") or {}).get(
            "camera_pose_convention", "mujoco-gl-forward"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "depth", "object_masks", "odom"):
        (output_dir / name).mkdir(exist_ok=True)

    first_rgb = Image.open(source_root / frames[0]["rgb_path"])
    width, height = first_rgb.size
    info = camera_info(width, height, episode["camera_intrinsics"])
    (output_dir / "rgb" / "camera_info.txt").write_text(info)
    (output_dir / "depth" / "camera_info.txt").write_text(info)

    pose_rows = []
    for index, frame in enumerate(frames):
        timestamp_ns = 1_000_000_000 + index * 66_666_667
        stem = str(timestamp_ns)
        Image.open(source_root / frame["rgb_path"]).convert("RGB").save(
            output_dir / "rgb" / f"rgb_{stem}.jpg", quality=95
        )
        depth = Image.open(source_root / frame["depth_path"])
        depth.save(output_dir / "depth" / f"depth_{stem}.png")
        mask = Image.open(source_root / frame["mask_path"]).convert("L")
        mask.save(output_dir / "object_masks" / f"mask_{stem}.png")

        pose = artipoint_camera_to_world(
            frame["camera_pose"], convention=source_pose_convention
        )
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        pose_rows.append(
            {
                "timestamp": timestamp_ns / 1e9,
                "x": pose[0, 3],
                "y": pose[1, 3],
                "z": pose[2, 3],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
            }
        )

    scene_name = output_dir.name
    with (output_dir / "odom" / f"{scene_name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pose_rows[0]))
        writer.writeheader()
        writer.writerows(pose_rows)

    identity_tf = {
        "depth_T_rgb": {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "base_T_depth": {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    (output_dir / "identity_tf.json").write_text(json.dumps(identity_tf, indent=2) + "\n")
    (output_dir / f"{scene_name}.json").write_text("{}\n")
    with (output_dir / "matched_cues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AXIS_NAME", "CUE_START", "CUE_END", "VERIFICATION"])
        writer.writeheader()
        writer.writerow(
            {
                "AXIS_NAME": "object_interaction",
                "CUE_START": 0,
                "CUE_END": len(frames) - 1,
                "VERIFICATION": "VERIFIED",
            }
        )

    manifest = {
        "source_episode": str(episode_path.resolve()),
        "object_id": episode["object_instance_id"],
        "frame_count": len(frames),
        "protocol": "artipoint_object_mask_adapter",
        "uses_object_mask": True,
        "uses_part_mask": False,
        "uses_gt_joint_parameters": False,
        "source_camera_pose_convention": source_pose_convention,
        "camera_pose_convention": "right-handed-camera-to-world",
    }
    (output_dir / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_episode(args.episode, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
