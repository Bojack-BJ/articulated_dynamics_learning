#!/usr/bin/env python3
"""Convert propagated SAM part labels into a calibrated tracking episode."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--camera-pose-order",
        default=None,
        help="Comma-separated calibration-pose indices corresponding to view_0,view_1,...",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_path = scene_dir / "pseudo_parts_sam2" / "episode.pseudo_parts.json"
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    with (scene_dir / "calibrate.pkl").open("rb") as stream:
        source_poses = pickle.load(stream)

    view_count = len(pseudo["frames"][0]["rgb_paths_by_view"])
    intrinsics = []
    for matrix in metadata["intrinsics"][:view_count]:
        intrinsics.append({"fx": matrix[0][0], "fy": matrix[1][1], "cx": matrix[0][2], "cy": matrix[1][2]})
    opencv_to_internal = np.diag([1.0, -1.0, 1.0, 1.0])
    if args.camera_pose_order is None:
        pose_order = list(range(view_count))
    else:
        pose_order = [int(value.strip()) for value in args.camera_pose_order.split(",")]
        if len(pose_order) != view_count or sorted(pose_order) != list(range(view_count)):
            raise ValueError(
                f"--camera-pose-order must be a permutation of 0..{view_count - 1}; got {pose_order}"
            )
    camera_poses = [
        (np.asarray(source_poses[pose_index], dtype=float) @ opencv_to_internal).tolist()
        for pose_index in pose_order
    ]

    mask_root = output_dir / "object_masks"
    frames = []
    mask_pixel_counts = []
    for frame_index, source_frame in enumerate(pseudo["frames"]):
        part_paths = source_frame.get("part_mask_paths_by_view") or []
        if len(part_paths) != view_count:
            raise ValueError(f"Frame {frame_index} has {len(part_paths)} pseudo masks, expected {view_count}")
        object_paths = []
        resolved_part_paths = []
        frame_counts = []
        for view_index, part_path in enumerate(part_paths):
            part_path = Path(part_path)
            if not part_path.is_absolute():
                part_path = pseudo_path.parent / part_path
            resolved_part_paths.append(str(part_path.resolve()))
            labels = np.asarray(Image.open(part_path))
            object_mask = labels > 0
            output_path = mask_root / f"view_{view_index}" / f"{frame_index:04d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(object_mask, 255, 0).astype(np.uint8), mode="L").save(output_path)
            object_paths.append(str(output_path))
            frame_counts.append(int(object_mask.sum()))
        mask_pixel_counts.append(frame_counts)
        rgb_paths = [str(scene_dir / "color" / str(view_index) / f"{frame_index}.png") for view_index in range(view_count)]
        depth_paths = []
        for view_index in range(view_count):
            depth_dir = scene_dir / "depth" / str(view_index)
            candidates = [depth_dir / f"{frame_index}.png", depth_dir / f"{frame_index}.npy"]
            depth_path = next((candidate for candidate in candidates if candidate.exists()), None)
            if depth_path is None:
                raise FileNotFoundError(f"Missing depth frame {frame_index} in {depth_dir}")
            depth_paths.append(str(depth_path))
        frames.append(
            {
                "timestamp_s": float(source_frame["timestamp_s"]),
                "rgb_path": rgb_paths[0],
                "depth_path": depth_paths[0],
                "mask_path": object_paths[0],
                "camera_pose": camera_poses[0],
                "rgb_paths_by_view": rgb_paths,
                "depth_paths_by_view": depth_paths,
                "mask_paths_by_view": object_paths,
                "part_mask_paths_by_view": resolved_part_paths,
                "camera_poses_by_view": camera_poses,
            }
        )

    totals = np.asarray(mask_pixel_counts, dtype=np.int64).sum(axis=1)
    sampled = np.arange(0, len(frames), max(1, args.frame_stride))
    reference_source_frame = int(sampled[np.argmax(totals[sampled])])
    episode = {
        "object_instance_id": pseudo["object_instance_id"],
        "category": pseudo["category"],
        "camera_intrinsics": intrinsics[0],
        "frames": frames,
        "metadata": {
            **pseudo.get("metadata", {}),
            "source": "phystwin-sam-pseudo-object-mask-calibrated-v1",
            "source_scene_dir": str(scene_dir),
            "camera_intrinsics_by_view": intrinsics,
            "camera_names": metadata.get("camera_names", [])[:view_count],
            "camera_pose_order": pose_order,
            "depth_convention": "z-depth",
            "depth_unit": "millimeter",
            "camera_pose_convention": "camera-to-world-forward-y-up",
            "source_camera_pose_convention": "opencv-camera-to-world-y-down",
            "fps": float(metadata["fps"]),
            "recommended_tracking": {
                "frame_stride": max(1, args.frame_stride),
                "reference_source_frame": reference_source_frame,
                "reference_sampled_frame": reference_source_frame // max(1, args.frame_stride),
            },
        },
    }
    output_path = output_dir / "episode.json"
    output_path.write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episode": str(output_path), "reference_source_frame": reference_source_frame}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
