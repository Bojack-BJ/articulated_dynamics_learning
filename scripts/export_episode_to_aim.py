#!/usr/bin/env python3
"""Export an RGB-D episode to AiM's three-stage Blender dataset layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-frame-count", type=int, default=3)
    parser.add_argument("--motion-frame-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _as_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {matrix.shape}")
    return matrix


def aim_camera_to_world(camera_to_world_forward: Any) -> list[list[float]]:
    """Convert the recorder's +Z-forward camera frame to Blender's -Z-forward frame."""
    matrix = _as_matrix(camera_to_world_forward)
    axis_change = np.diag([1.0, 1.0, -1.0, 1.0])
    return (matrix @ axis_change).tolist()


def _frame_views(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rgb_paths = frame.get("rgb_paths_by_view") or [frame["rgb_path"]]
    depth_paths = frame.get("depth_paths_by_view") or [frame["depth_path"]]
    mask_paths = frame.get("mask_paths_by_view") or [frame["mask_path"]]
    part_mask_paths = frame.get("part_mask_paths_by_view")
    if not part_mask_paths and frame.get("part_mask_path"):
        part_mask_paths = [frame["part_mask_path"]]
    if not part_mask_paths:
        part_mask_paths = [None] * len(rgb_paths)
    poses = frame.get("camera_poses_by_view") or [frame["camera_pose"]]
    lengths = {
        len(rgb_paths),
        len(depth_paths),
        len(mask_paths),
        len(part_mask_paths),
        len(poses),
    }
    if len(lengths) != 1:
        raise ValueError(
            "RGB, depth, object-mask, part-mask, and camera-pose view counts do not match"
        )
    return [
        {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "mask_path": mask_path,
            "part_mask_path": part_mask_path,
            "camera_pose": pose,
            "view_index": view_index,
        }
        for view_index, (rgb_path, depth_path, mask_path, part_mask_path, pose) in enumerate(
            zip(
                rgb_paths,
                depth_paths,
                mask_paths,
                part_mask_paths,
                poses,
                strict=True,
            )
        )
    ]


def _aim_scan_views(
    episode_path: Path,
    metadata: dict[str, Any],
    *,
    manifest_key: str,
) -> list[dict[str, Any]]:
    protocol = metadata.get("aim_protocol") or {}
    manifest_relative = protocol.get(manifest_key)
    if metadata.get("recording_protocol") not in {"aim_style", "aim_style_fixed_end"} or not manifest_relative:
        return []
    manifest_path = episode_path.parent / str(manifest_relative)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = list(payload.get("views", []))
    if not views:
        raise ValueError(f"AiM fixed-state scan contains no views: {manifest_path}")
    fixed_joint_positions = payload.get("joint_positions", {})
    for view in views:
        if view.get("joint_positions", {}) != fixed_joint_positions:
            raise ValueError("AiM fixed-state scan joint state changed between views")
    return [
        {
            "rgb_path": view["rgb_path"],
            "depth_path": view["depth_path"],
            "mask_path": view["mask_path"],
            "camera_pose": view["camera_pose"],
            "view_index": int(view.get("view_index", index)),
        }
        for index, view in enumerate(views)
    ]


def _aim_static_scan_views(episode_path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return _aim_scan_views(
        episode_path,
        metadata,
        manifest_key="static_scan_manifest",
    )


def _write_rgba(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if rgb.size != mask.size:
        raise ValueError(f"RGB/mask size mismatch: {rgb_path} {rgb.size} vs {mask_path} {mask.size}")
    rgba = rgb.copy()
    rgba.putalpha(mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path)


def _copy_depth(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _articulated_part_ids(metadata: dict[str, Any]) -> list[int]:
    segmentation = metadata.get("part_segmentation") or {}
    return sorted(
        int(part["part_id"])
        for part in segmentation.get("parts", [])
        if part.get("role") == "articulated"
    )


def _stage_indices(frame_count: int, boundary_count: int, motion_stride: int) -> dict[str, list[int]]:
    if frame_count < 3:
        raise ValueError("AiM export requires at least three episode frames")
    if boundary_count < 1 or motion_stride < 1:
        raise ValueError("boundary-frame-count and motion-frame-stride must be positive")
    boundary_count = min(boundary_count, max(1, frame_count // 3))
    return {
        "start": list(range(boundary_count)),
        "motion": list(range(0, frame_count, motion_stride)),
        "end": list(range(frame_count - boundary_count, frame_count)),
    }


def _minimum_observations_for_train_count(test_stride: int, train_count: int) -> int:
    observations = 1
    while observations - math.ceil(observations / test_stride) < train_count:
        observations += 1
    return observations


def export_episode(
    episode_path: Path,
    output_dir: Path,
    *,
    boundary_frame_count: int = 3,
    motion_frame_stride: int = 1,
    test_stride: int = 8,
    overwrite: bool = False,
) -> Path:
    episode_path = episode_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if test_stride < 2:
        raise ValueError("test-stride must be at least 2")

    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    frames = list(payload.get("frames", []))
    if not frames:
        raise ValueError("Episode contains no frames")
    metadata = dict(payload.get("metadata", {}))
    convention = str(metadata.get("camera_pose_convention", ""))
    if convention not in {"mujoco-gl-forward", "camera-to-world-forward"}:
        raise ValueError(f"Unsupported camera pose convention for AiM export: {convention!r}")

    intrinsics = payload["camera_intrinsics"]
    first_rgb = episode_path.parent / _frame_views(frames[0])[0]["rgb_path"]
    width, _ = Image.open(first_rgb).size
    camera_angle_x = 2.0 * math.atan(width / (2.0 * float(intrinsics["fx"])))
    stages = _stage_indices(len(frames), boundary_frame_count, motion_frame_stride)
    static_scan_views = _aim_static_scan_views(episode_path, metadata)
    end_scan_views = _aim_scan_views(
        episode_path,
        metadata,
        manifest_key="end_scan_manifest",
    )
    minimum_end_observations = None
    if static_scan_views and not end_scan_views:
        # Official AiM samples ten end-stage training cameras in one optimization step.
        minimum_end_observations = _minimum_observations_for_train_count(test_stride, 10)
        if len(frames) >= minimum_end_observations and len(stages["end"]) < minimum_end_observations:
            stages["end"] = list(range(len(frames) - minimum_end_observations, len(frames)))
    max_time = max(float(frame.get("timestamp_s", 0.0)) for frame in frames)
    max_time = max(max_time, 1e-9)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "aim-blender-three-stage-v1",
        "source_episode": str(episode_path),
        "uses_part_or_joint_ground_truth": False,
        "source_camera_pose_convention": convention,
        "target_camera_pose_convention": "blender-opengl-minus-z-forward",
        "camera_angle_x": camera_angle_x,
        "aim_official_minimum_end_train_cameras": 10 if static_scan_views else None,
        "minimum_end_observations": minimum_end_observations,
        "diagnostics": {
            "part_masks": {
                "diagnostic_only": True,
                "used_by_aim": False,
                "articulated_part_ids": _articulated_part_ids(metadata),
            }
        },
        "stages": {},
    }
    for stage, frame_indices in stages.items():
        observations: list[dict[str, Any]] = []
        scan_views = (
            static_scan_views
            if stage == "start"
            else end_scan_views if stage == "end" else []
        )
        if scan_views:
            for view in scan_views:
                observations.append(
                    {
                        "frame_index": 0 if stage == "start" else len(frames) - 1,
                        "view_index": view["view_index"],
                        "time": 0.0,
                        **view,
                    }
                )
        else:
            for frame_index in frame_indices:
                frame = frames[frame_index]
                normalized_time = float(frame.get("timestamp_s", 0.0)) / max_time
                if stage == "start":
                    normalized_time = 0.0
                elif stage == "end":
                    normalized_time = 1.0
                for view in _frame_views(frame):
                    observations.append(
                        {
                            "frame_index": frame_index,
                            "view_index": view["view_index"],
                            "time": normalized_time,
                            **view,
                        }
                    )

        train_frames: list[dict[str, Any]] = []
        test_frames: list[dict[str, Any]] = []
        for observation_index, observation in enumerate(observations):
            split = test_frames if observation_index % test_stride == 0 else train_frames
            image_stem = f"{observation_index:06d}"
            rgb_source = episode_path.parent / observation["rgb_path"]
            mask_source = episode_path.parent / observation["mask_path"]
            depth_source = episode_path.parent / observation["depth_path"]
            _write_rgba(rgb_source, mask_source, output_dir / stage / "rgb" / f"{image_stem}.png")
            _copy_depth(depth_source, output_dir / stage / "depth" / f"{image_stem}.png")
            frame_payload = {
                "file_path": f"./rgb/{image_stem}",
                "time": observation["time"],
                "transform_matrix": aim_camera_to_world(observation["camera_pose"]),
                "source_frame_index": observation["frame_index"],
                "source_view_index": observation["view_index"],
            }
            part_mask_path = observation.get("part_mask_path")
            if stage == "motion" and part_mask_path:
                diagnostic_relative = Path("diagnostics/part_mask") / f"{image_stem}.png"
                part_mask_source = episode_path.parent / part_mask_path
                _copy_depth(part_mask_source, output_dir / stage / diagnostic_relative)
                frame_payload["diagnostic_part_mask_path"] = f"./{diagnostic_relative}"
            split.append(frame_payload)

        if not train_frames or not test_frames:
            raise ValueError(f"Stage {stage} must contain both train and test observations")
        for split_name, split_frames in (("train", train_frames), ("test", test_frames)):
            transform_path = output_dir / stage / f"transforms_{split_name}.json"
            transform_path.write_text(
                json.dumps({"camera_angle_x": camera_angle_x, "frames": split_frames}, indent=2),
                encoding="utf-8",
            )
        manifest["stages"][stage] = {
            "source_frame_indices": sorted(
                {int(observation["frame_index"]) for observation in observations}
            ),
            "source": (
                "aim_static_scan"
                if stage == "start" and static_scan_views
                else "aim_end_scan"
                if stage == "end" and end_scan_views
                else "episode_frames"
            ),
            "observation_count": len(observations),
            "train_count": len(train_frames),
            "test_count": len(test_frames),
            "unique_camera_count": len(
                {json.dumps(frame["transform_matrix"], sort_keys=True) for frame in train_frames + test_frames}
            ),
        }

    manifest_path = output_dir / "aim_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    manifest = export_episode(
        args.episode,
        args.output_dir,
        boundary_frame_count=args.boundary_frame_count,
        motion_frame_stride=args.motion_frame_stride,
        test_stride=args.test_stride,
        overwrite=args.overwrite,
    )
    print(json.dumps({"aim_export_manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
