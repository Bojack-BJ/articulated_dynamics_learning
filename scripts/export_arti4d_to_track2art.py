#!/usr/bin/env python3
"""Export one Arti4D interaction and non-GT masks to a Track2Art episode."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def _camera_intrinsics(path: Path) -> dict[str, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("K:"):
            values = tuple(float(value) for value in ast.literal_eval(line.split(":", 1)[1].strip()))
            return {"fx": values[0], "fy": values[4], "cx": values[2], "cy": values[5]}
    raise ValueError(f"Missing K in {path}")


def _transform(payload: dict[str, object]) -> np.ndarray:
    translation = payload["translation"]
    rotation = payload["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation[key] for key in ("x", "y", "z", "w")]
    ).as_matrix()
    matrix[:3, 3] = [translation[key] for key in ("x", "y", "z")]
    return matrix


def _poses(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(
                [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
            ).as_matrix()
            matrix[:3, 3] = [float(row[key]) for key in ("x", "y", "z")]
            timestamps.append(float(row["timestamp"]))
            poses.append(matrix)
    return np.asarray(timestamps), poses


def official_optical_camera_pose(camera_to_world_base: np.ndarray) -> np.ndarray:
    """Apply the official ArtiPoint ``flipped=true`` camera convention."""
    image_flip = np.eye(4)
    image_flip[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    return np.asarray(camera_to_world_base, dtype=np.float64) @ image_flip


def export_scene(
    scene: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame: int,
    mask_dir: Path | None,
    mask_index_space: str = "source",
    object_instance_id: str | None = None,
    category: str = "articulated_object",
    object_mask_source: str = "artipoint_native_hand_mobile_sam",
) -> dict[str, object]:
    rgb_paths = sorted((scene / "rgb").glob("rgb_image_*.jpg"))
    depth_paths = sorted((scene / "depth").glob("depth_image_*.png"))
    if len(rgb_paths) != len(depth_paths):
        raise ValueError("RGB/depth frame counts differ")
    tf = json.loads((scene / "azure_tf.json").read_text(encoding="utf-8"))
    camera_calibration = _transform(tf["base_T_depth"]) @ _transform(tf["depth_T_rgb"])
    expected_odom = scene / "odom" / f"{scene.name}.csv"
    if not expected_odom.exists():
        odom_candidates = sorted((scene / "odom").glob("*.csv"))
        if len(odom_candidates) != 1:
            raise ValueError(f"Expected exactly one odometry CSV under {scene / 'odom'}, found {len(odom_candidates)}")
        expected_odom = odom_candidates[0]
    pose_timestamps, raw_poses = _poses(expected_odom)
    masks = (
        {int(path.stem.split("_")[-1]): path for path in mask_dir.glob("mask_*.png")}
        if mask_dir is not None else {}
    )

    frames = []
    aligned_mask_dir = output.parent / "object_masks_rgb_resolution"
    aligned_mask_dir.mkdir(parents=True, exist_ok=True)
    for local_index, source_index in enumerate(range(start_frame, end_frame + 1)):
        rgb = rgb_paths[source_index]
        depth = depth_paths[source_index]
        timestamp_ns = int(rgb.stem.split("_")[-1])
        pose_index = int(np.argmin(np.abs(pose_timestamps - timestamp_ns / 1e9)))
        frame = {
                "timestamp_s": local_index / 15.0,
                "rgb_path": str(rgb.resolve()),
                "depth_path": str(depth.resolve()),
                "camera_pose": official_optical_camera_pose(
                    raw_poses[pose_index] @ camera_calibration
                ).tolist(),
                "action_log": {"source_frame_index": source_index},
        }
        if masks:
            target_mask_index = local_index if mask_index_space == "local" else source_index
            nearest_mask_index = min(masks, key=lambda index: abs(index - target_mask_index))
            rgb_size = Image.open(rgb).size
            mask = Image.open(masks[nearest_mask_index]).convert("L")
            if mask.size != rgb_size:
                mask = mask.resize(rgb_size, resample=Image.Resampling.NEAREST)
            aligned_mask = aligned_mask_dir / f"mask_{local_index:06d}.png"
            mask.save(aligned_mask)
            frame["mask_path"] = str(aligned_mask.resolve())
        frames.append(frame)
    payload = {
        "object_instance_id": object_instance_id or f"arti4d_{scene.name}",
        "category": category,
        "camera_intrinsics": _camera_intrinsics(scene / "rgb" / "camera_info.txt"),
        "frames": frames,
        "metadata": {
            "source": "official_arti4d",
            "camera_pose_convention": "right-handed-camera-to-world",
            "source_camera_coordinate_convention": "opencv-optical-x-right-y-down-z-forward",
            "track2art_camera_coordinate_convention": "opencv-optical-x-right-y-down-z-forward",
            "depth_convention": "opencv-z-depth",
            "arti4d_flipped": True,
            "object_mask_source": object_mask_source if masks else "none_manual_annotation_required",
            "uses_gt_part_masks": False,
            "interaction_source_frames": [start_frame, end_frame],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--end-frame", type=int, default=85)
    parser.add_argument("--mask-index-space", choices=("source", "local"), default="source")
    parser.add_argument("--object-instance-id")
    parser.add_argument("--category", default="articulated_object")
    parser.add_argument("--object-mask-source", default="artipoint_native_hand_mobile_sam")
    args = parser.parse_args()
    payload = export_scene(
        args.scene.resolve(), args.output.resolve(), start_frame=args.start_frame,
        end_frame=args.end_frame, mask_dir=args.mask_dir.resolve() if args.mask_dir else None,
        mask_index_space=args.mask_index_space,
        object_instance_id=args.object_instance_id,
        category=args.category,
        object_mask_source=args.object_mask_source,
    )
    print(json.dumps({"episode": str(args.output.resolve()), "frames": len(payload["frames"])}, indent=2))


if __name__ == "__main__":
    main()
