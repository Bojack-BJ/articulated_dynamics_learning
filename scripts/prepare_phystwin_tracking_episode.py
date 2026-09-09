#!/usr/bin/env python3
"""Prepare a PhysTwin recording for safe object-mask point tracking."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--category", default="microwave")
    parser.add_argument("--object-mask-id", type=int, default=2)
    parser.add_argument("--hand-mask-ids", default="0,1")
    parser.add_argument(
        "--auto-mask-ids-from-info",
        action="store_true",
        help="Resolve per-view object and hand IDs from mask/mask_info_<view>.json.",
    )
    parser.add_argument("--views", default="0,1")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--object-erosion-px", type=int, default=4)
    parser.add_argument("--hand-exclusion-px", type=int, default=10)
    return parser.parse_args()


def _odd_filter_size(radius: int) -> int:
    return max(3, 2 * max(1, int(radius)) + 1)


def _binary_mask(path: Path) -> Image.Image:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array > 0, 255, 0).astype(np.uint8), mode="L")


def _resolve_mask_ids(
    scene_dir: Path,
    views: list[int],
    object_mask_id: int,
    hand_mask_ids: list[int],
    auto_from_info: bool,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    object_ids = {view: object_mask_id for view in views}
    hand_ids = {view: list(hand_mask_ids) for view in views}
    if not auto_from_info:
        return object_ids, hand_ids

    for view in views:
        info_path = scene_dir / "mask" / f"mask_info_{view}.json"
        labels = json.loads(info_path.read_text(encoding="utf-8"))
        view_hand_ids = sorted(int(mask_id) for mask_id, label in labels.items() if "hand" in label.lower())
        view_object_ids = sorted(
            int(mask_id) for mask_id, label in labels.items() if "hand" not in label.lower()
        )
        if len(view_object_ids) != 1:
            raise ValueError(
                f"Expected exactly one non-hand object in {info_path}, found {view_object_ids}: {labels}"
            )
        object_ids[view] = view_object_ids[0]
        hand_ids[view] = view_hand_ids
    return object_ids, hand_ids


def _safe_mask(
    object_path: Path,
    hand_paths: list[Path],
    depth_path: Path,
    object_erosion_px: int,
    hand_exclusion_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    object_raw = np.asarray(_binary_mask(object_path), dtype=np.uint8) > 0
    eroded = np.asarray(
        _binary_mask(object_path).filter(ImageFilter.MinFilter(_odd_filter_size(object_erosion_px))),
        dtype=np.uint8,
    ) > 0
    hand_union = np.zeros_like(object_raw)
    for path in hand_paths:
        if path.exists():
            hand_union |= np.asarray(_binary_mask(path), dtype=np.uint8) > 0
    hand_dilated = np.asarray(
        Image.fromarray(np.where(hand_union, 255, 0).astype(np.uint8), mode="L").filter(
            ImageFilter.MaxFilter(_odd_filter_size(hand_exclusion_px))
        ),
        dtype=np.uint8,
    ) > 0
    depth_valid = np.asarray(np.load(depth_path)) > 0
    safe = eroded & ~hand_dilated & depth_valid
    return safe, {
        "object_pixels": int(object_raw.sum()),
        "eroded_object_pixels": int(eroded.sum()),
        "excluded_hand_neighborhood_pixels": int((eroded & hand_dilated).sum()),
        "safe_pixels": int(safe.sum()),
    }


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [int(value) for value in args.views.split(",") if value.strip()]
    default_hand_ids = [int(value) for value in args.hand_mask_ids.split(",") if value.strip()]
    object_ids_by_view, hand_ids_by_view = _resolve_mask_ids(
        scene_dir,
        views,
        args.object_mask_id,
        default_hand_ids,
        args.auto_mask_ids_from_info,
    )

    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    frame_count = int(metadata["frame_num"])
    fps = float(metadata["fps"])
    intrinsics = []
    for view_index in views:
        matrix = metadata["intrinsics"][view_index]
        intrinsics.append(
            {"fx": matrix[0][0], "fy": matrix[1][1], "cx": matrix[0][2], "cy": matrix[1][2]}
        )
    with (scene_dir / "calibrate.pkl").open("rb") as stream:
        all_camera_poses = pickle.load(stream)
    # PhyTwin stores RealSense/OpenCV optical poses (+X right, +Y down,
    # +Z forward). The project backprojection convention uses +Y up.
    opencv_to_internal = np.diag([1.0, -1.0, 1.0, 1.0])
    camera_poses = [
        (np.asarray(all_camera_poses[index], dtype=float) @ opencv_to_internal).tolist()
        for index in views
    ]

    safe_root = output_dir / "safe_masks"
    audit_rows: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for frame_index in range(frame_count):
        safe_paths: list[str] = []
        per_view_stats = []
        for output_view_index, source_view_index in enumerate(views):
            object_mask_id = object_ids_by_view[source_view_index]
            hand_ids = hand_ids_by_view[source_view_index]
            object_path = scene_dir / "mask" / str(source_view_index) / str(object_mask_id) / f"{frame_index}.png"
            hand_paths = [
                scene_dir / "mask" / str(source_view_index) / str(hand_id) / f"{frame_index}.png"
                for hand_id in hand_ids
            ]
            depth_path = scene_dir / "depth" / str(source_view_index) / f"{frame_index}.npy"
            safe, stats = _safe_mask(
                object_path,
                hand_paths,
                depth_path,
                args.object_erosion_px,
                args.hand_exclusion_px,
            )
            safe_path = safe_root / f"view_{output_view_index}" / f"{frame_index:04d}.png"
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(safe, 255, 0).astype(np.uint8), mode="L").save(safe_path)
            safe_paths.append(str(safe_path))
            per_view_stats.append(
                {
                    "source_view": source_view_index,
                    "object_mask_id": object_mask_id,
                    "hand_mask_ids": hand_ids,
                    **stats,
                }
            )
        audit_rows.append(
            {
                "frame_index": frame_index,
                "sampled": frame_index % max(1, args.frame_stride) == 0,
                "safe_pixels_total": sum(int(row["safe_pixels"]) for row in per_view_stats),
                "views": per_view_stats,
            }
        )
        rgb_paths = [str(scene_dir / "color" / str(view) / f"{frame_index}.png") for view in views]
        depth_paths = [str(scene_dir / "depth" / str(view) / f"{frame_index}.npy") for view in views]
        frames.append(
            {
                "timestamp_s": frame_index / fps,
                "rgb_path": rgb_paths[0],
                "depth_path": depth_paths[0],
                "mask_path": safe_paths[0],
                "camera_pose": camera_poses[0],
                "rgb_paths_by_view": rgb_paths,
                "depth_paths_by_view": depth_paths,
                "mask_paths_by_view": safe_paths,
                "camera_poses_by_view": camera_poses,
            }
        )

    sampled_rows = [row for row in audit_rows if row["sampled"]]
    reference_row = max(sampled_rows, key=lambda row: int(row["safe_pixels_total"]))
    reference_source_frame = int(reference_row["frame_index"])
    reference_sampled_frame = reference_source_frame // max(1, args.frame_stride)
    episode = {
        "object_instance_id": scene_dir.name,
        "category": args.category,
        "camera_intrinsics": intrinsics[0],
        "frames": frames,
        "metadata": {
            "source": "phystwin-three-camera-mcap-v1",
            "source_scene_dir": str(scene_dir),
            "camera_intrinsics_by_view": intrinsics,
            "camera_names": [metadata["camera_names"][index] for index in views],
            "source_view_indices": views,
            "depth_convention": "z-depth",
            "depth_unit": "millimeter",
            "camera_pose_convention": "camera-to-world-forward-y-up",
            "source_camera_pose_convention": "opencv-camera-to-world-y-down",
            "fps": fps,
            "safe_mask": {
                "object_mask_ids_by_view": object_ids_by_view,
                "hand_mask_ids_by_view": hand_ids_by_view,
                "auto_mask_ids_from_info": args.auto_mask_ids_from_info,
                "object_erosion_px": args.object_erosion_px,
                "hand_exclusion_px": args.hand_exclusion_px,
            },
            "recommended_tracking": {
                "frame_stride": args.frame_stride,
                "reference_source_frame": reference_source_frame,
                "reference_sampled_frame": reference_sampled_frame,
            },
        },
    }
    episode_path = output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
    audit = {
        "episode_path": str(episode_path),
        "reference_source_frame": reference_source_frame,
        "reference_sampled_frame": reference_sampled_frame,
        "reference_safe_pixels": reference_row["safe_pixels_total"],
        "frames": audit_rows,
    }
    (output_dir / "seed_mask_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
