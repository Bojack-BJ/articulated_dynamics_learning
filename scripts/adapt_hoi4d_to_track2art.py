#!/usr/bin/env python3
"""Adapt extracted HOI4D articulation sequences to Track2Art episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Official HOI4D 2D motion-segmentation IDs are category-specific. Label 2 is
# the right hand in these categories; several categories also contain a handled
# distractor. Keep only articulated-object parts and preserve their mask IDs.
HOI4D_PART_LABELS = {
    "C3": {
        "parts": [(1, "screen", "moving"), (3, "keyboard", "base")],
        "ignored": [2],
    },
    "C4": {
        "parts": [(1, "door", "moving"), (3, "cabinet_body", "base"), (4, "drawer", "moving")],
        "ignored": [2, 5],
    },
    "C6": {
        "parts": [(1, "safe_body", "base"), (3, "door", "moving")],
        "ignored": [2, 4],
    },
    "C14": {
        "parts": [(1, "trash_can_body", "base"), (3, "lid", "moving")],
        "ignored": [2, 4],
    },
}

HOI4D_STATIC_POSE_LABEL = {
    "C3": "keyboard",
    "C4": "body",
    "C6": "box",
    "C14": "base",
}


def sequence_key(sequence_name: str) -> Path:
    fields = sequence_name.split("_")
    if len(fields) != 7:
        raise ValueError(f"Unexpected HOI4D sequence name: {sequence_name}")
    return Path(*fields)


def _metric_extrinsics(extrinsics: np.ndarray, annotation: Path, category_id: str) -> tuple[np.ndarray, float]:
    """Resolve monocular-SLAM translation scale from the annotated static part."""
    needle = HOI4D_STATIC_POSE_LABEL.get(category_id)
    if needle is None:
        raise ValueError(f"No static pose label configured for {category_id}")
    design, targets = [], []
    for pose_path in sorted((annotation / "objpose").glob("*.json")):
        payload = json.loads(pose_path.read_text())
        if not payload.get("isEffective"):
            continue
        part = next(
            (item for item in payload.get("dataList", []) if needle in str(item.get("label", "")).lower()),
            None,
        )
        if part is None:
            continue
        frame_index = int(payload["frameId"]) - 1
        if frame_index < 0 or frame_index >= len(extrinsics):
            continue
        matrix = extrinsics[frame_index]
        center = np.array([part["center"][axis] for axis in "xyz"], dtype=np.float64)
        for coordinate in range(3):
            row = np.zeros(4, dtype=np.float64)
            row[:3] = matrix[coordinate, :3]
            row[3] = matrix[coordinate, 3]
            design.append(row)
            targets.append(center[coordinate])
    if len(design) < 12:
        raise ValueError(f"Insufficient static-part poses for metric camera calibration: {annotation}")
    solution = np.linalg.lstsq(np.stack(design), np.asarray(targets), rcond=None)[0]
    scale = float(solution[3])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid HOI4D SLAM translation scale {scale} for {annotation}")
    metric = extrinsics.copy()
    metric[:, :3, 3] *= scale
    return metric, scale


def build_episode(
    sequence: Path,
    annotation_root: Path,
    *,
    frame_stride: int = 1,
    end_frame: int | None = None,
) -> dict:
    info = json.loads((sequence / "camera/recon/split_0/info.json").read_text())
    intrinsics = info.get("crop_intrinsic") or info["orig_intrinsic"]
    annotation = annotation_root / sequence_key(sequence.name)
    category_id = sequence.name.split("_")[2]
    # HOI4D reconstruction stores scale-ambiguous OpenCV world-to-camera
    # extrinsics. Calibrate their translation against the static object part.
    metric_extrinsics, slam_translation_scale = _metric_extrinsics(
        np.asarray(info["extrinsics"], dtype=np.float64), annotation, category_id
    )
    camera_to_world = np.linalg.inv(metric_extrinsics)
    stop = min(300, len(camera_to_world), end_frame if end_frame is not None else 300)
    frame_ids = range(0, stop, max(1, int(frame_stride)))
    frames = []
    for frame_id in frame_ids:
        stem = f"{frame_id:05d}"
        rgb = sequence / "images" / f"{stem}.png"
        depth = sequence / "raw_depth" / f"{stem}.png"
        part_mask = annotation / "2Dseg/mask" / f"{stem}.png"
        missing = [path for path in (rgb, depth, part_mask) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing HOI4D frame assets: {missing}")
        frames.append({
            "timestamp_s": frame_id / 30.0,
            "rgb_path": str(rgb.resolve()),
            "depth_path": str(depth.resolve()),
            "mask_path": str(part_mask.resolve()),
            "part_mask_path": str(part_mask.resolve()),
            "camera_pose": camera_to_world[frame_id].tolist(),
            "action_log": {"source_frame_index": frame_id},
        })
    label_spec = HOI4D_PART_LABELS.get(category_id)
    if label_spec is None:
        raise ValueError(
            f"No verified HOI4D 2Dseg label mapping for {category_id}; "
            "refusing to treat category-specific mask IDs as part IDs."
        )
    return {
        "object_instance_id": f"hoi4d_{sequence.name}",
        "category": category_id,
        "camera_intrinsics": {
            "width": 1920,
            "height": 1080,
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
        },
        "frames": frames,
        "metadata": {
            "source": "HOI4D",
            "source_sequence": sequence.name,
            "source_frame_range": [0, stop],
            "camera_pose_convention": "camera-to-world-forward",
            "depth_convention": "opencv-z-depth",
            "depth_scale_to_m": 0.001,
            "slam_translation_scale_to_m": slam_translation_scale,
            "part_segmentation": {
                "provider": "HOI4D-2Dseg",
                "background_part_id": 0,
                "ignored_part_ids": label_spec["ignored"],
                "parts": [
                    {"part_id": part_id, "name": name, "role": role}
                    for part_id, name, role in label_spec["parts"]
                ],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("annotation_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--end-frame",
        action="append",
        default=[],
        metavar="SEQUENCE=EXCLUSIVE_END",
        help="Trim a sequence before global-motion contamination; may be repeated.",
    )
    parser.add_argument("--sequence", action="append", default=[])
    args = parser.parse_args()
    selected = set(args.sequence)
    end_frames = {}
    for value in args.end_frame:
        name, separator, raw_end = value.rpartition("=")
        if not separator or not name:
            parser.error(f"invalid --end-frame value: {value!r}")
        end_frames[name] = int(raw_end)
    sequences = [
        path for path in sorted(args.raw_root.iterdir())
        if path.is_dir() and (not selected or path.name in selected)
    ]
    outputs = []
    for sequence in sequences:
        episode = build_episode(
            sequence,
            args.annotation_root,
            frame_stride=max(1, args.frame_stride),
            end_frame=end_frames.get(sequence.name),
        )
        destination = args.output_root / sequence.name / "episode.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(episode, indent=2) + "\n")
        outputs.append({"sequence": sequence.name, "episode": str(destination.resolve()), "frames": len(episode["frames"])})
    print(json.dumps({"episodes": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
