#!/usr/bin/env python3
"""Audit Arti4D camera conventions using static-scene RGB-D consistency."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _transform(payload: dict[str, object]) -> np.ndarray:
    translation = payload["translation"]
    rotation = payload["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation[key] for key in ("x", "y", "z", "w")]
    ).as_matrix()
    matrix[:3, 3] = [translation[key] for key in ("x", "y", "z")]
    return matrix


def _intrinsics(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("K:"):
            values = np.asarray(ast.literal_eval(line.split(":", 1)[1].strip()))
            return values.reshape(3, 3)
    raise ValueError(f"Missing K in {path}")


def _poses(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamps = []
    poses = []
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


def _frame_timestamp(path: Path) -> float:
    raw = path.stem.split("_")[-1]
    return float(f"{raw[:10]}.{raw[10:]}")


def _cloud(
    depth_path: Path,
    rgb_path: Path,
    mask_path: Path | None,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(Image.open(depth_path), dtype=np.float64) / 1000.0
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    valid = (depth > 0.25) & (depth < 3.0)
    if mask_path is not None and mask_path.is_file():
        mask = Image.open(mask_path).convert("L").resize(
            (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
        )
        valid &= np.asarray(mask) == 0
    vv, uu = np.nonzero(valid)
    keep = np.arange(0, len(uu), max(1, stride))
    uu, vv = uu[keep], vv[keep]
    z = depth[vv, uu]
    camera = np.column_stack(
        (
            (uu - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (vv - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return world, rgb[vv, uu]


def _symmetric_nn(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return float("inf")
    forward = cKDTree(first).query(second, k=1)[0]
    backward = cKDTree(second).query(first, k=1)[0]
    return float(0.5 * (np.median(forward) + np.median(backward)))


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float):
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[indices], colors[indices]


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[30, 47, 85])
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.015)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scene = args.scene.resolve()
    rgb_paths = sorted((scene / "rgb").glob("rgb_image_*.jpg"))
    depth_paths = sorted((scene / "depth").glob("depth_image_*.png"))
    timestamps, poses = _poses(scene / "odom" / f"{scene.name}.csv")
    intrinsics = _intrinsics(scene / "depth" / "camera_info.txt")
    tf = json.loads((scene / "azure_tf.json").read_text(encoding="utf-8"))
    calibration = _transform(tf["base_T_depth"]) @ _transform(tf["depth_T_rgb"])
    flip = np.eye(4)
    flip[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()

    raw_for_frame = []
    for frame_index in args.frames:
        timestamp = _frame_timestamp(rgb_paths[frame_index])
        raw_for_frame.append(poses[int(np.argmin(np.abs(timestamps - timestamp)))])
    candidates = {
        "official": [pose @ calibration @ flip for pose in raw_for_frame],
        "official_no_flip": [pose @ calibration for pose in raw_for_frame],
        "inverse_official": [np.linalg.inv(pose @ calibration @ flip) for pose in raw_for_frame],
        "raw_pose": raw_for_frame,
        "inverse_raw_pose": [np.linalg.inv(pose) for pose in raw_for_frame],
    }

    mask_paths = {}
    if args.mask_dir:
        mask_paths = {
            int(path.stem.split("_")[-1]): path for path in args.mask_dir.glob("mask_*.png")
        }
    diagnostics = {}
    candidate_clouds = {}
    for name, candidate_poses in candidates.items():
        clouds = []
        colors = []
        for frame_index, pose in zip(args.frames, candidate_poses, strict=True):
            mask_path = mask_paths.get(frame_index)
            points, rgb = _cloud(
                depth_paths[frame_index], rgb_paths[frame_index], mask_path,
                intrinsics, pose, stride=args.stride,
            )
            clouds.append(points)
            colors.append(rgb)
        pair_errors = [
            _symmetric_nn(clouds[0], cloud) for cloud in clouds[1:]
        ]
        diagnostics[name] = {
            "median_symmetric_nn_m": float(np.median(pair_errors)),
            "pair_errors_m": pair_errors,
            "rotation_determinants": [float(np.linalg.det(pose[:3, :3])) for pose in candidate_poses],
        }
        candidate_clouds[name] = (clouds, colors)

    best_name = min(diagnostics, key=lambda name: diagnostics[name]["median_symmetric_nn_m"])
    clouds, colors = candidate_clouds[best_name]
    fused_points, fused_colors = _voxel_downsample(
        np.concatenate(clouds), np.concatenate(colors), args.voxel_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = args.output_dir / "full_scene_static_fused.ply"
    _write_ply(ply_path, fused_points, fused_colors)
    report = {
        "scene": str(scene),
        "frames": args.frames,
        "best_candidate": best_name,
        "diagnostics": diagnostics,
        "fused_scene_ply": str(ply_path.resolve()),
        "fused_point_count": int(len(fused_points)),
    }
    report_path = args.output_dir / "camera_compensation_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
