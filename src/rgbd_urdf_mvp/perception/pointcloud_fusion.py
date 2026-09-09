from __future__ import annotations

import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from ..core.serialization import load_episode, save_json
from ..sim.mujoco_recorder import orbit_camera_pose
from .part_segmentation import part_name_lookup


@dataclass(slots=True)
class PointCloudFusionConfig:
    episode_path: str | Path
    output_dir: str | Path | None = None
    min_depth_m: float = 0.05
    max_depth_m: float = 6.0
    pixel_stride: int = 4
    central_crop_ratio: float = 0.6
    near_depth_band_m: float = 0.35
    bbox_margin_m: float = 0.20
    voxel_size_m: float = 0.015
    allow_shared_pose_fallback: bool = True
    mask_depth_percentile: float = 0.75
    mask_depth_margin_m: float = 0.25


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1)))))
    return ordered[index]


def _read_png_depth_u16(path: Path) -> list[list[int]]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG depth loading requires Pillow for .png depth files.") from exc
    image = Image.open(path)
    width, height = image.size
    values = list(image.getdata())
    rows: list[list[int]] = []
    for row_index in range(height):
        start = row_index * width
        rows.append([int(value) for value in values[start : start + width]])
    return rows


def _read_pgm_u16(path: Path) -> list[list[int]]:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic for {path}: {magic!r}")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        if len(tokens) < 3:
            raise ValueError(f"Incomplete PGM header in {path}")

        width = int(tokens[0])
        height = int(tokens[1])
        max_value = int(tokens[2])
        if max_value > 65535:
            raise ValueError(f"Unsupported PGM max value in {path}: {max_value}")

        raw = handle.read()
        if max_value <= 255:
            values = array("B", raw)
            ints = [int(value) for value in values]
        else:
            byte_count = width * height * 2
            raw = raw[:byte_count]
            ints = []
            for index in range(0, len(raw), 2):
                ints.append((raw[index] << 8) | raw[index + 1])

        rows: list[list[int]] = []
        offset = 0
        for _ in range(height):
            rows.append(ints[offset : offset + width])
            offset += width
        return rows


def _load_depth_u16(path: Path) -> list[list[int]]:
    if path.suffix.lower() == ".pgm":
        return _read_pgm_u16(path)
    if path.suffix.lower() == ".png":
        return _read_png_depth_u16(path)
    if path.suffix.lower() == ".npy":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("NPY depth loading requires NumPy.") from exc
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D depth array in {path}, got shape {values.shape}.")
        return values.astype(np.uint16, copy=False).tolist()
    raise ValueError(f"Unsupported depth format: {path.suffix}")


def _resolve_view_intrinsics(
    default_intrinsics: dict[str, float], metadata: dict[str, object], view_index: int
) -> dict[str, float]:
    per_view = metadata.get("camera_intrinsics_by_view")
    if not isinstance(per_view, list) or view_index >= len(per_view):
        return default_intrinsics
    values = per_view[view_index]
    if not isinstance(values, dict):
        raise ValueError(f"camera_intrinsics_by_view[{view_index}] must be an object.")
    required = ("fx", "fy", "cx", "cy")
    if any(key not in values for key in required):
        raise ValueError(f"camera_intrinsics_by_view[{view_index}] is missing fx/fy/cx/cy.")
    return {key: float(values[key]) for key in required}


def _image_shape(depth_u16: list[list[int]]) -> tuple[int, int]:
    height = len(depth_u16)
    width = len(depth_u16[0]) if height else 0
    return height, width


def _camera_to_world_point(
    u_coord: int,
    v_coord: int,
    depth_m: float,
    intrinsics: dict[str, float],
    camera_pose: list[list[float]],
    depth_convention: str = "z-depth",
) -> tuple[float, float, float]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    ray_x = (float(u_coord) - cx) / fx
    image_y = (float(v_coord) - cy) / fy
    ray_y = image_y if depth_convention == "opencv-z-depth" else -image_y
    if depth_convention == "ray-length":
        ray_norm = math.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
        x_cam = depth_m * ray_x / ray_norm
        y_cam = depth_m * ray_y / ray_norm
        z_cam = depth_m / ray_norm
    else:
        x_cam = ray_x * depth_m
        y_cam = ray_y * depth_m
        z_cam = depth_m

    rotation = [row[:3] for row in camera_pose[:3]]
    translation = [camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]]
    x_world = rotation[0][0] * x_cam + rotation[0][1] * y_cam + rotation[0][2] * z_cam + translation[0]
    y_world = rotation[1][0] * x_cam + rotation[1][1] * y_cam + rotation[1][2] * z_cam + translation[1]
    z_world = rotation[2][0] * x_cam + rotation[2][1] * y_cam + rotation[2][2] * z_cam + translation[2]
    return x_world, y_world, z_world


def _camera_pose_convention(episode_metadata: dict[str, object]) -> str:
    convention = episode_metadata.get("camera_pose_convention")
    if isinstance(convention, str) and convention:
        return convention
    if episode_metadata.get("source") == "mujoco-recorder":
        return "legacy-mujoco-orbit"
    return "camera-to-world-forward"


def _depth_convention(episode_metadata: dict[str, object]) -> str:
    convention = episode_metadata.get("depth_convention")
    if isinstance(convention, str) and convention:
        return convention
    if episode_metadata.get("source") == "mujoco-recorder":
        return "z-depth"
    return "z-depth"


def _is_valid_lookat(value: object) -> bool:
    return isinstance(value, list) and len(value) == 3


def _correct_legacy_mujoco_pose(
    pose: list[list[float]],
    lookat: list[float],
) -> list[list[float]]:
    corrected = [[float(v) for v in row] for row in pose]
    for index in range(3):
        corrected[index][2] = -corrected[index][2]
        corrected[index][3] = 2.0 * float(lookat[index]) - corrected[index][3]
    return corrected


def _resolve_view_depth_paths(frame, episode_root: Path) -> list[Path]:
    if frame.depth_paths_by_view:
        return [episode_root / depth_path for depth_path in frame.depth_paths_by_view]
    return [episode_root / frame.depth_path]


def _resolve_view_mask_paths(frame, episode_root: Path) -> list[Path]:
    if frame.mask_paths_by_view:
        return [episode_root / mask_path for mask_path in frame.mask_paths_by_view]
    if frame.part_mask_paths_by_view:
        return [episode_root / mask_path for mask_path in frame.part_mask_paths_by_view]
    if frame.mask_path:
        return [episode_root / frame.mask_path]
    if frame.part_mask_path:
        return [episode_root / frame.part_mask_path]
    return []


def _resolve_view_part_mask_paths(frame, episode_root: Path) -> list[Path]:
    if frame.part_mask_paths_by_view:
        return [episode_root / mask_path for mask_path in frame.part_mask_paths_by_view]
    if frame.part_mask_path:
        return [episode_root / frame.part_mask_path]
    return []


def _resolve_view_camera_poses(frame, episode_metadata: dict[str, object], view_count: int) -> tuple[list[list[list[float]]], str]:
    convention = _camera_pose_convention(episode_metadata)
    lookat = episode_metadata.get("lookat")

    if frame.camera_poses_by_view:
        if convention == "legacy-mujoco-orbit" and _is_valid_lookat(lookat):
            return [
                _correct_legacy_mujoco_pose(pose, [float(value) for value in lookat])
                for pose in frame.camera_poses_by_view
            ], "legacy-recorded-per-view-corrected"
        return frame.camera_poses_by_view, "recorded-per-view"
    if view_count == 1:
        if convention == "legacy-mujoco-orbit" and _is_valid_lookat(lookat):
            return [_correct_legacy_mujoco_pose(frame.camera_pose, [float(value) for value in lookat])], "legacy-single-view-corrected"
        return [frame.camera_pose], "single-view"

    camera_distance = episode_metadata.get("camera_distance")
    camera_elevation_deg = episode_metadata.get("camera_elevation_deg")
    azimuths = episode_metadata.get("camera_azimuths_deg")
    if (
        isinstance(lookat, list)
        and len(lookat) == 3
        and camera_distance is not None
        and camera_elevation_deg is not None
        and isinstance(azimuths, list)
        and len(azimuths) >= view_count
    ):
        poses = [
            orbit_camera_pose(
                lookat=[float(value) for value in lookat],
                distance=float(camera_distance),
                azimuth_deg=float(azimuths[index]),
                elevation_deg=float(camera_elevation_deg),
            )
            for index in range(view_count)
        ]
        return poses, "reconstructed-from-metadata"

    return [frame.camera_pose for _ in range(view_count)], "shared-central-fallback"


def _central_crop_bounds(width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    ratio = min(max(ratio, 0.1), 1.0)
    crop_width = max(1, int(round(width * ratio)))
    crop_height = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return left, top, left + crop_width, top + crop_height


def _bootstrap_candidates(
    depth_u16: list[list[int]],
    intrinsics: dict[str, float],
    camera_pose: list[list[float]],
    config: PointCloudFusionConfig,
    depth_convention: str,
) -> list[tuple[float, float, float]]:
    height, width = _image_shape(depth_u16)
    left, top, right, bottom = _central_crop_bounds(width, height, config.central_crop_ratio)
    valid_depths: list[float] = []
    for v_coord in range(top, bottom):
        row = depth_u16[v_coord]
        for u_coord in range(left, right):
            depth_m = row[u_coord] / 1000.0
            if config.min_depth_m <= depth_m <= config.max_depth_m:
                valid_depths.append(depth_m)
    if not valid_depths:
        return []

    min_depth = min(valid_depths)
    near_threshold = min_depth + config.near_depth_band_m
    candidates: list[tuple[float, float, float]] = []
    stride = max(1, config.pixel_stride)
    for v_coord in range(top, bottom, stride):
        row = depth_u16[v_coord]
        for u_coord in range(left, right, stride):
            depth_m = row[u_coord] / 1000.0
            if depth_m < config.min_depth_m or depth_m > min(near_threshold, config.max_depth_m):
                continue
            candidates.append(
                _camera_to_world_point(
                    u_coord,
                    v_coord,
                    depth_m,
                    intrinsics,
                    camera_pose,
                    depth_convention=depth_convention,
                )
            )
    return candidates


def _estimate_object_bounds(
    frames,
    episode_root: Path,
    intrinsics: dict[str, float],
    episode_metadata: dict[str, object],
    config: PointCloudFusionConfig,
) -> tuple[list[float], list[float], dict[str, object]]:
    candidates: list[tuple[float, float, float]] = []
    pose_modes: set[str] = set()
    depth_convention = _depth_convention(episode_metadata)

    for frame in frames[: min(len(frames), 3)]:
        depth_paths = _resolve_view_depth_paths(frame, episode_root)
        mask_paths = _resolve_view_mask_paths(frame, episode_root)
        camera_poses, pose_mode = _resolve_view_camera_poses(frame, episode_metadata, len(depth_paths))
        pose_modes.add(pose_mode)
        for view_index, (depth_path, camera_pose) in enumerate(zip(depth_paths, camera_poses)):
            view_intrinsics = _resolve_view_intrinsics(intrinsics, episode_metadata, view_index)
            depth_u16 = _load_depth_u16(depth_path)
            if view_index < len(mask_paths):
                mask_u16 = _load_depth_u16(mask_paths[view_index])
                masked = _bootstrap_candidates_from_mask(
                    depth_u16,
                    mask_u16,
                    view_intrinsics,
                    camera_pose,
                    config,
                    depth_convention,
                )
                candidates.extend(masked)
            else:
                candidates.extend(
                    _bootstrap_candidates(
                        depth_u16,
                        view_intrinsics,
                        camera_pose,
                        config,
                        depth_convention,
                    )
                )

    if not candidates:
        center = [0.0, 0.0, 0.0]
        lower = [-1.0, -1.0, -1.0]
        upper = [1.0, 1.0, 1.0]
    else:
        xs = [point[0] for point in candidates]
        ys = [point[1] for point in candidates]
        zs = [point[2] for point in candidates]
        center = [median(xs), median(ys), median(zs)]
        lower = [
            _percentile(xs, 0.05) - config.bbox_margin_m,
            _percentile(ys, 0.05) - config.bbox_margin_m,
            _percentile(zs, 0.05) - config.bbox_margin_m,
        ]
        upper = [
            _percentile(xs, 0.95) + config.bbox_margin_m,
            _percentile(ys, 0.95) + config.bbox_margin_m,
            _percentile(zs, 0.95) + config.bbox_margin_m,
        ]

    metadata = {
        "bootstrap_candidate_count": len(candidates),
        "pose_modes": sorted(pose_modes),
        "depth_convention": depth_convention,
        "estimated_center": center,
        "estimated_bounds": {"lower": lower, "upper": upper},
    }
    return lower, upper, metadata


def _point_in_bounds(point: tuple[float, float, float], lower: list[float], upper: list[float]) -> bool:
    return all(lower[index] <= point[index] <= upper[index] for index in range(3))


def _masked_depth_upper_bound(
    depth_u16: list[list[int]],
    mask_u16: list[list[int]],
    config: PointCloudFusionConfig,
) -> float | None:
    depths_m = [
        depth_u16[v_coord][u_coord] / 1000.0
        for v_coord in range(len(depth_u16))
        for u_coord in range(len(depth_u16[0]))
        if int(mask_u16[v_coord][u_coord]) > 0
        and config.min_depth_m <= (depth_u16[v_coord][u_coord] / 1000.0) <= config.max_depth_m
    ]
    if not depths_m:
        return None
    percentile = min(max(float(config.mask_depth_percentile), 0.05), 1.0)
    return _percentile(depths_m, percentile) + max(0.0, float(config.mask_depth_margin_m))


def _bootstrap_candidates_from_mask(
    depth_u16: list[list[int]],
    mask_u16: list[list[int]],
    intrinsics: dict[str, float],
    camera_pose: list[list[float]],
    config: PointCloudFusionConfig,
    depth_convention: str,
) -> list[tuple[float, float, float]]:
    candidates: list[tuple[float, float, float]] = []
    height, width = _image_shape(depth_u16)
    upper_depth = _masked_depth_upper_bound(depth_u16, mask_u16, config)
    stride = max(1, config.pixel_stride)
    for v_coord in range(0, height, stride):
        depth_row = depth_u16[v_coord]
        mask_row = mask_u16[v_coord]
        for u_coord in range(0, width, stride):
            if int(mask_row[u_coord]) <= 0:
                continue
            depth_m = depth_row[u_coord] / 1000.0
            if depth_m < config.min_depth_m or depth_m > config.max_depth_m:
                continue
            if upper_depth is not None and depth_m > upper_depth:
                continue
            candidates.append(
                _camera_to_world_point(
                    u_coord,
                    v_coord,
                    depth_m,
                    intrinsics,
                    camera_pose,
                    depth_convention=depth_convention,
                )
            )
    return candidates


def _fuse_view_points(
    points: Iterable[tuple[float, float, float, int]],
    voxel_size_m: float,
) -> list[tuple[float, float, float, int]]:
    if voxel_size_m <= 0.0:
        return list(points)

    buckets: dict[tuple[int, int, int], list[float]] = {}
    counts: dict[tuple[int, int, int], int] = {}
    part_counts: dict[tuple[int, int, int], dict[int, int]] = {}
    for x_coord, y_coord, z_coord, part_id in points:
        key = (
            int(math.floor(x_coord / voxel_size_m)),
            int(math.floor(y_coord / voxel_size_m)),
            int(math.floor(z_coord / voxel_size_m)),
        )
        if key not in buckets:
            buckets[key] = [x_coord, y_coord, z_coord]
            counts[key] = 1
        else:
            buckets[key][0] += x_coord
            buckets[key][1] += y_coord
            buckets[key][2] += z_coord
            counts[key] += 1
        bucket_part_counts = part_counts.setdefault(key, {})
        bucket_part_counts[int(part_id)] = bucket_part_counts.get(int(part_id), 0) + 1
    fused: list[tuple[float, float, float, int]] = []
    for key, sums in buckets.items():
        count = counts[key]
        label_counts = part_counts.get(key, {})
        fused_part_id = 0
        if label_counts:
            fused_part_id = max(
                sorted(label_counts),
                key=lambda candidate: (label_counts[candidate], candidate != 0, -candidate),
            )
        fused.append((sums[0] / count, sums[1] / count, sums[2] / count, int(fused_part_id)))
    return fused


def _write_pointcloud_ply(
    path: Path,
    points: list[tuple[float, float, float, float, int, int]],
) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float time",
        "property int frame_index",
        "property ushort part_id",
        "end_header",
    ]
    for x_coord, y_coord, z_coord, time_s, frame_index, part_id in points:
        lines.append(f"{x_coord:.6f} {y_coord:.6f} {z_coord:.6f} {time_s:.6f} {frame_index} {part_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frame_ply(path: Path, points: list[tuple[float, float, float, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property ushort part_id",
        "end_header",
    ]
    for x_coord, y_coord, z_coord, part_id in points:
        lines.append(f"{x_coord:.6f} {y_coord:.6f} {z_coord:.6f} {part_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class EpisodePointCloudFuser:
    def __init__(self, config: PointCloudFusionConfig) -> None:
        self.config = config

    def fuse(self) -> Path:
        episode_path = Path(self.config.episode_path).resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)

        output_dir = (
            Path(self.config.output_dir).resolve()
            if self.config.output_dir is not None
            else episode_root / "pointcloud_4d"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        per_frame_dir = output_dir / "frames"
        per_frame_dir.mkdir(parents=True, exist_ok=True)

        lower, upper, bootstrap_metadata = _estimate_object_bounds(
            episode.frames,
            episode_root,
            episode.camera_intrinsics,
            episode.metadata,
            self.config,
        )

        all_points_4d: list[tuple[float, float, float, float, int, int]] = []
        frame_counts: list[int] = []
        pose_sources: set[str] = set(bootstrap_metadata["pose_modes"])
        depth_convention = _depth_convention(episode.metadata)
        part_segmentation = (
            dict(episode.metadata.get("part_segmentation", {}))
            if isinstance(episode.metadata.get("part_segmentation"), dict)
            else None
        )
        part_labels = part_name_lookup(part_segmentation)
        part_point_counts: dict[int, int] = {}

        for frame_index, frame in enumerate(episode.frames):
            depth_paths = _resolve_view_depth_paths(frame, episode_root)
            mask_paths = _resolve_view_mask_paths(frame, episode_root)
            part_mask_paths = _resolve_view_part_mask_paths(frame, episode_root)
            camera_poses, pose_source = _resolve_view_camera_poses(frame, episode.metadata, len(depth_paths))
            if pose_source == "shared-central-fallback" and not self.config.allow_shared_pose_fallback:
                raise ValueError(
                    "Per-view camera poses are missing. Re-record with updated metadata or pass allow_shared_pose_fallback=True."
                )
            pose_sources.add(pose_source)

            merged_points: list[tuple[float, float, float, int]] = []
            used_masks_for_frame = 0
            for view_index, (depth_path, camera_pose) in enumerate(zip(depth_paths, camera_poses)):
                view_intrinsics = _resolve_view_intrinsics(
                    episode.camera_intrinsics, episode.metadata, view_index
                )
                depth_u16 = _load_depth_u16(depth_path)
                mask_u16 = (
                    _load_depth_u16(mask_paths[view_index])
                    if view_index < len(mask_paths)
                    else None
                )
                part_mask_u16 = (
                    _load_depth_u16(part_mask_paths[view_index])
                    if view_index < len(part_mask_paths)
                    else None
                )
                if mask_u16 is not None:
                    used_masks_for_frame += 1
                    masked_upper_depth = _masked_depth_upper_bound(depth_u16, mask_u16, self.config)
                else:
                    masked_upper_depth = None
                height, width = _image_shape(depth_u16)
                stride = max(1, self.config.pixel_stride)
                for v_coord in range(0, height, stride):
                    row = depth_u16[v_coord]
                    mask_row = mask_u16[v_coord] if mask_u16 is not None else None
                    part_mask_row = part_mask_u16[v_coord] if part_mask_u16 is not None else None
                    for u_coord in range(0, width, stride):
                        if mask_row is not None and int(mask_row[u_coord]) <= 0:
                            continue
                        part_id = int(part_mask_row[u_coord]) if part_mask_row is not None else 0
                        if part_mask_row is not None and part_id <= 0:
                            continue
                        depth_m = row[u_coord] / 1000.0
                        if depth_m < self.config.min_depth_m or depth_m > self.config.max_depth_m:
                            continue
                        if masked_upper_depth is not None and depth_m > masked_upper_depth:
                            continue
                        point = _camera_to_world_point(
                            u_coord=u_coord,
                            v_coord=v_coord,
                            depth_m=depth_m,
                            intrinsics=view_intrinsics,
                            camera_pose=camera_pose,
                            depth_convention=depth_convention,
                        )
                        if _point_in_bounds(point, lower, upper):
                            merged_points.append((point[0], point[1], point[2], part_id))

            fused_points = _fuse_view_points(merged_points, self.config.voxel_size_m)
            frame_counts.append(len(fused_points))
            for x_coord, y_coord, z_coord, part_id in fused_points:
                all_points_4d.append((x_coord, y_coord, z_coord, frame.timestamp_s, frame_index, part_id))
                if part_id > 0:
                    part_point_counts[part_id] = part_point_counts.get(part_id, 0) + 1
            _write_frame_ply(per_frame_dir / f"frame_{frame_index:04d}.ply", fused_points)

        pointcloud_path = output_dir / "pointcloud_4d.ply"
        _write_pointcloud_ply(pointcloud_path, all_points_4d)

        manifest_path = output_dir / "fusion_manifest.json"
        save_json(
            {
                "episode_path": str(episode_path),
                "pointcloud_4d_path": str(pointcloud_path),
                "per_frame_dir": str(per_frame_dir),
                "frame_count": len(episode.frames),
                "total_point_count": len(all_points_4d),
                "per_frame_point_counts": frame_counts,
                "bounds": {"lower": lower, "upper": upper},
                "config": {
                    "min_depth_m": self.config.min_depth_m,
                    "max_depth_m": self.config.max_depth_m,
                    "pixel_stride": self.config.pixel_stride,
                    "central_crop_ratio": self.config.central_crop_ratio,
                    "near_depth_band_m": self.config.near_depth_band_m,
                    "bbox_margin_m": self.config.bbox_margin_m,
                    "voxel_size_m": self.config.voxel_size_m,
                    "allow_shared_pose_fallback": self.config.allow_shared_pose_fallback,
                    "mask_depth_percentile": self.config.mask_depth_percentile,
                    "mask_depth_margin_m": self.config.mask_depth_margin_m,
                },
                "bootstrap": bootstrap_metadata,
                "pose_sources_used": sorted(pose_sources),
                "depth_convention": depth_convention,
                "mask_aware_fusion": any(bool(_resolve_view_mask_paths(frame, episode_root)) for frame in episode.frames),
                "part_segmentation": part_segmentation,
                "part_ids_present": sorted(part_point_counts),
                "part_point_counts": {
                    str(part_id): {
                        "name": part_labels.get(part_id, f"part_{part_id}"),
                        "count": count,
                    }
                    for part_id, count in sorted(part_point_counts.items())
                },
                "part_labeled_point_count": int(sum(part_point_counts.values())),
            },
            manifest_path,
        )
        return manifest_path
