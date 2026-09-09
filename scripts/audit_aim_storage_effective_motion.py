#!/usr/bin/env python3
"""Audit effective motion cues and AiM decomposition for one simulated episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from audit_aim_protocol import _read_ply_part_ids, _read_ply_xyz_rgb, _rgb_labels


GAUSSIAN_PATTERN = re.compile(
    r"(?P<static>\d+) Static Gaussians and (?P<moving>\d+) Moving Gaussians "
)
PALETTE = np.asarray(
    [
        [45, 55, 72],
        [72, 210, 166],
        [255, 112, 112],
        [190, 174, 255],
        [255, 202, 40],
        [48, 209, 194],
        [242, 133, 58],
        [232, 91, 145],
        [110, 168, 254],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("aim_run", type=Path)
    parser.add_argument("aim_metrics", type=Path, help="Per-object aim.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=12.0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_u16(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def _backproject(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float],
    camera_pose: np.ndarray,
    *,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    vv, uu = np.nonzero(mask & (depth > 0))
    if pixel_stride > 1:
        keep = (vv % pixel_stride == 0) & (uu % pixel_stride == 0)
        vv, uu = vv[keep], uu[keep]
    z = depth[vv, uu].astype(np.float64) / 1000.0
    x = (uu.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = -(vv.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    camera = np.column_stack((x, y, z))
    world = camera @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    return world, np.column_stack((vv, uu))


def _joint_series(frames: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted(
        {
            name
            for frame in frames
            for name in frame.get("action_log", {}).get("joint_positions", {})
        }
    )
    values = np.full((len(frames), len(names)), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        positions = frame.get("action_log", {}).get("joint_positions", {})
        for joint_index, name in enumerate(names):
            if name in positions:
                values[frame_index, joint_index] = float(positions[name])
    return names, values


def _camera_azimuths(frames: list[dict[str, Any]], lookat: np.ndarray) -> np.ndarray:
    positions = np.asarray([frame["camera_pose"] for frame in frames], dtype=np.float64)[:, :3, 3]
    relative = positions - lookat[None, :]
    return np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))


def _plot_schedule(
    path: Path,
    timestamps: np.ndarray,
    joint_names: list[str],
    joint_values: np.ndarray,
    camera_azimuth: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(joint_names) + 1
    figure, axes = plt.subplots(count, 1, figsize=(13, max(5, 1.55 * count)), sharex=True)
    axes = np.atleast_1d(axes)
    for index, name in enumerate(joint_names):
        series = joint_values[:, index]
        axes[index].plot(timestamps, series, color="#35b779", linewidth=1.8)
        axes[index].set_ylabel(name)
        span = float(np.nanmax(series) - np.nanmin(series))
        active = np.abs(np.diff(series, prepend=series[0])) > max(span * 1e-3, 1e-6)
        axes[index].fill_between(
            timestamps,
            np.nanmin(series),
            np.nanmax(series),
            where=active,
            color="#fde725",
            alpha=0.18,
        )
        axes[index].grid(alpha=0.2)
    axes[-1].plot(timestamps, np.degrees(camera_azimuth), color="#3b82f6", linewidth=1.8)
    axes[-1].set_ylabel("camera\nazimuth (deg)")
    axes[-1].set_xlabel("time (s)")
    axes[-1].grid(alpha=0.2)
    figure.suptitle("Storage-47648 joint schedule and camera trajectory")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_gaussian_counts(run_dir: Path) -> tuple[int, int]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    matches = list(GAUSSIAN_PATTERN.finditer(text))
    if not matches:
        raise RuntimeError("Could not find static/moving Gaussian counts in train.log")
    return int(matches[-1].group("static")), int(matches[-1].group("moving"))


def _part_gaussian_stats(
    run_dir: Path,
    reference_path: Path,
    static_count: int,
    moving_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_xyz, reference_part = _read_ply_part_ids(reference_path)
    initial_xyz, initial_rgb = _read_ply_xyz_rgb(run_dir / "seg_init_dual" / "segmented_point.ply")
    final_xyz, final_rgb = _read_ply_xyz_rgb(run_dir / "motion_seg_final" / "segmented_point.ply")
    bbox_diagonal = float(np.linalg.norm(np.ptp(reference_xyz, axis=0)))
    max_distance = max(0.02 * bbox_diagonal, 1e-9)
    distance, reference_index = cKDTree(reference_xyz).query(initial_xyz, k=1)
    valid = distance <= max_distance
    nearest_gaussian_gt = reference_part[reference_index]
    gaussian_gt = np.full(len(initial_xyz), -1, dtype=np.int64)
    gaussian_gt[valid] = nearest_gaussian_gt[valid]
    moving_slice = np.arange(len(initial_xyz)) >= static_count
    final_label = _rgb_labels(final_rgb) if final_rgb is not None else np.zeros(len(final_xyz), dtype=np.int64)
    final_distance, final_index = cKDTree(final_xyz).query(reference_xyz, k=1)
    final_valid = final_distance <= max_distance
    rows = []
    for part_id in sorted(int(value) for value in np.unique(reference_part)):
        selected_gaussians = gaussian_gt == part_id
        selected_moving = selected_gaussians & moving_slice
        selected_nearest = nearest_gaussian_gt == part_id
        selected_nearest_moving = selected_nearest & moving_slice
        selected_reference = reference_part == part_id
        mapped_components = final_label[final_index[selected_reference & final_valid]]
        component_counts = Counter(int(value) for value in mapped_components)
        rows.append(
            {
                "gt_part_id": part_id,
                "mapped_gaussian_count": int(np.sum(selected_gaussians)),
                "mapped_dynamic_gaussian_count": int(np.sum(selected_moving)),
                "nearest_assigned_dynamic_gaussian_count": int(
                    np.sum(selected_nearest_moving)
                ),
                "dynamic_ratio_within_part": (
                    float(np.mean(moving_slice[selected_gaussians]))
                    if np.any(selected_gaussians)
                    else None
                ),
                "fraction_of_initial_dynamic_set": (
                    float(np.sum(selected_moving) / max(moving_count, 1))
                ),
                "nearest_assigned_fraction_of_initial_dynamic_set": float(
                    np.sum(selected_nearest_moving) / max(moving_count, 1)
                ),
                "below_official_global_10pct_threshold": bool(
                    np.sum(selected_nearest_moving) < 0.1 * moving_count
                ),
                "nearest_assignment_distance_median_m": (
                    float(np.median(distance[selected_nearest_moving]))
                    if np.any(selected_nearest_moving)
                    else None
                ),
                "nearest_assignment_distance_p95_m": (
                    float(np.percentile(distance[selected_nearest_moving], 95))
                    if np.any(selected_nearest_moving)
                    else None
                ),
                "dominant_final_component": (
                    component_counts.most_common(1)[0][0] if component_counts else None
                ),
                "dominant_final_component_coverage": (
                    component_counts.most_common(1)[0][1] / max(sum(component_counts.values()), 1)
                    if component_counts
                    else None
                ),
            }
        )
    metadata = {
        "static_gaussian_count": static_count,
        "initial_dynamic_gaussian_count": moving_count,
        "official_min_inliers_fraction": 0.1,
        "official_min_inliers_count": int(math.ceil(0.1 * moving_count)),
        "bbox_diagonal_m": bbox_diagonal,
        "gaussian_to_gt_max_distance_m": max_distance,
        "strictly_mapped_dynamic_gaussian_count": int(np.sum(valid & moving_slice)),
        "strict_dynamic_mapping_ratio": float(
            np.mean(valid[moving_slice]) if np.any(moving_slice) else 0.0
        ),
        "nearest_assignment_note": (
            "Every dynamic Gaussian is assigned to its nearest observed GT surface only for "
            "support-size diagnostics. Distances are reported because outliers may be assigned."
        ),
    }
    return rows, metadata


def _observed_part_motion(
    episode_dir: Path,
    episode: dict[str, Any],
    *,
    pixel_stride: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    frames = list(episode["frames"])
    intrinsics = episode["camera_intrinsics"]
    per_part: dict[int, dict[str, list[Any]]] = {}
    frame_cache: dict[int, dict[str, np.ndarray]] = {}
    for frame_index, frame in enumerate(frames):
        depth = _load_u16(episode_dir / frame["depth_path"])
        part_mask = _load_u16(episode_dir / frame["part_mask_path"])
        pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        rgb = np.asarray(Image.open(episode_dir / frame["rgb_path"]).convert("RGB"))
        frame_cache[frame_index] = {"rgb": rgb, "depth": depth, "part_mask": part_mask, "pose": pose}
        for part_id in np.unique(part_mask):
            if part_id == 0:
                continue
            selected = part_mask == part_id
            world, pixels = _backproject(
                depth, selected, intrinsics, pose, pixel_stride=pixel_stride
            )
            if len(world) == 0:
                continue
            values = per_part.setdefault(
                int(part_id),
                {"frames": [], "pixels": [], "centroid_2d": [], "centroid_3d": [], "points": []},
            )
            values["frames"].append(frame_index)
            values["pixels"].append(int(np.sum(selected)))
            values["centroid_2d"].append(np.mean(pixels[:, ::-1], axis=0))
            values["centroid_3d"].append(np.mean(world, axis=0))
            values["points"].append(world)
    rows = []
    for part_id, values in sorted(per_part.items()):
        centroid_2d = np.asarray(values["centroid_2d"])
        centroid_3d = np.asarray(values["centroid_3d"])
        first_points = values["points"][0]
        first_tree = cKDTree(first_points)
        surface_displacement = []
        for points in values["points"]:
            distance, _ = first_tree.query(points, k=1)
            surface_displacement.append(float(np.median(distance)))
        rows.append(
            {
                "gt_part_id": part_id,
                "visible_frame_count": len(values["frames"]),
                "visible_frame_ratio": len(values["frames"]) / len(frames),
                "visible_pixel_count_median": float(np.median(values["pixels"])),
                "visible_pixel_count_min": int(np.min(values["pixels"])),
                "image_centroid_displacement_px_max": float(
                    np.max(np.linalg.norm(centroid_2d - centroid_2d[0], axis=1))
                ),
                "world_centroid_displacement_m_max": float(
                    np.max(np.linalg.norm(centroid_3d - centroid_3d[0], axis=1))
                ),
                "observed_surface_nn_displacement_m_median_max": float(
                    np.max(surface_displacement)
                ),
                "note_surface_metric": (
                    "Median nearest-neighbor distance to the first visible RGB-D surface; "
                    "it includes visibility changes and is not point correspondence."
                ),
            }
        )
    return rows, frame_cache


def _plot_dynamic_fraction(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part_ids = [str(row["gt_part_id"]) for row in rows]
    fractions = [
        float(row["nearest_assigned_fraction_of_initial_dynamic_set"]) for row in rows
    ]
    colors = ["#ef4444" if value < 0.1 else "#22c55e" for value in fractions]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(part_ids, fractions, color=colors)
    axis.axhline(0.1, color="#facc15", linestyle="--", label="official global 10% min-inliers")
    axis.set_xlabel("GT part ID")
    axis.set_ylabel("fraction of initial dynamic Gaussians")
    axis.set_title("Storage-47648 part support versus sequential-RANSAC threshold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _overlay_labels(
    rgb: np.ndarray,
    pixels: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.58,
) -> np.ndarray:
    result = rgb.copy()
    colors = PALETTE[np.mod(labels, len(PALETTE))]
    vv, uu = pixels[:, 0], pixels[:, 1]
    result[vv, uu] = (
        (1.0 - alpha) * result[vv, uu].astype(np.float32) + alpha * colors
    ).astype(np.uint8)
    return result


def _labeled_panel(image: np.ndarray, title: str) -> Image.Image:
    panel = Image.fromarray(image)
    canvas = Image.new("RGB", (panel.width, panel.height + 38), (12, 18, 27))
    canvas.paste(panel, (0, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(235, 241, 248), font=ImageFont.load_default())
    return canvas


def _write_video(
    path: Path,
    episode: dict[str, Any],
    frame_cache: dict[int, dict[str, np.ndarray]],
    run_dir: Path,
    static_count: int,
    *,
    frame_stride: int,
    pixel_stride: int,
    fps: float,
) -> None:
    import cv2

    initial_xyz, _ = _read_ply_xyz_rgb(run_dir / "seg_init_dual" / "segmented_point.ply")
    final_xyz, final_rgb = _read_ply_xyz_rgb(run_dir / "motion_seg_final" / "segmented_point.ply")
    if final_rgb is None:
        raise RuntimeError("Final segmented PLY has no RGB component labels")
    initial_tree = cKDTree(initial_xyz)
    final_tree = cKDTree(final_xyz)
    final_labels = _rgb_labels(final_rgb)
    intrinsics = episode["camera_intrinsics"]
    metadata = episode.get("metadata", {})
    bbox = float(metadata.get("camera_fit_radius", 1.0))
    query_limit = max(0.04 * bbox, 0.02)
    writer = None
    try:
        for frame_index in range(0, len(episode["frames"]), max(frame_stride, 1)):
            cached = frame_cache[frame_index]
            object_mask = cached["part_mask"] > 0
            world, pixels = _backproject(
                cached["depth"],
                object_mask,
                intrinsics,
                cached["pose"],
                pixel_stride=pixel_stride,
            )
            initial_distance, initial_index = initial_tree.query(world, k=1)
            final_distance, final_index = final_tree.query(world, k=1)
            dynamic_labels = np.zeros(len(world), dtype=np.int64)
            dynamic_labels[initial_index >= static_count] = 1
            dynamic_labels[initial_distance > query_limit] = 2
            component_labels = final_labels[final_index] + 1
            component_labels[final_distance > query_limit] = 0
            gt = cached["part_mask"]
            gt_image = cached["rgb"].copy()
            selected = gt > 0
            gt_image[selected] = (
                0.42 * gt_image[selected].astype(np.float32)
                + 0.58 * PALETTE[np.mod(gt[selected], len(PALETTE))]
            ).astype(np.uint8)
            panels = [
                _labeled_panel(cached["rgb"], f"RGB interaction | frame {frame_index}"),
                _labeled_panel(gt_image, "GT part segmentation"),
                _labeled_panel(
                    _overlay_labels(cached["rgb"], pixels, dynamic_labels),
                    "AiM initial static/dynamic (gray=unmapped)",
                ),
                _labeled_panel(
                    _overlay_labels(cached["rgb"], pixels, component_labels),
                    "AiM final RANSAC components",
                ),
            ]
            width, height = panels[0].size
            canvas = Image.new("RGB", (2 * width, 2 * height), (5, 8, 13))
            for index, panel in enumerate(panels):
                canvas.paste(panel, ((index % 2) * width, (index // 2) * height))
            if writer is None:
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    canvas.size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {path}")
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    metrics = json.loads(args.aim_metrics.read_text(encoding="utf-8"))
    frames = list(episode["frames"])
    timestamps = np.asarray(
        [float(frame.get("timestamp_s", index)) for index, frame in enumerate(frames)]
    )
    joint_names, joint_values = _joint_series(frames)
    lookat = np.asarray(episode.get("metadata", {}).get("lookat", [0.0, 0.0, 0.0]))
    camera_azimuth = _camera_azimuths(frames, lookat)
    _plot_schedule(
        args.output_dir / "motion_schedule.png",
        timestamps,
        joint_names,
        joint_values,
        camera_azimuth,
    )
    schedule_rows = []
    for index, name in enumerate(joint_names):
        series = joint_values[:, index]
        span = float(np.nanmax(series) - np.nanmin(series))
        active = np.abs(np.diff(series, prepend=series[0])) > max(span * 1e-3, 1e-6)
        schedule_rows.append(
            {
                "joint_name": name,
                "q_min": float(np.nanmin(series)),
                "q_max": float(np.nanmax(series)),
                "q_range": span,
                "active_frame_count": int(np.sum(active)),
                "first_active_frame": int(np.flatnonzero(active)[0]) if np.any(active) else None,
                "last_active_frame": int(np.flatnonzero(active)[-1]) if np.any(active) else None,
            }
        )
    _write_csv(args.output_dir / "joint_schedule.csv", schedule_rows)

    static_count, moving_count = _parse_gaussian_counts(args.aim_run)
    gaussian_rows, gaussian_metadata = _part_gaussian_stats(
        args.aim_run,
        Path(metrics["reference_ply"]),
        static_count,
        moving_count,
    )
    observed_rows, frame_cache = _observed_part_motion(
        args.episode.parent,
        episode,
        pixel_stride=args.pixel_stride,
    )
    observed_by_id = {int(row["gt_part_id"]): row for row in observed_rows}
    combined_rows = []
    for gaussian_row in gaussian_rows:
        combined_rows.append(
            {
                **gaussian_row,
                **observed_by_id.get(int(gaussian_row["gt_part_id"]), {}),
            }
        )
    _write_csv(args.output_dir / "per_part_effective_motion.csv", combined_rows)
    _plot_dynamic_fraction(args.output_dir / "ransac_part_support.png", gaussian_rows)
    _write_video(
        args.output_dir / "interaction_gt_aim_decomposition.mp4",
        episode,
        frame_cache,
        args.aim_run,
        static_count,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        fps=args.video_fps,
    )
    summary = {
        "episode": str(args.episode),
        "aim_run": str(args.aim_run),
        "frame_count": len(frames),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "effective_fps": float((len(frames) - 1) / max(timestamps[-1] - timestamps[0], 1e-9)),
        "camera_azimuth_sweep_deg": float(np.degrees(np.ptp(camera_azimuth))),
        "joint_schedule_exact_official_match": "unknown",
        "joint_schedule_note": (
            "The released AiM repository does not provide an object-specific temporal schedule "
            "for Storage-47648; this file audits the implemented schedule without claiming an exact match."
        ),
        "gaussian_support": gaussian_metadata,
        "parts": combined_rows,
        "artifacts": {
            "motion_schedule": "motion_schedule.png",
            "per_part_metrics": "per_part_effective_motion.csv",
            "ransac_support": "ransac_part_support.png",
            "four_panel_video": "interaction_gt_aim_decomposition.mp4",
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"audit": str(args.output_dir / "audit.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
