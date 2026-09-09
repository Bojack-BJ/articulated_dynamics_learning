#!/usr/bin/env python3
"""Export publication-ready assets from an articulated reconstruction run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageOps

from rgbd_urdf_mvp.perception.cotracker_features import load_cotracker_feature_map
from rgbd_urdf_mvp.perception.motion_part_slots import (
    _build_slot_model,
    _require_torch,
    _sample_from_artifact,
)


PAPER_COLORS = {
    "base": "#7B8188",
    "revolute": "#E8892D",
    "prismatic": "#3579B9",
    "additional_0": "#4C9A86",
    "additional_1": "#B06C8E",
    "additional_2": "#9A8A45",
    "track": "#2B6F77",
    "axis": "#252A2E",
    "pivot": "#D1495B",
}
PAPER_CAMERA = {"azimuth_deg": -131.0, "elevation_deg": 24.0}
PAPER_DPI = 360


@dataclass
class PaperData:
    object_id: str
    tracks_path: Path
    poses_path: Path
    joints_path: Path
    episode_path: Path
    checkpoint_path: Path
    features_path: Path
    tracks_artifact: dict[str, Any]
    poses_artifact: dict[str, Any]
    joints_artifact: dict[str, Any]
    episode: dict[str, Any]
    probabilities: np.ndarray
    active_slots: list[int]
    part_colors: dict[int, str]
    part_roles: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--part-poses", type=Path, required=True)
    parser.add_argument("--joints", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=1)
    parser.add_argument("--max-3d-tracks", type=int, default=240)
    parser.add_argument("--max-2d-tracks", type=int, default=72)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--camera-azimuth-deg", type=float, default=PAPER_CAMERA["azimuth_deg"])
    parser.add_argument("--camera-elevation-deg", type=float, default=PAPER_CAMERA["elevation_deg"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def resolve_artifact_path(raw: str | None, fallback: Path | None = None) -> Path:
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate.resolve()
    if fallback and fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(raw or str(fallback))


def load_paper_data(args: argparse.Namespace) -> PaperData:
    tracks_path = args.tracks.expanduser().resolve()
    poses_path = args.part_poses.expanduser().resolve()
    joints_path = args.joints.expanduser().resolve()
    tracks = load_json(tracks_path)
    poses = load_json(poses_path)
    joints = load_json(joints_path)
    episode_path = resolve_artifact_path(
        str(args.episode) if args.episode else tracks.get("input_episode_path"),
        tracks_path.parent.parent.parent.parent
        / "recordings_refrigerators_staged_dense_mps"
        / tracks_path.parent.name
        / "episode.json",
    )
    episode = load_json(episode_path)
    motion_meta = tracks.get("motion_segmentation", {})
    checkpoint = resolve_artifact_path(
        str(args.checkpoint) if args.checkpoint else motion_meta.get("model_path"),
        tracks_path.parent.parent / "slot_model_cross_object_balanced_v2" / "motion_part_slots.pt",
    )
    features = resolve_artifact_path(
        str(args.features) if args.features else None,
        tracks_path.parent / "cotracker_features.npz",
    )
    probabilities, active_slots = slot_probabilities(tracks, checkpoint, features)
    roles = infer_part_roles(poses, joints)
    colors = assign_part_colors(poses, joints, roles)
    return PaperData(
        object_id=str(episode.get("object_instance_id", tracks_path.parent.name)),
        tracks_path=tracks_path,
        poses_path=poses_path,
        joints_path=joints_path,
        episode_path=episode_path,
        checkpoint_path=checkpoint,
        features_path=features,
        tracks_artifact=tracks,
        poses_artifact=poses,
        joints_artifact=joints,
        episode=episode,
        probabilities=probabilities,
        active_slots=active_slots,
        part_colors=colors,
        part_roles=roles,
    )


def slot_probabilities(
    artifact: dict[str, Any], checkpoint_path: Path, features_path: Path
) -> tuple[np.ndarray, list[int]]:
    torch = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sample = _sample_from_artifact(
        artifact,
        load_cotracker_feature_map(features_path),
        object_id=str(artifact.get("object_instance_id", "paper_object")),
        require_labels=False,
        canonicalize_geometry=bool(checkpoint.get("canonicalize_geometry", False)),
    )
    model = _build_slot_model(
        torch,
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_slots=int(checkpoint["max_slots"]),
        encoder_layers=int(checkpoint["encoder_layers"]),
        decoder_layers=int(checkpoint["decoder_layers"]),
        attention_heads=int(checkpoint["attention_heads"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values = torch.from_numpy(sample["features"])
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]
    with torch.no_grad():
        logits, existence = model((values - mean) / std)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        existence_probabilities = torch.sigmoid(existence).cpu().numpy()
    active = np.flatnonzero(
        existence_probabilities
        >= float(artifact.get("motion_segmentation", {}).get("slot_existence_threshold", 0.5))
    ).astype(int).tolist()
    if not active:
        active = [int(np.argmax(existence_probabilities))]
    return probabilities, active


def infer_part_roles(
    poses: dict[str, Any], joints: dict[str, Any]
) -> dict[int, str]:
    anchor = int(joints.get("anchor_part_id", poses.get("anchor_part_id", -1)))
    roles = {int(part["part_id"]): "base" if int(part["part_id"]) == anchor else "additional"
             for part in poses.get("parts", [])}
    for joint in joints.get("joints", []):
        roles[int(joint["child_part_id"])] = str(joint["joint_type"])
    return roles


def assign_part_colors(
    poses: dict[str, Any], joints: dict[str, Any], roles: dict[int, str]
) -> dict[int, str]:
    colors: dict[int, str] = {}
    additional_index = 0
    for part in poses.get("parts", []):
        part_id = int(part["part_id"])
        role = roles.get(part_id, "additional")
        if role in PAPER_COLORS:
            colors[part_id] = PAPER_COLORS[role]
        else:
            colors[part_id] = PAPER_COLORS[f"additional_{additional_index % 3}"]
            additional_index += 1
    return colors


def selected_frame_indices(frame_count: int) -> list[int]:
    return sorted(set(int(round(value)) for value in np.linspace(0, frame_count - 1, 4)))


def episode_rgb_path(data: PaperData, source_frame: int, view_index: int) -> Path:
    frame = data.episode["frames"][source_frame]
    paths = frame.get("rgb_paths_by_view")
    raw = paths[view_index] if paths else frame["rgb_path"]
    return (data.episode_path.parent / raw).resolve()


def episode_mask_path(data: PaperData, source_frame: int, view_index: int) -> Path | None:
    frame = data.episode["frames"][source_frame]
    paths = frame.get("mask_paths_by_view")
    raw = paths[view_index] if paths else frame.get("mask_path")
    return (data.episode_path.parent / raw).resolve() if raw else None


def common_object_crop(
    data: PaperData, frame_indices: list[int], view_index: int
) -> tuple[int, int, int, int] | None:
    boxes = []
    image_size = None
    for frame_index in frame_indices:
        mask_path = episode_mask_path(data, frame_index, view_index)
        if mask_path is None or not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=2)
        else:
            mask = mask > 0
        rows, columns = np.nonzero(mask)
        image_size = (mask.shape[1], mask.shape[0])
        if len(columns):
            boxes.append((int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)))
    if not boxes or image_size is None:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    margin = int(round(0.09 * max(right - left, bottom - top)))
    width, height = image_size
    return max(0, left - margin), max(0, top - margin), min(width, right + margin), min(height, bottom + margin)


def save_rgb_assets(
    data: PaperData, output: Path, view_index: int
) -> tuple[list[int], tuple[int, int, int, int] | None]:
    source_count = len(data.episode["frames"])
    indices = selected_frame_indices(source_count)
    crop = common_object_crop(data, indices, view_index)
    images = []
    for order, frame_index in enumerate(indices):
        image = Image.open(episode_rgb_path(data, frame_index, view_index)).convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        image.save(output / f"rgb_t{order}.png")
        images.append(image)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    gap = max(8, width // 40)
    canvas = Image.new("RGB", (width * 4 + gap * 3, height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, (index * (width + gap), 0))
    canvas.save(output / "input_rgb.png")
    return indices, crop


def visible_samples(track: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sample for sample in track.get("samples", [])
        if sample.get("visible", True) and sample.get("depth_valid", True)
    ]


def part_tracks(data: PaperData) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for track in data.tracks_artifact.get("tracks", []):
        result.setdefault(int(track["part_id"]), []).append(track)
    return result


def sample_tracks_by_part(
    tracks: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for track in tracks:
        if len(visible_samples(track)) >= 3:
            grouped.setdefault(int(track["part_id"]), []).append(track)
    chosen = []
    per_part = max(1, limit // max(1, len(grouped)))
    for part_id in sorted(grouped):
        candidates = grouped[part_id]
        indices = rng.choice(len(candidates), min(per_part, len(candidates)), replace=False)
        chosen.extend(candidates[int(index)] for index in indices)
    if len(chosen) > limit:
        indices = rng.choice(len(chosen), limit, replace=False)
        chosen = [chosen[int(index)] for index in indices]
    return chosen


def clean_3d_axis(ax: Any, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    low, high = bounds
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def set_camera(ax: Any, args: argparse.Namespace) -> None:
    ax.view_init(elev=args.camera_elevation_deg, azim=args.camera_azimuth_deg)


def trajectory_bounds(tracks: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [sample["xyz_world"] for track in tracks for sample in visible_samples(track)],
        dtype=float,
    )
    return points.min(axis=0), points.max(axis=0)


def save_figure(fig: Any, output: Path, *, svg: bool = False) -> None:
    fig.savefig(output, dpi=PAPER_DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    if svg:
        fig.savefig(output.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_tracks_2d(
    data: PaperData,
    output: Path,
    frame_index: int,
    view_index: int,
    limit: int,
    seed: int,
    crop: tuple[int, int, int, int] | None,
) -> None:
    image_pil = Image.open(episode_rgb_path(data, frame_index, view_index)).convert("RGB")
    if crop is not None:
        image_pil = image_pil.crop(crop)
    image = np.asarray(image_pil)
    offset = np.asarray(crop[:2] if crop is not None else (0, 0), dtype=float)
    candidates = [
        track for track in data.tracks_artifact["tracks"]
        if int(track.get("view_index", -1)) == view_index
    ]
    chosen = sample_tracks_by_part(candidates, limit, seed)
    palette = plt.get_cmap("turbo")(np.linspace(0.05, 0.92, max(1, len(chosen))))
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.imshow(image)
    for color, track in zip(palette, chosen):
        samples = [sample for sample in visible_samples(track) if int(sample["source_frame_index"]) <= frame_index]
        if len(samples) < 2:
            continue
        uv = np.asarray([sample["uv"] for sample in samples]) - offset
        ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=0.8, alpha=0.78)
        selected = uv[np.linspace(0, len(uv) - 1, min(12, len(uv))).astype(int)]
        ax.scatter(selected[:, 0], selected[:, 1], color=color, s=6, edgecolors="white", linewidths=0.15)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    save_figure(fig, output / "tracks_2d.png")


def plot_tracks_4d(data: PaperData, output: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    tracks = sample_tracks_by_part(data.tracks_artifact["tracks"], args.max_3d_tracks, args.seed)
    bounds = trajectory_bounds(data.tracks_artifact["tracks"])
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for track in tracks:
        points = np.asarray([sample["xyz_world"] for sample in visible_samples(track)], dtype=float)
        color = data.part_colors[int(track["part_id"])]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=0.45, alpha=0.42)
        sampled = points[np.linspace(0, len(points) - 1, min(14, len(points))).astype(int)]
        ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], color=color, s=3.5, alpha=0.86)
    clean_3d_axis(ax, bounds)
    set_camera(ax, args)
    save_figure(fig, output / "tracks_4d.png", svg=True)
    return bounds


def points_at_frame(
    tracks: list[dict[str, Any]], frame_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, predicted, gt = [], [], []
    for track in tracks:
        sample = next(
            (row for row in track.get("samples", []) if int(row["frame_index"]) == frame_index
             and row.get("visible", True) and row.get("depth_valid", True)),
            None,
        )
        if sample is None:
            continue
        points.append(sample["xyz_world"])
        predicted.append(int(track["part_id"]))
        gt.append(int(track.get("original_part_id", -1)))
    return np.asarray(points), np.asarray(predicted), np.asarray(gt)


def plot_segmentation(
    data: PaperData,
    output: Path,
    args: argparse.Namespace,
    bounds: tuple[np.ndarray, np.ndarray],
    frame_index: int,
) -> None:
    points, predicted, gt = points_at_frame(data.tracks_artifact["tracks"], frame_index)
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for part_id in sorted(set(predicted.tolist())):
        mask = predicted == part_id
        ax.scatter(*points[mask].T, s=5, color=data.part_colors[part_id], alpha=0.9, depthshade=False)
    clean_3d_axis(ax, bounds)
    set_camera(ax, args)
    save_figure(fig, output / "part_segmentation.png")
    if np.any(gt >= 0):
        gt_ids = sorted(set(gt[gt >= 0].tolist()))
        gt_colors = {}
        for gt_id in gt_ids:
            matches = predicted[gt == gt_id]
            dominant = int(np.bincount(matches).argmax()) if len(matches) else -1
            gt_colors[gt_id] = data.part_colors.get(dominant, PAPER_COLORS["additional_0"])
        fig = plt.figure(figsize=(7.0, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        for gt_id in gt_ids:
            mask = gt == gt_id
            ax.scatter(*points[mask].T, s=5, color=gt_colors[gt_id], alpha=0.9, depthshade=False)
        clean_3d_axis(ax, bounds)
        set_camera(ax, args)
        save_figure(fig, output / "part_segmentation_gt.png")


def plot_slot_assignment(data: PaperData, output: Path) -> None:
    active = data.active_slots
    matrix = data.probabilities[:, active]
    order = np.lexsort((np.arange(len(matrix)), np.argmax(matrix, axis=1)))
    matrix = matrix[order]
    cmap = LinearSegmentedColormap.from_list("paper_blues", ["#F7FAFC", "#BFD5E6", "#3579B9", "#173F5F"])
    fig, ax = plt.subplots(figsize=(5.7, 6.0))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(active)), [f"S{index + 1}" for index in range(len(active))], fontsize=10)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07)
    save_figure(fig, output / "slot_assignment.png", svg=True)


def concise_part_label(part_id: int, role: str) -> str:
    return {"base": "Base", "revolute": "Door", "prismatic": "Drawer"}.get(role, f"P{part_id}")


def plot_kinematic_graph(data: PaperData, output: Path) -> None:
    parts = [int(part["part_id"]) for part in data.poses_artifact["parts"]]
    anchor = int(data.joints_artifact["anchor_part_id"])
    levels = {anchor: 0}
    changed = True
    while changed:
        changed = False
        for joint in data.joints_artifact["joints"]:
            parent, child = int(joint["parent_part_id"]), int(joint["child_part_id"])
            if parent in levels and child not in levels:
                levels[child] = levels[parent] + 1
                changed = True
    positions: dict[int, tuple[float, float]] = {}
    for level in sorted(set(levels.get(part, 1) for part in parts)):
        members = [part for part in parts if levels.get(part, 1) == level]
        for index, part in enumerate(members):
            positions[part] = (float(level), float(index - (len(members) - 1) / 2))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for joint in data.joints_artifact["joints"]:
        parent, child = int(joint["parent_part_id"]), int(joint["child_part_id"])
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.annotate("", xy=(x1 - 0.12, y1), xytext=(x0 + 0.12, y0),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#454B50"})
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.13,
                "R" if joint["joint_type"] == "revolute" else "P",
                ha="center", va="center", fontsize=11, color=data.part_colors[child], weight="bold")
    for part in parts:
        x, y = positions[part]
        circle = plt.Circle((x, y), 0.13, facecolor=data.part_colors[part], edgecolor="white", linewidth=1.2)
        ax.add_patch(circle)
        ax.text(x, y - 0.25, concise_part_label(part, data.part_roles[part]),
                ha="center", va="top", fontsize=10, color="#303438")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.22)
    ax.set_axis_off()
    save_figure(fig, output / "kinematic_graph.png", svg=True)


def pose_matrix(part: dict[str, Any], frame_index: int) -> np.ndarray:
    sample = next(
        (row for row in part["samples"] if int(row["frame_index"]) == frame_index and row.get("valid", True)),
        None,
    )
    if sample is None:
        raise ValueError(f"No valid pose for part {part['part_id']} at frame {frame_index}")
    return np.asarray(sample["matrix4"], dtype=float)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :3]


def part_reference_points(data: PaperData, part_id: int, limit: int = 600) -> np.ndarray:
    points = np.asarray(
        [track["reference_xyz_world"] for track in data.tracks_artifact["tracks"]
         if int(track["part_id"]) == part_id],
        dtype=float,
    )
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit).astype(int)
        points = points[indices]
    return points


def child_gt_support_center(
    data: PaperData,
    child_part_id: int,
    frame_index: int,
) -> np.ndarray:
    """Return the GT support center associated with a predicted child cluster."""
    child_tracks = [
        track for track in data.tracks_artifact["tracks"]
        if int(track["part_id"]) == child_part_id
    ]
    gt_ids = [
        int(track.get("original_part_id", -1))
        for track in child_tracks
        if int(track.get("original_part_id", -1)) >= 0
    ]
    dominant_gt = max(set(gt_ids), key=gt_ids.count) if gt_ids else None
    candidates = []
    for track in data.tracks_artifact["tracks"]:
        if dominant_gt is not None and int(track.get("original_part_id", -1)) != dominant_gt:
            continue
        if dominant_gt is None and int(track["part_id"]) != child_part_id:
            continue
        sample = next(
            (
                row for row in track.get("samples", [])
                if int(row["frame_index"]) == frame_index
                and row.get("visible", True)
                and row.get("depth_valid", True)
            ),
            None,
        )
        if sample is not None:
            candidates.append(sample["xyz_world"])
    if candidates:
        return np.mean(np.asarray(candidates, dtype=float), axis=0)
    return np.mean(part_reference_points(data, child_part_id), axis=0)


def project_point_onto_axis(point: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Move a display anchor along an axis without changing the represented line."""
    return pivot + axis * float(np.dot(point - pivot, axis))


def joint_parent_frame_geometry(
    data: PaperData, joint: dict[str, Any], frame_indices: list[int]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    parts = {int(part["part_id"]): part for part in data.poses_artifact["parts"]}
    parent_id, child_id = int(joint["parent_part_id"]), int(joint["child_part_id"])
    child_points = part_reference_points(data, child_id)
    transformed = []
    for frame_index in frame_indices:
        parent = pose_matrix(parts[parent_id], frame_index)
        child = pose_matrix(parts[child_id], frame_index)
        relative = np.linalg.inv(parent) @ child
        transformed.append(transform_points(relative, child_points))
    parent_zero_inv = np.linalg.inv(pose_matrix(parts[parent_id], frame_indices[0]))
    axis = parent_zero_inv[:3, :3] @ np.asarray(joint["axis"], dtype=float)
    pivot = transform_points(parent_zero_inv, np.asarray([joint["pivot"]], dtype=float))[0]
    return transformed, axis / max(np.linalg.norm(axis), 1e-12), pivot


def plot_relative_motion(
    data: PaperData, output: Path, args: argparse.Namespace, joint: dict[str, Any]
) -> None:
    frame_count = int(data.poses_artifact["frame_count"])
    frames = np.linspace(0, frame_count - 1, 10).astype(int).tolist()
    geometries, axis, pivot = joint_parent_frame_geometry(data, joint, frames)
    all_points = np.concatenate(geometries)
    bounds = (all_points.min(axis=0), all_points.max(axis=0))
    child_color = data.part_colors[int(joint["child_part_id"])]
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for index, points in enumerate(geometries):
        progress = index / max(len(geometries) - 1, 1)
        ax.scatter(
            *points.T,
            s=4,
            color=child_color,
            alpha=0.12 + 0.72 * progress,
            depthshade=False,
        )
    correspondence_indices = np.linspace(
        0, len(geometries[0]) - 1, min(16, len(geometries[0]))
    ).astype(int)
    for point_index in correspondence_indices:
        path = np.asarray([points[point_index] for points in geometries])
        ax.plot(*path.T, color=child_color, linewidth=0.75, alpha=0.72)
    extent = max(float(np.max(bounds[1] - bounds[0])), 0.25)
    center = np.mean(all_points, axis=0)
    if joint["joint_type"] == "revolute":
        display_pivot = project_point_onto_axis(center, pivot, axis)
        line = np.stack([display_pivot - axis * extent * 0.8, display_pivot + axis * extent * 0.8])
        ax.plot(*line.T, color=child_color, linewidth=2.0)
        ax.quiver(
            *display_pivot,
            *(axis * extent * 0.72),
            color=child_color,
            arrow_length_ratio=0.10,
            linewidth=1.9,
        )
        ax.scatter(*display_pivot, color=child_color, s=32, edgecolor="white", linewidth=0.7)
        name = "relative_motion_revolute.png"
    else:
        line = np.stack([center - axis * extent * 0.65, center + axis * extent * 0.65])
        ax.plot(*line.T, color=child_color, linewidth=2.0)
        ax.quiver(*line[0], *(line[1] - line[0]), color=child_color,
                  arrow_length_ratio=0.08, linewidth=1.8)
        name = "relative_motion_prismatic.png"
    expanded = (
        np.minimum(bounds[0], line.min(axis=0)),
        np.maximum(bounds[1], line.max(axis=0)),
    )
    clean_3d_axis(ax, expanded)
    set_camera(ax, args)
    save_figure(fig, output / name, svg=True)


def plot_joint_overlay(
    data: PaperData,
    output: Path,
    args: argparse.Namespace,
    bounds: tuple[np.ndarray, np.ndarray],
    frame_index: int,
    final_name: str = "joint_axis_overlay.png",
) -> None:
    points, labels, _ = points_at_frame(data.tracks_artifact["tracks"], frame_index)
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for part_id in sorted(set(labels.tolist())):
        mask = labels == part_id
        ax.scatter(*points[mask].T, s=5, color=data.part_colors[part_id], alpha=0.72, depthshade=False)
    extent = max(float(np.max(bounds[1] - bounds[0])), 0.25)
    for joint in data.joints_artifact["joints"]:
        axis = np.asarray(joint["axis"], dtype=float)
        axis /= max(np.linalg.norm(axis), 1e-12)
        pivot = np.asarray(joint["pivot"], dtype=float)
        color = data.part_colors[int(joint["child_part_id"])]
        child_center = child_gt_support_center(
            data,
            int(joint["child_part_id"]),
            frame_index,
        )
        if joint["joint_type"] == "revolute":
            display_pivot = project_point_onto_axis(child_center, pivot, axis)
            line = np.stack([
                display_pivot - axis * extent * 0.55,
                display_pivot + axis * extent * 0.55,
            ])
            ax.plot(*line.T, color=color, linewidth=2.2)
            ax.quiver(
                *display_pivot,
                *(axis * extent * 0.48),
                color=color,
                arrow_length_ratio=0.11,
                linewidth=1.9,
            )
            ax.scatter(*display_pivot, color=color, s=30, edgecolor="white", linewidth=0.7)
        else:
            start = child_center - axis * extent * 0.38
            delta = axis * extent * 0.7
            ax.plot(*np.stack([start, start + delta]).T, color=color, linewidth=2.2)
            ax.quiver(*start, *delta, color=color, arrow_length_ratio=0.09, linewidth=1.8)
    clean_3d_axis(ax, bounds)
    set_camera(ax, args)
    save_figure(fig, output / final_name, svg=True)


def plot_final_views(
    data: PaperData,
    output: Path,
    args: argparse.Namespace,
    bounds: tuple[np.ndarray, np.ndarray],
    frame_index: int,
) -> None:
    plot_joint_overlay(data, output, args, bounds, frame_index, "final_articulated_model.png")
    original_azimuth = args.camera_azimuth_deg
    original_elevation = args.camera_elevation_deg
    args.camera_azimuth_deg = original_azimuth + 75.0
    args.camera_elevation_deg = original_elevation + 4.0
    plot_joint_overlay(data, output, args, bounds, frame_index, "final_articulated_model_alt_view.png")
    args.camera_azimuth_deg = original_azimuth
    args.camera_elevation_deg = original_elevation


def build_contact_sheet(output: Path) -> None:
    names = [
        "input_rgb.png", "tracks_2d.png", "tracks_4d.png", "part_segmentation.png",
        "part_segmentation_gt.png", "slot_assignment.png", "kinematic_graph.png",
        "relative_motion_revolute.png", "relative_motion_prismatic.png",
        "joint_axis_overlay.png", "final_articulated_model.png",
        "final_articulated_model_alt_view.png",
    ]
    entries = [(name, Image.open(output / name).convert("RGB")) for name in names if (output / name).exists()]
    thumb_w, thumb_h, label_h, gap = 620, 480, 34, 18
    columns = 3
    rows = math.ceil(len(entries) / columns)
    sheet = Image.new("RGB", (columns * (thumb_w + gap) + gap, rows * (thumb_h + label_h + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(entries):
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), "white")
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        x = gap + (index % columns) * (thumb_w + gap)
        y = gap + (index // columns) * (thumb_h + label_h + gap)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + thumb_h + 7), name, fill="#252A2E")
    sheet.save(output / "contact_sheet.png", dpi=(180, 180))


def write_manifest(
    data: PaperData,
    output: Path,
    args: argparse.Namespace,
    rgb_frames: list[int],
    representative_track_frame: int,
    tracking_source_frame: int,
    crop: tuple[int, int, int, int],
) -> None:
    joints = [
        {
            "name": joint["name"],
            "parent_part_id": int(joint["parent_part_id"]),
            "child_part_id": int(joint["child_part_id"]),
            "joint_type": joint["joint_type"],
            "axis": joint["axis"],
            "pivot": joint["pivot"],
        }
        for joint in data.joints_artifact["joints"]
    ]
    payload = {
        "schema": "paper-pipeline-assets-v1",
        "object_id": data.object_id,
        "checkpoint": str(data.checkpoint_path),
        "tracks": str(data.tracks_path),
        "part_poses": str(data.poses_path),
        "joint_inference": str(data.joints_path),
        "episode": str(data.episode_path),
        "selected_rgb_source_frame_indices": rgb_frames,
        "representative_track_frame_index": representative_track_frame,
        "tracking_2d_source_frame_index": tracking_source_frame,
        "rgb_crop_box_xyxy": list(crop),
        "predicted_part_count": len(data.part_colors),
        "active_slot_indices": data.active_slots,
        "joints": joints,
        "plotted_parent_child_pairs": [
            [joint["parent_part_id"], joint["child_part_id"]] for joint in joints
        ],
        "camera": {
            "projection": "orthographic",
            "azimuth_deg": args.camera_azimuth_deg,
            "elevation_deg": args.camera_elevation_deg,
            "equal_xyz_scale": True,
        },
        "part_colors": {str(key): value for key, value in data.part_colors.items()},
        "part_roles": {str(key): value for key, value in data.part_roles.items()},
        "coordinate_conventions": {
            "track_and_joint_frame": "world/source canonical frame",
            "relative_motion": "T_rel(t) = inv(T_parent(t)) @ T_child(t)",
            "relative_joint_geometry": "axis/pivot transformed into parent reference frame",
            "axis_display_anchor": (
                "Revolute anchors are shifted only along the predicted axis line toward the "
                "child GT-support center; prismatic direction arrows are translated to that center."
            ),
        },
        "gt_usage": "GT labels are used only in part_segmentation_gt.png.",
        "assets": [
            "input_rgb.png",
            "rgb_t0.png",
            "rgb_t1.png",
            "rgb_t2.png",
            "rgb_t3.png",
            "tracks_2d.png",
            "tracks_4d.png",
            "tracks_4d.svg",
            "part_segmentation.png",
            "part_segmentation_gt.png",
            "slot_assignment.png",
            "slot_assignment.svg",
            "kinematic_graph.png",
            "kinematic_graph.svg",
            "relative_motion_revolute.png",
            "relative_motion_revolute.svg",
            "relative_motion_prismatic.png",
            "relative_motion_prismatic.svg",
            "joint_axis_overlay.png",
            "joint_axis_overlay.svg",
            "final_articulated_model.png",
            "final_articulated_model.svg",
            "final_articulated_model_alt_view.png",
            "final_articulated_model_alt_view.svg",
            "contact_sheet.png",
        ],
    }
    (output / "figure_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_paper_data(args)
    rgb_frames, crop = save_rgb_assets(data, output, args.view_index)
    tracking_frame = int(data.tracks_artifact["sampled_frame_indices"][-1])
    plot_tracks_2d(
        data, output, tracking_frame, args.view_index, args.max_2d_tracks, args.seed, crop
    )
    bounds = plot_tracks_4d(data, output, args)
    representative_track_frame = 0
    plot_segmentation(data, output, args, bounds, representative_track_frame)
    plot_slot_assignment(data, output)
    plot_kinematic_graph(data, output)
    for joint in data.joints_artifact["joints"]:
        if joint["joint_type"] in {"revolute", "prismatic"}:
            plot_relative_motion(data, output, args, joint)
    plot_joint_overlay(data, output, args, bounds, representative_track_frame)
    plot_final_views(data, output, args, bounds, representative_track_frame)
    write_manifest(
        data,
        output,
        args,
        rgb_frames,
        representative_track_frame,
        tracking_frame,
        crop,
    )
    build_contact_sheet(output)
    print(json.dumps({"output_dir": str(output), "object_id": data.object_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
