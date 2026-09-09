#!/usr/bin/env python3
"""Render the open-state Figure-1 panel from fused RGB-D and CoTracker data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAME_INDEX = 180
FUSED_PLY = ROOT / f"outputs/recordings_refrigerators_staged_dense_mps/refrigerator045/pointcloud_4d_partseg/frames/frame_{FRAME_INDEX:04d}.ply"
TRACKS_JSON = ROOT / "outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/learning_features_v1/refrigerator045/predicted_slots_balanced_v2.json"
OUTPUT = ROOT / "outputs/figure1_open_panels"

COLORS = {1: "#858B93", 2: "#397BC5", 3: "#F28E2B", 4: "#397BC5"}
AZIMUTH_DEG = -143.0
ELEVATION_DEG = 24.0


def load_ascii_part_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            if line.strip() == "end_header":
                break
    data = np.loadtxt(path, skiprows=header_lines)
    return data[:, :3], data[:, 3].astype(int)


def visible_points(track: dict) -> np.ndarray:
    return np.asarray(
        [
            sample["xyz_world"] for sample in track.get("samples", [])
            if sample.get("visible", True) and sample.get("depth_valid", True)
        ],
        dtype=float,
    )


def quality_score(points: np.ndarray, part_id: int) -> float | None:
    if len(points) < 24:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    endpoint = float(np.linalg.norm(points[-1] - points[0]))
    if endpoint < 0.10 or np.quantile(steps, 0.95) > 0.15:
        return None
    path = float(steps.sum())
    excess_path = max(0.0, path / max(endpoint, 1e-8) - 1.0)
    directions = np.diff(points, axis=0)
    norms = np.linalg.norm(directions, axis=1)
    unit = directions[norms > 1e-8] / norms[norms > 1e-8, None]
    smoothness = float(np.mean(np.sum(unit[:-1] * unit[1:], axis=1))) if len(unit) > 1 else 1.0
    if part_id == 3:
        return endpoint + 1.4 * min(excess_path, 0.35) + 0.25 * smoothness
    return endpoint + 0.65 * smoothness - 1.5 * excess_path


def select_tracks(artifact: dict) -> list[tuple[int, np.ndarray]]:
    selected: list[tuple[int, np.ndarray]] = []
    limits = {2: 9, 3: 13, 4: 8}
    for part_id, limit in limits.items():
        candidates = []
        for track in artifact["tracks"]:
            if int(track.get("original_part_id", -1)) != part_id:
                continue
            points = visible_points(track)
            score = quality_score(points, part_id)
            if score is not None:
                candidates.append((score, points))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected.extend((part_id, points) for _, points in candidates[:limit])
    return selected


def temporal_samples(points: np.ndarray, count: int = 12) -> np.ndarray:
    indices = np.linspace(0, len(points) - 1, min(count, len(points))).round().astype(int)
    return points[np.unique(indices)]


def draw_track(ax, points: np.ndarray, color: str) -> None:
    points = temporal_samples(points)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    rgb = to_rgba(color)[:3]
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=[(*rgb, float(a)) for a in np.linspace(0.24, 0.98, len(segments))],
            linewidths=np.linspace(0.8, 2.35, len(segments)),
        )
    )
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color=[(*rgb, float(a)) for a in np.linspace(0.25, 0.90, len(points))],
        s=np.linspace(3.0, 9.0, len(points)), edgecolors="none", depthshade=False,
    )


def configure_axis(ax, low: np.ndarray, high: np.ndarray) -> None:
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION_DEG, azim=AZIMUTH_DEG)
    ax.set_axis_off()
    ax.grid(False)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=650, transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox:
            left, top, right, bottom = bbox
            pad = max(24, int(0.035 * max(right - left, bottom - top)))
            rgba.crop((max(0, left - pad), max(0, top - pad), min(rgba.width, right + pad), min(rgba.height, bottom + pad))).save(path, dpi=(650, 650))


def render(points: np.ndarray, part_ids: np.ndarray, tracks: list[tuple[int, np.ndarray]]) -> None:
    low, high = points.min(axis=0), points.max(axis=0)

    fig = plt.figure(figsize=(5.8, 5.8), dpi=650)
    ax = fig.add_subplot(111, projection="3d")
    for part_id in sorted(np.unique(part_ids)):
        mask = part_ids == part_id
        ax.scatter(*points[mask].T, s=6.5, c=COLORS.get(part_id, "#858B93"), alpha=0.92, edgecolors="none", depthshade=False)
    configure_axis(ax, low, high)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    save(fig, OUTPUT / "open_fused_pointcloud.png")

    fig = plt.figure(figsize=(5.8, 5.8), dpi=650)
    ax = fig.add_subplot(111, projection="3d")
    for part_id in sorted(np.unique(part_ids)):
        mask = part_ids == part_id
        ax.scatter(*points[mask].T, s=3.4, c=COLORS.get(part_id, "#858B93"), alpha=0.22, edgecolors="none", depthshade=False)
    for part_id, trajectory in tracks:
        draw_track(ax, trajectory, COLORS[part_id])
    configure_axis(ax, low, high)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    save(fig, OUTPUT / "open_fused_cotracker_trajectories.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points, part_ids = load_ascii_part_ply(FUSED_PLY)
    artifact = json.loads(TRACKS_JSON.read_text(encoding="utf-8"))
    tracks = select_tracks(artifact)
    render(points, part_ids, tracks)
    metadata = {
        "fused_frame": str(FUSED_PLY),
        "frame_index": FRAME_INDEX,
        "joint_state": {
            "freezer0_door_joint": 0.45,
            "fridge_door_joint": 1.570796,
            "fridge_drawer0_joint": 0.45,
        },
        "track_source": str(TRACKS_JSON),
        "tracker": artifact.get("estimator"),
        "selected_track_count": len(tracks),
        "camera": {"azimuth_deg": AZIMUTH_DEG, "elevation_deg": ELEVATION_DEG},
        "outputs": [
            str(OUTPUT / "open_fused_pointcloud.png"),
            str(OUTPUT / "open_fused_cotracker_trajectories.png"),
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
