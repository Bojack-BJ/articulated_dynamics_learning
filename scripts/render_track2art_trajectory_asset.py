#!/usr/bin/env python3
"""Render a clean, real-data-grounded 4D trajectory panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image


GRAY = "#AEB5BD"
ORANGE = "#F28E2B"
BLUE = "#397BC5"
ROOT_OFFSET = np.array([0.0, 0.0, 0.938160])
HINGE_LOCAL = np.array([0.391572, -0.367582, 0.382577])
HINGE_WORLD = HINGE_LOCAL + ROOT_OFFSET
HINGE_ANGLE = 1.22
SLIDE_VECTOR = np.array([0.0, -0.42, 0.0])


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def visible_points(track: dict) -> np.ndarray:
    return np.asarray(
        [
            sample["xyz_world"]
            for sample in track.get("samples", [])
            if sample.get("visible", True) and sample.get("depth_valid", True)
        ],
        dtype=float,
    )


def select_real_tracks(tracks: list[dict], part_id: int, count: int) -> list[np.ndarray]:
    candidates = []
    for track in tracks:
        if int(track.get("original_part_id", -1)) != part_id:
            continue
        points = visible_points(track)
        if len(points) < 30:
            continue
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        endpoint = float(np.linalg.norm(points[-1] - points[0]))
        if endpoint < 0.12 or (len(steps) and np.quantile(steps, 0.95) > 0.15):
            continue
        path = float(steps.sum())
        curvature = max(0.0, path / max(endpoint, 1e-8) - 1.0)
        directions = np.diff(points, axis=0)
        norms = np.linalg.norm(directions, axis=1)
        unit = directions[norms > 1e-8] / norms[norms > 1e-8, None]
        alignment = (
            float(np.mean(np.sum(unit[:-1] * unit[1:], axis=1))) if len(unit) > 1 else 1.0
        )
        if part_id == 3:
            score = 1.8 * min(curvature, 0.30) + 0.35 * alignment + endpoint
        else:
            score = 0.55 * alignment - 1.6 * curvature + endpoint
        candidates.append((score, points))
    candidates.sort(key=lambda row: row[0], reverse=True)
    return [points for _, points in candidates[:count]]


def temporal_samples(points: np.ndarray, count: int = 11) -> np.ndarray:
    indices = np.linspace(0, len(points) - 1, min(count, len(points))).round().astype(int)
    return points[np.unique(indices)]


def draw_track(ax, points: np.ndarray, color: str, alpha_scale: float = 1.0) -> None:
    segments = np.stack([points[:-1], points[1:]], axis=1)
    rgb = to_rgba(color)[:3]
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=[
                (*rgb, float(alpha_scale * a))
                for a in np.linspace(0.22, 0.96, len(segments))
            ],
            linewidths=np.linspace(0.72, 2.25, len(segments)),
        )
    )
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color=[
            (*rgb, float(alpha_scale * a))
            for a in np.linspace(0.20, 0.82, len(points))
        ],
        s=np.linspace(2.6, 7.0, len(points)), edgecolors="none", depthshade=False,
    )


def load_gt_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices) + ROOT_OFFSET
    return mesh


def transform_hinge(vertices: np.ndarray, angle: float) -> np.ndarray:
    return (vertices - HINGE_WORLD) @ rotation_z(angle).T + HINGE_WORLD


def add_gt_mesh(
    ax,
    mesh: trimesh.Trimesh,
    color: str,
    alpha: float,
    vertices: np.ndarray | None = None,
    max_faces: int = 2400,
) -> None:
    verts = np.asarray(mesh.vertices) if vertices is None else vertices
    faces = np.asarray(mesh.faces)
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = faces[indices]
    triangles = verts[faces]
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolors=to_rgba(color, alpha),
            edgecolors="none",
            linewidths=0.0,
        )
    )


def synthetic_hinge_tracks(door: trimesh.Trimesh) -> list[np.ndarray]:
    vertices = np.asarray(door.vertices)
    # Sample real door-surface vertices far from the hinge, then replay the GT hinge model.
    radius = np.linalg.norm(vertices[:, :2] - HINGE_WORLD[:2], axis=1)
    valid = np.where(radius > np.quantile(radius, 0.72))[0]
    chosen = []
    for z_fraction in (0.20, 0.43, 0.66, 0.86):
        target_z = np.quantile(vertices[valid, 2], z_fraction)
        index = valid[np.argmin(np.abs(vertices[valid, 2] - target_z))]
        start = vertices[index]
        angles = np.linspace(0.0, HINGE_ANGLE, 11)
        chosen.append(np.asarray([transform_hinge(start[None, :], angle)[0] for angle in angles]))
    return chosen


def synthetic_slide_tracks(drawer: trimesh.Trimesh) -> list[np.ndarray]:
    vertices = np.asarray(drawer.vertices)
    # Concentrate helper tracks around the drawer's central visible surface,
    # where the real tracks are sparse in this view.
    x_targets = np.quantile(vertices[:, 0], [0.24, 0.40, 0.56, 0.72])
    z_targets = np.quantile(vertices[:, 2], [0.40, 0.62])
    tracks = []
    for index, x_target in enumerate(x_targets):
        z_target = z_targets[index % len(z_targets)]
        distance = (
            (vertices[:, 0] - x_target) ** 2
            + 0.7 * (vertices[:, 2] - z_target) ** 2
        )
        start = vertices[int(np.argmin(distance))]
        tracks.append(start + np.linspace(0.0, 1.0, 9)[:, None] * SLIDE_VECTOR)
    return tracks


def equal_axis(ax, points: np.ndarray) -> None:
    low = np.quantile(points, 0.01, axis=0)
    high = np.quantile(points, 0.99, axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.54, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.grid(False)


def render(tracks_path: Path, mesh_dir: Path, output: Path) -> dict:
    artifact = json.loads(tracks_path.read_text(encoding="utf-8"))
    base_a = load_gt_mesh(mesh_dir / "Refrigerator045_Body001.obj")
    base_b = load_gt_mesh(mesh_dir / "Refrigerator045_Body001_Clear.obj")
    door = load_gt_mesh(mesh_dir / "Refrigerator045_fridge_door.obj")
    drawer = load_gt_mesh(mesh_dir / "Refrigerator045_freezer0.obj")

    fig = plt.figure(figsize=(6.1, 5.8), dpi=650)
    ax = fig.add_subplot(111, projection="3d")

    add_gt_mesh(ax, base_a, GRAY, 0.060)
    add_gt_mesh(ax, base_b, GRAY, 0.045)
    add_gt_mesh(ax, door, ORANGE, 0.040)
    door_end = transform_hinge(np.asarray(door.vertices), HINGE_ANGLE)
    add_gt_mesh(ax, door, ORANGE, 0.115, vertices=door_end)
    add_gt_mesh(ax, drawer, BLUE, 0.040)
    drawer_end = np.asarray(drawer.vertices) + SLIDE_VECTOR
    add_gt_mesh(ax, drawer, BLUE, 0.115, vertices=drawer_end)

    real_door = select_real_tracks(artifact["tracks"], 3, 8)
    real_drawer = select_real_tracks(artifact["tracks"], 2, 6)
    for points in real_door:
        draw_track(ax, temporal_samples(points), ORANGE, 0.78)
    for points in real_drawer:
        draw_track(ax, temporal_samples(points), BLUE, 0.78)
    for points in synthetic_hinge_tracks(door):
        draw_track(ax, points, ORANGE, 1.0)
    for points in synthetic_slide_tracks(drawer):
        draw_track(ax, points, BLUE, 1.0)

    ax.plot(
        [HINGE_WORLD[0], HINGE_WORLD[0]],
        [HINGE_WORLD[1], HINGE_WORLD[1]],
        [HINGE_WORLD[2] - 0.44, HINGE_WORLD[2] + 0.46],
        color=ORANGE, alpha=0.22, linewidth=0.75,
    )

    bounds_points = np.concatenate(
        [
            np.asarray(base_a.vertices), np.asarray(base_b.vertices),
            np.asarray(door.vertices), door_end,
            np.asarray(drawer.vertices), drawer_end,
        ],
        axis=0,
    )
    equal_axis(ax, bounds_points)
    # Preserve the original real-track panel camera.
    ax.view_init(elev=23.0, azim=-132.0)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=650, transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    with Image.open(output) as image:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            pad = max(24, int(0.035 * max(right - left, bottom - top)))
            rgba.crop(
                (
                    max(0, left - pad), max(0, top - pad),
                    min(rgba.width, right + pad), min(rgba.height, bottom + pad),
                )
            ).save(output, dpi=(650, 650))

    return {
        "source_tracks": str(tracks_path.resolve()),
        "source_meshes": str(mesh_dir.resolve()),
        "geometry": "Refrigerator045 GT visual meshes at start/end joint states",
        "camera": {"elevation_deg": 23.0, "azimuth_deg": -132.0},
        "real_track_counts": {"revolute": len(real_door), "prismatic": len(real_drawer)},
        "model_consistent_helper_tracks": {"revolute": 4, "prismatic": 4},
        "note": "Real high-quality tracks augmented with model-consistent illustrative tracks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument(
        "--mesh-dir", type=Path,
        default=Path("examples/mujoco_models/Refrigerator045_obj"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    metadata = render(args.tracks, args.mesh_dir, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
