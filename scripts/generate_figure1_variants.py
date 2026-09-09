#!/usr/bin/env python3
"""Generate two compact Figure-1 candidates for Track2Art."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets" / "refrigerator045"
OUTPUT = ROOT / "outputs" / "figure1_variants"
MESH_DIR = ROOT / "examples" / "mujoco_models" / "Refrigerator045_obj"

GRAY = "#858B93"
ORANGE = "#F28E2B"
BLUE = "#397BC5"
NAVY = "#315A9D"
ROOT_OFFSET = np.array([0.0, 0.0, 0.938160])
HINGE = np.array([0.391572, -0.367582, 0.382577]) + ROOT_OFFSET
ANGLE = 1.22
SLIDE = np.array([0.0, -0.42, 0.0])


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def load_mesh(name: str) -> trimesh.Trimesh:
    mesh = trimesh.load(MESH_DIR / name, force="mesh", process=False).copy()
    mesh.vertices = np.asarray(mesh.vertices) + ROOT_OFFSET
    return mesh


def transform_door(points: np.ndarray, angle: float = ANGLE) -> np.ndarray:
    return (points - HINGE) @ rotation_z(angle).T + HINGE


def sample(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    return np.asarray(mesh.sample(count))


def equal_axes(ax, points: np.ndarray) -> None:
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    radius = np.max(high - low) * 0.57
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def save_transparent(fig, path: Path) -> None:
    fig.savefig(path, dpi=450, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox:
            l, t, r, b = bbox
            pad = 24
            rgba.crop((max(0, l - pad), max(0, t - pad), min(rgba.width, r + pad), min(rgba.height, b + pad))).save(path)


def render_open_model(path: Path) -> None:
    base = [load_mesh("Refrigerator045_Body001.obj"), load_mesh("Refrigerator045_Body001_Clear.obj")]
    door = load_mesh("Refrigerator045_fridge_door.obj")
    drawer = load_mesh("Refrigerator045_freezer0.obj")
    base_pts = np.concatenate([sample(base[0], 1100, 1), sample(base[1], 650, 2)])
    door_pts = transform_door(sample(door, 720, 3))
    drawer_pts = sample(drawer, 480, 4) + SLIDE
    all_pts = np.concatenate([base_pts, door_pts, drawer_pts])

    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*base_pts.T, s=4.0, c=GRAY, alpha=0.78, depthshade=False)
    ax.scatter(*door_pts.T, s=4.8, c=ORANGE, alpha=0.92, depthshade=False)
    ax.scatter(*drawer_pts.T, s=4.8, c=BLUE, alpha=0.92, depthshade=False)
    ax.plot([HINGE[0], HINGE[0]], [HINGE[1], HINGE[1]], [HINGE[2] - 0.52, HINGE[2] + 0.56], color=ORANGE, lw=2.2)
    drawer_center = np.median(drawer_pts, axis=0)
    ax.quiver(*drawer_center, *np.array([0, -0.62, 0]), color=BLUE, linewidth=2.0, arrow_length_ratio=0.18)
    equal_axes(ax, all_pts)
    ax.view_init(elev=23, azim=-132)
    save_transparent(fig, path)


def render_two_state_cloud(path: Path) -> None:
    base = load_mesh("Refrigerator045_Body001.obj")
    door = load_mesh("Refrigerator045_fridge_door.obj")
    drawer = load_mesh("Refrigerator045_freezer0.obj")
    base_pts = sample(base, 850, 11)
    door_start = sample(door, 420, 12)
    drawer_start = sample(drawer, 320, 13)
    door_end = transform_door(door_start)
    drawer_end = drawer_start + SLIDE
    all_pts = np.concatenate([base_pts, door_start, drawer_start, door_end, drawer_end])

    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*base_pts.T, s=3.5, c=GRAY, alpha=0.34, depthshade=False)
    ax.scatter(*door_start.T, s=3.8, c=ORANGE, alpha=0.18, depthshade=False)
    ax.scatter(*drawer_start.T, s=3.8, c=BLUE, alpha=0.18, depthshade=False)
    ax.scatter(*door_end.T, s=4.3, c=ORANGE, alpha=0.78, depthshade=False)
    ax.scatter(*drawer_end.T, s=4.3, c=BLUE, alpha=0.78, depthshade=False)
    equal_axes(ax, all_pts)
    ax.view_init(elev=23, azim=-132)
    save_transparent(fig, path)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def contain(image: Image.Image, box: tuple[int, int, int, int], padding: int = 12) -> Image.Image:
    x0, y0, x1, y1 = box
    width, height = x1 - x0 - 2 * padding, y1 - y0 - 2 * padding
    result = image.copy().convert("RGBA")
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (x1 - x0, y1 - y0), (255, 255, 255, 0))
    canvas.alpha_composite(result, ((canvas.width - result.width) // 2, (canvas.height - result.height) // 2))
    return canvas


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=NAVY, width=6)
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    vector /= np.linalg.norm(vector)
    normal = np.array([-vector[1], vector[0]])
    tip = np.asarray(end, dtype=float)
    base = tip - 22 * vector
    polygon = [tuple(tip), tuple(base + 10 * normal), tuple(base - 10 * normal)]
    draw.polygon(polygon, fill=NAVY)


def build_figure(representation: Path, output: Path, representation_title: str) -> None:
    width, height = 1800, 1180
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    margin, gap = 70, 95
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = 470
    boxes = {
        "input": (margin, 75, margin + panel_w, 75 + panel_h),
        "repr": (margin + panel_w + gap, 75, width - margin, 75 + panel_h),
        "parts": (margin, 635, margin + panel_w, 635 + panel_h),
        "model": (margin + panel_w + gap, 635, width - margin, 635 + panel_h),
    }
    titles = {
        "input": "RGB-D Interaction",
        "repr": representation_title,
        "parts": "Motion-aware Part Decomposition",
        "model": "Recovered Articulated Model",
    }
    images = {
        "input": ASSETS / "input_rgb.png",
        "repr": representation,
        "parts": ASSETS / "part_segmentation_gt.png",
        "model": OUTPUT / "gt_open_articulated_model.png",
    }
    title_font = font(31, True)
    for key, box in boxes.items():
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=24, fill="#FBFCFE", outline="#D5DFEC", width=3)
        text_box = draw.textbbox((0, 0), titles[key], font=title_font)
        text_w = text_box[2] - text_box[0]
        draw.text(((x0 + x1 - text_w) / 2, y0 + 17), titles[key], fill="#17253A", font=title_font)
        image = Image.open(images[key]).convert("RGBA")
        inner = (x0 + 12, y0 + 62, x1 - 12, y1 - 12)
        fitted = contain(image, inner, padding=10)
        canvas.paste(fitted, (inner[0], inner[1]), fitted)

    # Z-shaped feed-forward reading order; no cyclic implication.
    arrow(draw, (margin + panel_w + 15, 310), (margin + panel_w + gap - 15, 310))
    arrow(draw, (width - margin - 70, 75 + panel_h + 18), (margin + 70, 635 - 18))
    arrow(draw, (margin + panel_w + 15, 870), (margin + panel_w + gap - 15, 870))

    canvas.save(output, dpi=(300, 300))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    open_model = OUTPUT / "gt_open_articulated_model.png"
    pointcloud = OUTPUT / "two_state_pointcloud.png"
    render_open_model(open_model)
    render_two_state_cloud(pointcloud)
    build_figure(pointcloud, OUTPUT / "figure1_pointcloud_version.png", "Two-state 4D Point Clouds")
    build_figure(ASSETS / "tracks_4d_clean.png", OUTPUT / "figure1_pointflow_version.png", "Persistent 4D Point Flows")
    manifest = {
        "layout": "2x2 Z-flow",
        "gt_open_model": str(open_model),
        "variants": [
            str(OUTPUT / "figure1_pointcloud_version.png"),
            str(OUTPUT / "figure1_pointflow_version.png"),
        ],
        "note": "GT semantic visualization; door and drawer are both open in the final model.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
