#!/usr/bin/env python3
"""Generate reproducible Track2Art experiment figures from retained artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/track2art-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from scipy.optimize import linear_sum_assignment


PAPER_COLORS = {
    "ink": "#243447",
    "muted": "#657587",
    "light": "#DCE4EA",
    "iou": "#315B7D",
    "count": "#C47A3A",
    "joint": "#3E8177",
    "base": "#8D959D",
    "revolute": "#E67E22",
    "revolute_2": "#C55A45",
    "prismatic": "#3578B8",
    "prismatic_2": "#59A5D8",
    "extra_1": "#6E5AA7",
    "extra_2": "#4C9A70",
    "extra_3": "#D6A72C",
    "extra_4": "#B65C85",
    "extra_5": "#7A6855",
    "failure": "#B44B4B",
}

BUCKET_ORDER = {"2": 0, "3-4": 1, ">=5": 2}
BUCKET_LABELS = {"2": "2 parts", "3-4": "3--4 parts", ">=5": r"$\geq$5 parts"}
MAIN_CONFIGURATION = "cotracker_only + full_neural"
REPRESENTATIVE_FRAME = 60


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: mpl.figure.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=350, facecolor="white")
    plt.close(fig)


def float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def main_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("configuration") == MAIN_CONFIGURATION]
    if not selected and rows and "ours_main" in rows[0]:
        selected = [row for row in rows if row.get("ours_main", "").lower() == "true"]
    if not selected:
        raise ValueError(f"No rows found for validation-selected config: {MAIN_CONFIGURATION}")
    return selected


def make_complexity_figure(source: Path, output_dir: Path) -> None:
    rows = main_rows(read_csv(source))
    rows.sort(key=lambda row: BUCKET_ORDER[row["difficulty_bucket"]])
    counts = [int(row["object_count"]) for row in rows]
    if len(rows) != 3 or sum(counts) != 116 or any(count <= 0 for count in counts):
        raise ValueError(f"Unexpected complexity counts {counts}; expected three nonempty buckets totaling 116")

    metrics = [
        ("Point IoU", "point_iou", PAPER_COLORS["iou"]),
        ("Exact part count", "exact_part_count_rate", PAPER_COLORS["count"]),
        (r"Joint@$20^\circ$", "joint_success_at_20_deg", PAPER_COLORS["joint"]),
    ]
    x = np.arange(3)
    width = 0.235
    fig, ax = plt.subplots(figsize=(3.45, 2.38))
    for index, (label, column, color) in enumerate(metrics):
        values = [float(row[column]) for row in rows]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=label,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.022,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=PAPER_COLORS["ink"],
            )
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Score")
    ax.set_xticks(x, [f"{BUCKET_LABELS[row['difficulty_bucket']]}\n$N={count}$" for row, count in zip(rows, counts, strict=True)])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.yaxis.grid(True, color=PAPER_COLORS["light"], linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.9)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.20)
    save_figure(fig, output_dir / "complexity_scaling")


def select_examples(
    rows: list[dict[str, str]], overrides: dict[str, str | None]
) -> list[dict[str, str]]:
    by_object = {row["object_id"]: row for row in rows}
    selected: list[dict[str, str]] = []
    specs = [
        ("simple", "2", True),
        ("medium", "3-4", False),
        ("complex", ">=5", False),
    ]
    for role, bucket, require_exact in specs:
        override = overrides.get(role)
        if override:
            if override not in by_object:
                raise ValueError(f"Override {override!r} not present in main-method rows")
            candidate = by_object[override]
            if candidate["difficulty_bucket"] != bucket:
                raise ValueError(f"Override {override} is not in requested bucket {bucket}")
            selected.append(candidate)
            continue

        bucket_rows = [row for row in rows if row["difficulty_bucket"] == bucket]
        median_iou = float(np.median([float(row["point_iou"]) for row in bucket_rows]))
        median_joint = float(np.median([float(row["joint_success_at_20_deg"]) for row in bucket_rows]))
        candidates = bucket_rows
        if require_exact:
            candidates = [row for row in candidates if row["exact_part_count"].lower() == "true" and float(row["point_iou"]) >= 0.9]
        if role == "complex":
            underseg = [row for row in candidates if int(row["predicted_part_count"]) < int(row["gt_part_count"])]
            if underseg:
                candidates = underseg
        candidates.sort(
            key=lambda row: (
                abs(float(row["point_iou"]) - median_iou)
                + abs(float(row["joint_success_at_20_deg"]) - median_joint),
                row["object_id"],
            )
        )
        selected.append(candidates[0])
    return selected


def track_point(track: dict[str, Any], frame: int) -> np.ndarray | None:
    visible = [s for s in track["samples"] if s.get("visible") and s.get("xyz_world") is not None]
    if not visible:
        return None
    sample = min(visible, key=lambda s: abs(int(s["frame_index"]) - frame))
    return np.asarray(sample["xyz_world"], dtype=float)


def track_path(track: dict[str, Any]) -> np.ndarray:
    points = [s["xyz_world"] for s in track["samples"] if s.get("visible") and s.get("xyz_world") is not None]
    return np.asarray(points, dtype=float)


def hungarian_color_mapping(tracks: list[dict[str, Any]]) -> tuple[dict[int, int], np.ndarray, list[int], list[int]]:
    gt_ids = sorted({int(track["original_part_id"]) for track in tracks})
    pred_ids = sorted({int(track["part_id"]) for track in tracks})
    overlap = np.zeros((len(pred_ids), len(gt_ids)), dtype=int)
    pred_index = {part_id: i for i, part_id in enumerate(pred_ids)}
    gt_index = {part_id: i for i, part_id in enumerate(gt_ids)}
    for track in tracks:
        overlap[pred_index[int(track["part_id"])], gt_index[int(track["original_part_id"])]] += 1
    row_ids, col_ids = linear_sum_assignment(-overlap)
    return ({pred_ids[r]: gt_ids[c] for r, c in zip(row_ids, col_ids, strict=True)}, overlap, pred_ids, gt_ids)


def assign_colors(
    tracks: list[dict[str, Any]], joints: list[dict[str, Any]], anchor_part_id: int
) -> tuple[dict[int, str], dict[int, str], dict[str, Any]]:
    pred_to_gt, overlap, pred_ids, gt_ids = hungarian_color_mapping(tracks)
    child_types = {int(j["child_part_id"]): j["joint_type"] for j in joints}
    pred_colors: dict[int, str] = {anchor_part_id: PAPER_COLORS["base"]}
    rev_palette = [PAPER_COLORS["revolute"], PAPER_COLORS["revolute_2"], PAPER_COLORS["extra_3"]]
    pri_palette = [PAPER_COLORS["prismatic"], PAPER_COLORS["prismatic_2"]]
    extra_palette = [PAPER_COLORS[f"extra_{i}"] for i in range(1, 6)]
    rev_i = pri_i = extra_i = 0
    for pred_id in pred_ids:
        if pred_id == anchor_part_id:
            continue
        joint_type = child_types.get(pred_id)
        if joint_type == "revolute":
            color = rev_palette[rev_i % len(rev_palette)]
            rev_i += 1
        elif joint_type == "prismatic":
            color = pri_palette[pri_i % len(pri_palette)]
            pri_i += 1
        else:
            color = extra_palette[extra_i % len(extra_palette)]
            extra_i += 1
        pred_colors[pred_id] = color

    gt_colors: dict[int, str] = {}
    for pred_id, gt_id in pred_to_gt.items():
        gt_colors[gt_id] = pred_colors[pred_id]
    for gt_id in gt_ids:
        if gt_id not in gt_colors:
            gt_colors[gt_id] = extra_palette[extra_i % len(extra_palette)]
            extra_i += 1
    metadata = {
        "purpose": "visual_color_alignment_only",
        "predicted_to_gt": {str(k): int(v) for k, v in pred_to_gt.items()},
        "predicted_colors": {str(k): v for k, v in pred_colors.items()},
        "gt_colors": {str(k): v for k, v in gt_colors.items()},
        "overlap_matrix": overlap.tolist(),
        "predicted_ids": pred_ids,
        "gt_ids": gt_ids,
    }
    return pred_colors, gt_colors, metadata


def camera_view(episode: dict[str, Any], points: np.ndarray, frame: int) -> dict[str, Any]:
    pose = np.asarray(episode["frames"][frame]["camera_poses_by_view"][0], dtype=float)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    camera_center = -rotation.T @ translation
    center = np.nanmedian(points, axis=0)
    delta = camera_center - center
    azim = float(np.degrees(np.arctan2(delta[1], delta[0])))
    elev = float(np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]) + 1e-12)))
    return {
        "frame_index": frame,
        "view_index": 0,
        "world_to_camera": pose.tolist(),
        "camera_center_world": camera_center.tolist(),
        "matplotlib_view": {"elev_deg": elev, "azim_deg": azim},
    }


def set_equal_3d(ax: Any, points: np.ndarray, camera: dict[str, Any]) -> None:
    finite = points[np.all(np.isfinite(points), axis=1)]
    low, high = np.percentile(finite, [1, 99], axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.58, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=camera["matplotlib_view"]["elev_deg"], azim=camera["matplotlib_view"]["azim_deg"])
    ax.set_axis_off()
    ax.set_facecolor("white")


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs: list[float], ys: list[float], zs: list[float], *args: Any, **kwargs: Any) -> None:
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer: Any = None) -> float:
        xs, ys, zs = self._verts3d
        x2, y2, z2 = proj3d.proj_transform(xs, ys, zs, self.axes.get_proj())
        self.set_positions((x2[0], y2[0]), (x2[1], y2[1]))
        return float(np.min(z2))


def add_joint_geometry(ax: Any, joints: list[dict[str, Any]], points: np.ndarray, pred_colors: dict[int, str]) -> None:
    center = np.nanmedian(points, axis=0)
    diag = float(np.linalg.norm(np.nanpercentile(points, 97, axis=0) - np.nanpercentile(points, 3, axis=0)))
    length = max(0.45 * diag, 0.05)
    for joint in joints:
        axis = np.asarray(joint["axis"], dtype=float)
        axis /= np.linalg.norm(axis) + 1e-12
        color = pred_colors[int(joint["child_part_id"])]
        if joint["joint_type"] == "revolute":
            pivot = np.asarray(joint["pivot"], dtype=float)
            nearest = pivot + axis * np.dot(center - pivot, axis)
            ends = np.vstack([nearest - axis * length, nearest + axis * length])
            line_width = 1.55 if len(joints) <= 3 else 0.95
            ax.plot(ends[:, 0], ends[:, 1], ends[:, 2], color=color, linewidth=line_width, alpha=0.88, zorder=20)
            if np.linalg.norm(pivot - center) < 1.5 * diag:
                ax.scatter(*pivot, color=color, edgecolor="white", linewidth=0.35, s=14, zorder=21)
            # The compact arc is symbolic; the fitted axis and line remain exact.
            basis = np.cross(axis, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(basis) < 1e-5:
                basis = np.cross(axis, np.array([0.0, 1.0, 0.0]))
            basis /= np.linalg.norm(basis)
            basis2 = np.cross(axis, basis)
            theta = np.linspace(0.15, 1.65 * np.pi, 34)
            arc = nearest + 0.16 * diag * (np.cos(theta)[:, None] * basis + np.sin(theta)[:, None] * basis2)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, linewidth=0.85, alpha=0.9)
        else:
            ends = np.vstack([center - axis * length, center + axis * length])
            arrow = Arrow3D(
                ends[:, 0].tolist(), ends[:, 1].tolist(), ends[:, 2].tolist(),
                arrowstyle="<|-|>", mutation_scale=8, linewidth=1.35 if len(joints) <= 3 else 0.9, color=color,
            )
            ax.add_artist(arrow)


def draw_graph_inset(ax: Any, joints: list[dict[str, Any]], anchor_part_id: int, pred_colors: dict[int, str]) -> None:
    inset = ax.inset_axes([0.02, 0.01, 0.72, 0.34])
    inset.set_axis_off()
    children = [int(j["child_part_id"]) for j in joints]
    positions = {anchor_part_id: (0.5, 0.82)}
    for i, child in enumerate(children):
        positions[child] = ((i + 1) / (len(children) + 1), 0.18)
    for joint in joints:
        parent, child = int(joint["parent_part_id"]), int(joint["child_part_id"])
        if parent not in positions:
            positions[parent] = (0.5, 0.5)
        p0, p1 = positions[parent], positions[child]
        color = pred_colors[child]
        inset.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": color})
        label = "R" if joint["joint_type"] == "revolute" else "P"
        if len(children) <= 4:
            inset.text((p0[0] + p1[0]) / 2 + 0.025, (p0[1] + p1[1]) / 2, label, color=color, fontsize=5.2, weight="bold")
    for part_id, (x, y) in positions.items():
        inset.scatter([x], [y], s=42, color=pred_colors.get(part_id, PAPER_COLORS["base"]), edgecolor="white", linewidth=0.6, zorder=3)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)


def crop_rgb_to_mask(rgb_path: Path, mask_path: Path, padding_ratio: float = 0.14) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path))
    foreground = mask > 0
    if foreground.ndim == 3:
        foreground = np.any(foreground, axis=2)
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        return rgb
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = int(max(x1 - x0, y1 - y0) * padding_ratio)
    x0, x1 = max(0, x0 - pad), min(rgb.width, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(rgb.height, y1 + pad)
    return rgb.crop((x0, y0, x1, y1))


def plot_qualitative_row(
    fig: mpl.figure.Figure,
    axes: list[Any],
    row: dict[str, str],
    raw_root: Path,
) -> dict[str, Any]:
    object_id = row["object_id"]
    artifact_root = raw_root / object_id
    tracks_path = artifact_root / "hybrid" / "motion_part_tracks_slots.json"
    joints_path = artifact_root / "hybrid" / "joint_inference.json"
    episode_path = artifact_root / "recording" / "episode.json"
    rgb_path = artifact_root / "recording" / "rgb.png"
    mask_path = artifact_root / "recording" / "part_mask.png"
    for path in (tracks_path, joints_path, episode_path, rgb_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing qualitative artifact: {path}")

    track_data = json.loads(tracks_path.read_text())
    joint_data = json.loads(joints_path.read_text())
    episode = json.loads(episode_path.read_text())
    tracks = track_data["tracks"]
    joints = joint_data["joints"]
    anchor_part_id = int(joint_data["anchor_part_id"])
    pred_colors, gt_colors, matching = assign_colors(tracks, joints, anchor_part_id)

    points, gt_ids, pred_ids = [], [], []
    for track in tracks:
        point = track_point(track, REPRESENTATIVE_FRAME)
        if point is not None:
            points.append(point)
            gt_ids.append(int(track["original_part_id"]))
            pred_ids.append(int(track["part_id"]))
    points_array = np.asarray(points)
    camera = camera_view(episode, points_array, REPRESENTATIVE_FRAME)

    axes[0].imshow(crop_rgb_to_mask(rgb_path, mask_path))
    axes[0].set_axis_off()

    trajectory_segments, trajectory_colors = [], []
    stride = max(1, math.ceil(len(tracks) / 180))
    for track in tracks[::stride]:
        path = track_path(track)
        if len(path) >= 2:
            trajectory_segments.append(path)
            trajectory_colors.append(pred_colors[int(track["part_id"])])
    axes[1].add_collection3d(Line3DCollection(trajectory_segments, colors=trajectory_colors, linewidths=0.65, alpha=0.55))
    axes[1].scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c=[pred_colors[x] for x in pred_ids], s=2.0, alpha=0.8)
    set_equal_3d(axes[1], points_array, camera)

    axes[2].scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c=[gt_colors[x] for x in gt_ids], s=3.2, alpha=0.92, depthshade=False)
    set_equal_3d(axes[2], points_array, camera)
    axes[3].scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c=[pred_colors[x] for x in pred_ids], s=3.2, alpha=0.92, depthshade=False)
    set_equal_3d(axes[3], points_array, camera)
    axes[4].scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c=[pred_colors[x] for x in pred_ids], s=2.2, alpha=0.28, depthshade=False)
    add_joint_geometry(axes[4], joints, points_array, pred_colors)
    draw_graph_inset(axes[4], joints, anchor_part_id, pred_colors)
    set_equal_3d(axes[4], points_array, camera)

    bucket = row["difficulty_bucket"]
    axes[0].text(
        -0.04, 0.5, BUCKET_LABELS[bucket], transform=axes[0].transAxes, rotation=90,
        ha="right", va="center", fontsize=7.0, color=PAPER_COLORS["ink"], weight="bold",
    )
    metrics = (
        f"{object_id.replace('partnet_', 'PN-')}\n"
        f"IoU {float(row['point_iou']):.2f}  |  Parts {row['predicted_part_count']}/{row['gt_part_count']}\n"
        f"J@$20^\\circ$ {float(row['joint_success_at_20_deg']):.2f}"
    )
    axes[0].text(
        0.025, 0.02, metrics, transform=axes[0].transAxes, ha="left", va="bottom",
        fontsize=5.0, linespacing=1.16, color="white",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
    )

    return {
        "object_id": object_id,
        "category": row["category"],
        "bucket": bucket,
        "gt_part_count": int(row["gt_part_count"]),
        "predicted_part_count": int(row["predicted_part_count"]),
        "metrics": {
            "point_iou": float(row["point_iou"]),
            "joint_success_at_20_deg": float(row["joint_success_at_20_deg"]),
            "failure_penalized_axis_error_mean_deg": float_or_nan(row["failure_penalized_axis_error_mean_deg"]),
        },
        "prediction_artifacts": {
            "rgb": str(rgb_path.resolve()),
            "tracks": str(tracks_path.resolve()),
            "part_poses": str((artifact_root / "hybrid" / "part_poses.json").resolve()),
            "joint_inference": str(joints_path.resolve()),
            "episode": str(episode_path.resolve()),
        },
        "camera": camera,
        "color_matching": matching,
    }


def make_qualitative_figure(
    per_object_source: Path,
    raw_root: Path,
    output_dir: Path,
    overrides: dict[str, str | None],
) -> list[dict[str, Any]]:
    rows = main_rows(read_csv(per_object_source))
    selected = select_examples(rows, overrides)
    fig = plt.figure(figsize=(7.08, 5.55))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.10, 1.08, 1.0, 1.0, 1.20], wspace=0.015, hspace=0.075)
    column_headers = ["RGB", "4D trajectories", "GT parts", "Track2Art", "Graph + joints"]
    metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected):
        axes: list[Any] = []
        axes.append(fig.add_subplot(gs[row_index, 0]))
        for column_index in range(1, 5):
            axes.append(fig.add_subplot(gs[row_index, column_index], projection="3d"))
        if row_index == 0:
            for ax, label in zip(axes, column_headers, strict=True):
                ax.text2D(0.5, 1.03, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=PAPER_COLORS["ink"], weight="bold") if hasattr(ax, "text2D") else ax.text(0.5, 1.03, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=PAPER_COLORS["ink"], weight="bold")
        metadata.append(plot_qualitative_row(fig, axes, row, raw_root))
    fig.subplots_adjust(left=0.04, right=0.995, top=0.965, bottom=0.02)
    save_figure(fig, output_dir / "qualitative_results")

    selection_payload = {
        "main_configuration": MAIN_CONFIGURATION,
        "configuration_selection": "validation-selected; test-set metrics were not used to select the configuration",
        "selection_rule": (
            "Within each required GT-part bucket, minimize L1 distance to the bucket median in Point IoU and Joint@20; "
            "the simple example additionally requires exact part count and IoU >= 0.9; the complex example prefers "
            "under-segmented predictions; object_id is the deterministic tie-breaker. CLI overrides replace this rule."
        ),
        "representative_frame": REPRESENTATIVE_FRAME,
        "gt_usage": "GT part IDs are used only for evaluation and Hungarian visualization color matching, never to alter predictions.",
        "selected": metadata,
    }
    (output_dir / "qualitative_selection.json").write_text(json.dumps(selection_payload, indent=2, allow_nan=False) + "\n")
    return metadata


def make_difficulty_figure(source: Path, output_dir: Path) -> None:
    rows = main_rows(read_csv(source))
    rows.sort(key=lambda row: (BUCKET_ORDER[row["difficulty_bucket"]], -float(row["point_iou"]), row["object_id"]))
    if len(rows) != 116:
        raise ValueError(f"Expected 116 main-method objects, found {len(rows)}")
    x = np.arange(len(rows))
    colors = [PAPER_COLORS["iou"] if r["difficulty_bucket"] == "2" else PAPER_COLORS["joint"] if r["difficulty_bucket"] == "3-4" else PAPER_COLORS["count"] for r in rows]
    iou = np.array([float(row["point_iou"]) for row in rows])
    joint = np.array([float_or_nan(row["joint_success_at_20_deg"]) for row in rows])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7.08, 3.10), sharex=True, gridspec_kw={"height_ratios": [1.0, 0.78], "hspace": 0.12})
    ax_top.bar(x, iou, width=0.88, color=colors, edgecolor="none")
    ax_top.set_ylabel("Point IoU")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_yticks([0, 0.5, 1.0])
    ax_bottom.scatter(x, joint, c=colors, s=7, linewidths=0, alpha=0.9)
    ax_bottom.set_ylabel(r"Joint@$20^\circ$")
    ax_bottom.set_ylim(-0.04, 1.05)
    ax_bottom.set_yticks([0, 0.5, 1.0])
    ax_bottom.set_xlabel("Held-out objects (easy-to-hard within each complexity bucket)")

    boundaries, centers = [], []
    cursor = 0
    for bucket in ("2", "3-4", ">=5"):
        count = sum(row["difficulty_bucket"] == bucket for row in rows)
        centers.append((cursor + (count - 1) / 2, bucket, count))
        cursor += count
        if cursor < len(rows):
            boundaries.append(cursor - 0.5)
    for ax in (ax_top, ax_bottom):
        for boundary in boundaries:
            ax.axvline(boundary, color=PAPER_COLORS["muted"], linewidth=0.65, linestyle=(0, (2, 2)))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    for center, bucket, count in centers:
        ax_top.text(center, 1.015, f"{BUCKET_LABELS[bucket]}  ($N={count}$)", ha="center", va="bottom", fontsize=7.0, color=PAPER_COLORS["ink"])
    ax_bottom.set_xticks([])

    # Label only representative large segmentation failures, not every object.
    failure_indices = sorted(range(len(rows)), key=lambda i: (iou[i], rows[i]["object_id"]))[:3]
    for label_index, i in enumerate(failure_indices):
        ax_top.annotate(
            rows[i]["object_id"].replace("partnet_", "PN-"), xy=(i, iou[i]), xytext=(-8 + 10 * label_index, 8 + 9 * label_index),
            textcoords="offset points", ha="center", va="bottom", fontsize=5.4, rotation=35,
            color=PAPER_COLORS["failure"], arrowprops={"arrowstyle": "-", "lw": 0.45, "color": PAPER_COLORS["failure"]},
        )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.16)
    save_figure(fig, output_dir / "per_object_difficulty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controlled-dir",
        type=Path,
        default=Path("outputs/partnet_core_v1_training/ours_controlled_common_domain_v1"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("paper_assets/experiments/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_assets/experiments"))
    parser.add_argument("--simple-object")
    parser.add_argument("--medium-object")
    parser.add_argument("--complex-object")
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_complexity_figure(args.controlled_dir / "ours_five_config_by_complexity.csv", args.output_dir)
    selected: list[dict[str, Any]] = []
    if not args.skip_qualitative:
        selected = make_qualitative_figure(
            args.controlled_dir / "ours_five_config_per_object.csv",
            args.raw_root,
            args.output_dir,
            {"simple": args.simple_object, "medium": args.medium_object, "complex": args.complex_object},
        )
    make_difficulty_figure(args.controlled_dir / "ours_five_config_per_object.csv", args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "selected_objects": [item["object_id"] for item in selected]}, indent=2))


if __name__ == "__main__":
    main()
