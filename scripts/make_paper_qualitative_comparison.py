#!/usr/bin/env python3
"""Build the object-aligned Track2Art qualitative comparison figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


DEFAULT_OBJECTS = {
    "simple": "partnet_46556",
    "medium": "partnet_10638",
    "complex": "partnet_103069",
}
METHODS = ("gt", "track2art", "artgs", "videoartgs", "dta", "aim", "reart", "paris")
METHOD_LABELS = {
    "gt": "GT",
    "track2art": "Track2Art",
    "artgs": "ArtGS",
    "videoartgs": "VideoArtGS",
    "dta": "DTA",
    "aim": "AiM",
    "reart": "ReArt",
    "paris": "PARIS",
}
METHOD_MANIFEST_NAMES = {
    "track2art": "ours_hybrid",
    "artgs": "artgs",
    "videoartgs": "videoartgs",
    "dta": "dta",
    "aim": "aim_aligned",
    "reart": "reart",
    "paris": "paris",
}
PALETTE = (
    "#90989D",  # base/static
    "#E6862D",  # first revolute child
    "#3979B9",  # first prismatic child
    "#4F9D69",
    "#8066A8",
    "#9A6B4F",
    "#36A6A6",
    "#C95F62",
    "#B49A35",
    "#5E7895",
)
UNMATCHED = "#D18BC2"
REVOLUTE = "#E67E22"
PRISMATIC = "#2676B8"


@dataclass
class Cloud:
    points: np.ndarray
    pred_labels: np.ndarray
    canonical_gt_labels: np.ndarray
    native_path: str
    joint_axes: list[dict[str, Any]]
    coordinate_contract: str
    axis_contract: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    with path.open() as handle:
        for line in handle:
            header_lines += 1
            if line.strip() == "end_header":
                break
    values = np.loadtxt(path, skiprows=header_lines)
    points = values[:, :3].astype(float)
    rgb = values[:, 3:6].astype(int)
    _, labels = np.unique(rgb, axis=0, return_inverse=True)
    return points, labels.astype(int)


def deterministic_sample(*arrays: np.ndarray, limit: int) -> tuple[np.ndarray, ...]:
    count = len(arrays[0])
    if count <= limit:
        return arrays
    indices = np.linspace(0, count - 1, limit, dtype=int)
    return tuple(array[indices] for array in arrays)


def canonicalize_labels(points: np.ndarray, gt_points: np.ndarray, gt_labels: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    distances, indices = cKDTree(gt_points).query(points, k=1)
    diagonal = max(float(np.linalg.norm(np.ptp(gt_points, axis=0))), 1e-8)
    return gt_labels[indices], {
        "mean_nearest_distance_bbox": float(np.mean(distances) / diagonal),
        "p95_nearest_distance_bbox": float(np.percentile(distances, 95) / diagonal),
    }


def hungarian_mapping(pred: np.ndarray, gt: np.ndarray) -> tuple[dict[int, int], dict[str, Any]]:
    pred_ids = np.unique(pred)
    gt_ids = np.unique(gt)
    overlap = np.zeros((len(pred_ids), len(gt_ids)), dtype=int)
    for row, pred_id in enumerate(pred_ids):
        for column, gt_id in enumerate(gt_ids):
            overlap[row, column] = int(np.sum((pred == pred_id) & (gt == gt_id)))
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {int(pred_ids[row]): int(gt_ids[column]) for row, column in zip(rows, columns) if overlap[row, column] > 0}
    return mapping, {
        "predicted_ids": pred_ids.astype(int).tolist(),
        "gt_ids": gt_ids.astype(int).tolist(),
        "overlap": overlap.tolist(),
        "mapping": {str(key): value for key, value in mapping.items()},
    }


def axis_rows(path: Path, object_id: str, gt: bool) -> list[dict[str, Any]]:
    rows = load_json(path)["joints"]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("object_id") != object_id or row.get("setting") != "full_neural":
            continue
        joint_id = str(row.get("joint_id"))
        if gt and joint_id in seen:
            continue
        if gt:
            axis = row.get("gt_axis")
            origin = row.get("gt_line_point")
            joint_type = row.get("joint_type")
            seen.add(joint_id)
        else:
            if not row.get("edge_detected") or not row.get("valid"):
                continue
            axis = row.get("estimated_axis")
            origin = row.get("estimated_line_point")
            joint_type = row.get("estimated_joint_type")
        if axis is None or origin is None or joint_type not in {"revolute", "prismatic"}:
            continue
        selected.append({
            "joint_id": joint_id,
            "joint_type": joint_type,
            "axis": axis,
            "origin": origin,
            "parent_slot": row.get("parent_slot"),
            "child_slot": row.get("child_slot"),
            "parent_hard_assigned_track_count": row.get("parent_hard_assigned_track_count"),
            "child_hard_assigned_track_count": row.get("child_hard_assigned_track_count"),
            "source": "GT evaluation metadata" if gt else "Hybrid full-neural relation output",
        })
    return selected


def attach_relation_part_ids(
    joints: list[dict[str, Any]],
    pred_labels: np.ndarray,
) -> dict[str, Any]:
    """Map relation-head slot IDs to displayed part IDs without relying on row order."""
    part_counts = {
        int(part_id): int(np.sum(pred_labels == part_id))
        for part_id in np.unique(pred_labels)
    }
    slot_counts: dict[int, list[int]] = {}
    for joint in joints:
        for role in ("parent", "child"):
            slot = joint.get(f"{role}_slot")
            count = joint.get(f"{role}_hard_assigned_track_count")
            if slot is not None and count is not None:
                slot_counts.setdefault(int(slot), []).append(int(count))
    representative_counts = {
        slot: int(round(float(np.median(counts))))
        for slot, counts in slot_counts.items()
    }
    slots = sorted(representative_counts)
    parts = sorted(part_counts)
    slot_to_part: dict[int, int] = {}
    if slots and parts:
        cost = np.asarray([
            [abs(representative_counts[slot] - part_counts[part]) for part in parts]
            for slot in slots
        ], dtype=float)
        rows, columns = linear_sum_assignment(cost)
        slot_to_part = {slots[row]: parts[column] for row, column in zip(rows, columns)}
    for joint in joints:
        parent_slot = joint.get("parent_slot")
        child_slot = joint.get("child_slot")
        if parent_slot is not None:
            joint["parent_part_id"] = slot_to_part.get(int(parent_slot))
        if child_slot is not None:
            joint["child_part_id"] = slot_to_part.get(int(child_slot))
        joint["relation_part_mapping"] = "slot_track_count_hungarian"
    return {
        "slot_to_part_id": {str(slot): part for slot, part in slot_to_part.items()},
        "slot_track_counts": {str(slot): count for slot, count in representative_counts.items()},
        "part_track_counts": {str(part): count for part, count in part_counts.items()},
    }


def load_metric_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["object_id"], row["method"]): row for row in rows}


def build_gt_palette(gt_labels: np.ndarray, gt_axes: list[dict[str, Any]]) -> dict[int, str]:
    counts = {int(label): int(np.sum(gt_labels == label)) for label in np.unique(gt_labels)}
    base = max(counts, key=counts.get)
    remaining = sorted((label for label in counts if label != base), key=lambda label: (-counts[label], label))
    colors = {base: PALETTE[0]}
    types = [axis["joint_type"] for axis in gt_axes]
    revolute_used = False
    prismatic_used = False
    extras = iter(PALETTE[3:])
    for index, label in enumerate(remaining):
        joint_type = types[index] if index < len(types) else None
        if joint_type == "revolute" and not revolute_used:
            colors[label] = PALETTE[1]
            revolute_used = True
        elif joint_type == "prismatic" and not prismatic_used:
            colors[label] = PALETTE[2]
            prismatic_used = True
        else:
            colors[label] = next(extras, PALETTE[(index + 3) % len(PALETTE)])
    return colors


def load_cloud(
    method: str,
    object_id: str,
    raw_root: Path,
    baseline_root: Path,
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    analytic_axes: Path,
) -> tuple[Cloud, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    if method in {"gt", "track2art", "aim", "reart"}:
        path = raw_root / object_id / f"{method}.json"
        data = load_json(path)
        points = np.asarray(data["points"], dtype=float)
        if method in {"gt", "aim", "reart"}:
            native_gt = np.asarray(data.get("gt_part_id"), dtype=int)
            valid = native_gt >= 0
            points = points[valid]
            pred = np.asarray(data.get("pred_part_id"), dtype=int)[valid]
            canonical_gt = native_gt[valid]
            canonicalized, diagnostics = canonicalize_labels(points, gt_points, gt_labels)
            if method != "gt":
                canonical_gt = canonicalized
        else:
            pred = np.asarray(data["pred_part_id"], dtype=int)
            canonical_gt, diagnostics = canonicalize_labels(points, gt_points, gt_labels)
        diagnostics["compact_artifact_path"] = str(path)
        if method == "gt":
            pred = canonical_gt.copy()
            joints = axis_rows(analytic_axes, object_id, gt=True)
            axis_contract = "GT axes from common evaluation metadata"
        elif method == "track2art":
            joints = axis_rows(analytic_axes, object_id, gt=False)
            diagnostics["relation_slot_mapping"] = attach_relation_part_ids(joints, pred)
            axis_contract = "Hybrid + full-neural predictions; missing edges remain missing"
        else:
            joints = data.get("joint_axes", [])
            axis_contract = (
                "native AiM motion JSON" if method == "aim" else "converted from native ReArt part poses"
            )
        cloud = Cloud(
            points,
            pred,
            canonical_gt,
            data.get("source_path", str(path)),
            joints,
            "common object/world frame",
            axis_contract,
        )
        return cloud, diagnostics

    if method in {"artgs", "videoartgs", "paris"}:
        adapter = baseline_root / object_id / method / "adapter"
        path = adapter / (
            "start_labeled_gaussians.ply" if method == "artgs" else
            "labeled_parts.ply" if method == "paris" else
            "start_labeled_parts.ply"
        )
        points, pred = read_ascii_ply(path)
        joints_data = load_json(adapter / "predictions.json").get("joints", [])
        joints = [{
            "joint_id": f"{method}_{index}",
            "joint_type": "revolute" if item.get("type") in {"r", "revolute"} else "prismatic",
            "axis": item.get("axis_direction"),
            "origin": item.get("axis_position"),
            "child_part_id": item.get("child_part_id", item.get("child")),
            "source": f"native {METHOD_LABELS[method]} prediction adapter",
        } for index, item in enumerate(joints_data)]
        canonical_gt, diagnostics = canonicalize_labels(points, gt_points, gt_labels)
        return Cloud(
            points,
            pred,
            canonical_gt,
            str(path),
            joints,
            "object-aligned native adapter frame",
            f"native {METHOD_LABELS[method]} joint output",
        ), diagnostics

    if method == "dta":
        adapter = baseline_root / object_id / method / "adapter"
        path = adapter / "start_labeled_parts.ply"
        points, pred = read_ascii_ply(path)
        canonical_gt, diagnostics = canonicalize_labels(points, gt_points, gt_labels)
        return Cloud(
            points,
            pred,
            canonical_gt,
            str(path),
            [],
            "two-state object frame",
            "unsupported: released non-GT path exposes dual hypotheses without native selection",
        ), diagnostics
    raise KeyError(method)


def normalize_axis(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str] | None:
    axis = item.get("axis") or item.get("axis_direction")
    origin = item.get("origin") or item.get("pivot") or item.get("axis_position")
    joint_type = item.get("joint_type") or item.get("type")
    if axis is None or origin is None or joint_type not in {"revolute", "prismatic", "r", "p"}:
        return None
    direction = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    return np.asarray(origin, dtype=float), direction / norm, "revolute" if joint_type in {"revolute", "r"} else "prismatic"


def draw_joint(ax: Any, item: dict[str, Any], diagonal: float) -> None:
    normalized = normalize_axis(item)
    if normalized is None:
        return
    origin, direction, joint_type = normalized
    length = diagonal * 0.72
    color = REVOLUTE if joint_type == "revolute" else PRISMATIC
    start = origin - direction * length / 2
    end = origin + direction * length / 2
    ax.plot(*zip(start, end), color=color, lw=1.45, zorder=10, solid_capstyle="round")
    if joint_type == "revolute":
        ax.scatter(*origin, c=color, s=11, edgecolor="white", linewidth=0.35, depthshade=False, zorder=11)
        helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        u = np.cross(direction, helper)
        u /= max(np.linalg.norm(u), 1e-8)
        v = np.cross(direction, u)
        angles = np.linspace(math.radians(25), math.radians(285), 35)
        radius = diagonal * 0.085
        arc = origin + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, lw=1.0, zorder=10)
    else:
        for sign in (-1.0, 1.0):
            ax.quiver(
                *(origin - sign * direction * length * 0.36),
                *(sign * direction),
                length=length * 0.35,
                normalize=True,
                color=color,
                linewidth=1.25,
                arrow_length_ratio=0.18,
                zorder=10,
            )


def style_axis(ax: Any, center: np.ndarray, span: float, azimuth: float, elevation: float) -> None:
    ax.set_proj_type("ortho")
    ax.view_init(elev=elevation, azim=azimuth)
    half = span / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_facecolor("white")


def render_cloud(
    ax: Any,
    cloud: Cloud,
    color_map: dict[int, str],
    mapping: dict[int, int],
    center: np.ndarray,
    span: float,
    azimuth: float,
    elevation: float,
    point_limit: int,
    show_joints: bool,
) -> None:
    points, pred = deterministic_sample(cloud.points, cloud.pred_labels, limit=point_limit)
    colors = [color_map.get(mapping.get(int(label), -999), UNMATCHED) for label in pred]
    unmatched = np.array([int(label) not in mapping for label in pred])
    matched = ~unmatched
    if np.any(matched):
        ax.scatter(
            points[matched, 0], points[matched, 1], points[matched, 2],
            c=np.asarray(colors)[matched], s=2.1, alpha=0.93, linewidths=0, depthshade=False,
        )
    if np.any(unmatched):
        ax.scatter(
            points[unmatched, 0], points[unmatched, 1], points[unmatched, 2],
            c=UNMATCHED, s=3.0, alpha=0.95, edgecolors="#3C3240", linewidths=0.18, depthshade=False,
        )
    if show_joints:
        for joint in cloud.joint_axes:
            draw_joint(ax, joint, span)
    style_axis(ax, center, span, azimuth, elevation)


def save_cell(
    path: Path,
    cloud: Cloud,
    color_map: dict[int, str],
    mapping: dict[int, int],
    center: np.ndarray,
    span: float,
    azimuth: float,
    elevation: float,
    point_limit: int,
    show_joints: bool,
    note: str | None,
) -> None:
    figure = plt.figure(figsize=(2.0, 2.0), dpi=300)
    ax = figure.add_subplot(111, projection="3d")
    render_cloud(ax, cloud, color_map, mapping, center, span, azimuth, elevation, point_limit, show_joints)
    if note:
        ax.text2D(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#6C7378")
    figure.subplots_adjust(0, 0, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def off_frame_axis_count(joints: list[dict[str, Any]], center: np.ndarray, span: float) -> int:
    half = span / 2
    count = 0
    for joint in joints:
        normalized = normalize_axis(joint)
        if normalized is None:
            continue
        origin = normalized[0]
        if np.any(np.abs(origin - center) > half):
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-object", default=DEFAULT_OBJECTS["simple"])
    parser.add_argument("--medium-object", default=DEFAULT_OBJECTS["medium"])
    parser.add_argument("--complex-object", default=DEFAULT_OBJECTS["complex"])
    parser.add_argument("--output-dir", type=Path, default=Path("paper_assets/experiments"))
    parser.add_argument("--raw-root", type=Path, default=Path("paper_assets/experiments/qualitative_comparison_raw"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/external_baseline_suite_v1/per_object"))
    parser.add_argument(
        "--visualization-manifest",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/appendix_visualization_v1/visualization_manifest.csv"),
    )
    parser.add_argument(
        "--analytic-axis-json",
        type=Path,
        default=Path("outputs/partnet_core_v1_training/ours_controlled_ablation_v1/hybrid/test/analytic_axis_per_joint.json"),
    )
    parser.add_argument(
        "--aligned-manifest",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/aligned_v1/full20_objects.csv"),
    )
    parser.add_argument("--azimuth", type=float, default=-58.0)
    parser.add_argument("--elevation", type=float, default=18.0)
    parser.add_argument("--point-limit", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objects = [
        ("Simple", args.simple_object),
        ("Medium", args.medium_object),
        ("Complex", args.complex_object),
    ]
    metrics = load_metric_rows(args.visualization_manifest)
    with args.aligned_manifest.open(newline="") as handle:
        aligned_objects = {row["object_id"] for row in csv.DictReader(handle)}
    missing_from_aligned = [object_id for _, object_id in objects if object_id not in aligned_objects]
    if missing_from_aligned:
        raise ValueError(f"Objects are not in the aligned external suite: {missing_from_aligned}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_root = args.output_dir / "qualitative_cells"

    figure = plt.figure(figsize=(7.16, 4.55))
    grid = figure.add_gridspec(3, 6, left=0.105, right=0.995, top=0.92, bottom=0.045, wspace=0.01, hspace=0.09)
    selection: dict[str, Any] = {
        "selection_rule": (
            "Deterministic aligned-suite defaults: all five methods successful; simple has exactly 2 GT parts, "
            "medium exposes decomposition disagreement, and complex has >=5 parts with an honest Track2Art limitation."
        ),
        "methods": [METHOD_LABELS[method] for method in METHODS],
        "objects": [],
    }
    manifest: dict[str, Any] = {
        "figure_contract": {
            "camera": {"azimuth_deg": args.azimuth, "elevation_deg": args.elevation, "projection": "orthographic"},
            "point_limit_per_cell": args.point_limit,
            "point_size": 2.1,
            "matching": "Hungarian maximum overlap after common-domain nearest-neighbor GT label transfer; visualization only",
            "palette": {"base": PALETTE[0], "revolute": PALETTE[1], "prismatic": PALETTE[2], "unmatched": UNMATCHED},
        },
        "objects": {},
        "omitted_methods": {
            "VideoArtGS": "Omitted to keep at most five predicted methods; ArtGS provides the representative Gaussian two-state baseline.",
            "GaussianArt": "Compatible segmentation geometry was not synced for all selected objects.",
            "PARIS": "Native protocol is only applicable to two-part objects and failed on the selected simple object.",
            "Ditto": "Official checkpoint URL was unavailable and multi-part rows are outside its native contract.",
        },
    }

    for row_index, (difficulty, object_id) in enumerate(objects):
        gt_data = load_json(args.raw_root / object_id / "gt.json")
        gt_points = np.asarray(gt_data["points"], dtype=float)
        gt_labels = np.asarray(gt_data["gt_part_id"], dtype=int)
        valid = gt_labels >= 0
        gt_points, gt_labels = gt_points[valid], gt_labels[valid]
        gt_axes = axis_rows(args.analytic_axis_json, object_id, gt=True)
        color_map = build_gt_palette(gt_labels, gt_axes)
        center = (gt_points.min(axis=0) + gt_points.max(axis=0)) / 2
        span = float(np.max(np.ptp(gt_points, axis=0)) * 1.16)
        gt_part_count = len(np.unique(gt_labels))
        category = metrics.get((object_id, "ours_hybrid"), {}).get("category", "object")
        display_category = "Coffee machine" if category == "coffeemachine" else category.title()
        figure.text(
            0.008,
            0.78 - row_index * 0.31,
            f"{difficulty}\n{display_category}\n{gt_part_count} parts",
            ha="left",
            va="center",
            fontsize=7.2,
            color="#27333A",
            linespacing=1.28,
        )
        object_record: dict[str, Any] = {
            "object_id": object_id,
            "difficulty": difficulty.lower(),
            "category": category,
            "gt_part_count": gt_part_count,
            "camera": {"center": center.tolist(), "span": span, "azimuth_deg": args.azimuth, "elevation_deg": args.elevation},
            "gt_color_map": {str(key): value for key, value in color_map.items()},
            "methods": {},
        }

        for column_index, method in enumerate(METHODS):
            ax = figure.add_subplot(grid[row_index, column_index], projection="3d")
            try:
                cloud, coordinate_diagnostics = load_cloud(
                    method, object_id, args.raw_root, args.baseline_root, gt_points, gt_labels, args.analytic_axis_json
                )
                mapping, overlap = hungarian_mapping(cloud.pred_labels, cloud.canonical_gt_labels)
                note = None
                show_joints = bool(cloud.joint_axes)
                if method == "dta":
                    note = "Axis unavailable"
                else:
                    off_frame = off_frame_axis_count(cloud.joint_axes, center, span)
                    if off_frame:
                        note = f"{off_frame} axis off-frame"
                render_cloud(
                    ax, cloud, color_map, mapping, center, span, args.azimuth, args.elevation, args.point_limit, show_joints
                )
                metric_row = metrics.get((object_id, METHOD_MANIFEST_NAMES.get(method, "")), {})
                if method != "gt":
                    iou = metric_row.get("point_iou")
                    pred_count = metric_row.get("predicted_part_count") or str(len(np.unique(cloud.pred_labels)))
                    label = f"IoU {float(iou):.2f}  |  {pred_count}/{gt_part_count}" if iou not in {None, ""} else f"{pred_count}/{gt_part_count} parts"
                    ax.text2D(0.5, -0.015, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#4B555B")
                if note:
                    ax.text2D(0.5, 0.045, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color="#6D7377")
                save_cell(
                    cell_root / object_id / f"{method}.png",
                    cloud,
                    color_map,
                    mapping,
                    center,
                    span,
                    args.azimuth,
                    args.elevation,
                    args.point_limit,
                    show_joints,
                    note,
                )
                object_record["methods"][method] = {
                    "status": "success",
                    "native_output_path": cloud.native_path,
                    "coordinate_frame_compatibility": cloud.coordinate_contract,
                    "coordinate_diagnostics": coordinate_diagnostics,
                    "available_joint_fields": ["type", "axis", "origin"] if show_joints else [],
                    "axis_contract": cloud.axis_contract,
                    "off_frame_axis_count": off_frame_axis_count(cloud.joint_axes, center, span),
                    "part_matching": overlap,
                    "cell_path": str(cell_root / object_id / f"{method}.png"),
                    "metrics": {
                        "point_iou": metric_row.get("point_iou"),
                        "predicted_part_count": metric_row.get("predicted_part_count"),
                        "gt_part_count": gt_part_count,
                    },
                }
            except Exception as error:
                style_axis(ax, center, span, args.azimuth, args.elevation)
                ax.text2D(0.5, 0.5, "Failed", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#858B8F")
                object_record["methods"][method] = {"status": "failed", "reason": str(error)}

            if row_index == 0:
                background = "#EAF3FA" if method == "track2art" else "white"
                ax.set_title(METHOD_LABELS[method], fontsize=8.3, fontweight="bold", pad=3, color="#1F2D35", backgroundcolor=background)

        selection["objects"].append({
            "difficulty": difficulty.lower(),
            "object_id": object_id,
            "category": category,
            "gt_part_count": gt_part_count,
            "selection_criteria": (
                "all five selected prediction methods succeeded; difficulty bucket and decomposition behavior match the fixed rule"
            ),
            "methods": {
                method: {
                    "status": data["status"],
                    "native_output_path": data.get("native_output_path"),
                    "coordinate_frame_compatibility": data.get("coordinate_frame_compatibility"),
                    "available_joint_fields": data.get("available_joint_fields", []),
                    "axis_contract": data.get("axis_contract"),
                }
                for method, data in object_record["methods"].items()
            },
        })
        manifest["objects"][object_id] = object_record

    for y in (0.345, 0.655):
        figure.add_artist(plt.Line2D([0.105, 0.995], [y, y], transform=figure.transFigure, color="#DCE1E4", lw=0.55))
    figure.savefig(args.output_dir / "qualitative_comparison.pdf", facecolor="white")
    figure.savefig(args.output_dir / "qualitative_comparison.svg", facecolor="white")
    figure.savefig(args.output_dir / "qualitative_comparison.png", dpi=400, facecolor="white")
    plt.close(figure)

    (args.output_dir / "qualitative_comparison_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.output_dir / "qualitative_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "objects": [object_id for _, object_id in objects],
        "methods": [METHOD_LABELS[method] for method in METHODS],
        "output": str(args.output_dir / "qualitative_comparison.pdf"),
    }, indent=2))


if __name__ == "__main__":
    main()
