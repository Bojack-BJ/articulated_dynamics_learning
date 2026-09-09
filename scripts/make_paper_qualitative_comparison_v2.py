#!/usr/bin/env python3
"""Render upright main/supplementary Track2Art qualitative comparisons."""

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
from matplotlib.patches import FancyArrowPatch
import numpy as np

from make_paper_qualitative_comparison import (
    METHOD_LABELS,
    METHOD_MANIFEST_NAMES,
    PALETTE,
    PRISMATIC,
    REVOLUTE,
    UNMATCHED,
    axis_rows,
    build_gt_palette,
    deterministic_sample,
    hungarian_mapping,
    load_cloud,
    load_json,
    load_metric_rows,
    normalize_axis,
)


METHODS = ("gt", "track2art", "artgs", "videoartgs", "dta", "aim", "reart", "paris")
MAIN_DEFAULTS = ("partnet_9388", "partnet_10638", "partnet_7179")
SUPP_DEFAULTS = ("partnet_46556", "partnet_103118", "partnet_102055")
UPRIGHT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


@dataclass(frozen=True)
class Camera:
    azimuth: float
    elevation: float
    roll: float
    score: float
    score_terms: dict[str, float]


def upright(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ UPRIGHT_MATRIX.T


def camera_basis(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = math.radians(camera.azimuth)
    elevation = math.radians(camera.elevation)
    camera_direction = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )
    forward = -camera_direction
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= max(np.linalg.norm(right), 1e-8)
    screen_up = np.cross(forward, right)
    if camera.roll:
        angle = math.radians(camera.roll)
        right, screen_up = (
            math.cos(angle) * right + math.sin(angle) * screen_up,
            -math.sin(angle) * right + math.cos(angle) * screen_up,
        )
    return right, screen_up, forward


def project(points: np.ndarray, center: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    right, screen_up, forward = camera_basis(camera)
    relative = upright(points) - center
    xy = np.column_stack((relative @ right, relative @ screen_up))
    return xy, relative @ forward


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    low = np.maximum(box_a[0], box_b[0])
    high = np.minimum(box_a[1], box_b[1])
    intersection = float(np.prod(np.maximum(high - low, 0.0)))
    area_a = float(np.prod(np.maximum(box_a[1] - box_a[0], 0.0)))
    area_b = float(np.prod(np.maximum(box_b[1] - box_b[0], 0.0)))
    return intersection / max(area_a + area_b - intersection, 1e-8)


def candidate_score(
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    joints: list[dict[str, Any]],
    category: str,
    camera: Camera,
) -> Camera:
    display_points = upright(gt_points)
    center = (display_points.min(axis=0) + display_points.max(axis=0)) / 2
    xy, _ = project(gt_points, center, camera)
    extent = np.ptp(xy, axis=0)
    projected_diagonal = max(float(np.linalg.norm(extent)), 1e-8)
    object_diagonal = max(float(np.linalg.norm(np.ptp(display_points, axis=0))), 1e-8)
    area_score = float(np.prod(extent) / (object_diagonal * object_diagonal))

    labels = np.unique(gt_labels)
    centroids = []
    boxes = []
    part_areas = []
    for label in labels:
        part_xy = xy[gt_labels == label]
        if not len(part_xy):
            continue
        centroids.append(part_xy.mean(axis=0))
        box = np.stack((part_xy.min(axis=0), part_xy.max(axis=0)))
        boxes.append(box)
        part_areas.append(float(np.prod(np.maximum(box[1] - box[0], 0.0))))
    separations = [
        np.linalg.norm(centroids[i] - centroids[j]) / projected_diagonal
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ]
    centroid_separation = float(np.mean(separations)) if separations else 0.0
    overlap_values = [
        bbox_iou(boxes[i], boxes[j])
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
    ]
    overlap_penalty = float(np.mean(overlap_values)) if overlap_values else 0.0
    total_area = max(float(np.prod(extent)), 1e-8)
    small_part_visibility = float(np.percentile(np.asarray(part_areas) / total_area, 30)) if part_areas else 0.0

    _, _, forward = camera_basis(camera)
    camera_direction = -forward
    projected_axes = []
    prismatic_axis = None
    for joint in joints:
        normalized = normalize_axis(joint)
        if normalized is None:
            continue
        axis = upright(normalized[1][None, :])[0]
        axis /= max(np.linalg.norm(axis), 1e-8)
        projected_axes.append(math.sqrt(max(0.0, 1.0 - float(np.dot(axis, camera_direction)) ** 2)))
        if normalized[2] == "prismatic" and prismatic_axis is None:
            prismatic_axis = axis
    axis_visibility = float(np.mean(projected_axes)) if projected_axes else 0.5

    horizontal_camera = camera_direction.copy()
    horizontal_camera[2] = 0.0
    horizontal_camera /= max(np.linalg.norm(horizontal_camera), 1e-8)
    if prismatic_axis is not None:
        front = prismatic_axis.copy()
        front[2] = 0.0
        front /= max(np.linalg.norm(front), 1e-8)
        signed_front = float(np.dot(horizontal_camera, front))
        front_score = max(0.0, 1.0 - abs(signed_front - 0.72))
        displacement_score = math.sqrt(max(0.0, 1.0 - signed_front * signed_front))
    else:
        # PartNet assets use a consistent -X-facing canonical front for the
        # selected door/refrigerator/oven/coffee-machine categories.
        signed_front = float(np.dot(horizontal_camera, np.array([-1.0, 0.0, 0.0])))
        front_score = max(0.0, 1.0 - abs(signed_front - 0.72))
        displacement_score = axis_visibility

    elevation_score = max(0.0, 1.0 - abs(camera.elevation - 20.0) / 15.0)
    score_terms = {
        "projected_area": area_score,
        "part_centroid_separation": centroid_separation,
        "small_part_visibility": small_part_visibility,
        "joint_axis_visibility": axis_visibility,
        "front_oblique": front_score,
        "motion_image_plane": displacement_score,
        "component_overlap_penalty": overlap_penalty,
        "elevation_preference": elevation_score,
    }
    score = (
        1.35 * area_score
        + 1.15 * centroid_separation
        + 0.95 * small_part_visibility
        + 0.75 * axis_visibility
        + 1.25 * front_score
        + (1.0 if category in {"drawer", "door"} else 0.55) * displacement_score
        + 0.25 * elevation_score
        - 0.9 * overlap_penalty
    )
    return Camera(camera.azimuth, camera.elevation, camera.roll, float(score), score_terms)


def rank_cameras(
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    joints: list[dict[str, Any]],
    category: str,
) -> list[Camera]:
    candidates = [
        candidate_score(gt_points, gt_labels, joints, category, Camera(float(azimuth), float(elevation), 0.0, 0.0, {}))
        for azimuth in range(0, 360, 30)
        for elevation in (15, 20, 25)
    ]
    return sorted(candidates, key=lambda item: (-item.score, item.azimuth, item.elevation))


def projected_bounds(gt_points: np.ndarray, center: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, float]:
    xy, _ = project(gt_points, center, camera)
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    midpoint = (low + high) / 2
    half = np.maximum((high - low) * 0.56, 1e-5)  # 12% total expansion.
    low, high = midpoint - half, midpoint + half
    occupancy = float(max(np.ptp(xy, axis=0) / np.maximum(high - low, 1e-8)))
    return low, high, occupancy


def projected_joint(
    joint: dict[str, Any],
    center: np.ndarray,
    camera: Camera,
    object_diagonal: float,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray] | None:
    normalized = normalize_axis(joint)
    if normalized is None:
        return None
    origin_world, axis_world, joint_type = normalized
    total_length = object_diagonal * (0.74 if joint_type == "revolute" else 0.64) * length_scale
    segment = np.stack((origin_world - axis_world * total_length / 2, origin_world + axis_world * total_length / 2))
    segment_xy, depth = project(segment, center, camera)
    origin_xy, _ = project(origin_world[None, :], center, camera)
    return segment_xy[0], segment_xy[1], joint_type, origin_xy[0], depth


def draw_joint_overlay(
    ax: Any,
    joint: dict[str, Any],
    center: np.ndarray,
    camera: Camera,
    object_diagonal: float,
    length_scale: float,
    color: str,
) -> None:
    item = projected_joint(joint, center, camera, object_diagonal, length_scale)
    if item is None:
        return
    start, end, joint_type, origin, _ = item
    offset_index = float(joint.get("display_parallel_offset", 0.0))
    if offset_index:
        direction_2d = end - start
        direction_2d /= max(np.linalg.norm(direction_2d), 1e-8)
        screen_normal = np.array([-direction_2d[1], direction_2d[0]])
        display_offset = screen_normal * object_diagonal * 0.012 * offset_index
        start, end, origin = start + display_offset, end + display_offset, origin + display_offset
    if joint_type == "revolute":
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=0.88, alpha=0.86, zorder=10)
        ax.scatter([origin[0]], [origin[1]], s=6, c=color, edgecolors="white", linewidths=0.28, zorder=11)
        direction = end - start
        direction /= max(np.linalg.norm(direction), 1e-8)
        normal = np.array([-direction[1], direction[0]])
        radius = object_diagonal * 0.045
        angles = np.linspace(math.radians(20), math.radians(285), 28)
        arc = origin + radius * (
            np.cos(angles)[:, None] * direction + 0.56 * np.sin(angles)[:, None] * normal
        )
        ax.plot(arc[:, 0], arc[:, 1], color=color, lw=0.68, alpha=0.82, zorder=10)
    else:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>",
            mutation_scale=5.2,
            linewidth=0.86,
            color=color,
            alpha=0.84,
            zorder=10,
        )
        ax.add_patch(arrow)


def render_cell(
    ax: Any,
    cloud: Any,
    color_map: dict[int, str],
    mapping: dict[int, int],
    center: np.ndarray,
    camera: Camera,
    bounds: tuple[np.ndarray, np.ndarray],
    object_diagonal: float,
    point_limit: int,
    show_joints: bool,
) -> None:
    points, pred = deterministic_sample(cloud.points, cloud.pred_labels, limit=point_limit)
    xy, depth = project(points, center, camera)
    order = np.argsort(depth)
    xy, pred = xy[order], pred[order]
    matched = np.array([int(label) in mapping for label in pred])
    colors = np.array([color_map.get(mapping.get(int(label), -999), UNMATCHED) for label in pred])
    if np.any(matched):
        ax.scatter(xy[matched, 0], xy[matched, 1], c=colors[matched], s=3.5, alpha=0.80, linewidths=0, zorder=3)
    if np.any(~matched):
        ax.scatter(
            xy[~matched, 0],
            xy[~matched, 1],
            c=UNMATCHED,
            s=4.1,
            alpha=0.82,
            edgecolors="#443747",
            linewidths=0.22,
            zorder=4,
        )
    if show_joints:
        length_scale = 0.76 if len(cloud.joint_axes) >= 4 else 1.0
        for joint in cloud.joint_axes:
            child_part_id = joint.get("child_part_id", joint.get("child"))
            child_gt_id = mapping.get(int(child_part_id)) if child_part_id is not None else None
            overlay_color = color_map.get(child_gt_id) if child_gt_id is not None else None
            if overlay_color is None:
                overlay_color = REVOLUTE if joint.get("joint_type") == "revolute" else PRISMATIC
            draw_joint_overlay(ax, joint, center, camera, object_diagonal, length_scale, overlay_color)
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def anchor_gt_prismatic_joints(
    joints: list[dict[str, Any]],
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Place GT prismatic direction arrows on distinct visible moving parts."""
    counts = {int(label): int(np.sum(gt_labels == label)) for label in np.unique(gt_labels)}
    base_label = max(counts, key=counts.get)
    moving_labels = sorted(
        (label for label in counts if label != base_label),
        key=lambda label: (-counts[label], label),
    )
    anchored: list[dict[str, Any]] = []
    for index, joint in enumerate(joints):
        rendered = dict(joint)
        if index < len(moving_labels):
            rendered["child_part_id"] = int(moving_labels[index])
        if rendered.get("joint_type") == "prismatic" and index < len(moving_labels):
            label = moving_labels[index]
            rendered["origin"] = np.mean(gt_points[gt_labels == label], axis=0).tolist()
            rendered["display_origin_policy"] = "visible_child_part_centroid"
            rendered["display_child_part_id"] = int(label)
        else:
            rendered["display_origin_policy"] = "native_axis_line_point"
        anchored.append(rendered)
    coincident_groups: dict[tuple[float, ...], list[int]] = {}
    for index, joint in enumerate(anchored):
        normalized = normalize_axis(joint)
        if normalized is None:
            continue
        origin, axis, _ = normalized
        if tuple(np.round(axis, 5)) > tuple(np.round(-axis, 5)):
            axis = -axis
        key = tuple(np.round(np.concatenate((origin, axis)), 5))
        coincident_groups.setdefault(key, []).append(index)
    for indices in coincident_groups.values():
        if len(indices) <= 1:
            continue
        midpoint = (len(indices) - 1) / 2
        for order, index in enumerate(indices):
            anchored[index]["display_parallel_offset"] = float(order - midpoint)
            anchored[index]["display_overlap_policy"] = "parallel_screen_offset_for_coincident_gt_axes"
    return anchored


def load_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        payload = {
            "schema": "track2art-qualitative-camera-overrides-v1",
            "description": "Optional per-object azimuth/elevation/roll/center/scale overrides.",
            "objects": {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    return load_json(path)


def camera_from_override(candidate: Camera, override: dict[str, Any] | None) -> Camera:
    if not override:
        return candidate
    return Camera(
        float(override.get("azimuth", candidate.azimuth)),
        float(override.get("elevation", candidate.elevation)),
        float(override.get("roll", candidate.roll)),
        candidate.score,
        candidate.score_terms,
    )


def write_camera_contact_sheet(
    output: Path,
    objects: list[dict[str, Any]],
    raw_root: Path,
    analytic_axes: Path,
) -> None:
    figure, axes = plt.subplots(len(objects), 5, figsize=(10.8, 8.6), squeeze=False)
    for row_index, item in enumerate(objects):
        object_id = item["object_id"]
        data = load_json(raw_root / object_id / "gt.json")
        points = np.asarray(data["points"], dtype=float)
        labels = np.asarray(data["gt_part_id"], dtype=int)
        valid = labels >= 0
        points, labels = points[valid], labels[valid]
        joints = axis_rows(analytic_axes, object_id, gt=True)
        colors = build_gt_palette(labels, joints)
        display = upright(points)
        center = (display.min(axis=0) + display.max(axis=0)) / 2
        cameras = list(item["camera_candidates"][:5])
        final_camera = item["camera"]
        is_listed = any(
            camera.azimuth == final_camera.azimuth
            and camera.elevation == final_camera.elevation
            and camera.roll == final_camera.roll
            for camera in cameras
        )
        if item.get("override") and not is_listed:
            cameras[-1] = final_camera
        for column_index, camera in enumerate(cameras):
            ax = axes[row_index, column_index]
            xy, depth = project(points, center, camera)
            order = np.argsort(depth)
            ax.scatter(
                xy[order, 0],
                xy[order, 1],
                c=[colors[int(label)] for label in labels[order]],
                s=1.3,
                alpha=0.85,
                linewidths=0,
            )
            low, high, _ = projected_bounds(points, center, camera)
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            selected = (
                camera.azimuth == final_camera.azimuth
                and camera.elevation == final_camera.elevation
                and camera.roll == final_camera.roll
            )
            suffix = " [selected]" if selected else ""
            ax.set_title(
                f"az {camera.azimuth:.0f} / el {camera.elevation:.0f}{suffix}\nscore {camera.score:.3f}",
                fontsize=7,
            )
            if column_index == 0:
                ax.text(-0.05, 0.5, f"{item['category']} · {item['gt_part_count']} parts", transform=ax.transAxes, ha="right", va="center", fontsize=7.4)
    figure.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.03, wspace=0.08, hspace=0.22)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def save_cell(
    path: Path,
    cloud: Any,
    color_map: dict[int, str],
    mapping: dict[int, int],
    center: np.ndarray,
    camera: Camera,
    bounds: tuple[np.ndarray, np.ndarray],
    object_diagonal: float,
    point_limit: int,
    show_joints: bool,
    marker: str | None,
) -> None:
    figure, ax = plt.subplots(figsize=(2.0, 2.0), dpi=300)
    render_cell(ax, cloud, color_map, mapping, center, camera, bounds, object_diagonal, point_limit, show_joints)
    if marker:
        ax.text(0.96, 0.94, marker, transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#858B8F")
    figure.subplots_adjust(0, 0, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def create_comparison(
    output_stem: Path,
    assignment: str,
    object_items: list[dict[str, Any]],
    args: argparse.Namespace,
    metrics: dict[tuple[str, str], dict[str, str]],
    manifest: dict[str, Any],
) -> None:
    figure = plt.figure(figsize=(7.16, 4.18))
    grid = figure.add_gridspec(3, len(METHODS), left=0.105, right=0.995, top=0.91, bottom=0.06, wspace=0.018, hspace=0.16)
    row_centers = []
    for row_index, item in enumerate(object_items):
        object_id = item["object_id"]
        gt_data = load_json(args.raw_root / object_id / "gt.json")
        gt_points = np.asarray(gt_data["points"], dtype=float)
        gt_labels = np.asarray(gt_data["gt_part_id"], dtype=int)
        valid = gt_labels >= 0
        gt_points, gt_labels = gt_points[valid], gt_labels[valid]
        gt_axes = axis_rows(args.analytic_axis_json, object_id, gt=True)
        color_map = build_gt_palette(gt_labels, gt_axes)
        display = upright(gt_points)
        override = item.get("override") or {}
        center = np.asarray(override.get("center", (display.min(axis=0) + display.max(axis=0)) / 2), dtype=float)
        camera = item["camera"]
        low, high, occupancy = projected_bounds(gt_points, center, camera)
        if "scale" in override:
            midpoint = (low + high) / 2
            half = (high - low) / 2 * float(override["scale"])
            low, high = midpoint - half, midpoint + half
        object_diagonal = float(np.linalg.norm(np.ptp(display, axis=0)))
        category_label = "Coffee machine" if item["category"] == "coffeemachine" else item["category"].title()
        row_y = 0.755 - row_index * 0.306
        row_centers.append(row_y)
        figure.text(
            0.099,
            row_y,
            f"{category_label}\n{item['gt_part_count']} parts",
            ha="right",
            va="center",
            fontsize=6.0,
            linespacing=1.12,
            color="#28363D",
        )

        object_record = manifest["objects"].setdefault(object_id, {})
        object_record.update(
            {
                "assignment": assignment,
                "category": item["category"],
                "gt_part_count": item["gt_part_count"],
                "camera_candidates": [
                    {"azimuth": c.azimuth, "elevation": c.elevation, "roll": c.roll, "score": c.score, "score_terms": c.score_terms}
                    for c in item["camera_candidates"][:5]
                ],
                "final_camera": {"azimuth": camera.azimuth, "elevation": camera.elevation, "roll": camera.roll, "center": center.tolist(), "projected_bounds": [low.tolist(), high.tolist()]},
                "manual_override": item.get("override"),
                "projected_fill_ratio": occupancy,
                "upright_transform": UPRIGHT_MATRIX.tolist(),
                "gt_joint_count": len(gt_axes),
                "gt_joints_fully_rendered": True,
                "gt_prismatic_display_origin": "visible child-part centroid; direction remains GT",
                "videoartgs_artifact_available": False,
                "methods": {},
            }
        )

        for column_index, method in enumerate(METHODS):
            ax = figure.add_subplot(grid[row_index, column_index])
            marker = None
            if method == "paris" and item["gt_part_count"] != 2:
                ax.set_axis_off()
                ax.text(0.5, 0.52, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=7.4, color="#888E92")
                ax.text(0.5, 0.40, "2-part only", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color="#A0A5A8")
                object_record["methods"][method] = {
                    "status": "not_applicable",
                    "reason": "PARIS is evaluated only within its native two-part scope",
                }
                if row_index == 0:
                    ax.set_title(METHOD_LABELS[method], fontsize=7.5, fontweight="bold", pad=4, color="#1F2D35")
                continue
            try:
                cloud, coordinate_diagnostics = load_cloud(
                    method,
                    object_id,
                    args.raw_root,
                    args.baseline_root,
                    gt_points,
                    gt_labels,
                    args.analytic_axis_json,
                )
                if method == "gt":
                    cloud.joint_axes = anchor_gt_prismatic_joints(cloud.joint_axes, gt_points, gt_labels)
                mapping, overlap = hungarian_mapping(cloud.pred_labels, cloud.canonical_gt_labels)
                show_joints = bool(cloud.joint_axes) and method != "dta"
                if method != "dta" and method != "gt" and not show_joints:
                    marker = "†"
                render_cell(ax, cloud, color_map, mapping, center, camera, (low, high), object_diagonal, args.point_limit, show_joints)
                if marker:
                    ax.text(0.96, 0.94, marker, transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color="#858B8F")
                metric = metrics.get((object_id, METHOD_MANIFEST_NAMES.get(method, "")), {})
                predicted = (
                    metric.get("predicted_part_count") or str(len(np.unique(cloud.pred_labels)))
                    if method != "gt"
                    else str(item["gt_part_count"])
                )
                cell_path = args.output_dir / "qualitative_cells" / object_id / f"{method}.png"
                save_cell(cell_path, cloud, color_map, mapping, center, camera, (low, high), object_diagonal, args.point_limit, show_joints, marker)
                object_record["methods"][method] = {
                    "status": "success",
                    "native_output_path": cloud.native_path,
                    "coordinate_contract": cloud.coordinate_contract,
                    "coordinate_diagnostics": coordinate_diagnostics,
                    "axis_contract": cloud.axis_contract,
                    "joint_overlay_available": show_joints,
                    "rendered_joint_count": len(cloud.joint_axes) if show_joints else 0,
                    "native_joint_overlay": show_joints and method != "gt",
                    "omitted_overlay_reason": (
                        "compatible native joint geometry unavailable; segmentation only"
                        if method == "dta"
                        else None if show_joints else "no native joint overlay in artifact"
                    ),
                    "part_matching": overlap,
                    "cell_path": str(cell_path),
                    "metrics_not_rendered": {
                        "point_iou": metric.get("point_iou"),
                        "predicted_part_count": predicted,
                    },
                }
                if method == "videoartgs":
                    object_record["videoartgs_artifact_available"] = True
            except Exception as error:
                ax.set_axis_off()
                metric = metrics.get((object_id, METHOD_MANIFEST_NAMES.get(method, "")), {})
                status = metric.get("status", "")
                label = "Failed" if status in {"failed", "pending"} else "N/A"
                ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.4, color="#888E92")
                object_record["methods"][method] = {"status": label.lower(), "reason": str(error), "reported_status": status}
            if row_index == 0:
                title = "DTA†" if method == "dta" else METHOD_LABELS[method]
                ax.set_title(
                    title,
                    fontsize=7.5 if method != "videoartgs" else 6.9,
                    fontweight="bold",
                    pad=4,
                    color="#1F2D35",
                    backgroundcolor="#EAF3FA" if method == "track2art" else "white",
                )

    for y in ((row_centers[0] + row_centers[1]) / 2, (row_centers[1] + row_centers[2]) / 2):
        figure.add_artist(plt.Line2D([0.105, 0.995], [y - 0.012, y - 0.012], transform=figure.transFigure, color="#DCE1E4", lw=0.55))
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(output_stem.with_suffix(".png"), dpi=400, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-simple-object", default=MAIN_DEFAULTS[0])
    parser.add_argument("--main-medium-object", default=MAIN_DEFAULTS[1])
    parser.add_argument("--main-complex-object", default=MAIN_DEFAULTS[2])
    parser.add_argument("--supp-simple-object", default=SUPP_DEFAULTS[0])
    parser.add_argument("--supp-medium-object", default=SUPP_DEFAULTS[1])
    parser.add_argument("--supp-complex-object", default=SUPP_DEFAULTS[2])
    parser.add_argument("--output-dir", type=Path, default=Path("paper_assets/experiments"))
    parser.add_argument("--raw-root", type=Path, default=Path("paper_assets/experiments/qualitative_comparison_raw"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/external_baseline_suite_v1/per_object"))
    parser.add_argument("--camera-overrides", type=Path, default=Path("paper_assets/experiments/qualitative_camera_overrides.json"))
    parser.add_argument("--visualization-manifest", type=Path, default=Path("outputs/external_baseline_suite_v1/appendix_visualization_v1/visualization_manifest.csv"))
    parser.add_argument("--aligned-manifest", type=Path, default=Path("outputs/external_baseline_suite_v1/aligned_v1/full20_objects.csv"))
    parser.add_argument("--analytic-axis-json", type=Path, default=Path("outputs/partnet_core_v1_training/ours_controlled_ablation_v1/hybrid/test/analytic_axis_per_joint.json"))
    parser.add_argument("--point-limit", type=int, default=760)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = {
        "main": [args.main_simple_object, args.main_medium_object, args.main_complex_object],
        "supp": [args.supp_simple_object, args.supp_medium_object, args.supp_complex_object],
    }
    with args.aligned_manifest.open(newline="") as handle:
        aligned_rows = {row["object_id"]: row for row in csv.DictReader(handle)}
    for object_id in requested["main"] + requested["supp"]:
        if object_id not in aligned_rows:
            raise ValueError(f"{object_id} is not in the aligned external suite")
    if len(set(requested["main"] + requested["supp"])) != 6:
        raise ValueError("Main and supplementary selections must contain six distinct objects")

    overrides = load_overrides(args.camera_overrides).get("objects", {})
    all_items = []
    for assignment, object_ids in requested.items():
        for object_id in object_ids:
            row = aligned_rows[object_id]
            gt = load_json(args.raw_root / object_id / "gt.json")
            points = np.asarray(gt["points"], dtype=float)
            labels = np.asarray(gt["gt_part_id"], dtype=int)
            valid = labels >= 0
            points, labels = points[valid], labels[valid]
            joints = axis_rows(args.analytic_axis_json, object_id, gt=True)
            candidates = rank_cameras(points, labels, joints, row["category"])
            override = overrides.get(object_id)
            final_camera = camera_from_override(candidates[0], override)
            if override:
                final_camera = candidate_score(points, labels, joints, row["category"], final_camera)
            all_items.append(
                {
                    "assignment": assignment,
                    "object_id": object_id,
                    "category": row["category"],
                    "gt_part_count": int(row["gt_part_count"]),
                    "camera_candidates": candidates,
                    "camera": final_camera,
                    "override": override,
                }
            )

    write_camera_contact_sheet(
        args.output_dir / "qualitative_camera_candidates.pdf",
        all_items,
        args.raw_root,
        args.analytic_axis_json,
    )
    metrics = load_metric_rows(args.visualization_manifest)
    manifest: dict[str, Any] = {
        "schema": "track2art-qualitative-comparison-v2",
        "selection_rule": {
            "main": "clean common-success representative in each complexity bucket",
            "supp": "additional common-success case exposing under/over-segmentation or incorrect part count",
            "manual_score_selection": False,
        },
        "complex_object_change": {
            "changed": args.main_complex_object != "partnet_45261",
            "previous_object_id": "partnet_45261",
            "final_object_id": args.main_complex_object,
            "reason": "The former drawer collapses from six to four observable kinematic parts. The replacement is the strongest fully covered >=5-part case under the updated observable-domain metrics.",
        },
        "methods": [METHOD_LABELS[method] for method in METHODS],
        "joint_overlay_summary": {
            "available": ["GT", "Track2Art", "ArtGS", "VideoArtGS", "AiM", "ReArt", "PARIS"],
            "omitted": {"DTA": "compatible native joint geometry unavailable; segmentation only"},
        },
        "visual_contract": {
            "upright_transform": "display_xyz = world_(x,z,-y)",
            "projection": "orthographic",
            "shared_camera_per_object": True,
            "crop": "GT projected bounds with 12% total expansion",
            "point_limit": args.point_limit,
            "matching": "Hungarian maximum overlap for colors only",
            "per_cell_metrics_rendered": False,
            "joint_overlay_lengths_bbox": {"revolute": 0.74, "prismatic": 0.64},
            "complex_joint_overlay_scale": 0.76,
            "point_style": {"limit": args.point_limit, "size": 3.5, "alpha": 0.8},
            "gt_joint_policy": "all GT joints rendered; prismatic directions are visually anchored at distinct visible child-part centroids",
            "coincident_gt_axis_policy": "coincident GT axes retain their geometry and receive a small parallel screen-space display offset",
            "joint_overlay_color": "matched child-part color when child identity is available; joint-type color only as fallback",
            "prediction_modification": "none; Hungarian matching affects colors only",
            "dta_marker": "† compatible native joint geometry unavailable; segmentation only",
        },
        "objects": {},
    }
    create_comparison(
        args.output_dir / "qualitative_comparison_main",
        "main",
        [item for item in all_items if item["assignment"] == "main"],
        args,
        metrics,
        manifest,
    )
    create_comparison(
        args.output_dir / "qualitative_comparison_supp",
        "supp",
        [item for item in all_items if item["assignment"] == "supp"],
        args,
        metrics,
        manifest,
    )

    selection = {
        "main": [item["object_id"] for item in all_items if item["assignment"] == "main"],
        "supp": [item["object_id"] for item in all_items if item["assignment"] == "supp"],
        "criteria": manifest["selection_rule"],
        "methods": manifest["methods"],
    }
    (args.output_dir / "qualitative_comparison_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (args.output_dir / "qualitative_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
