#!/usr/bin/env python3
"""Evaluate AiM's native screw-motion output against simulation GT joints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", action="append", nargs=4, metavar=("ID", "AIM_RUN", "AIM_IOU", "EPISODE"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _joint_defs_from_mjcf(model_path: Path) -> dict[str, dict[str, Any]]:
    """Read GT joint lines directly from MuJoCo without pipeline imports.

    This evaluator is intentionally standalone: baseline evaluation must not be
    blocked by unrelated experimental edits in the main inference package.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints: dict[str, dict[str, Any]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        body_position = np.asarray(data.xpos[body_id], dtype=np.float64)
        axis = rotation @ np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        pivot = body_position + rotation @ np.asarray(model.jnt_pos[joint_id], dtype=np.float64)
        joint_kind = int(model.jnt_type[joint_id])
        joints[str(name)] = {
            "name": str(name),
            "joint_type": (
                "prismatic" if joint_kind == int(mujoco.mjtJoint.mjJNT_SLIDE)
                else "revolute" if joint_kind == int(mujoco.mjtJoint.mjJNT_HINGE)
                else "unsupported"
            ),
            "axis_world": axis.tolist(),
            "pivot_world": pivot.tolist(),
        }
    return joints


def _read_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    colors = None
    vertex_colors = getattr(cloud.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(points):
        colors = np.asarray(vertex_colors[:, :3], dtype=np.int64)
    return points, colors


def _motion_to_predicted_label(run: Path) -> dict[int, int]:
    segmented_points, colors = _read_xyz_rgb(run / "motion_seg_final/segmented_point.ply")
    if colors is None:
        raise ValueError("AiM segmented point cloud has no RGB labels")
    palette = sorted({tuple(int(value) for value in row) for row in colors})
    color_to_label = {color: index for index, color in enumerate(palette)}
    tree = cKDTree(segmented_points)
    output: dict[int, int] = {}
    for path in sorted(run.glob("sub_*_point_cloud_end.ply")):
        motion_id = int(path.name.split("_")[1])
        points, _ = _read_xyz_rgb(path)
        sample = points[:: max(1, len(points) // 2048)]
        distances, indices = tree.query(sample, k=1)
        valid = distances <= max(1e-5, float(np.quantile(distances, 0.9)) + 1e-8)
        labels = [color_to_label[tuple(int(value) for value in colors[index])] for index in indices[valid]]
        if not labels:
            raise ValueError(f"Could not map AiM motion component {motion_id} to its PLY label")
        output[motion_id] = max(set(labels), key=labels.count)
    return output


def _matching(iou: dict[str, Any]) -> dict[int, int]:
    primary = iou["primary"]
    metrics = primary.get("coverage_aware_metrics") or primary.get(
        "covered_only_metrics"
    )
    if metrics is None:
        raise ValueError("AiM segmentation evaluation contains no matching metrics")
    rows = metrics["matching"]
    return {
        int(row["pred_part"] if "pred_part" in row else row["pred_slot"]): int(
            row["gt_part"]
        )
        for row in rows
    }


def _part_joints(episode: dict[str, Any]) -> dict[int, list[str]]:
    parts = episode["metadata"]["part_segmentation"]["parts"]
    return {
        int(part["part_id"]): [str(name) for name in part.get("joint_names", [])]
        for part in parts
    }


def _joint_travel(episode: dict[str, Any], joint_name: str) -> float | None:
    """Return the observed joint travel over the recording in native SI units."""
    values = [
        frame.get("action_log", {}).get("joint_positions", {}).get(joint_name)
        for frame in episode.get("frames", [])
    ]
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) - min(finite) if finite else None


def _axis_error_deg(predicted: list[float], target: list[float]) -> float:
    a = np.asarray(predicted, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def _axis_line_distance(
    predicted_point: list[float],
    predicted_axis: list[float],
    target_point: list[float],
    target_axis: list[float],
) -> float:
    p1 = np.asarray(predicted_point, dtype=np.float64)
    p2 = np.asarray(target_point, dtype=np.float64)
    u = np.asarray(predicted_axis, dtype=np.float64)
    v = np.asarray(target_axis, dtype=np.float64)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    normal = np.cross(u, v)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-8:
        return float(np.linalg.norm(np.cross(p2 - p1, u)))
    return abs(float(np.dot(p2 - p1, normal / normal_norm)))


def evaluate_object(
    object_id: str,
    run: Path,
    iou_path: Path,
    episode_path: Path,
) -> list[dict[str, Any]]:
    motions = json.loads((run / "motion.json").read_text(encoding="utf-8"))
    iou = json.loads(iou_path.read_text(encoding="utf-8"))
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    model_path = Path(episode["metadata"]["model_path"])
    joint_defs = _joint_defs_from_mjcf(model_path)
    component_labels = _motion_to_predicted_label(run)
    label_to_gt = _matching(iou)
    part_joints = _part_joints(episode)
    bbox_diagonal = float(iou["reference_bbox_diagonal_m"])

    rows: list[dict[str, Any]] = []
    for motion_id, motion in enumerate(motions):
        # AiM reserves component zero for the static identity motion.
        if motion_id == 0:
            continue
        predicted_label = component_labels.get(motion_id)
        gt_part_id = label_to_gt.get(predicted_label) if predicted_label is not None else None
        joint_names = part_joints.get(gt_part_id, []) if gt_part_id is not None else []
        joint = next((joint_defs.get(name) for name in joint_names if joint_defs.get(name)), None)
        predicted_type = "prismatic" if int(motion["motion_type"]) == 0 else "revolute"
        type_correct = joint is not None and predicted_type == joint["joint_type"]
        axis_error = (
            _axis_error_deg(motion["axis"], joint["axis_world"])
            if joint is not None and type_correct
            else None
        )
        axis_line_m = (
            _axis_line_distance(
                motion["center"],
                motion["axis"],
                joint["pivot_world"],
                joint["axis_world"],
            )
            if joint is not None and type_correct and predicted_type == "revolute"
            else None
        )
        axis_line = axis_line_m / bbox_diagonal if axis_line_m is not None else None
        gt_travel = _joint_travel(episode, joint["name"]) if joint is not None else None
        predicted_travel = float(motion["theta"] if predicted_type == "revolute" else motion["phi"])
        motion_error = (
            abs(predicted_travel - gt_travel)
            if gt_travel is not None and type_correct
            else None
        )
        rows.append(
            {
                "object_id": object_id,
                "motion_id": motion_id,
                "predicted_part": predicted_label,
                "matched_gt_part": gt_part_id,
                "matched_joint": joint["name"] if joint else None,
                "predicted_type": predicted_type,
                "gt_type": joint["joint_type"] if joint else None,
                "type_correct": type_correct if joint else None,
                "predicted_axis": motion["axis"],
                "gt_axis": joint["axis_world"] if joint else None,
                "axis_error_deg_type_correct": axis_error,
                "axis_line_error_m": axis_line_m,
                "axis_line_error_bbox_normalized": axis_line,
                "theta_rad": float(motion["theta"]),
                "translation_m": float(motion["phi"]),
                "gt_part_motion": gt_travel,
                "predicted_part_motion": predicted_travel,
                "part_motion_abs_error": motion_error,
                "part_motion_unit": (
                    "rad" if predicted_type == "revolute" else "m"
                ),
                "evaluation_scope": "AiM native screw output, conditional on point-IoU part matching",
            }
        )
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _typed_summary(rows: list[dict[str, Any]], joint_type: str) -> dict[str, Any]:
    """Summarize homogeneous SI units only; radians and metres never mix."""
    typed = [row for row in rows if row.get("gt_type") == joint_type]
    correct = [row for row in typed if row.get("type_correct")]
    motion = [float(row["part_motion_abs_error"]) for row in correct if row.get("part_motion_abs_error") is not None]
    axis = [float(row["axis_error_deg_type_correct"]) for row in correct if row.get("axis_error_deg_type_correct") is not None]
    line = [float(row["axis_line_error_bbox_normalized"]) for row in correct if row.get("axis_line_error_bbox_normalized") is not None]
    return {
        "matched_joint_count": len(typed),
        "type_correct_count": len(correct),
        "joint_type_accuracy": len(correct) / len(typed) if typed else None,
        "axis_error_mean_deg": statistics.fmean(axis) if axis else None,
        "axis_error_median_deg": statistics.median(axis) if axis else None,
        "axis_line_error_mean_m": _mean(correct, "axis_line_error_m"),
        "axis_line_error_mean_bbox_normalized": statistics.fmean(line) if line else None,
        "part_motion_abs_error_mean": statistics.fmean(motion) if motion else None,
        "part_motion_abs_error_median": statistics.median(motion) if motion else None,
        "part_motion_unit": "rad" if joint_type == "revolute" else "m",
    }


def main() -> int:
    args = parse_args()
    if not args.object:
        raise ValueError("At least one --object ID AIM_RUN AIM_IOU EPISODE is required")
    rows = [
        row
        for object_id, run, iou, episode in args.object
        for row in evaluate_object(object_id, Path(run), Path(iou), Path(episode))
    ]
    matched = [row for row in rows if row["matched_joint"] is not None]
    typed = [row for row in matched if row["type_correct"]]
    summary = {
        "object_count": len({row["object_id"] for row in rows}),
        "predicted_motion_count": len(rows),
        "matched_joint_count": len(matched),
        "type_correct_count": len(typed),
        "joint_type_accuracy": sum(bool(row["type_correct"]) for row in matched) / len(matched) if matched else None,
        "axis_error_mean_deg": _mean(typed, "axis_error_deg_type_correct"),
        "axis_error_median_deg": (
            statistics.median(
                float(row["axis_error_deg_type_correct"])
                for row in typed
                if row["axis_error_deg_type_correct"] is not None
            )
            if any(row["axis_error_deg_type_correct"] is not None for row in typed)
            else None
        ),
        "axis_line_error_mean_m": _mean(typed, "axis_line_error_m"),
        "axis_line_error_mean_bbox_normalized": _mean(typed, "axis_line_error_bbox_normalized"),
        "by_joint_type": {
            "revolute": _typed_summary(matched, "revolute"),
            "prismatic": _typed_summary(matched, "prismatic"),
        },
        "metric_scope": (
            "AiM motion.json screw type/axis, conditional on Hungarian-matched predicted components. "
            "Axis error is reported only for type-correct joints."
        ),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "aim_axis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "aim_axis_per_joint.json").write_text(json.dumps({"summary": summary, "joints": rows}, indent=2) + "\n", encoding="utf-8")
    with (output / "aim_axis_per_joint.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
