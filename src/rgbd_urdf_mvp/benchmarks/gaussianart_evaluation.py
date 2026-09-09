"""Common kinematics evaluation for GaussianArt transform predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class JointTransform:
    """One joint represented by its end-state rigid transform."""

    joint_type: str
    axis: np.ndarray
    pivot: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    motion: float


def transform_to_joint(rotation: np.ndarray, translation: np.ndarray) -> JointTransform:
    """Interpret a canonical-to-end transform using GaussianArt's type rule."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if float(np.abs(rotation - np.eye(3)).mean()) < 5e-2:
        magnitude = float(np.linalg.norm(translation))
        axis = translation / magnitude if magnitude > 1e-8 else np.zeros(3)
        return JointTransform(
            joint_type="prismatic",
            axis=axis,
            pivot=np.zeros(3),
            rotation=rotation,
            translation=translation,
            motion=magnitude,
        )

    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    magnitude = float(np.linalg.norm(rotvec))
    axis = rotvec / magnitude if magnitude > 1e-8 else np.zeros(3)
    # (I - R) is singular along the rotation axis. The least-squares solution
    # gives the point on the axis closest to the origin.
    pivot = np.linalg.lstsq(np.eye(3) - rotation, translation, rcond=None)[0]
    pivot = pivot - axis * float(np.dot(axis, pivot))
    return JointTransform(
        joint_type="revolute",
        axis=axis,
        pivot=pivot,
        rotation=rotation,
        translation=translation,
        motion=magnitude,
    )


def evaluate_joint_transforms(
    predicted: list[JointTransform],
    ground_truth: list[JointTransform],
    *,
    bbox_diagonal: float,
    type_mismatch_cost: float = 2.0,
) -> dict[str, Any]:
    """Match arbitrary joint orders and compute suite-common metrics."""
    if bbox_diagonal <= 0:
        raise ValueError("bbox_diagonal must be positive")
    if not predicted or not ground_truth:
        return {
            "predicted_joint_count": len(predicted),
            "gt_joint_count": len(ground_truth),
            "matched_joint_count": 0,
            "joint_type_accuracy": None,
            "axis_angle_error_deg_type_correct": None,
            "revolute_axis_line_error_bbox_normalized": None,
            "revolute_motion_error_rad": None,
            "prismatic_motion_error_m": None,
            "matches": [],
        }

    cost = np.empty((len(predicted), len(ground_truth)), dtype=float)
    for pred_index, pred in enumerate(predicted):
        for gt_index, gt in enumerate(ground_truth):
            angle = _axis_angle_deg(pred.axis, gt.axis) / 90.0
            mismatch = type_mismatch_cost if pred.joint_type != gt.joint_type else 0.0
            motion_scale = max(abs(gt.motion), 1e-6)
            motion = min(abs(pred.motion - gt.motion) / motion_scale, 2.0)
            cost[pred_index, gt_index] = mismatch + angle + 0.1 * motion

    pred_indices, gt_indices = linear_sum_assignment(cost)
    matches: list[dict[str, Any]] = []
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        pred = predicted[pred_index]
        gt = ground_truth[gt_index]
        type_correct = pred.joint_type == gt.joint_type
        axis_error = _axis_angle_deg(pred.axis, gt.axis) if type_correct else None
        line_error = (
            _line_distance(pred.pivot, pred.axis, gt.pivot, gt.axis) / bbox_diagonal
            if type_correct and gt.joint_type == "revolute"
            else None
        )
        matches.append(
            {
                "predicted_index": pred_index,
                "gt_index": gt_index,
                "predicted_type": pred.joint_type,
                "gt_type": gt.joint_type,
                "type_correct": type_correct,
                "axis_angle_error_deg_type_correct": axis_error,
                "axis_line_error_bbox_normalized": line_error,
                "motion_abs_error": abs(pred.motion - gt.motion) if type_correct else None,
            }
        )

    type_correct = [row for row in matches if row["type_correct"]]
    revolute = [row for row in type_correct if row["gt_type"] == "revolute"]
    prismatic = [row for row in type_correct if row["gt_type"] == "prismatic"]
    return {
        "predicted_joint_count": len(predicted),
        "gt_joint_count": len(ground_truth),
        "matched_joint_count": len(matches),
        "joint_type_accuracy": _mean([float(row["type_correct"]) for row in matches]),
        "axis_angle_error_deg_type_correct": _mean(
            [float(row["axis_angle_error_deg_type_correct"]) for row in type_correct]
        ),
        "revolute_axis_line_error_bbox_normalized": _mean(
            [float(row["axis_line_error_bbox_normalized"]) for row in revolute]
        ),
        "revolute_motion_error_rad": _mean(
            [float(row["motion_abs_error"]) for row in revolute]
        ),
        "prismatic_motion_error_m": _mean(
            [float(row["motion_abs_error"]) for row in prismatic]
        ),
        "matches": matches,
    }


def _axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 90.0
    cosine = np.clip(abs(float(np.dot(first, second))) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _line_distance(
    first_point: np.ndarray,
    first_axis: np.ndarray,
    second_point: np.ndarray,
    second_axis: np.ndarray,
) -> float:
    first_axis = np.asarray(first_axis, dtype=float)
    second_axis = np.asarray(second_axis, dtype=float)
    normal = np.cross(first_axis, second_axis)
    normal_norm = float(np.linalg.norm(normal))
    offset = np.asarray(first_point, dtype=float) - np.asarray(second_point, dtype=float)
    if normal_norm <= 1e-8:
        axis_norm = max(float(np.linalg.norm(first_axis)), 1e-12)
        return float(np.linalg.norm(np.cross(offset, first_axis)) / axis_norm)
    return abs(float(np.dot(normal, offset))) / normal_norm


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
