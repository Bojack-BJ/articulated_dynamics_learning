from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rgbd_urdf_mvp.benchmarks.gaussianart_evaluation import (
    JointTransform,
    evaluate_joint_transforms,
    transform_to_joint,
)


def _revolute(axis: list[float], pivot: list[float], angle: float) -> JointTransform:
    axis_array = np.asarray(axis, dtype=float)
    pivot_array = np.asarray(pivot, dtype=float)
    rotation = Rotation.from_rotvec(axis_array * angle).as_matrix()
    translation = (np.eye(3) - rotation) @ pivot_array
    return transform_to_joint(rotation, translation)


def _prismatic(axis: list[float], distance: float) -> JointTransform:
    translation = np.asarray(axis, dtype=float) * distance
    return transform_to_joint(np.eye(3), translation)


def test_hungarian_matching_is_invariant_to_joint_order() -> None:
    gt = [
        _revolute([0, 0, 1], [1, 0, 0], 0.8),
        _prismatic([1, 0, 0], 0.3),
    ]
    result = evaluate_joint_transforms(
        [gt[1], gt[0]],
        gt,
        bbox_diagonal=2.0,
    )
    assert result["joint_type_accuracy"] == 1.0
    assert result["axis_angle_error_deg_type_correct"] == 0.0
    assert result["revolute_axis_line_error_bbox_normalized"] < 1e-8


def test_type_mismatch_is_not_included_in_axis_error() -> None:
    result = evaluate_joint_transforms(
        [_prismatic([0, 0, 1], 0.2)],
        [_revolute([0, 0, 1], [0, 0, 0], 0.5)],
        bbox_diagonal=1.0,
    )
    assert result["joint_type_accuracy"] == 0.0
    assert result["axis_angle_error_deg_type_correct"] is None
    assert result["revolute_axis_line_error_bbox_normalized"] is None


def test_revolute_line_distance_is_bbox_normalized() -> None:
    gt = _revolute([0, 0, 1], [0, 0, 0], 0.5)
    pred = _revolute([0, 0, 1], [0.2, 0, 0], 0.5)
    result = evaluate_joint_transforms([pred], [gt], bbox_diagonal=2.0)
    assert np.isclose(result["revolute_axis_line_error_bbox_normalized"], 0.1)
