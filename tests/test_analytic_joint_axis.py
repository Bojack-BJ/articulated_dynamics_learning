from __future__ import annotations

import math
import json
import unittest

import numpy as np

from rgbd_urdf_mvp.kinematics.analytic_joint_axis import (
    AnalyticAxisConfig,
    axis_angle_error_deg,
    estimate_analytic_joint_axis,
    select_analytic_joint_model,
)
from rgbd_urdf_mvp.kinematics.so3_augmentation import sample_uniform_so3_batch


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _synthetic_points(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(48, 3)) * np.asarray([0.4, 0.25, 0.2])


def _revolute_tracks(
    axis: np.ndarray, pivot: np.ndarray, *, frames: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent_reference = _synthetic_points(1)
    child_reference = _synthetic_points(2) + np.asarray([0.8, 0.1, 0.0])
    parent = np.repeat(parent_reference[:, None, :], frames, axis=1)
    child = np.zeros((len(child_reference), frames, 3), dtype=float)
    for frame, angle in enumerate(np.linspace(0.0, 0.8, frames)):
        rotation = _rotation(axis, float(angle))
        child[:, frame] = (child_reference - pivot) @ rotation.T + pivot
    return parent, np.ones(parent.shape[:2], bool), child, np.ones(child.shape[:2], bool)


class AnalyticJointAxisTests(unittest.TestCase):
    def test_tracks_may_first_appear_after_frame_zero(self) -> None:
        parent, parent_visibility, child, child_visibility = _revolute_tracks(
            np.asarray([0.0, 0.0, 1.0]), np.asarray([0.2, 0.0, 0.0])
        )
        child_visibility[:, 0] = False
        estimate = estimate_analytic_joint_axis(
            parent, parent_visibility, child, child_visibility, "revolute"
        )
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, [0.0, 0.0, 1.0]), 1.0)

    def test_noiseless_revolute_motion(self) -> None:
        axis = np.asarray([0.2, -0.4, 0.9])
        axis /= np.linalg.norm(axis)
        pivot = np.asarray([0.1, -0.2, 0.3])
        estimate = estimate_analytic_joint_axis(*_revolute_tracks(axis, pivot), "revolute")
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 0.05)
        self.assertLess(np.linalg.norm(np.cross(np.asarray(estimate.line_point) - pivot, axis)), 1e-4)

    def test_noiseless_prismatic_motion(self) -> None:
        axis = np.asarray([0.3, 0.8, -0.2])
        axis /= np.linalg.norm(axis)
        frames = 10
        parent_reference = _synthetic_points(3)
        child_reference = _synthetic_points(4)
        parent = np.repeat(parent_reference[:, None, :], frames, axis=1)
        child = np.stack([child_reference + axis * q for q in np.linspace(0.0, 0.5, frames)], axis=1)
        estimate = estimate_analytic_joint_axis(
            parent, np.ones(parent.shape[:2], bool), child, np.ones(child.shape[:2], bool), "prismatic"
        )
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, -axis), 0.05)
        self.assertGreater(estimate.eigenvalue_ratio, 0.999)

    def test_model_selection_recovers_noiseless_revolute(self) -> None:
        axis = np.asarray([0.2, -0.4, 0.9])
        axis /= np.linalg.norm(axis)
        selection = select_analytic_joint_model(
            *_revolute_tracks(axis, np.asarray([0.1, -0.2, 0.3]))
        )
        self.assertTrue(selection.valid)
        self.assertEqual(selection.selected_type, "revolute")
        self.assertLess(selection.revolute_replay_rmse, selection.prismatic_replay_rmse)

    def test_model_selection_recovers_noiseless_prismatic(self) -> None:
        axis = np.asarray([0.3, 0.8, -0.2])
        axis /= np.linalg.norm(axis)
        frames = 10
        parent_reference = _synthetic_points(3)
        child_reference = _synthetic_points(4)
        parent = np.repeat(parent_reference[:, None, :], frames, axis=1)
        child = np.stack(
            [child_reference + axis * q for q in np.linspace(0.0, 0.5, frames)], axis=1
        )
        selection = select_analytic_joint_model(
            parent, np.ones(parent.shape[:2], bool),
            child, np.ones(child.shape[:2], bool),
        )
        self.assertTrue(selection.valid)
        self.assertEqual(selection.selected_type, "prismatic")
        self.assertLess(selection.prismatic_replay_rmse, selection.revolute_replay_rmse)

    def test_invalid_model_selection_is_strict_json_serializable(self) -> None:
        points = np.zeros((2, 2, 3), dtype=float)
        visibility = np.ones((2, 2), dtype=bool)
        selection = select_analytic_joint_model(
            points, visibility, points, visibility,
        )
        self.assertFalse(selection.valid)
        json.dumps(selection.to_dict(), allow_nan=False)

    def test_axis_sign_is_undirected(self) -> None:
        self.assertAlmostEqual(axis_angle_error_deg([1, 0, 0], [-1, 0, 0]), 0.0)

    def test_low_motion_is_rejected(self) -> None:
        axis = np.asarray([0.0, 0.0, 1.0])
        tracks = _revolute_tracks(axis, np.zeros(3), frames=8)
        parent, parent_vis, child, child_vis = tracks
        child = child[:, :1] + (child - child[:, :1]) * 0.005
        estimate = estimate_analytic_joint_axis(parent, parent_vis, child, child_vis, "revolute")
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "insufficient_rotation")

    def test_missing_visibility_is_supported(self) -> None:
        axis = np.asarray([0.4, 0.1, 0.8])
        axis /= np.linalg.norm(axis)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, np.asarray([0.1, 0.2, 0.0]))
        parent_vis[:12, 3:7] = False
        child_vis[12:24, 5:9] = False
        estimate = estimate_analytic_joint_axis(parent, parent_vis, child, child_vis, "revolute")
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 0.2)

    def test_pose_chain_recovers_after_fully_missing_frame(self) -> None:
        axis = np.asarray([0.2, 0.7, -0.1])
        axis /= np.linalg.norm(axis)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, np.zeros(3))
        parent_vis[:, 5] = False
        child_vis[:, 5] = False
        estimate = estimate_analytic_joint_axis(parent, parent_vis, child, child_vis, "revolute")
        self.assertTrue(estimate.valid)
        self.assertGreaterEqual(estimate.valid_frame_count, 8)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 0.2)

    def test_outlier_frames_are_trimmed_by_axis_consensus(self) -> None:
        axis = np.asarray([0.1, 0.9, 0.3])
        axis /= np.linalg.norm(axis)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, np.asarray([0.2, 0.0, -0.1]), frames=15)
        wrong_rotation = _rotation(np.asarray([1.0, 0.0, 0.0]), 0.7)
        child[:, 8] = child[:, 7] @ wrong_rotation.T
        estimate = estimate_analytic_joint_axis(parent, parent_vis, child, child_vis, "revolute")
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 5.0)

    def test_segmentwise_fit_recovers_after_persistent_surface_jump(self) -> None:
        axis = np.asarray([0.2, 0.7, -0.1]); axis /= np.linalg.norm(axis)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, np.zeros(3), frames=18)
        child[:, 8:] += np.asarray([0.45, -0.25, 0.2])
        estimate = estimate_analytic_joint_axis(
            parent, parent_vis, child, child_vis, "revolute",
            AnalyticAxisConfig(robust_segment_fitting=True, max_segment_translation_jump=0.08),
        )
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 3.0)

    def test_huber_irls_rejects_id_switched_tracks(self) -> None:
        axis = np.asarray([0.1, 0.9, 0.3]); axis /= np.linalg.norm(axis)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, np.zeros(3), frames=16)
        child[:8, 5:] += np.asarray([0.5, -0.4, 0.3])
        estimate = estimate_analytic_joint_axis(
            parent, parent_vis, child, child_vis, "revolute",
            AnalyticAxisConfig(robust_segment_fitting=True, huber_delta_m=0.015),
        )
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 5.0)

    def test_short_disconnected_segments_are_hard_rejected(self) -> None:
        relative = [(0, np.eye(4)), (1, np.eye(4)), (9, np.eye(4)), (12, np.eye(4))]
        config = AnalyticAxisConfig(robust_segment_fitting=True, min_valid_frames=3)
        from rgbd_urdf_mvp.kinematics.analytic_joint_axis import _rebase_relative_segments
        self.assertEqual(_rebase_relative_segments(relative, config), [])

    def test_so3_rotated_example(self) -> None:
        base_axis = np.asarray([0.0, 0.0, 1.0])
        global_rotation = _rotation(np.asarray([0.3, 0.7, 0.2]), 1.1)
        axis = global_rotation @ base_axis
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, global_rotation @ np.asarray([0.2, 0.1, 0.0]))
        estimate = estimate_analytic_joint_axis(parent, parent_vis, child, child_vis, "revolute")
        self.assertTrue(estimate.valid)
        self.assertLess(axis_angle_error_deg(estimate.axis, axis), 0.1)

    def test_revolute_solver_is_equivariant_over_haar_so3(self) -> None:
        axis = np.asarray([0.2, -0.4, 0.9], dtype=float)
        axis /= np.linalg.norm(axis)
        pivot = np.asarray([0.1, -0.2, 0.3], dtype=float)
        parent, parent_vis, child, child_vis = _revolute_tracks(axis, pivot)
        baseline = estimate_analytic_joint_axis(
            parent, parent_vis, child, child_vis, "revolute"
        )
        self.assertTrue(baseline.valid)
        errors = []
        for rotation in sample_uniform_so3_batch(100, seed=7):
            rotated = estimate_analytic_joint_axis(
                parent @ rotation.T,
                parent_vis,
                child @ rotation.T,
                child_vis,
                "revolute",
            )
            self.assertTrue(rotated.valid)
            errors.append(
                axis_angle_error_deg(rotated.axis, rotation @ np.asarray(baseline.axis))
            )
        self.assertLess(float(np.mean(errors)), 1e-4)
        self.assertLess(float(np.median(errors)), 1e-4)
        self.assertLess(float(np.max(errors)), 1e-3)


if __name__ == "__main__":
    unittest.main()
