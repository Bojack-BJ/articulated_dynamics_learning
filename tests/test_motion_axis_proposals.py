from __future__ import annotations

import numpy as np

from rgbd_urdf_mvp.kinematics.group_motion_axis import (
    build_group_motion_proposals,
    consensus_group_axis,
)
from rgbd_urdf_mvp.kinematics.track_motion_voting import (
    build_track_motion_proposals,
    consensus_track_axes,
)


def _error(a, b):
    return np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def _revolute_tracks():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(24, 3)) * 0.2
    axis = np.asarray([0.2, -0.3, 0.932]); axis /= np.linalg.norm(axis)
    pivot = np.asarray([0.1, -0.2, 0.05])
    frames = []
    for angle in np.linspace(0, 1.0, 16):
        cross = np.asarray([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
        frames.append((reference - pivot) @ rotation.T + pivot)
    return np.stack(frames, axis=1), axis


def test_group_proposals_recover_revolute_axis():
    points, axis = _revolute_tracks()
    proposals = build_group_motion_proposals(points, np.ones(points.shape[:2], bool), strides=(2, 4))
    predicted, _ = consensus_group_axis(proposals)
    assert _error(predicted, axis) < 0.1


def test_track_voting_recovers_revolute_axis_without_part_pose():
    points, axis = _revolute_tracks()
    proposals = build_track_motion_proposals(points, np.ones(points.shape[:2], bool))
    predicted, _ = consensus_track_axes(proposals, "revolute")
    assert _error(predicted, axis) < 0.1


def test_track_voting_recovers_prismatic_axis():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(20, 3))
    axis = np.asarray([0.4, 0.8, -0.2]); axis /= np.linalg.norm(axis)
    points = np.stack([reference + value * axis for value in np.linspace(0, 0.5, 12)], axis=1)
    proposals = build_track_motion_proposals(points, np.ones(points.shape[:2], bool))
    predicted, _ = consensus_track_axes(proposals, "prismatic")
    assert _error(predicted, axis) < 0.1


def test_track_voting_splits_visibility_gaps():
    points, _ = _revolute_tracks()
    visibility = np.ones(points.shape[:2], bool)
    visibility[:, 6:10] = False
    proposals = build_track_motion_proposals(
        points, visibility, min_segment_frames=4, max_gap=1
    )
    assert proposals
    assert all(row.end_frame < 6 or row.start_frame >= 10 for row in proposals)


def test_motion_proposals_reject_mismatched_visibility():
    points = np.zeros((4, 8, 3))
    visibility = np.ones((4, 7), bool)
    for builder in (build_group_motion_proposals, build_track_motion_proposals):
        try:
            builder(points, visibility)
        except ValueError as error:
            assert "visibility" in str(error)
        else:
            raise AssertionError("mismatched visibility should fail")
