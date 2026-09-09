from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rgbd_urdf_mvp.benchmarks.dta_hypothesis_selection import (
    replay_hypothesis,
    select_joint_hypothesis,
)


def _hypotheses() -> tuple[dict, dict]:
    rotation = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(45.0)]).as_matrix()
    revolute = {
        "rotation": rotation.tolist(),
        "translation": [0.0, 0.0, 0.0],
        "axis_position": [0.0, 0.0, 0.0],
    }
    prismatic = {
        "rotation": rotation.tolist(),
        "translation": [0.6, 0.0, 0.0],
        "axis_position": [0.0, 0.0, 0.0],
    }
    return prismatic, revolute


def test_selects_revolute_from_target_replay() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(500, 3)) + np.asarray([2.0, 0.0, 0.0])
    prismatic, revolute = _hypotheses()
    target = replay_hypothesis(source, revolute, "revolute")
    result = select_joint_hypothesis(
        source, target, prismatic=prismatic, revolute=revolute
    )
    assert result["selected_type"] == "revolute"
    assert result["residual_m"]["revolute"] < result["residual_m"]["prismatic"]


def test_selects_prismatic_from_target_replay() -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(size=(500, 3))
    prismatic, revolute = _hypotheses()
    target = replay_hypothesis(source, prismatic, "prismatic")
    result = select_joint_hypothesis(
        source, target, prismatic=prismatic, revolute=revolute
    )
    assert result["selected_type"] == "prismatic"


def test_marks_indistinguishable_low_motion_as_ambiguous() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    identity = {
        "rotation": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "axis_position": [0.0, 0.0, 0.0],
    }
    result = select_joint_hypothesis(
        source, source, prismatic=identity, revolute=identity
    )
    assert result["selected_type"] == "ambiguous"
    assert "low_motion" in result["ambiguity_reasons"]
