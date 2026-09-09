from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path("scripts/evaluate_videoartgs_common.py")
    spec = importlib.util.spec_from_file_location("evaluate_videoartgs_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_axis_angle_is_sign_invariant() -> None:
    module = _load_script()
    assert module.axis_angle_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(0.0)
    assert module.axis_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)


def test_parallel_axis_line_distance() -> None:
    module = _load_script()
    distance = module.axis_line_distance(
        [0, 0, 0],
        [0, 0, 1],
        [0.25, 0, 2],
        [0, 0, -1],
    )
    assert distance == pytest.approx(0.25)


def test_joint_evaluation_conditions_axis_on_correct_type() -> None:
    module = _load_script()
    result = module.evaluate_joints(
        [
            {
                "joint": "slider",
                "jointData": {
                    "axis": {"origin": [0, 0, 0], "direction": [1, 0, 0]}
                },
            }
        ],
        [
            {"joint_type": "s"},
            {
                "joint_type": "r",
                "origin": [0, 0, 0],
                "direction": [1, 0, 0],
            },
        ],
        bbox_diagonal=1.0,
    )
    assert result["joint_type_accuracy"] == pytest.approx(0.0)
    assert result["axis_angle_deg_type_correct"] is None
    assert result["revolute_axis_line_bbox"] is None
