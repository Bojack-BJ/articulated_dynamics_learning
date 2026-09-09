from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_motion_axis_proposals.py"
    spec = importlib.util.spec_from_file_location("evaluate_motion_axis_proposals", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_penalizes_missing_axes_and_splits_joint_types():
    metrics = _module()._metrics([
        {"valid": True, "axis_error_deg": 10.0, "line_error": 0.2, "joint_type": "revolute"},
        {"valid": False, "axis_error_deg": None, "line_error": None, "joint_type": "prismatic"},
    ])
    assert metrics["coverage"] == 0.5
    assert metrics["axis_mean_deg"] == 10.0
    assert metrics["penalized_axis_mean_deg"] == 50.0
    assert metrics["by_type"]["prismatic"]["coverage"] == 0.0


def test_track_voting_does_not_inherit_group_minimum_track_count():
    import numpy as np

    module = _module()
    points = np.zeros((1, 6, 3), dtype=float)
    points[0, :, 0] = np.linspace(0.0, 1.0, 6)
    axis, _, proposal_count = module._estimate(
        "track_voting",
        points,
        np.ones((1, 6), dtype=bool),
        np.ones((1, 6), dtype=float),
        np.ones(1, dtype=float),
        "prismatic",
        3,
        5,
        1,
    )
    assert proposal_count == 1
    assert axis is not None
