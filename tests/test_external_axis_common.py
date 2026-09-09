import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("external_axis", Path(__file__).parents[1]/"scripts/evaluate_external_axis_common.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_axes_cannot_change_part_matching():
    points = np.array([[0,0,0],[0,.1,0],[1,0,0],[1,.1,0]], dtype=float)
    mapping, _ = module.part_mapping(points, np.array([7,7,2,2]), points, np.array([1,1,3,3]), 2)
    assert mapping == {7:1, 2:3}
    gt = [{"name":"j", "child":3,"type":"prismatic","axis":[1,0,0],"origin":[0,0,0]}]
    pred = [{"child":7,"type":"prismatic","axis":[1,0,0]},
            {"child":2,"type":"prismatic","axis":[0,1,0]}]
    result = module.joint_metrics(pred, gt, mapping, 2)[0]
    assert result["axis_deg"] == 90


def test_missing_and_duplicate_child_are_penalized():
    gt = [{"name":"j", "child":3,"type":"prismatic","axis":[1,0,0]}]
    pred = [{"child":2,"type":"prismatic","axis":[1,0,0]}]*2
    for predictions in ([], pred):
        result = module.joint_metrics(predictions, gt, {2:3}, 2)[0]
        assert not result["matched"]
        assert result["axis_penalty_deg"] == 90
