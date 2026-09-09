import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "export_rbo_mocap_masks", Path(__file__).parents[1] / "scripts" / "export_rbo_mocap_masks.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_transform_and_render_one_point():
    pose = MODULE.transform([1, 2, 3], rpy=[0, 0, 0])
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
    labels, depth = MODULE.render_labels(
        [(1, np.asarray([[0.0, 0.0, 1.0]]))],
        np.eye(4),
        (100.0, 100.0, 2.0, 2.0),
        (5, 5),
        np.ones((5, 5), dtype=np.float32),
        0.01,
    )
    assert labels[2, 2] == 1
    assert depth[2, 2] == 1.0


def test_nearest_complete_pose_skips_partial_packet():
    rows = [(1.0, {1: "partial"}), (1.1, {1: "a", 2: "b"})]
    assert MODULE.nearest_complete_pose(rows, 1.0, {1, 2}) == {1: "a", 2: "b"}
