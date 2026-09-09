from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import read_ascii_labeled_ply


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_artgs_predictions.py"
SPEC = importlib.util.spec_from_file_location("export_artgs_predictions", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_write_labeled_ply_preserves_slot_partition(tmp_path: Path) -> None:
    path = tmp_path / "labels.ply"
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    MODULE.write_labeled_ply(path, points, np.asarray([0, 1, 1]))

    loaded_points, encoded_labels = read_ascii_labeled_ply(path, label_mode="rgb")
    assert np.allclose(loaded_points, points)
    assert encoded_labels[0] != encoded_labels[1]
    assert encoded_labels[1] == encoded_labels[2]
