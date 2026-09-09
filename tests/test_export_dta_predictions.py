from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_dta_predictions.py"
SPEC = importlib.util.spec_from_file_location("export_dta_predictions", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sample_part_meshes_preserves_two_labels(tmp_path: Path) -> None:
    paths = []
    for index, offset in enumerate((0.0, 2.0)):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.apply_translation((offset, 0.0, 0.0))
        path = tmp_path / f"part_{index}.obj"
        mesh.export(path)
        paths.append(path)

    points, labels = MODULE.sample_part_meshes(paths, sample_count=200, seed=3)
    assert points.shape == (200, 3)
    assert set(labels.tolist()) == {0, 1}


def test_latest_result_step_skips_newer_incomplete_directory(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "results" / "step_0002000"
    incomplete = tmp_path / "results" / "step_0002500"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (complete / "init_part_0_clustered.obj").write_text("", encoding="utf-8")

    assert MODULE.latest_result_step(tmp_path) == complete
