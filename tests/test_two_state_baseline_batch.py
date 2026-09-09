from __future__ import annotations

import csv
import sys
from pathlib import Path

import importlib.util


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_two_state_baseline_batch.py"
_SPEC = importlib.util.spec_from_file_location("run_two_state_baseline_batch", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_manifest(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("object_id", "category", "gt_part_count"))
        writer.writerows(rows)


def test_priority_manifest_orders_core_before_full_extras(tmp_path: Path) -> None:
    full = tmp_path / "full.csv"
    core = tmp_path / "core.csv"
    _write_manifest(
        full,
        [("extra", "door", 2), ("core_b", "drawer", 3), ("core_a", "oven", 2)],
    )
    _write_manifest(core, [("core_a", "oven", 2), ("core_b", "drawer", 3)])
    jobs = _MODULE.load_jobs(full, priority_manifest=core)
    assert [job.object_id for job in jobs] == ["core_a", "core_b", "extra"]


def test_manifest_sharding_is_disjoint_and_complete(tmp_path: Path) -> None:
    manifest = tmp_path / "full.csv"
    rows = [(f"object_{index}", "door", 2) for index in range(7)]
    _write_manifest(manifest, rows)
    shards = [
        _MODULE.load_jobs(manifest, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    flattened = [job.object_id for shard in shards for job in shard]
    assert len(flattened) == len(set(flattened)) == len(rows)
    assert set(flattened) == {row[0] for row in rows}


def test_dta_completion_requires_every_oracle_part(tmp_path: Path) -> None:
    job = _MODULE.ObjectJob("partnet_1", "drawer", 3)
    step = (
        tmp_path
        / "runs"
        / "external_baseline_suite_v1"
        / job.object_id
        / "results"
        / "step_0004000"
    )
    step.mkdir(parents=True)
    for index in range(2):
        (step / f"init_part_{index}_clustered.obj").write_text("", encoding="utf-8")
    for kind in ("prismatic", "revolute"):
        (step / f"init_{kind}_motion.json").write_text("[]", encoding="utf-8")
    assert not _MODULE.native_complete("dta", tmp_path, job)
    (step / "init_part_2_clustered.obj").write_text("", encoding="utf-8")
    assert _MODULE.native_complete("dta", tmp_path, job)
