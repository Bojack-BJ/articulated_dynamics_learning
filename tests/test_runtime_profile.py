from __future__ import annotations

from rgbd_urdf_mvp.benchmarks.runtime_profile import hardware_snapshot


def test_hardware_snapshot_without_torch_is_serializable() -> None:
    snapshot = hardware_snapshot()
    assert snapshot["hostname"]
    assert snapshot["machine"]
    assert isinstance(snapshot["cpu_logical_count"], int)
    assert set(snapshot["thread_environment"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
