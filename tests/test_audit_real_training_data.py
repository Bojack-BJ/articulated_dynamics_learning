from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "audit_real_training_data.py"
    spec = importlib.util.spec_from_file_location("audit_real_training_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_track_stats_reports_fragmentation_and_jumps() -> None:
    module = _module()
    payload = {
        "frame_count": 5,
        "tracks": [{
            "track_quality_score": 0.8,
            "samples": [
                {"frame_index": 0, "visible": True, "xyz_world": [0, 0, 0], "timestep_quality_score": 0.9},
                {"frame_index": 1, "visible": True, "xyz_world": [0.01, 0, 0], "timestep_quality_score": 0.7},
                {"frame_index": 2, "visible": False, "xyz_world": None},
                {"frame_index": 3, "visible": True, "xyz_world": [1, 0, 0]},
                {"frame_index": 4, "visible": True, "xyz_world": [1.2, 0, 0]},
            ],
        }],
    }
    result = module._track_stats(payload, jump_m=0.05)
    assert result["track_count"] == 1
    assert result["min_part_track_count"] == 1
    assert result["track_lifetime_median"] == 4
    assert result["longest_contiguous_run_median"] == 2
    assert result["jump_ratio"] == 0.5
    assert result["track_quality_median"] == 0.8
