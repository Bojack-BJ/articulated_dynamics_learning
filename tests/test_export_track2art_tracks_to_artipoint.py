import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_track2art_tracks_to_artipoint.py"
SPEC = importlib.util.spec_from_file_location("track2art_artipoint_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exports_only_xyz_and_visibility(tmp_path: Path) -> None:
    source = tmp_path / "tracks.json"
    source.write_text(
        json.dumps(
            {
                "frame_count": 2,
                "sampled_frame_indices": [0, 4],
                "tracks": [
                    {
                        "part_id": 7,
                        "original_part_id": 3,
                        "samples": [
                            {"frame_index": 0, "visible": True, "xyz_world": [1, 2, 3]},
                            {"frame_index": 1, "visible": False, "xyz_world": [4, 5, 6]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tracks_world.npz"
    manifest_path = tmp_path / "manifest.json"
    manifest = MODULE.write_artipoint_tracks(source, output, manifest_path)

    loaded = np.load(output, allow_pickle=True)
    assert loaded["tracks"].shape == (1,)
    assert loaded["visibility"].shape == (1,)
    tracks = np.asarray(loaded["tracks"][0], dtype=float)
    visibility = np.asarray(loaded["visibility"][0], dtype=bool)
    assert np.asarray(loaded["visibility"][0]).dtype == np.bool_
    assert tracks.shape == (2, 1, 3)
    assert visibility.tolist() == [[True], [False]]
    assert manifest["passes_track2art_part_assignments"] is False
    assert manifest["passes_gt_part_or_joint_information"] is False
    assert "part_id" not in manifest_path.read_text(encoding="utf-8")
