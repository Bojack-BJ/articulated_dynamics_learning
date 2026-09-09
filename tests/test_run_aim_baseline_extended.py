from pathlib import Path

from scripts.run_aim_baseline_extended import _reference_frame_path, _source_frame_index


def test_reference_frame_path_uses_requested_source_frame() -> None:
    root = Path("/tmp/episode/pointcloud_4d_partseg")

    assert _reference_frame_path(root, 0) == root / "frames/frame_0000.ply"
    assert _reference_frame_path(root, 116) == root / "frames/frame_0116.ply"


def test_source_frame_index_prefers_per_object_manifest_value() -> None:
    args = type("Args", (), {"source_frame_index": 116})()

    assert _source_frame_index(args, {"source_frame_index": "499"}) == 499
    assert _source_frame_index(args, {}) == 116
