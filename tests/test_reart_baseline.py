from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from rgbd_urdf_mvp.benchmarks.reart_baseline import evaluate_reart_segmentation
from rgbd_urdf_mvp.perception.reart_adapter import ReArtSequenceExportConfig, ReArtSequenceExporter
from rgbd_urdf_mvp.perception.reart_run_wrapper import _ball_query_fallback


def _write_part_ply(path: Path, points: list[tuple[float, float, float, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property ushort part_id",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z} {part}" for x, y, z, part in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_object_mask_only_reart_export_keeps_zero_labeled_foreground(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    points = [(0.0, 0.0, 0.0, 0), (1.0, 0.0, 0.0, 0)]
    _write_part_ply(frames / "frame_0000.ply", points)
    _write_part_ply(frames / "frame_0001.ply", points)
    manifest = tmp_path / "fusion_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "per_frame_dir": str(frames),
                "mask_aware_fusion": True,
                "part_ids_present": [],
            }
        ),
        encoding="utf-8",
    )

    output = ReArtSequenceExporter().export(
        ReArtSequenceExportConfig(
            fusion_manifest_path=manifest,
            output_dir=tmp_path / "reart",
            foreground_only=True,
        )
    )

    exported = (output / "frame_0000.ply").read_text(encoding="utf-8")
    metadata = json.loads((output / "reart_sequence_manifest.json").read_text(encoding="utf-8"))
    assert "part_id" not in exported
    assert metadata["foreground_filter_mode"] == "object-mask-prefiltered"
    assert metadata["frames"][0]["point_count"] == 2


def test_reart_export_supports_explicit_source_frame_indices(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    points = [(0.0, 0.0, 0.0, 0), (1.0, 0.0, 0.0, 0)]
    for index in range(6):
        _write_part_ply(frames / f"frame_{index:04d}.ply", points)
    manifest = tmp_path / "fusion_manifest.json"
    manifest.write_text(
        json.dumps({"per_frame_dir": str(frames), "mask_aware_fusion": True, "part_ids_present": []}),
        encoding="utf-8",
    )

    output = ReArtSequenceExporter().export(
        ReArtSequenceExportConfig(
            fusion_manifest_path=manifest,
            output_dir=tmp_path / "reart",
            frame_indices=(0, 2, 4, 5),
            protocol_profile="sapien_count_matched_4frame",
        )
    )

    metadata = json.loads((output / "reart_sequence_manifest.json").read_text(encoding="utf-8"))
    assert metadata["source_frame_indices"] == [0, 2, 4, 5]
    assert metadata["temporal_sampling_provenance"] == "explicit_source_frame_indices"
    assert metadata["protocol_profile"] == "sapien_count_matched_4frame"


def test_reart_count_matched_profile_requires_four_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    points = [(0.0, 0.0, 0.0, 0), (1.0, 0.0, 0.0, 0)]
    for index in range(3):
        _write_part_ply(frames / f"frame_{index:04d}.ply", points)
    manifest = tmp_path / "fusion_manifest.json"
    manifest.write_text(
        json.dumps({"per_frame_dir": str(frames), "mask_aware_fusion": True, "part_ids_present": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly four"):
        ReArtSequenceExporter().export(
            ReArtSequenceExportConfig(
                fusion_manifest_path=manifest,
                output_dir=tmp_path / "reart",
                protocol_profile="sapien_count_matched_4frame",
            )
        )


def test_reart_segmentation_uses_common_point_domain_evaluator(tmp_path: Path) -> None:
    reference = tmp_path / "reference.ply"
    points = [
        (0.0, 0.0, 0.0, 1),
        (0.1, 0.0, 0.0, 1),
        (1.0, 0.0, 0.0, 2),
        (1.1, 0.0, 0.0, 2),
    ]
    _write_part_ply(reference, points)
    result = tmp_path / "result.pkl"
    with result.open("wb") as stream:
        pickle.dump(
            {
                "cano_pc": np.asarray([row[:3] for row in points], dtype=np.float32),
                "pred_cano_part": np.asarray([7, 7, 3, 3], dtype=np.int64),
                "pred_pose_list": np.repeat(np.eye(4)[None, None], 2, axis=0),
                "cano_idx": 0,
                "joint_connection": [[7, 3]],
            },
            stream,
        )

    evaluation = evaluate_reart_segmentation(
        result,
        reference,
        primary_distance_ratio=1.0,
    )

    metrics = evaluation["primary"]["covered_only_metrics"]
    assert metrics["one_to_one_mean_iou"] == 1.0
    assert metrics["adjusted_rand_index"] == 1.0
    assert evaluation["native_outputs"]["joint_type"] is False
    assert evaluation["metric_support"]["directed_edge_f1"] == "unsupported_native"


def test_reart_ball_query_fallback_repeats_nearest_for_empty_balls() -> None:
    torch = pytest.importorskip("torch")
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    queries = torch.tensor([[[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]])

    indices = _ball_query_fallback(0.25, 2, xyz, queries)

    assert indices.tolist() == [[[0, 0], [2, 2]]]
