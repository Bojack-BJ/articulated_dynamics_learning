from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.tapip3d_adapter import (
    TAPIP3DImportConfig,
    TAPIP3DInputConfig,
    TAPIP3DInputPreparer,
    TAPIP3DMergeConfig,
    TAPIP3DMultiViewMerger,
    TAPIP3DTrackImporter,
    _tapip_world_to_camera,
)


class TAPIP3DAdapterTests(unittest.TestCase):
    def test_cli_parsers_accept_remote_tapip3d_round_trip(self) -> None:
        parser = build_parser()
        prepare = parser.parse_args(
            [
                "prepare-tapip3d-input", "episode.json", "--output-npz", "tapip_input.npz",
                "--view-index", "1", "--frame-stride", "4", "--seed-tracks", "tracks.json",
            ]
        )
        imported = parser.parse_args(
            [
                "import-tapip3d-tracks", "tapip_input.npz", "result.npz", "tracks.json",
                "--output-json", "tapip_tracks.json",
            ]
        )
        self.assertEqual(prepare.command, "prepare-tapip3d-input")
        self.assertEqual(prepare.view_index, 1)
        self.assertEqual(imported.command, "import-tapip3d-tracks")
        self.assertEqual(imported.output_json, Path("tapip_tracks.json"))
        probe = parser.parse_args(["probe-track-features", "tracks.json", "features.npz"])
        self.assertEqual(probe.command, "probe-track-features")
        merge = parser.parse_args([
            "merge-tapip3d-views", "view0.json", "view1.json",
            "--output-tracks", "merged.json",
        ])
        self.assertEqual(merge.command, "merge-tapip3d-views")

    def test_pose_conversion_uses_cv_camera_y_and_world_to_camera(self) -> None:
        camera_to_world = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
        actual = _tapip_world_to_camera(camera_to_world)
        expected = np.asarray([[1.0, 0.0, 0.0, -1.0], [0.0, -1.0, 0.0, 2.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
        np.testing.assert_allclose(actual, expected)

    def test_prepare_and_import_preserves_seed_track_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames = []
            for index in range(2):
                Image.fromarray(np.full((3, 4, 3), 30 + index, dtype=np.uint8)).save(root / f"rgb_{index}.png")
                Image.fromarray(np.full((3, 4), 1000, dtype=np.uint16)).save(root / f"depth_{index}.png")
                frames.append({"timestamp_s": index * 0.1, "rgb_path": f"rgb_{index}.png", "depth_path": f"depth_{index}.png", "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
            episode_path = root / "episode.json"
            episode_path.write_text(json.dumps({"object_instance_id": "demo", "category": "door", "camera_intrinsics": {"fx": 2, "fy": 2, "cx": 1.5, "cy": 1}, "frames": frames}), encoding="utf-8")
            seed_path = root / "seeds.json"
            seed_path.write_text(json.dumps({"source_frame_count": 2, "tracks": [{"track_id": 7, "part_id": 2, "part_name": "door", "view_index": 0, "query_source_frame_index": 0, "reference_xyz_world": [0.0, 0.0, 1.0], "samples": [{"source_frame_index": 0, "timestamp_s": 0.0}, {"source_frame_index": 1, "timestamp_s": 0.1}]}]}), encoding="utf-8")
            input_path = TAPIP3DInputPreparer(TAPIP3DInputConfig(episode_path=episode_path, output_npz=root / "input.npz", seed_tracks=seed_path)).prepare()
            with np.load(input_path) as data:
                self.assertEqual(data["video"].shape, (2, 3, 4, 3))
                self.assertEqual(data["query_point"].shape, (1, 4))
                np.testing.assert_array_equal(data["track_ids"], [7])
            result_path = root / "result.npz"
            np.savez(result_path, coords=np.asarray([[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.0]]]), visibs=np.asarray([[True], [True]]))
            output_path = TAPIP3DTrackImporter(TAPIP3DImportConfig(tapip_input_npz=input_path, tapip_result_npz=result_path, seed_tracks=seed_path, output_json=root / "tracks.json")).import_tracks()
            imported = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(imported["estimator"], "tapip3d-world-track-import")
            self.assertEqual(imported["tracks"][0]["track_id"], 7)
            self.assertEqual(imported["tracks"][0]["samples"][1]["xyz_world"], [0.1, 0.0, 1.0])

    def test_multiview_merge_preserves_world_tracks_and_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_paths = []
            feature_paths = []
            for view_index, track_id in enumerate((10, 20)):
                track_path = root / f"view{view_index}.json"
                track_path.write_text(json.dumps({
                    "frame_count": 2,
                    "tracks": [{"track_id": track_id, "view_index": view_index, "samples": []}],
                }), encoding="utf-8")
                feature_path = root / f"view{view_index}.npz"
                np.savez_compressed(
                    feature_path,
                    track_ids=np.asarray([track_id]),
                    embeddings=np.full((1, 6), view_index + 1, dtype=np.float32),
                    temporal_tokens=np.full((1, 2, 2), view_index + 1, dtype=np.float16),
                    valid_timestep_count=np.asarray([2]),
                )
                track_paths.append(track_path)
                feature_paths.append(feature_path)
            tracks_output, features_output = TAPIP3DMultiViewMerger(TAPIP3DMergeConfig(
                track_paths=track_paths,
                output_tracks=root / "merged.json",
                feature_paths=feature_paths,
                output_features=root / "merged.npz",
            )).merge()
            merged_tracks = json.loads(tracks_output.read_text(encoding="utf-8"))
            self.assertEqual(merged_tracks["view_count"], 2)
            self.assertEqual([track["track_id"] for track in merged_tracks["tracks"]], [10, 20])
            self.assertIsNotNone(features_output)
            with np.load(features_output) as merged_features:
                np.testing.assert_array_equal(merged_features["track_ids"], [10, 20])
                self.assertEqual(merged_features["temporal_tokens"].shape, (2, 2, 2))
