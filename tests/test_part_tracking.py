from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.part_tracking import (
    TrackPartPoseEstimationConfig,
    TrackPartPoseEstimator,
    _backproject_track_sample,
    _sample_foreground_seed_pixels,
)


def _rotation_z(angle_rad: float) -> list[list[float]]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _transform(point: list[float], rotation: list[list[float]], translation: list[float]) -> list[float]:
    return [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2] + translation[0],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2] + translation[1],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2] + translation[2],
    ]


def _track(track_id: int, part_id: int, source: list[float], target: list[float]) -> dict:
    return {
        "track_id": track_id,
        "part_id": part_id,
        "part_name": f"part_{part_id}",
        "view_index": 0,
        "query_frame_index": 0,
        "query_uv": [0.0, 0.0],
        "reference_xyz_world": source,
        "samples": [
            {
                "frame_index": 0,
                "timestamp_s": 0.0,
                "uv": [0.0, 0.0],
                "xyz_world": source,
                "visible": True,
                "confidence": 1.0,
            },
            {
                "frame_index": 1,
                "timestamp_s": 0.1,
                "uv": [0.0, 0.0],
                "xyz_world": target,
                "visible": True,
                "confidence": 1.0,
            },
        ],
    }


class PartTrackingTests(unittest.TestCase):
    def test_track_part_pixels_parser_accepts_mac_device(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "track-part-pixels",
                "episode.json",
                "--output-json",
                "part_tracks.json",
                "--device",
                "mps",
                "--cotracker-repo",
                "co-tracker",
                "--cotracker-checkpoint",
                "co-tracker/ckpt/scaled_offline.pth",
                "--unsafe-force-mps",
                "--reference-frame",
                "0",
                "--frame-stride",
                "4",
                "--seed-stride-px",
                "12",
                "--no-progress",
            ]
        )
        self.assertEqual(args.command, "track-part-pixels")
        self.assertEqual(args.device, "mps")
        self.assertTrue(args.unsafe_force_mps)
        self.assertEqual(args.frame_stride, 4)
        self.assertEqual(args.seed_stride_px, 12)
        self.assertTrue(args.no_progress)
        self.assertEqual(Path(args.cotracker_checkpoint), Path("co-tracker/ckpt/scaled_offline.pth"))

    def test_estimate_part_poses_parser_accepts_tracks_method(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "estimate-part-poses",
                "part_tracks.json",
                "--method",
                "tracks",
                "--min-tracks-per-part",
                "5",
            ]
        )
        self.assertEqual(args.command, "estimate-part-poses")
        self.assertEqual(args.method, "tracks")
        self.assertEqual(args.min_tracks_per_part, 5)

    def test_probe_torch_mps_parser_accepts_force_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-torch-mps",
                "--unsafe-force-mps",
            ]
        )
        self.assertEqual(args.command, "probe-torch-mps")
        self.assertTrue(args.unsafe_force_mps)

    def test_object_mask_helpers_treat_any_nonzero_label_as_foreground(self) -> None:
        mask = [
            [0, 0, 0, 0],
            [0, 255, 0, 2],
            [0, 0, 0, 0],
        ]
        seeds = _sample_foreground_seed_pixels(mask, stride_px=1, max_points=8)
        self.assertEqual(seeds, [(1, 1), (3, 1)])
        xyz, depth_valid, mask_consistent = _backproject_track_sample(
            u_float=1.0,
            v_float=1.0,
            depth_u16=[
                [1000, 1000, 1000, 1000],
                [1000, 1200, 1000, 1000],
                [1000, 1000, 1000, 1000],
            ],
            part_mask_u16=mask,
            expected_part_id=-1,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            camera_pose=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            depth_convention="opengl",
            min_depth_m=0.05,
            max_depth_m=6.0,
            require_part_mask_consistency=True,
        )
        self.assertTrue(depth_valid)
        self.assertTrue(mask_consistent)
        self.assertIsNotNone(xyz)

    def test_track_based_pose_estimation_recovers_rigid_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_points = [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4],
            ]
            rotation = _rotation_z(math.radians(30.0))
            translation = [0.4, -0.2, 0.1]
            base_tracks = [
                _track(index, 1, point, point)
                for index, point in enumerate(source_points)
            ]
            door_tracks = [
                _track(10 + index, 2, point, _transform(point, rotation, translation))
                for index, point in enumerate(source_points)
            ]
            track_path = root / "part_tracks.json"
            track_path.write_text(
                json.dumps(
                    {
                        "input_episode_path": str(root / "episode.json"),
                        "estimator": "cotracker-depth-backprojection",
                        "frame_count": 2,
                        "source_frame_count": 8,
                        "sampled_frame_indices": [0, 4],
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base"},
                                {"part_id": 2, "name": "door", "role": "articulated"},
                            ]
                        },
                        "tracks": base_tracks + door_tracks,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(
                    input_path=track_path,
                    min_tracks_per_part=3,
                )
            ).estimate()

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["estimator"], "cotracker-depth-rigid-registration")
            self.assertEqual(artifact["anchor_part_id"], 1)
            self.assertEqual(artifact["source_frame_count"], 8)
            self.assertEqual(artifact["sampled_frame_indices"], [0, 4])
            tracks = {track["part_id"]: track for track in artifact["parts"]}
            door_frame1 = next(sample for sample in tracks[2]["samples"] if sample["frame_index"] == 1)
            self.assertTrue(door_frame1["valid"])
            self.assertEqual(door_frame1["source_frame_index"], 4)
            self.assertIn("centroid_world", door_frame1)
            self.assertNotAlmostEqual(door_frame1["centroid_world"][0], 0.0, places=6)
            self.assertAlmostEqual(door_frame1["translation"][0], translation[0], places=6)
            self.assertAlmostEqual(door_frame1["translation"][1], translation[1], places=6)
            self.assertAlmostEqual(door_frame1["translation"][2], translation[2], places=6)
            self.assertGreater(tracks[2]["relative_motion_summary"]["rotation_angle_range_rad"], 0.4)


if __name__ == "__main__":
    unittest.main()
