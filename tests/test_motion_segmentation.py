from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.motion_segmentation import MotionPartSegmentationConfig, MotionPartSegmenter
from rgbd_urdf_mvp.perception.part_tracking import TrackPartPoseEstimationConfig, TrackPartPoseEstimator


def _rotate_about_z(point: list[float], pivot: list[float], angle_rad: float) -> list[float]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    x_coord = point[0] - pivot[0]
    y_coord = point[1] - pivot[1]
    return [
        pivot[0] + cosine * x_coord - sine * y_coord,
        pivot[1] + sine * x_coord + cosine * y_coord,
        point[2],
    ]


def _track(track_id: int, reference: list[float], points: list[list[float]]) -> dict:
    return {
        "track_id": track_id,
        "part_id": 1,
        "part_name": "object",
        "view_index": 0,
        "query_frame_index": 0,
        "query_uv": [float(track_id), 0.0],
        "reference_xyz_world": reference,
        "samples": [
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index * 0.1,
                "uv": [float(track_id), float(frame_index)],
                "xyz_world": point,
                "visible": True,
                "depth_valid": True,
                "mask_consistent": True,
                "confidence": 1.0,
            }
            for frame_index, point in enumerate(points)
        ],
    }


class MotionSegmentationTests(unittest.TestCase):
    def test_parser_accepts_segment_motion_parts(self) -> None:
        args = build_parser().parse_args(
            [
                "segment-motion-parts",
                "part_tracks.json",
                "--output-json",
                "motion_part_tracks.json",
                "--rigidity-threshold-m",
                "0.02",
                "--max-neighbor-distance-m",
                "0.5",
                "--min-common-frames",
                "3",
            ]
        )
        self.assertEqual(args.command, "segment-motion-parts")
        self.assertEqual(args.rigidity_threshold_m, 0.02)
        self.assertEqual(args.max_neighbor_distance_m, 0.5)
        self.assertEqual(args.min_common_frames, 3)

    def test_motion_segmentation_relabels_object_tracks_into_rigid_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_points = [
                [-0.2, -0.1, 0.0],
                [-0.2, 0.2, 0.0],
                [0.1, -0.1, 0.0],
                [0.1, 0.2, 0.2],
            ]
            door_reference = [
                [1.0, -0.2, 0.0],
                [1.0, 0.2, 0.0],
                [1.3, -0.2, 0.0],
                [1.3, 0.2, 0.2],
            ]
            pivot = [0.8, 0.0, 0.0]
            angles = [0.0, math.radians(25.0), math.radians(50.0)]
            tracks = []
            for index, point in enumerate(base_points):
                tracks.append(_track(index, point, [point, point, point]))
            for index, point in enumerate(door_reference, start=10):
                tracks.append(_track(index, point, [_rotate_about_z(point, pivot, angle) for angle in angles]))

            input_path = root / "object_tracks.json"
            input_path.write_text(
                json.dumps(
                    {
                        "input_episode_path": str(root / "episode.json"),
                        "estimator": "cotracker-depth-backprojection",
                        "frame_count": 3,
                        "source_frame_count": 3,
                        "sampled_frame_indices": [0, 1, 2],
                        "part_segmentation": {
                            "source": "object-mask",
                            "parts": [{"part_id": 1, "name": "object", "role": "object"}],
                        },
                        "tracks": tracks,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = MotionPartSegmenter(
                MotionPartSegmentationConfig(
                    input_tracks=input_path,
                    rigidity_threshold_m=0.01,
                    max_neighbor_distance_m=0.75,
                    min_common_frames=3,
                    min_tracks_per_part=3,
                    static_motion_threshold_m=0.01,
                )
            ).segment()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["estimator"], "motion-rigidity-cotracker-clustering")
            self.assertEqual(payload["motion_segmentation"]["part_count"], 2)
            part_ids = sorted({int(track["part_id"]) for track in payload["tracks"]})
            self.assertEqual(part_ids, [1, 2])
            counts = {int(part_id): item["count"] for part_id, item in payload["part_track_counts"].items()}
            self.assertEqual(counts, {1: 4, 2: 4})
            self.assertEqual(payload["part_segmentation"]["parts"][0]["role"], "base")

            pose_path = TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(input_path=output_path, min_tracks_per_part=3)
            ).estimate()
            pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
            self.assertEqual(len(pose_payload["parts"]), 2)
            moving = next(part for part in pose_payload["parts"] if part["part_id"] == 2)
            self.assertGreater(moving["relative_motion_summary"]["rotation_angle_range_rad"], 0.5)


if __name__ == "__main__":
    unittest.main()
