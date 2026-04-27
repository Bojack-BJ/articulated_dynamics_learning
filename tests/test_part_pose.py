from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.part_pose import PartPoseEstimationConfig, PartPoseEstimator


def _rotation_z(angle_rad: float) -> list[list[float]]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _transform_points(points: list[list[float]], rotation: list[list[float]], translation: list[float]) -> list[list[float]]:
    transformed: list[list[float]] = []
    for point in points:
        x_coord = rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2] + translation[0]
        y_coord = rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2] + translation[1]
        z_coord = rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2] + translation[2]
        transformed.append([x_coord, y_coord, z_coord])
    return transformed


def _box_grid(size: list[float], steps: tuple[int, int, int]) -> list[list[float]]:
    xs = [
        -0.5 * size[0] + size[0] * index / max(1, steps[0] - 1)
        for index in range(steps[0])
    ]
    ys = [
        -0.5 * size[1] + size[1] * index / max(1, steps[1] - 1)
        for index in range(steps[1])
    ]
    zs = [
        -0.5 * size[2] + size[2] * index / max(1, steps[2] - 1)
        for index in range(steps[2])
    ]
    return [[x_coord, y_coord, z_coord] for x_coord in xs for y_coord in ys for z_coord in zs]


class PartPoseTests(unittest.TestCase):
    def test_estimate_part_poses_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "estimate-part-poses",
                "fusion_manifest.json",
                "--output-json",
                "part_poses.json",
                "--min-points-per-part",
                "32",
                "--anchor-part-id",
                "2",
            ]
        )
        self.assertEqual(args.command, "estimate-part-poses")
        self.assertEqual(args.min_points_per_part, 32)
        self.assertEqual(args.anchor_part_id, 2)

    def test_estimate_part_poses_from_part_labeled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_points = _box_grid([1.0, 0.6, 0.8], (5, 4, 4))
            door_points = _box_grid([0.8, 0.2, 1.0], (5, 3, 5))

            frame0_base = _transform_points(base_points, _rotation_z(0.0), [0.0, 0.0, 0.0])
            frame1_base = _transform_points(base_points, _rotation_z(0.0), [0.0, 0.0, 0.0])
            frame0_door = _transform_points(door_points, _rotation_z(0.0), [0.7, 0.0, 0.0])
            frame1_door = _transform_points(door_points, _rotation_z(math.radians(35.0)), [0.4, 0.45, 0.0])

            ply_path = root / "pointcloud_4d.ply"
            lines = [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(frame0_base) + len(frame1_base) + len(frame0_door) + len(frame1_door)}",
                "property float x",
                "property float y",
                "property float z",
                "property float time",
                "property int frame_index",
                "property ushort part_id",
                "end_header",
            ]
            for point in frame0_base:
                lines.append(f"{point[0]} {point[1]} {point[2]} 0.0 0 1")
            for point in frame0_door:
                lines.append(f"{point[0]} {point[1]} {point[2]} 0.0 0 2")
            for point in frame1_base:
                lines.append(f"{point[0]} {point[1]} {point[2]} 0.1 1 1")
            for point in frame1_door:
                lines.append(f"{point[0]} {point[1]} {point[2]} 0.1 1 2")
            ply_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            manifest_path = root / "fusion_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pointcloud_4d_path": str(ply_path),
                        "frame_count": 2,
                        "part_ids_present": [1, 2],
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base"},
                                {"part_id": 2, "name": "door", "role": "articulated"},
                            ]
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = PartPoseEstimator(
                PartPoseEstimationConfig(
                    input_path=manifest_path,
                    min_points_per_part=16,
                )
            ).estimate()

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["anchor_part_id"], 1)
            self.assertEqual(len(artifact["parts"]), 2)

            tracks = {track["part_id"]: track for track in artifact["parts"]}
            door_track = tracks[2]
            self.assertEqual(door_track["name"], "door")
            frame1 = next(sample for sample in door_track["samples"] if sample["frame_index"] == 1)
            self.assertTrue(frame1["valid"])
            self.assertIn("centroid_world", frame1)
            self.assertAlmostEqual(frame1["translation"][0], 0.4, places=1)
            self.assertAlmostEqual(frame1["translation"][1], 0.45, places=1)
            self.assertIsNotNone(frame1["relative_to_anchor"])
            self.assertGreater(door_track["relative_motion_summary"]["rotation_angle_range_rad"], 0.4)


if __name__ == "__main__":
    unittest.main()
