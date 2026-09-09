from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.core.models import FrameObservation
from rgbd_urdf_mvp.perception.pointcloud_fusion import (
    EpisodePointCloudFuser,
    PointCloudFusionConfig,
    _camera_to_world_point,
    _resolve_view_camera_poses,
)
from rgbd_urdf_mvp.sim.mujoco_recorder import orbit_camera_pose


def _write_pgm_u16(path: Path, width: int, height: int, values: list[int]) -> None:
    header = f"P5\n{width} {height}\n65535\n".encode("ascii")
    payload = bytearray()
    for value in values:
        payload.extend(int(value).to_bytes(2, byteorder="big", signed=False))
    path.write_bytes(header + payload)


class PointCloudFusionTests(unittest.TestCase):
    def test_opencv_z_depth_uses_image_y_down(self) -> None:
        point = _camera_to_world_point(
            u_coord=1,
            v_coord=2,
            depth_m=1.0,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            camera_pose=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            depth_convention="opencv-z-depth",
        )
        self.assertEqual(point, (1.0, 2.0, 1.0))

    def test_camera_to_world_point_maps_center_ray_to_lookat_for_mujoco_orbit_pose(self) -> None:
        intrinsics = {"fx": 240.0, "fy": 240.0, "cx": 319.5, "cy": 239.5}
        lookat = [0.0, 0.0, 1.0]
        pose = orbit_camera_pose(
            lookat=lookat,
            distance=2.2,
            azimuth_deg=50.0,
            elevation_deg=-28.0,
        )
        point = _camera_to_world_point(
            u_coord=320,
            v_coord=240,
            depth_m=2.2,
            intrinsics=intrinsics,
            camera_pose=pose,
        )
        for actual, expected in zip(point, lookat):
            self.assertAlmostEqual(actual, expected, places=2)

    def test_resolve_view_camera_poses_corrects_legacy_mujoco_recorder_pose(self) -> None:
        frame = FrameObservation(
            timestamp_s=0.0,
            rgb_path="rgb.ppm",
            depth_path="depth.pgm",
            camera_pose=[
                [0.7660444431, 0.3017705037, -0.5675477727, 1.2486050999],
                [-0.6427876097, 0.3596360819, -0.6763770971, 1.4880296136],
                [0.0, 0.8829475929, 0.4694715628, -0.0328374381],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        poses, source = _resolve_view_camera_poses(
            frame=frame,
            episode_metadata={
                "source": "mujoco-recorder",
                "lookat": [0.0, 0.0, 1.0],
            },
            view_count=1,
        )
        self.assertEqual(source, "legacy-single-view-corrected")
        corrected = poses[0]
        self.assertAlmostEqual(corrected[0][3], -1.2486050999)
        self.assertAlmostEqual(corrected[1][3], -1.4880296136)
        self.assertAlmostEqual(corrected[2][3], 2.0328374381)
        self.assertAlmostEqual(corrected[0][2], 0.5675477727)
        self.assertAlmostEqual(corrected[1][2], 0.6763770971)
        self.assertAlmostEqual(corrected[2][2], -0.4694715628)

    def test_fuse_triview_episode_to_4d_pointcloud(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_dir = root / "episode"
            assets_dir = episode_dir / "assets"
            view_dirs = [assets_dir / f"view_{index}" for index in range(3)]
            for view_dir in view_dirs:
                view_dir.mkdir(parents=True, exist_ok=True)

            width = 4
            height = 4
            depth_values = [1000] * (width * height)
            part_mask_values = [
                1, 1, 2, 2,
                1, 1, 2, 2,
                1, 1, 2, 2,
                1, 1, 2, 2,
            ]
            for view_index, view_dir in enumerate(view_dirs):
                _write_pgm_u16(view_dir / "frame_0000_depth.pgm", width, height, depth_values)
                _write_pgm_u16(view_dir / "frame_0000_part_mask.pgm", width, height, part_mask_values)

            episode_payload = {
                "object_instance_id": "synthetic-triview",
                "category": "door",
                "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.5},
                "frames": [
                    {
                        "timestamp_s": 0.0,
                        "rgb_path": "assets/view_1/frame_0000_rgb.ppm",
                        "depth_path": "assets/view_1/frame_0000_depth.pgm",
                        "rgb_paths_by_view": [
                            "assets/view_0/frame_0000_rgb.ppm",
                            "assets/view_1/frame_0000_rgb.ppm",
                            "assets/view_2/frame_0000_rgb.ppm",
                        ],
                        "depth_paths_by_view": [
                            "assets/view_0/frame_0000_depth.pgm",
                            "assets/view_1/frame_0000_depth.pgm",
                            "assets/view_2/frame_0000_depth.pgm",
                        ],
                        "part_mask_paths_by_view": [
                            "assets/view_0/frame_0000_part_mask.pgm",
                            "assets/view_1/frame_0000_part_mask.pgm",
                            "assets/view_2/frame_0000_part_mask.pgm",
                        ],
                        "camera_pose": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "camera_poses_by_view": [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [1.0, 0.0, 0.0, 0.1],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.1],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        ],
                        "action_log": {"target_open_fraction": 0.0},
                        "observation_confidence": 1.0,
                    }
                ],
                "metadata": {
                    "camera_mode": "triview",
                    "part_segmentation": {
                        "provider": "test-prior",
                        "parts": [
                            {"part_id": 1, "name": "base"},
                            {"part_id": 2, "name": "door"},
                        ],
                    },
                },
            }
            episode_path = episode_dir / "episode.json"
            episode_path.write_text(json.dumps(episode_payload, indent=2) + "\n", encoding="utf-8")

            manifest_path = EpisodePointCloudFuser(
                PointCloudFusionConfig(
                    episode_path=episode_path,
                    output_dir=episode_dir / "pointcloud_4d",
                    pixel_stride=1,
                    voxel_size_m=0.0,
                )
            ).fuse()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertGreater(manifest["total_point_count"], 0)
            self.assertEqual(manifest["frame_count"], 1)
            self.assertIn("recorded-per-view", manifest["pose_sources_used"])

            ply_path = Path(manifest["pointcloud_4d_path"])
            self.assertTrue(ply_path.exists())
            header = ply_path.read_text(encoding="utf-8").splitlines()[:10]
            self.assertIn("property float time", header)
            self.assertIn("property ushort part_id", header)
            self.assertEqual(manifest["part_ids_present"], [1, 2])
            self.assertEqual(manifest["part_point_counts"]["1"]["name"], "base")
            self.assertEqual(manifest["part_point_counts"]["2"]["name"], "door")


if __name__ == "__main__":
    unittest.main()
