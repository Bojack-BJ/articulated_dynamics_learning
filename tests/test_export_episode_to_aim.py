from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_episode_to_aim.py"
SPEC = importlib.util.spec_from_file_location("export_episode_to_aim", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportEpisodeToAimTest(unittest.TestCase):
    def test_minimum_observation_count_accounts_for_test_split(self) -> None:
        count = MODULE._minimum_observations_for_train_count(test_stride=8, train_count=10)
        self.assertEqual(count - int(np.ceil(count / 8)), 10)
        self.assertEqual(count, 12)

    def test_forward_camera_is_converted_to_blender_minus_z(self) -> None:
        pose = np.diag([1.0, 1.0, -1.0, 1.0])
        converted = np.asarray(MODULE.aim_camera_to_world(pose))
        np.testing.assert_allclose(converted, np.eye(4))
        self.assertAlmostEqual(np.linalg.det(converted[:3, :3]), 1.0)

    def test_export_uses_object_mask_as_alpha_without_part_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            frames = []
            for index in range(3):
                rgb = assets / f"rgb_{index}.png"
                depth = assets / f"depth_{index}.png"
                mask = assets / f"mask_{index}.png"
                part_mask = assets / f"part_mask_{index}.png"
                Image.new("RGB", (4, 3), (10, 20, 30)).save(rgb)
                Image.new("I;16", (4, 3), 1000).save(depth)
                Image.new("L", (4, 3), 255 if index else 0).save(mask)
                Image.new("I", (4, 3), 3).save(part_mask)
                frames.append(
                    {
                        "timestamp_s": float(index),
                        "rgb_paths_by_view": [str(rgb.relative_to(root))] * 2,
                        "depth_paths_by_view": [str(depth.relative_to(root))] * 2,
                        "mask_paths_by_view": [str(mask.relative_to(root))] * 2,
                        "part_mask_paths_by_view": [
                            str(part_mask.relative_to(root))
                        ]
                        * 2,
                        "camera_poses_by_view": [np.diag([1.0, 1.0, -1.0, 1.0]).tolist()] * 2,
                    }
                )
            episode = root / "episode.json"
            episode.write_text(
                json.dumps(
                    {
                        "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.0},
                        "frames": frames,
                        "metadata": {
                            "camera_pose_convention": "mujoco-gl-forward",
                            "part_segmentation": {
                                "parts": [
                                    {"part_id": 1, "role": "base"},
                                    {"part_id": 3, "role": "articulated"},
                                ]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = MODULE.export_episode(
                episode, root / "aim", boundary_frame_count=1, test_stride=2
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["uses_part_or_joint_ground_truth"])
            self.assertFalse(
                manifest["diagnostics"]["part_masks"]["used_by_aim"]
            )
            self.assertEqual(
                manifest["diagnostics"]["part_masks"]["articulated_part_ids"], [3]
            )
            self.assertEqual(Image.open(root / "aim/start/rgb/000000.png").getextrema()[3], (0, 0))
            self.assertTrue((root / "aim/motion/transforms_train.json").is_file())
            self.assertTrue(
                (root / "aim/motion/diagnostics/part_mask/000000.png").is_file()
            )
            self.assertTrue((root / "aim/end/transforms_test.json").is_file())
            motion_frames = json.loads(
                (root / "aim/motion/transforms_train.json").read_text(encoding="utf-8")
            )["frames"] + json.loads(
                (root / "aim/motion/transforms_test.json").read_text(encoding="utf-8")
            )["frames"]
            grouped_times: dict[int, set[float]] = {}
            for frame in motion_frames:
                grouped_times.setdefault(frame["source_frame_index"], set()).add(
                    frame["time"]
                )
            self.assertTrue(all(len(times) == 1 for times in grouped_times.values()))

    def test_aim_style_export_uses_static_scan_for_start_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            frames = []
            for index in range(6):
                rgb = assets / f"rgb_{index}.png"
                depth = assets / f"depth_{index}.png"
                mask = assets / f"mask_{index}.png"
                Image.new("RGB", (4, 3), (10, 20, 30)).save(rgb)
                Image.new("I;16", (4, 3), 1000).save(depth)
                Image.new("L", (4, 3), 255).save(mask)
                frames.append(
                    {
                        "timestamp_s": float(index),
                        "rgb_path": str(rgb.relative_to(root)),
                        "depth_path": str(depth.relative_to(root)),
                        "mask_path": str(mask.relative_to(root)),
                        "camera_pose": np.diag([1.0, 1.0, -1.0, 1.0]).tolist(),
                    }
                )

            static_dir = root / "aim_protocol" / "static_scan"
            static_dir.mkdir(parents=True)
            static_views = []
            for index in range(4):
                view_dir = static_dir / f"view_{index:03d}"
                view_dir.mkdir()
                rgb = view_dir / "rgb.png"
                depth = view_dir / "depth.png"
                mask = view_dir / "mask.png"
                Image.new("RGB", (4, 3), (index * 20, 0, 0)).save(rgb)
                Image.new("I;16", (4, 3), 1000).save(depth)
                Image.new("L", (4, 3), 255).save(mask)
                static_views.append(
                    {
                        "view_index": index,
                        "rgb_path": str(rgb.relative_to(root)),
                        "depth_path": str(depth.relative_to(root)),
                        "mask_path": str(mask.relative_to(root)),
                        "camera_pose": np.diag([1.0, 1.0, -1.0, 1.0]).tolist(),
                        "joint_positions": {"door": 0.0},
                    }
                )
            (static_dir / "cameras.json").write_text(
                json.dumps({"joint_positions": {"door": 0.0}, "views": static_views}),
                encoding="utf-8",
            )
            episode = root / "episode.json"
            episode.write_text(
                json.dumps(
                    {
                        "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.0},
                        "frames": frames,
                        "metadata": {
                            "camera_pose_convention": "mujoco-gl-forward",
                            "recording_protocol": "aim_style",
                            "aim_protocol": {
                                "static_scan_manifest": "aim_protocol/static_scan/cameras.json"
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = MODULE.export_episode(
                episode,
                root / "aim",
                boundary_frame_count=2,
                test_stride=2,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stages"]["start"]["source"], "aim_static_scan")
            self.assertEqual(manifest["stages"]["start"]["observation_count"], 4)
            self.assertEqual(manifest["stages"]["motion"]["observation_count"], 6)

    def test_fixed_end_export_uses_static_final_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            frames = []
            for index in range(6):
                rgb = assets / f"rgb_{index}.png"
                depth = assets / f"depth_{index}.png"
                mask = assets / f"mask_{index}.png"
                Image.new("RGB", (4, 3), (10, 20, 30)).save(rgb)
                Image.new("I;16", (4, 3), 1000).save(depth)
                Image.new("L", (4, 3), 255).save(mask)
                frames.append({
                    "timestamp_s": float(index),
                    "rgb_path": str(rgb.relative_to(root)),
                    "depth_path": str(depth.relative_to(root)),
                    "mask_path": str(mask.relative_to(root)),
                    "camera_pose": np.diag([1.0, 1.0, -1.0, 1.0]).tolist(),
                })
            protocol = {}
            for stage, joint_position, view_count in (
                ("static_scan", 0.0, 4),
                ("end_scan", 1.0, 12),
            ):
                stage_dir = root / "aim_protocol" / stage
                stage_dir.mkdir(parents=True)
                views = []
                for index in range(view_count):
                    view_dir = stage_dir / f"view_{index:03d}"
                    view_dir.mkdir()
                    for name, mode in (("rgb.png", "RGB"), ("depth.png", "I;16"), ("mask.png", "L")):
                        Image.new(mode, (4, 3), 255).save(view_dir / name)
                    views.append({
                        "view_index": index,
                        "rgb_path": str((view_dir / "rgb.png").relative_to(root)),
                        "depth_path": str((view_dir / "depth.png").relative_to(root)),
                        "mask_path": str((view_dir / "mask.png").relative_to(root)),
                        "camera_pose": np.diag([1.0, 1.0, -1.0, 1.0]).tolist(),
                        "joint_positions": {"door": joint_position},
                    })
                manifest = stage_dir / "cameras.json"
                manifest.write_text(
                    json.dumps({"joint_positions": {"door": joint_position}, "views": views}),
                    encoding="utf-8",
                )
                protocol[f"{'static' if stage == 'static_scan' else 'end'}_scan_manifest"] = str(
                    manifest.relative_to(root)
                )
            episode = root / "episode.json"
            episode.write_text(json.dumps({
                "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.0},
                "frames": frames,
                "metadata": {
                    "camera_pose_convention": "mujoco-gl-forward",
                    "recording_protocol": "aim_style_fixed_end",
                    "aim_protocol": protocol,
                },
            }), encoding="utf-8")
            manifest_path = MODULE.export_episode(
                episode, root / "aim", boundary_frame_count=2, test_stride=2
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stages"]["end"]["source"], "aim_end_scan")
            self.assertEqual(manifest["stages"]["end"]["observation_count"], 12)


if __name__ == "__main__":
    unittest.main()
