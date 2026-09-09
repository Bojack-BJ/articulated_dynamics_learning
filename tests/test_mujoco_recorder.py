from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.core.serialization import load_json
from rgbd_urdf_mvp.sim.mujoco_recorder import (
    MuJoCoEpisodeCompactConfig,
    MuJoCoEpisodeCompactor,
    MuJoCoEpisodeRecorder,
    MuJoCoEpisodeRepackConfig,
    MuJoCoEpisodeRepacker,
    MuJoCoRecordConfig,
    _fit_camera_to_bounding_spheres,
    _fovy_deg_from_intrinsics,
    aim_interaction_camera_angles,
    aim_static_scan_camera_angles,
    orbit_camera_pose,
)


class MuJoCoRecorderTests(unittest.TestCase):
    def test_aim_static_scan_covers_full_orbit_without_duplicate_endpoint(self) -> None:
        angles = aim_static_scan_camera_angles(
            24,
            azimuth_start_deg=20.0,
            elevation_deg=15.0,
        )
        self.assertEqual(len(angles), 24)
        self.assertAlmostEqual(angles[0][0], 20.0)
        self.assertAlmostEqual(angles[-1][0], 365.0)
        self.assertEqual({elevation for _, elevation in angles}, {15.0})
        self.assertEqual(len({round(azimuth % 360.0, 6) for azimuth, _ in angles}), 24)

    def test_aim_interaction_camera_path_is_not_linear_in_joint_open_fraction(self) -> None:
        samples = []
        joint_fractions = []
        for index in range(21):
            alpha = index / 20.0
            azimuth, elevation = aim_interaction_camera_angles(
                alpha,
                azimuth_start_deg=10.0,
                elevation_deg=15.0,
                orbit_count=1.0,
                elevation_amplitude_deg=10.0,
            )
            samples.append((azimuth, elevation))
            joint_fractions.append(0.5 - 0.5 * math.cos(math.pi * alpha))
        camera_steps = np.diff([azimuth for azimuth, _ in samples])
        self.assertGreater(float(np.std(camera_steps)), 0.1)
        linear_fit = np.polyfit(joint_fractions, [azimuth for azimuth, _ in samples], deg=1)
        residual = np.asarray([azimuth for azimuth, _ in samples]) - np.polyval(linear_fit, joint_fractions)
        self.assertGreater(float(np.max(np.abs(residual))), 5.0)
        self.assertGreater(float(np.ptp([elevation for _, elevation in samples])), 10.0)

    def test_front_loaded_orbit_uses_front_arc_then_completes_back_scan(self) -> None:
        split = 0.94
        samples = [
            aim_interaction_camera_angles(
                alpha,
                azimuth_start_deg=300.0,
                elevation_deg=-15.0,
                orbit_count=1.0,
                elevation_amplitude_deg=0.0,
                trajectory="front_loaded_orbit",
                motion_end_fraction=split,
            )[0]
            for alpha in (0.0, split, 1.0)
        ]
        self.assertEqual(samples, [300.0, 420.0, 660.0])

    def test_front_loaded_orbit_is_monotonic_and_speed_continuous(self) -> None:
        split = 0.8
        alpha = np.linspace(0.0, 1.0, 1001)
        azimuth = np.asarray(
            [
                aim_interaction_camera_angles(
                    value,
                    azimuth_start_deg=-90.0,
                    elevation_deg=-15.0,
                    orbit_count=1.0,
                    elevation_amplitude_deg=0.0,
                    trajectory="front_loaded_orbit",
                    motion_end_fraction=split,
                )[0]
                for value in alpha
            ]
        )
        self.assertTrue(np.all(np.diff(azimuth) >= 0.0))
        split_index = int(split * 1000)
        left_speed = azimuth[split_index] - azimuth[split_index - 1]
        right_speed = azimuth[split_index + 1] - azimuth[split_index]
        self.assertLess(abs(left_speed), 0.02)
        self.assertLess(abs(right_speed), 0.2)

    def test_front_oscillate_remains_within_front_arc(self) -> None:
        samples = [
            aim_interaction_camera_angles(
                alpha,
                azimuth_start_deg=50.0,
                elevation_deg=-15.0,
                orbit_count=35.0 / 360.0,
                elevation_amplitude_deg=0.0,
                trajectory="front_oscillate",
            )[0]
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        np.testing.assert_allclose(samples, [50.0, 85.0, 50.0, 15.0, 50.0])

    def test_record_parser_accepts_aim_style_protocol(self) -> None:
        args = build_parser().parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "aim-style-test",
                "--recording-protocol",
                "aim_style",
                "--aim-static-scan-views",
                "32",
                "--aim-static-scan-elevation-deg",
                "12",
                "--aim-interaction-camera-orbits",
                "1.5",
                "--aim-interaction-elevation-amplitude-deg",
                "8",
                "--interaction-camera-trajectory",
                "front_loaded_orbit",
                "--aim-interaction-motion-end-fraction",
                "0.9",
                "--aim-interaction-fixed-view-azimuths-deg",
                "-45",
                "45",
                "135",
                "225",
                "--aim-interaction-fixed-view-elevations-deg",
                "-10",
                "-10",
                "-10",
                "25",
            ]
        )
        self.assertEqual(args.recording_protocol, "aim_style")
        self.assertEqual(args.aim_static_scan_views, 32)
        self.assertAlmostEqual(args.aim_static_scan_elevation_deg, 12.0)
        self.assertAlmostEqual(args.aim_interaction_camera_orbits, 1.5)
        self.assertAlmostEqual(args.aim_interaction_elevation_amplitude_deg, 8.0)
        self.assertEqual(args.interaction_camera_trajectory, "front_loaded_orbit")
        self.assertAlmostEqual(args.aim_interaction_motion_end_fraction, 0.9)
        self.assertEqual(
            args.aim_interaction_fixed_view_azimuths_deg,
            [-45.0, 45.0, 135.0, 225.0],
        )
        self.assertEqual(
            args.aim_interaction_fixed_view_elevations_deg,
            [-10.0, -10.0, -10.0, 25.0],
        )

    def test_record_parser_accepts_single_fixed_aim_interaction_view(self) -> None:
        args = build_parser().parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "aim-fixed-front-test",
                "--recording-protocol",
                "aim_style_fixed_end",
                "--aim-interaction-fixed-view-azimuths-deg",
                "50",
                "--aim-interaction-fixed-view-elevations-deg",
                "15",
            ]
        )
        self.assertEqual(args.aim_interaction_fixed_view_azimuths_deg, [50.0])
        self.assertEqual(args.aim_interaction_fixed_view_elevations_deg, [15.0])

    def test_camera_fit_scales_with_object_bounds(self) -> None:
        lookat, distance, radius = _fit_camera_to_bounding_spheres(
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [0.25, 0.25],
            fovy_deg=60.0,
            fill_ratio=0.6,
        )
        self.assertEqual(lookat, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(radius, 0.75)
        self.assertGreater(distance, radius)

    def test_record_parser_accepts_auto_camera_fit(self) -> None:
        args = build_parser().parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "camera-fit-test",
                "--auto-camera-fit",
                "--camera-fit-fill-ratio",
                "0.55",
            ]
        )
        self.assertTrue(args.auto_camera_fit)
        self.assertAlmostEqual(args.camera_fit_fill_ratio, 0.55)

    def test_fovy_inferred_from_intrinsics(self) -> None:
        fovy_deg = _fovy_deg_from_intrinsics(
            height=480,
            camera_intrinsics={"fy": 240.0},
        )
        self.assertIsNotNone(fovy_deg)
        self.assertAlmostEqual(float(fovy_deg), 90.0, places=4)

    def test_orbit_camera_pose_has_valid_transform_shape(self) -> None:
        pose = orbit_camera_pose(
            lookat=[0.0, 0.0, 0.8],
            distance=2.0,
            azimuth_deg=30.0,
            elevation_deg=15.0,
        )
        self.assertEqual(len(pose), 4)
        self.assertTrue(all(len(row) == 4 for row in pose))
        self.assertEqual(pose[3], [0.0, 0.0, 0.0, 1.0])

    def test_record_mujoco_parser_accepts_required_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--rgb-format",
                "png",
                "--depth-format",
                "png",
            ]
        )
        self.assertEqual(args.command, "record-mujoco")
        self.assertEqual(args.model.as_posix(), "model.xml")
        self.assertEqual(args.category, "door")
        self.assertEqual(args.object_id, "door-unit-test-001")
        self.assertEqual(args.rgb_format, "png")
        self.assertEqual(args.depth_format, "png")

    def test_record_mujoco_parser_accepts_staggered_control(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "multi-joint-test",
                "--control-mode",
                "staggered",
            ]
        )
        self.assertEqual(args.control_mode, "staggered")

    def test_paper_simultaneous_target_uses_shared_smoothstep(self) -> None:
        target = MuJoCoEpisodeRecorder._paper_simultaneous_target
        self.assertAlmostEqual(target(time_s=0.0, duration_s=4.0, start_q=0.0, end_q=1.0), 0.0)
        self.assertAlmostEqual(target(time_s=2.0, duration_s=4.0, start_q=0.0, end_q=1.0), 0.5)
        self.assertAlmostEqual(target(time_s=4.0, duration_s=4.0, start_q=0.0, end_q=1.0), 1.0)
        self.assertAlmostEqual(target(time_s=2.0, duration_s=4.0, start_q=0.0, end_q=0.7), 0.35)

    def test_paper_sequential_target_uses_same_range_in_nonoverlapping_slots(self) -> None:
        target = MuJoCoEpisodeRecorder._paper_sequential_target
        common = {"duration_s": 4.0, "joint_count": 2, "start_q": 0.0, "end_q": 1.0}
        self.assertAlmostEqual(target(time_s=1.0, joint_index=0, **common), 0.5)
        self.assertAlmostEqual(target(time_s=1.0, joint_index=1, **common), 0.0)
        self.assertAlmostEqual(target(time_s=3.0, joint_index=0, **common), 1.0)
        self.assertAlmostEqual(target(time_s=3.0, joint_index=1, **common), 0.5)

    def test_record_mujoco_parser_accepts_paper_simultaneous_control(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "refrigerator",
                "--object-id",
                "paper-test",
                "--control-mode",
                "paper_simultaneous",
                "--paper-simultaneous-joint",
                "hinge",
                "0",
                "1.570796",
            ]
        )
        self.assertEqual(args.control_mode, "paper_simultaneous")
        self.assertEqual(args.paper_simultaneous_joint, [["hinge", "0", "1.570796"]])

    def test_staggered_targets_use_isolated_slots_and_alternating_directions(self) -> None:
        target = MuJoCoEpisodeRecorder._staggered_target
        self.assertAlmostEqual(
            target(time_s=0.0, duration_s=6.0, joint_index=0, joint_count=2, lower=0.0, upper=1.0),
            0.0,
        )
        self.assertAlmostEqual(
            target(time_s=0.0, duration_s=6.0, joint_index=1, joint_count=2, lower=0.0, upper=1.0),
            1.0,
        )
        self.assertAlmostEqual(
            target(time_s=2.5, duration_s=6.0, joint_index=0, joint_count=2, lower=0.0, upper=1.0),
            1.0,
        )
        self.assertAlmostEqual(
            target(time_s=2.5, duration_s=6.0, joint_index=1, joint_count=2, lower=0.0, upper=1.0),
            1.0,
        )
        self.assertLess(
            target(time_s=4.0, duration_s=6.0, joint_index=1, joint_count=2, lower=0.0, upper=1.0),
            1.0,
        )

    def test_record_mujoco_parser_accepts_video_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--video",
                "--video-fps",
                "15",
            ]
        )
        self.assertEqual(args.command, "record-mujoco")
        self.assertTrue(args.video)
        self.assertEqual(args.video_fps, 15.0)

    def test_record_mujoco_parser_accepts_segmentation_mask_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--segmentation-masks",
                "--mask-format",
                "png",
            ]
        )
        self.assertTrue(args.segmentation_masks)
        self.assertEqual(args.mask_format, "png")

    def test_record_mujoco_parser_accepts_write_concat_assets(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--write-concat-assets",
            ]
        )
        self.assertTrue(args.write_concat_assets)

    def test_record_mujoco_parser_accepts_part_segmentation_mask_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--part-segmentation-masks",
            ]
        )
        self.assertTrue(args.part_segmentation_masks)

    def test_record_mujoco_parser_accepts_disable_target_mesh_collision(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "microwave",
                "--object-id",
                "microwave-unit-test-001",
                "--disable-target-mesh-collision",
            ]
        )
        self.assertTrue(args.disable_target_mesh_collision)

    def test_record_mujoco_parser_accepts_disable_target_collision(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "microwave",
                "--object-id",
                "microwave-unit-test-001",
                "--disable-target-collision",
            ]
        )
        self.assertTrue(args.disable_target_collision)

    def test_record_mujoco_parser_accepts_hide_clear_meshes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "microwave",
                "--object-id",
                "microwave-unit-test-001",
                "--hide-clear-meshes",
            ]
        )
        self.assertTrue(args.hide_clear_meshes)

    def test_render_mujoco_masks_parser_accepts_required_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render-mujoco-masks",
                "episode.json",
                "--mask-format",
                "png",
                "--output-episode",
                "episode_with_masks.json",
            ]
        )
        self.assertEqual(args.command, "render-mujoco-masks")
        self.assertEqual(args.episode.as_posix(), "episode.json")
        self.assertEqual(args.mask_format, "png")
        self.assertEqual(args.output_episode.as_posix(), "episode_with_masks.json")

    def test_render_mujoco_masks_parser_accepts_part_segmentation_masks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "render-mujoco-masks",
                "episode.json",
                "--part-segmentation-masks",
            ]
        )
        self.assertTrue(args.part_segmentation_masks)

    def test_compact_recording_parser_accepts_required_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compact-mujoco-recording",
                "episode.json",
                "--remove-concat-video",
            ]
        )
        self.assertEqual(args.command, "compact-mujoco-recording")
        self.assertEqual(args.episode.as_posix(), "episode.json")
        self.assertTrue(args.remove_concat_video)

    def test_repack_recording_parser_accepts_required_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "repack-mujoco-recording",
                "episode.json",
                "--keep-originals",
            ]
        )
        self.assertEqual(args.command, "repack-mujoco-recording")
        self.assertEqual(args.episode.as_posix(), "episode.json")
        self.assertTrue(args.keep_originals)

    def test_episode_compactor_rewrites_triview_paths_and_removes_concat_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_dir = root / "episode"
            concat_dir = episode_dir / "assets" / "concat"
            view_dirs = [episode_dir / "assets" / f"view_{index}" for index in range(3)]
            concat_dir.mkdir(parents=True)
            for view_dir in view_dirs:
                view_dir.mkdir(parents=True)
            (concat_dir / "frame_0000_rgb.png").write_bytes(b"concat-rgb")
            (concat_dir / "frame_0000_depth.png").write_bytes(b"concat-depth")
            (concat_dir / "frame_0000_mask.png").write_bytes(b"concat-mask")
            (concat_dir / "frame_0000_part_mask.png").write_bytes(b"concat-part-mask")
            (episode_dir / "episode_concat.mp4").write_bytes(b"video")
            for view_dir in view_dirs:
                for suffix in ("rgb.png", "depth.png", "mask.png", "part_mask.png"):
                    (view_dir / f"frame_0000_{suffix}").write_bytes(b"view")

            episode_path = episode_dir / "episode.json"
            episode_path.write_text(
                """{
  "object_instance_id": "unit",
  "category": "microwave",
  "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
  "frames": [
    {
      "timestamp_s": 0.0,
      "rgb_path": "assets/concat/frame_0000_rgb.png",
      "depth_path": "assets/concat/frame_0000_depth.png",
      "mask_path": "assets/concat/frame_0000_mask.png",
      "part_mask_path": "assets/concat/frame_0000_part_mask.png",
      "rgb_paths_by_view": [
        "assets/view_0/frame_0000_rgb.png",
        "assets/view_1/frame_0000_rgb.png",
        "assets/view_2/frame_0000_rgb.png"
      ],
      "depth_paths_by_view": [
        "assets/view_0/frame_0000_depth.png",
        "assets/view_1/frame_0000_depth.png",
        "assets/view_2/frame_0000_depth.png"
      ],
      "mask_paths_by_view": [
        "assets/view_0/frame_0000_mask.png",
        "assets/view_1/frame_0000_mask.png",
        "assets/view_2/frame_0000_mask.png"
      ],
      "part_mask_paths_by_view": [
        "assets/view_0/frame_0000_part_mask.png",
        "assets/view_1/frame_0000_part_mask.png",
        "assets/view_2/frame_0000_part_mask.png"
      ],
      "camera_pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "camera_poses_by_view": [],
      "action_log": {}
    }
  ],
  "metadata": {
    "camera_mode": "triview",
    "triview_asset_layout": "concat+views"
  }
}
""",
                encoding="utf-8",
            )

            result = MuJoCoEpisodeCompactor(
                MuJoCoEpisodeCompactConfig(
                    episode_path=episode_path,
                    remove_concat_dir=True,
                    remove_concat_video=False,
                    dry_run=False,
                )
            ).compact()

            payload = load_json(episode_path)
            frame = payload["frames"][0]
            self.assertEqual(frame["rgb_path"], "assets/view_1/frame_0000_rgb.png")
            self.assertEqual(frame["depth_path"], "assets/view_1/frame_0000_depth.png")
            self.assertEqual(frame["mask_path"], "assets/view_1/frame_0000_mask.png")
            self.assertEqual(frame["part_mask_path"], "assets/view_1/frame_0000_part_mask.png")
            self.assertEqual(payload["metadata"]["triview_asset_layout"], "views-only")
            self.assertFalse(concat_dir.exists())
            self.assertGreaterEqual(int(result["estimated_bytes_reclaimed"]), 1)

    def test_episode_repacker_converts_ppm_and_pgm_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_dir = root / "episode"
            view_dir = episode_dir / "assets" / "view_0"
            view_dir.mkdir(parents=True)
            (view_dir / "frame_0000_rgb.ppm").write_bytes(
                b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
            )
            (view_dir / "frame_0000_depth.pgm").write_bytes(
                b"P5\n2 1\n65535\n" + bytes([0x00, 0x01, 0x00, 0x02])
            )
            (view_dir / "frame_0000_mask.pgm").write_bytes(
                b"P5\n2 1\n65535\n" + bytes([0x00, 0x00, 0xFF, 0xFF])
            )
            (view_dir / "frame_0000_part_mask.pgm").write_bytes(
                b"P5\n2 1\n65535\n" + bytes([0x00, 0x03, 0x00, 0x04])
            )

            episode_path = episode_dir / "episode.json"
            episode_path.write_text(
                """{
  "object_instance_id": "unit",
  "category": "microwave",
  "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
  "frames": [
    {
      "timestamp_s": 0.0,
      "rgb_path": "assets/view_0/frame_0000_rgb.ppm",
      "depth_path": "assets/view_0/frame_0000_depth.pgm",
      "mask_path": "assets/view_0/frame_0000_mask.pgm",
      "part_mask_path": "assets/view_0/frame_0000_part_mask.pgm",
      "rgb_paths_by_view": ["assets/view_0/frame_0000_rgb.ppm"],
      "depth_paths_by_view": ["assets/view_0/frame_0000_depth.pgm"],
      "mask_paths_by_view": ["assets/view_0/frame_0000_mask.pgm"],
      "part_mask_paths_by_view": ["assets/view_0/frame_0000_part_mask.pgm"],
      "camera_pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "camera_poses_by_view": [],
      "action_log": {}
    }
  ],
  "metadata": {
    "camera_mode": "triview",
    "rgb_format": "ppm",
    "depth_format": "pgm",
    "mask_format": "pgm",
    "segmentation_masks": true,
    "part_segmentation_masks": true
  }
}
""",
                encoding="utf-8",
            )

            result = MuJoCoEpisodeRepacker(
                MuJoCoEpisodeRepackConfig(
                    episode_path=episode_path,
                    keep_originals=False,
                    dry_run=False,
                )
            ).repack()

            payload = load_json(episode_path)
            frame = payload["frames"][0]
            self.assertEqual(frame["rgb_path"], "assets/view_0/frame_0000_rgb.png")
            self.assertEqual(frame["depth_path"], "assets/view_0/frame_0000_depth.png")
            self.assertEqual(frame["mask_path"], "assets/view_0/frame_0000_mask.png")
            self.assertEqual(frame["part_mask_path"], "assets/view_0/frame_0000_part_mask.png")
            self.assertEqual(payload["metadata"]["rgb_format"], "png")
            self.assertEqual(payload["metadata"]["depth_format"], "png")
            self.assertEqual(payload["metadata"]["mask_format"], "png")
            self.assertTrue((view_dir / "frame_0000_rgb.png").exists())
            self.assertTrue((view_dir / "frame_0000_depth.png").exists())
            self.assertFalse((view_dir / "frame_0000_rgb.ppm").exists())
            self.assertFalse((view_dir / "frame_0000_depth.pgm").exists())
            self.assertGreaterEqual(int(result["converted_paths"]), 4)

    def test_microwave_defaults_to_door_joint_over_turntable(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError:
            self.skipTest("mujoco not installed")

        model_path = Path(__file__).resolve().parents[1] / "examples" / "mujoco_models" / "Microwave041.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path=model_path,
                output_dir=Path("outputs"),
                object_instance_id="microwave041-test",
                category="microwave",
            )
        )
        joint_id = recorder._resolve_joint_id(mujoco, model)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        self.assertEqual(joint_name, "microjoint")

    def test_disable_target_mesh_collision_zeros_target_mesh_contacts(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError:
            self.skipTest("mujoco not installed")

        model_path = Path(__file__).resolve().parents[1] / "examples" / "mujoco_models" / "Microwave041.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path=model_path,
                output_dir=Path("outputs"),
                object_instance_id="microwave041-test",
                category="microwave",
                disable_target_mesh_collision=True,
            )
        )
        joint_id = recorder._resolve_joint_id(mujoco, model)
        target_root_body_id = recorder._infer_object_root_body_id(
            model,
            [(joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))],
        )
        target_geom_ids = recorder._collect_target_geom_ids(model, target_root_body_id)
        recorder._disable_target_mesh_collision(model, target_geom_ids)
        for geom_id in target_geom_ids:
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            self.assertEqual(int(model.geom_contype[geom_id]), 0)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)

    def test_disable_target_collision_zeros_all_target_contacts(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError:
            self.skipTest("mujoco not installed")

        model_path = Path(__file__).resolve().parents[1] / "examples" / "mujoco_models" / "Microwave041.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(model_path, Path("outputs"), "microwave-test", "microwave")
        )
        target_geom_ids = set(range(model.ngeom))
        recorder._disable_target_collision(model, target_geom_ids)
        self.assertTrue(all(int(model.geom_contype[geom_id]) == 0 for geom_id in target_geom_ids))
        self.assertTrue(all(int(model.geom_conaffinity[geom_id]) == 0 for geom_id in target_geom_ids))

    def test_collect_clear_mesh_geom_ids_finds_microwave_clear_door(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError:
            self.skipTest("mujoco not installed")

        model_path = Path(__file__).resolve().parents[1] / "examples" / "mujoco_models" / "Microwave011.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path=model_path,
                output_dir=Path("outputs"),
                object_instance_id="microwave011-test",
                category="microwave",
                hide_clear_meshes=True,
            )
        )
        joint_id = recorder._resolve_joint_id(mujoco, model)
        target_root_body_id = recorder._infer_object_root_body_id(
            model,
            [(joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))],
        )
        target_geom_ids = recorder._collect_target_geom_ids(model, target_root_body_id)
        clear_geom_ids = recorder._collect_clear_mesh_geom_ids(mujoco, model, target_geom_ids)
        self.assertTrue(clear_geom_ids)

    def test_record_mujoco_parser_accepts_free_dynamics_controls(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "record-mujoco",
                "model.xml",
                "--category",
                "door",
                "--object-id",
                "door-unit-test-001",
                "--control-mode",
                "free",
                "--initial-qvel",
                "1.25",
                "--kick-force",
                "3.0",
                "--kick-start-s",
                "0.2",
                "--kick-duration-s",
                "0.05",
                "--excitation-mode",
                "pulse",
                "--excitation-force",
                "-2.0",
                "--excitation-start-s",
                "0.1",
                "--excitation-duration-s",
                "0.25",
            ]
        )
        self.assertEqual(args.command, "record-mujoco")
        self.assertEqual(args.control_mode, "free")
        self.assertAlmostEqual(args.initial_qvel, 1.25)
        self.assertAlmostEqual(args.kick_force, 3.0)
        self.assertAlmostEqual(args.kick_start_s, 0.2)
        self.assertAlmostEqual(args.kick_duration_s, 0.05)
        self.assertEqual(args.excitation_mode, "pulse")
        self.assertAlmostEqual(args.excitation_force, -2.0)
        self.assertAlmostEqual(args.excitation_start_s, 0.1)
        self.assertAlmostEqual(args.excitation_duration_s, 0.25)

    def test_excitation_force_profiles_are_deterministic(self) -> None:
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path="model.xml",
                output_dir=Path("outputs"),
                object_instance_id="door-forced-response",
                category="door",
                excitation_mode="pulse",
                excitation_force=2.0,
                excitation_start_s=0.1,
                excitation_duration_s=0.2,
            )
        )
        self.assertEqual(recorder._excitation_force(0.05, 0), 0.0)
        self.assertEqual(recorder._excitation_force(0.1, 0), 2.0)
        self.assertEqual(recorder._excitation_force(0.29, 0), 2.0)
        self.assertEqual(recorder._excitation_force(0.31, 0), 0.0)

        prbs_a = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path="model.xml",
                output_dir=Path("outputs"),
                object_instance_id="door-prbs-a",
                category="door",
                seed=7,
                excitation_mode="prbs",
                excitation_force=3.0,
                excitation_start_s=0.0,
                excitation_duration_s=1.0,
                excitation_period_s=0.1,
            )
        )
        prbs_b = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path="model.xml",
                output_dir=Path("outputs"),
                object_instance_id="door-prbs-b",
                category="door",
                seed=7,
                excitation_mode="prbs",
                excitation_force=3.0,
                excitation_start_s=0.0,
                excitation_duration_s=1.0,
                excitation_period_s=0.1,
            )
        )
        values_a = [prbs_a._excitation_force(index * 0.05, 0) for index in range(8)]
        values_b = [prbs_b._excitation_force(index * 0.05, 0) for index in range(8)]
        self.assertEqual(values_a, values_b)
        self.assertTrue(all(abs(value) == 3.0 for value in values_a))

    def test_forced_response_recording_respects_zero_initial_qvel(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError:
            self.skipTest("mujoco not installed")

        model_path = Path(__file__).resolve().parents[1] / "examples" / "mujoco_models" / "Microwave011.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path=model_path,
                output_dir=Path("outputs"),
                object_instance_id="microwave011-force-test",
                category="microwave",
                control_mode="free",
                joint_name="microjoint",
                initial_joint_qvel=0.0,
                excitation_mode="pulse",
                excitation_force=-2.0,
                excitation_start_s=0.1,
                excitation_duration_s=0.25,
            )
        )
        joint_id = recorder._resolve_joint_id(mujoco, model)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        recorder._initialize_free_joint_state(
            mujoco,
            model,
            data,
            joint_id,
            qpos_adr,
            dof_adr,
            primary_joint_id=joint_id,
            joint_index=0,
            rng=random.Random(0),
        )
        self.assertAlmostEqual(float(data.qpos[qpos_adr]), float(model.qpos0[qpos_adr]))
        self.assertAlmostEqual(float(data.qvel[dof_adr]), 0.0)


if __name__ == "__main__":
    unittest.main()
