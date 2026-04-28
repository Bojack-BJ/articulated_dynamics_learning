from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.core.serialization import load_json
from rgbd_urdf_mvp.sim.mujoco_recorder import (
    MuJoCoEpisodeCompactConfig,
    MuJoCoEpisodeCompactor,
    MuJoCoEpisodeRecorder,
    MuJoCoEpisodeRepackConfig,
    MuJoCoEpisodeRepacker,
    MuJoCoRecordConfig,
    _fovy_deg_from_intrinsics,
    orbit_camera_pose,
)


class MuJoCoRecorderTests(unittest.TestCase):
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
