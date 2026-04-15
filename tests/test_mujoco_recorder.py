from __future__ import annotations

import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.sim.mujoco_recorder import (
    MuJoCoEpisodeRecorder,
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
            ]
        )
        self.assertEqual(args.command, "record-mujoco")
        self.assertEqual(args.control_mode, "free")
        self.assertAlmostEqual(args.initial_qvel, 1.25)
        self.assertAlmostEqual(args.kick_force, 3.0)
        self.assertAlmostEqual(args.kick_start_s, 0.2)
        self.assertAlmostEqual(args.kick_duration_s, 0.05)


if __name__ == "__main__":
    unittest.main()
