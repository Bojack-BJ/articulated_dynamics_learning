from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.manipulation.arx_x5 import ARXX5ModelPreparer, ARXX5PrepareConfig
from rgbd_urdf_mvp.manipulation.contact_il import ContactILDatasetConfig, ContactILDatasetGenerator, ContactILEvalConfig, ContactILEvaluator
from rgbd_urdf_mvp.manipulation.il_animation import ILRolloutRenderer
from rgbd_urdf_mvp.manipulation.il_train import BallisticILTrainConfig, BallisticILTrainer


OBJECT_XML = """
<mujoco model="contact_il_test">
  <option timestep="0.01" gravity="0 0 0" />
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="door" pos="0 0 0">
        <inertial pos="0.2 0 0" quat="1 0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1" />
        <joint name="door_joint" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.2" frictionloss="0.0" />
        <geom type="box" pos="0.25 0 0" size="0.25 0.03 0.35" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


X5_MINI_URDF = """
<robot name="X5A">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <visual><geometry><mesh filename="package://X5A/meshes/base_link.STL"/></geometry></visual>
    <collision><geometry><mesh filename="package://X5A/meshes/base_link.STL"/></geometry></collision>
  </link>
  <link name="link6">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><mesh filename="package://X5A/meshes/link6.STL"/></geometry></visual>
    <collision><geometry><mesh filename="package://X5A/meshes/link6.STL"/></geometry></collision>
  </link>
  <joint name="joint6" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="link6"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
</robot>
""".strip()


ASCII_STL = """solid box
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 0.01 0 0
vertex 0 0.01 0
endloop
endfacet
endsolid box
"""


class ContactILTests(unittest.TestCase):
    def test_contact_cli_parsers_accept_arguments(self) -> None:
        parser = build_parser()
        x5 = parser.parse_args(["prepare-arx-x5-model", "--output-dir", "out", "--source-dir", "/tmp/ARX_Model"])
        self.assertEqual(x5.command, "prepare-arx-x5-model")
        contact = parser.parse_args(
            [
                "generate-contact-il-demos",
                "model.xml",
                "--output-dir",
                "out",
                "--robot-model",
                "X5A.urdf",
                "--joint-name",
                "door_joint",
            ]
        )
        self.assertEqual(contact.command, "generate-contact-il-demos")
        self.assertEqual(contact.robot_end_effector_body, "link6")
        contact_eval = parser.parse_args(["eval-contact-il-policy", "policy.pt", "model.xml", "--output-dir", "out"])
        self.assertEqual(contact_eval.command, "eval-contact-il-policy")
        render = parser.parse_args(
            [
                "render-il-rollout",
                "contact_il_dataset.npz",
                "model.xml",
                "--output-path",
                "demo.mp4",
                "--robot-model",
                "X5A.urdf",
                "--dynamics-artifact",
                "dynamics_identification.json",
                "--robot-base-pos",
                "-0.5",
                "-0.4",
                "0.0",
                "--object-visual-scale",
                "0.6",
                "--playback-slowdown",
                "2.5",
                "--self-collision-penetration-tolerance-m",
                "0.001",
                "--object-collision-penetration-tolerance-m",
                "0.002",
            ]
        )
        self.assertEqual(render.command, "render-il-rollout")
        self.assertEqual(render.trajectory_source, "dataset")
        self.assertEqual(render.robot_model, Path("X5A.urdf"))
        self.assertAlmostEqual(render.object_visual_scale, 0.6)
        self.assertAlmostEqual(render.playback_slowdown, 2.5)
        self.assertFalse(render.allow_robot_self_collision)
        self.assertAlmostEqual(render.self_collision_penetration_tolerance_m, 0.001)
        self.assertFalse(render.allow_robot_object_collision)
        self.assertAlmostEqual(render.object_collision_penetration_tolerance_m, 0.002)

    def test_rollout_renderer_interpolates_dataset_states(self) -> None:
        response = np.asarray(
            [
                [0.1, 0.2, 1.0],
                [0.3, 0.4, 0.5],
            ],
            dtype=np.float32,
        )
        states = ILRolloutRenderer()._dataset_states(response, np.asarray([0.0, 0.2, 0.3], dtype=float))
        self.assertEqual(len(states), 3)
        self.assertAlmostEqual(states[0][1], 0.2)
        self.assertAlmostEqual(states[1][1], 0.3, places=6)
        self.assertFalse(states[2][3])

    def test_prepare_x5_patches_package_mesh_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "ARX_Model"
            meshes = source / "X5" / "X5A" / "meshes"
            urdf = source / "X5" / "X5A" / "urdf"
            meshes.mkdir(parents=True)
            urdf.mkdir(parents=True)
            (meshes / "base_link.STL").write_text(ASCII_STL, encoding="utf-8")
            (meshes / "link6.STL").write_text(ASCII_STL, encoding="utf-8")
            (urdf / "X5A.urdf").write_text(X5_MINI_URDF, encoding="utf-8")
            manifest = ARXX5ModelPreparer().prepare(
                ARXX5PrepareConfig(output_dir=root / "prepared", source_dir=source, validate_mujoco=False)
            )
            text = (root / "prepared" / "X5A" / "urdf" / "X5A.mujoco.urdf").read_text(encoding="utf-8")
            self.assertIn("../meshes/base_link.STL", text)
            self.assertTrue(manifest.exists())

    def test_contact_dataset_writes_schema(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_available = False
        else:
            torch_available = True
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "object.xml"
            model_path.write_text(OBJECT_XML + "\n", encoding="utf-8")
            dataset = ContactILDatasetGenerator().generate(
                ContactILDatasetConfig(
                    mjcf_path=model_path,
                    output_dir=root / "contact_dataset",
                    joint_name="door_joint",
                    num_episodes=12,
                    release_duration_s=0.4,
                    contact_duration_s=0.05,
                    max_force=10.0,
                    num_teacher_candidates=21,
                    response_samples=10,
                    seed=5,
                )
            )
            payload = np.load(dataset)
            self.assertEqual(payload["obs"].shape, (12, 4))
            self.assertEqual(payload["condition"].shape, (12, 7))
            self.assertEqual(payload["action"].shape, (12, 8))
            self.assertEqual(payload["free_response"].shape, (12, 10, 3))
            self.assertGreater(float(np.max(np.abs(payload["action"][:, 6]))), 0.0)
            if not torch_available:
                return
            policy = BallisticILTrainer().train(
                BallisticILTrainConfig(
                    dataset_path=dataset,
                    output_dir=root / "policy",
                    condition_mode="oracle",
                    epochs=2,
                    batch_size=4,
                    hidden_dim=32,
                )
            )
            eval_path = ContactILEvaluator().evaluate(
                ContactILEvalConfig(
                    policy_path=policy,
                    mjcf_path=model_path,
                    output_dir=root / "eval",
                    joint_name="door_joint",
                    num_episodes=3,
                    release_duration_s=0.4,
                    response_samples=10,
                )
            )
            self.assertTrue(eval_path.exists())


if __name__ == "__main__":
    unittest.main()
