from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.manipulation.mujoco_push_planner import MuJoCoPushPlanConfig, MuJoCoPushPlanner


MODEL_XML = """
<mujoco model="push_plan_test">
  <option timestep="0.01" gravity="0 0 0" />
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="door" pos="0 0 0">
        <inertial pos="0 0 0" quat="1 0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1" />
        <joint name="door_joint" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.45" frictionloss="0.0" />
        <geom type="box" pos="0.25 0 0" size="0.25 0.03 0.35" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


class MuJoCoPushPlannerTests(unittest.TestCase):
    def test_plan_manipulation_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "plan-manipulation",
                "model.mjcf.xml",
                "--joint-name",
                "door_joint",
                "--target-q",
                "0.5",
                "--mode",
                "pulse_force",
                "--max-pulse-force",
                "3.0",
                "--gravity-mode",
                "zero",
            ]
        )
        self.assertEqual(args.command, "plan-manipulation")
        self.assertEqual(args.mjcf, Path("model.mjcf.xml"))
        self.assertEqual(args.joint_name, "door_joint")
        self.assertAlmostEqual(args.target_q, 0.5)
        self.assertEqual(args.mode, "pulse_force")
        self.assertAlmostEqual(args.max_pulse_force, 3.0)
        self.assertEqual(args.gravity_mode, "zero")

    def test_initial_velocity_planner_writes_policy_conditioned_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.mjcf.xml"
            model_path.write_text(MODEL_XML + "\n", encoding="utf-8")

            artifact_path = MuJoCoPushPlanner().run(
                MuJoCoPushPlanConfig(
                    mjcf_path=model_path,
                    output_dir=root / "manipulation",
                    joint_name="door_joint",
                    target_q=0.5,
                    duration_s=0.5,
                    max_initial_qvel=3.0,
                    num_candidates=61,
                    tolerance=0.08,
                    gravity_mode="zero",
                )
            )

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact["task"]["joint_name"], "door_joint")
            self.assertEqual(artifact["command"]["type"], "initial_joint_velocity")
            self.assertIn("policy_conditioning", artifact)
            self.assertIn("joint_damping", artifact["policy_conditioning"]["features"])
            self.assertLess(artifact["result"]["abs_error"], 0.15)
            self.assertGreater(len(artifact["rollout"]["best"]), 1)


if __name__ == "__main__":
    unittest.main()
