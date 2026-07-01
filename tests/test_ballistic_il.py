from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.manipulation.ballistic import condition_vector, dynamics_values, load_model, resolve_joint
from rgbd_urdf_mvp.manipulation.il_dataset import BallisticILDatasetConfig, BallisticILDatasetGenerator
from rgbd_urdf_mvp.manipulation.il_eval import BallisticILEvalConfig, BallisticILEvaluator
from rgbd_urdf_mvp.manipulation.il_policy import BCPolicyConfig, build_policy
from rgbd_urdf_mvp.manipulation.il_train import BallisticILTrainConfig, BallisticILTrainer


MODEL_XML = """
<mujoco model="ballistic_il_test">
  <option timestep="0.01" gravity="0 0 0" />
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="door" pos="0 0 0">
        <inertial pos="0 0 0" quat="1 0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1" />
        <joint name="door_joint" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.35" frictionloss="0.0" />
        <geom type="box" pos="0.25 0 0" size="0.25 0.03 0.35" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


class BallisticILTests(unittest.TestCase):
    def test_il_cli_parsers_accept_arguments(self) -> None:
        parser = build_parser()
        demo = parser.parse_args(
            [
                "generate-il-demos",
                "model.xml",
                "--output-dir",
                "out",
                "--task",
                "impulse-to-target",
                "--condition-source",
                "estimated",
            ]
        )
        self.assertEqual(demo.command, "generate-il-demos")
        self.assertEqual(demo.condition_source, "estimated")

        train = parser.parse_args(["train-il-policy", "il_dataset.npz", "--output-dir", "policy", "--condition-mode", "none"])
        self.assertEqual(train.command, "train-il-policy")
        self.assertEqual(train.condition_mode, "none")

        eval_args = parser.parse_args(["eval-il-policy", "policy.pt", "model.xml", "--output-dir", "eval"])
        self.assertEqual(eval_args.command, "eval-il-policy")
        self.assertEqual(eval_args.policy, Path("policy.pt"))

    def test_condition_vector_is_finite_with_zero_friction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.xml"
            model_path.write_text(MODEL_XML + "\n", encoding="utf-8")
            model = load_model(model_path)
            joint = resolve_joint(model, joint_name="door_joint")
            cond = condition_vector(dynamics_values(model, joint), joint)
            self.assertEqual(cond.shape, (7,))
            self.assertTrue(np.isfinite(cond).all())

    def test_dataset_train_and_eval_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            model_path.write_text(MODEL_XML + "\n", encoding="utf-8")

            dataset_path = BallisticILDatasetGenerator().generate(
                BallisticILDatasetConfig(
                    mjcf_path=model_path,
                    output_dir=root / "dataset",
                    joint_name="door_joint",
                    num_episodes=24,
                    release_duration_s=0.4,
                    response_samples=12,
                    num_teacher_candidates=31,
                    seed=3,
                )
            )
            payload = np.load(dataset_path)
            self.assertEqual(payload["obs"].shape, (24, 4))
            self.assertEqual(payload["condition"].shape, (24, 7))
            self.assertEqual(payload["action"].shape, (24, 1))
            self.assertEqual(payload["free_response"].shape, (24, 12, 3))
            response = payload["free_response"][0]
            self.assertGreater(response[-1, 0], response[0, 0])

            policy_path = BallisticILTrainer().train(
                BallisticILTrainConfig(
                    dataset_path=dataset_path,
                    output_dir=root / "policy",
                    condition_mode="oracle",
                    epochs=2,
                    batch_size=8,
                    hidden_dim=32,
                )
            )
            self.assertTrue(policy_path.exists())

            eval_path = BallisticILEvaluator().evaluate(
                BallisticILEvalConfig(
                    policy_path=policy_path,
                    mjcf_path=model_path,
                    output_dir=root / "eval",
                    joint_name="door_joint",
                    num_episodes=4,
                    release_duration_s=0.4,
                    response_samples=12,
                    seed=4,
                )
            )
            artifact = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertIn("success_rate", artifact)
            self.assertIn("final_error_mean", artifact)
            self.assertIn("settling_qdot_mean", artifact)
            self.assertEqual(len(artifact["episodes"]), 4)

    def test_bc_policy_forward_shape(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        policy = build_policy(BCPolicyConfig(obs_dim=4, condition_dim=7, action_dim=1, condition_mode="oracle", hidden_dim=16))
        output = policy(torch.zeros((5, 11), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
