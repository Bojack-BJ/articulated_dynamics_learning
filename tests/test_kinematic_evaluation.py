from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.kinematics.evaluation import (
    KinematicModelEvaluationConfig,
    KinematicModelEvaluator,
)


class KinematicEvaluationTests(unittest.TestCase):
    def test_parser_accepts_kinematic_eval_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate-kinematic-model",
                "joint_inference.json",
                "--part-poses",
                "part_poses.json",
                "--output-json",
                "eval.json",
            ]
        )
        self.assertEqual(args.command, "evaluate-kinematic-model")
        self.assertEqual(args.part_poses, Path("part_poses.json"))

    def test_evaluates_joint_against_mujoco_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            episode_path = root / "episode.json"
            manifest_path = root / "fusion_manifest.json"
            part_pose_path = root / "part_poses.json"
            joint_path = root / "joint_inference.json"

            model_path.write_text(
                """
<mujoco model="eval_test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="door" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 0 1" pos="0.1 0.2 0.3" range="-1 0" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
                + "\n",
                encoding="utf-8",
            )
            episode_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"action_log": {"joint_positions": {"hinge": value}}}
                            for value in [-0.1, -0.2, -0.3]
                        ],
                        "metadata": {"model_path": str(model_path)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "episode_path": str(episode_path),
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base", "joint_names": []},
                                {
                                    "part_id": 2,
                                    "name": "door",
                                    "role": "articulated",
                                    "joint_names": ["hinge"],
                                },
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            part_pose_path.write_text(
                json.dumps(
                    {
                        "input_path": str(manifest_path),
                        "anchor_part_id": 1,
                        "anchor_part_name": "base",
                        "parts": [
                            {
                                "part_id": 1,
                                "name": "base",
                                "canonical_frame": {
                                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    "translation": [0.0, 0.0, 0.0],
                                },
                                "samples": [],
                            },
                            {"part_id": 2, "name": "door", "samples": []},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            joint_path.write_text(
                json.dumps(
                    {
                        "input_path": str(part_pose_path),
                        "joints": [
                            {
                                "name": "door_joint",
                                "child_part_id": 2,
                                "child_name": "door",
                                "joint_type": "revolute",
                                "axis": [0.0, 0.0, -1.0],
                                "pivot": [0.1, 0.2, 0.3],
                                "limits": [-1.0, 0.0],
                                "q_samples": [
                                    {"frame_index": index, "timestamp_s": index * 0.1, "q": value}
                                    for index, value in enumerate([-0.1, -0.2, -0.3])
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = KinematicModelEvaluator().evaluate(
                KinematicModelEvaluationConfig(joint_inference_path=joint_path)
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["ground_truth_available"])
            self.assertEqual(payload["summary"]["matched_joint_count"], 1)
            self.assertEqual(payload["summary"]["joint_type_accuracy"], 1.0)
            self.assertAlmostEqual(payload["summary"]["axis_angle_error_deg_mean"], 0.0)
            self.assertAlmostEqual(payload["summary"]["pivot_error_m_mean"], 0.0)
            self.assertAlmostEqual(payload["summary"]["q_rmse_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

