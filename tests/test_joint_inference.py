from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.kinematics.joint_inference import JointInferenceConfig, JointInferencer


def _rotation_z(angle_rad: float) -> list[list[float]]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]


class JointInferenceTests(unittest.TestCase):
    def test_infer_joints_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "infer-joints",
                "part_poses.json",
                "--output-json",
                "joint_inference.json",
                "--rotation-threshold-rad",
                "0.3",
                "--translation-threshold-m",
                "0.05",
                "--mujoco-prior",
                "off",
            ]
        )
        self.assertEqual(args.command, "infer-joints")
        self.assertAlmostEqual(args.rotation_threshold_rad, 0.3)
        self.assertAlmostEqual(args.translation_threshold_m, 0.05)
        self.assertEqual(args.mujoco_prior, "off")

    def test_infer_revolute_and_prismatic_joints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            part_pose_path = root / "part_poses.json"

            door_angles = [0.0, 0.3, 0.6]
            drawer_offsets = [0.0, 0.08, 0.16]
            part_pose_path.write_text(
                json.dumps(
                    {
                        "anchor_part_id": 1,
                        "anchor_part_name": "base",
                        "parts": [
                            {
                                "part_id": 1,
                                "name": "base",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                            "translation": [0.0, 0.0, 0.0],
                                        },
                                    }
                                    for index in range(3)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.0, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                            {
                                "part_id": 2,
                                "name": "door",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": _rotation_z(angle),
                                            "translation": [
                                                0.5 * math.cos(angle),
                                                0.5 * math.sin(angle),
                                                0.0,
                                            ],
                                        },
                                    }
                                    for index, angle in enumerate(door_angles)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.08733219254516086, 0.2823212366975177, 0.0],
                                    "rotation_angle_range_rad": 0.6,
                                },
                            },
                            {
                                "part_id": 3,
                                "name": "drawer",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                            "translation": [offset, 0.2, 0.0],
                                        },
                                    }
                                    for index, offset in enumerate(drawer_offsets)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.16, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = JointInferencer(JointInferenceConfig(input_path=part_pose_path)).infer()
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(artifact["joints"]), 2)

            by_child = {joint["child_name"]: joint for joint in artifact["joints"]}
            self.assertEqual(by_child["door"]["joint_type"], "revolute")
            self.assertEqual(by_child["drawer"]["joint_type"], "prismatic")
            self.assertAlmostEqual(abs(by_child["door"]["axis"][2]), 1.0, places=1)
            self.assertAlmostEqual(by_child["drawer"]["axis"][0], 1.0, places=1)
            self.assertAlmostEqual(by_child["door"]["pivot"][0], 0.0, places=1)
            self.assertAlmostEqual(by_child["door"]["pivot"][1], 0.0, places=1)
            self.assertAlmostEqual(by_child["drawer"]["limits"][1], 0.16, places=2)

    def test_mujoco_joint_prior_overrides_noisy_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            episode_path = root / "episode.json"
            manifest_path = root / "fusion_manifest.json"
            part_pose_path = root / "part_poses.json"

            model_path.write_text(
                """
<mujoco model="prior_test">
  <worldbody>
    <body name="base" pos="1 2 3">
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
                            for value in [-0.2, -0.5, -0.8]
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
                                    "translation": [1.0, 2.0, 3.0],
                                },
                                "samples": [],
                            },
                            {
                                "part_id": 2,
                                "name": "door",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": _rotation_z(angle),
                                            "translation": [0.5 * math.cos(angle), 0.5 * math.sin(angle), 0.0],
                                        },
                                    }
                                    for index, angle in enumerate([0.0, 0.3, 0.6])
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.1, 0.2, 0.0],
                                    "rotation_angle_range_rad": 0.6,
                                },
                            },
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = JointInferencer(JointInferenceConfig(input_path=part_pose_path)).infer()
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            joint = artifact["joints"][0]

            self.assertEqual(artifact["mujoco_prior_count"], 1)
            self.assertEqual(joint["prior"]["source"], "mujoco-mjcf")
            self.assertEqual(joint["joint_type"], "revolute")
            self.assertEqual(joint["axis"], [0.0, 0.0, 1.0])
            self.assertAlmostEqual(joint["pivot"][0], 0.1)
            self.assertAlmostEqual(joint["pivot"][1], 0.2)
            self.assertAlmostEqual(joint["pivot"][2], 0.3)
            self.assertEqual(joint["limits"], [-1.0, 0.0])
            self.assertEqual([sample["q"] for sample in joint["q_samples"]], [-0.2, -0.5, -0.8])


if __name__ == "__main__":
    unittest.main()
