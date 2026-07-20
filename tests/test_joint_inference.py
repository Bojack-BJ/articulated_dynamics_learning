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
                "--robust-track-model-trim-ratio",
                "0.2",
                "--orient-parent-by-motion",
                "--parent-orientation-motion-margin-m",
                "0.01",
            ]
        )
        self.assertEqual(args.command, "infer-joints")
        self.assertAlmostEqual(args.rotation_threshold_rad, 0.3)
        self.assertAlmostEqual(args.translation_threshold_m, 0.05)
        self.assertEqual(args.mujoco_prior, "off")
        self.assertAlmostEqual(args.robust_track_model_trim_ratio, 0.2)
        self.assertTrue(args.orient_parent_by_motion)
        self.assertAlmostEqual(args.parent_orientation_motion_margin_m, 0.01)

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

    def test_motion_parent_orientation_can_flip_anchor_child_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_path = root / "part_tracks.json"
            part_pose_path = root / "part_poses.json"
            track_path.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "track_id": 1,
                                "part_id": 1,
                                "reference_xyz_world": [0.0, 0.0, 0.0],
                                "samples": [
                                    {"frame_index": 0, "xyz_world": [0.0, 0.0, 0.0], "visible": True, "depth_valid": True},
                                    {"frame_index": 1, "xyz_world": [0.2, 0.0, 0.0], "visible": True, "depth_valid": True},
                                ],
                            },
                            {
                                "track_id": 2,
                                "part_id": 2,
                                "reference_xyz_world": [1.0, 0.0, 0.0],
                                "samples": [
                                    {"frame_index": 0, "xyz_world": [1.0, 0.0, 0.0], "visible": True, "depth_valid": True},
                                    {"frame_index": 1, "xyz_world": [1.0, 0.0, 0.0], "visible": True, "depth_valid": True},
                                ],
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            part_pose_path.write_text(
                json.dumps(
                    {
                        "input_path": str(track_path),
                        "anchor_part_id": 1,
                        "anchor_part_name": "moving_anchor",
                        "parts": [
                            {
                                "part_id": 1,
                                "name": "moving_anchor",
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
                                    for index in range(2)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.0, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                            {
                                "part_id": 2,
                                "name": "static_child",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                            "translation": [0.1 * index, 0.0, 0.0],
                                        },
                                    }
                                    for index in range(2)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.1, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = JointInferencer(
                JointInferenceConfig(
                    input_path=part_pose_path,
                    mujoco_prior="off",
                    orient_parent_by_motion=True,
                    parent_orientation_motion_margin_m=0.01,
                )
            ).infer()
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            joint = artifact["joints"][0]
            self.assertEqual(joint["parent_part_id"], 2)
            self.assertEqual(joint["child_part_id"], 1)
            self.assertTrue(joint["metrics"]["parent_orientation"]["applied"])

    def test_robust_track_model_trim_drops_largest_replay_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_path = root / "part_tracks.json"
            part_pose_path = root / "part_poses.json"
            tracks = []
            for track_id in range(5):
                reference = [float(track_id), 0.0, 0.0]
                end = [float(track_id) + 0.2, 0.0, 0.0]
                if track_id == 4:
                    end = [float(track_id) + 0.2, 2.0, 0.0]
                tracks.append(
                    {
                        "track_id": track_id,
                        "part_id": 2,
                        "reference_xyz_world": reference,
                        "samples": [
                            {"frame_index": 0, "xyz_world": reference, "visible": True, "depth_valid": True},
                            {"frame_index": 1, "xyz_world": end, "visible": True, "depth_valid": True},
                        ],
                    }
                )
            track_path.write_text(json.dumps({"tracks": tracks}) + "\n", encoding="utf-8")
            part_pose_path.write_text(
                json.dumps(
                    {
                        "input_path": str(track_path),
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
                                    for index in range(2)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.0, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                            {
                                "part_id": 2,
                                "name": "drawer_like",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                            "translation": [0.2 * index, 0.0, 0.0],
                                        },
                                    }
                                    for index in range(2)
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.2, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.0,
                                },
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = JointInferencer(
                JointInferenceConfig(
                    input_path=part_pose_path,
                    mujoco_prior="off",
                    min_track_residual_tracks=1,
                    min_track_residual_samples=1,
                    robust_track_model_trim_ratio=0.2,
                    quality_weighted_replay=True,
                )
            ).infer()
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            comparison = artifact["joints"][0]["metrics"]["track_model_comparison"]
            prismatic = comparison["prismatic"]
            self.assertEqual(comparison["robust_trim_ratio"], 0.2)
            self.assertGreater(prismatic["raw_sample_count"], prismatic["sample_count"])
            self.assertGreater(prismatic["raw_rmse_m"], prismatic["rmse_m"])
            self.assertGreater(prismatic["weighted_replay_sample_count_raw"], prismatic["weighted_replay_sample_count_trimmed"])
            self.assertGreater(prismatic["weighted_replay_rmse_raw"], prismatic["weighted_replay_rmse_trimmed"])
            self.assertAlmostEqual(prismatic["rmse_m"], prismatic["weighted_replay_rmse_trimmed"])

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

    def test_mujoco_joint_prior_accepts_tracking_manifest_input_episode_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            episode_path = root / "episode.json"
            manifest_path = root / "part_tracks.json"
            part_pose_path = root / "part_poses.json"

            model_path.write_text(
                """
<mujoco model="prior_test">
  <worldbody>
    <body name="base">
      <body name="drawer">
        <joint name="slide" type="slide" axis="1 0 0" range="0 0.5" />
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
                            {"action_log": {"joint_positions": {"slide": value}}}
                            for value in [0.0, 0.2, 0.4]
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
                        "input_episode_path": str(episode_path),
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base", "joint_names": []},
                                {
                                    "part_id": 2,
                                    "name": "drawer",
                                    "role": "articulated",
                                    "joint_names": ["slide"],
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
                            {
                                "part_id": 2,
                                "name": "drawer",
                                "samples": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": index * 0.1,
                                        "valid": True,
                                        "relative_to_anchor": {
                                            "rotation_matrix": _rotation_z(angle),
                                            "translation": [offset, 0.0, 0.0],
                                        },
                                    }
                                    for index, (angle, offset) in enumerate([(0.0, 0.0), (0.4, 0.2), (0.8, 0.4)])
                                ],
                                "relative_motion_summary": {
                                    "translation_range": [0.4, 0.0, 0.0],
                                    "rotation_angle_range_rad": 0.8,
                                },
                            },
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = JointInferencer(JointInferenceConfig(input_path=part_pose_path, mujoco_prior="required")).infer()
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            joint = artifact["joints"][0]

            self.assertEqual(artifact["mujoco_prior_count"], 1)
            self.assertEqual(joint["joint_type"], "prismatic")
            self.assertEqual(joint["prior"]["joint_name"], "slide")
            self.assertEqual([sample["q"] for sample in joint["q_samples"]], [0.0, 0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
