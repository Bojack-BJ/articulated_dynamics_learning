from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.kinematics.inferred_articulation import (
    InferredArticulationPipeline,
    InferredArticulationPipelineConfig,
)


class InferredArticulationTests(unittest.TestCase):
    def test_export_inferred_articulation_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "export-inferred-articulation",
                "episode.json",
                "part_poses.json",
                "joint_inference.json",
                "--output-dir",
                "outputs/inferred",
            ]
        )
        self.assertEqual(args.command, "export-inferred-articulation")
        self.assertEqual(args.episode, Path("episode.json"))
        self.assertEqual(args.part_poses, Path("part_poses.json"))
        self.assertEqual(args.joint_inference, Path("joint_inference.json"))

    def test_export_inferred_articulation_builds_urdf_and_mjcf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_path = root / "episode.json"
            part_pose_path = root / "part_poses.json"
            joint_inference_path = root / "joint_inference.json"
            output_dir = root / "inferred_articulation"

            episode_path.write_text(
                json.dumps(
                    {
                        "object_instance_id": "test-door",
                        "category": "door",
                        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 160.0, "cy": 120.0},
                        "frames": [
                            {
                                "timestamp_s": 0.0,
                                "rgb_path": "rgb.png",
                                "depth_path": "depth.pgm",
                                "camera_pose": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            part_pose_path.write_text(
                json.dumps(
                    {
                        "anchor_part_id": 1,
                        "anchor_part_name": "base",
                        "parts": [
                            {
                                "part_id": 1,
                                "name": "base",
                                "reference_point_count": 1200,
                                "canonical_frame": {
                                    "translation": [0.0, 0.0, 0.0],
                                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                },
                                "reference_local_bounds": {
                                    "lower": [-0.5, -0.3, -0.8],
                                    "upper": [0.5, 0.3, 0.8],
                                },
                                "samples": [
                                    {"frame_index": 0, "timestamp_s": 0.0, "valid": True, "confidence": 0.9},
                                    {"frame_index": 1, "timestamp_s": 0.1, "valid": True, "confidence": 0.9},
                                ],
                            },
                            {
                                "part_id": 2,
                                "name": "door",
                                "reference_point_count": 420,
                                "canonical_frame": {
                                    "translation": [0.45, 0.0, 0.0],
                                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                },
                                "reference_local_bounds": {
                                    "lower": [-0.05, -0.25, -0.75],
                                    "upper": [0.05, 0.25, 0.75],
                                },
                                "samples": [
                                    {"frame_index": 0, "timestamp_s": 0.0, "valid": True, "confidence": 0.85},
                                    {"frame_index": 1, "timestamp_s": 0.1, "valid": True, "confidence": 0.88},
                                ],
                            },
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            joint_inference_path.write_text(
                json.dumps(
                    {
                        "anchor_part_id": 1,
                        "anchor_part_name": "base",
                        "joints": [
                            {
                                "name": "door_joint",
                                "parent_part_id": 1,
                                "parent_name": "base",
                                "child_part_id": 2,
                                "child_name": "door",
                                "joint_type": "revolute",
                                "axis": [0.0, 0.0, 1.0],
                                "pivot": [0.5, 0.0, 0.0],
                                "limits": [0.0, 1.2],
                                "confidence": 0.92,
                                "metrics": {"rotation_range_rad": 1.2},
                                "q_samples": [
                                    {"frame_index": 0, "timestamp_s": 0.0, "q": 0.0},
                                    {"frame_index": 1, "timestamp_s": 0.1, "q": 0.4},
                                ],
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            result = InferredArticulationPipeline().run(
                InferredArticulationPipelineConfig(
                    episode_path=episode_path,
                    part_pose_path=part_pose_path,
                    joint_inference_path=joint_inference_path,
                    output_dir=output_dir,
                )
            )

            self.assertTrue(Path(result.articulation_artifact_path).exists())
            self.assertTrue(Path(result.urdf_package.urdf_path).exists())
            self.assertTrue(Path(result.urdf_package.mjcf_path).exists())

            articulation_text = Path(result.articulation_artifact_path).read_text(encoding="utf-8")
            self.assertIn('"joint_type": "revolute"', articulation_text)
            self.assertIn('"joint_state_tracks"', articulation_text)
            self.assertIn('"door_joint"', articulation_text)

            mjcf_tree = ET.parse(result.urdf_package.mjcf_path)
            bodies = {node.attrib.get("name") for node in mjcf_tree.findall(".//body")}
            joints = mjcf_tree.findall(".//joint")
            self.assertIn("base", bodies)
            self.assertIn("door", bodies)
            self.assertTrue(any(node.attrib.get("name") == "door_joint" for node in joints))
            self.assertTrue(any(node.attrib.get("type") == "hinge" for node in joints))


if __name__ == "__main__":
    unittest.main()
