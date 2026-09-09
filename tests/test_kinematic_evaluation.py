from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.kinematics.evaluation import (
    KinematicModelEvaluationConfig,
    KinematicModelEvaluator,
    ObjectMaskKinematicEvaluationConfig,
    ObjectMaskKinematicEvaluator,
    _joint_cleanup_recommendation,
    _joint_failure_reason_guess,
    _object_mask_summary,
)
from rgbd_urdf_mvp.kinematics.local_split_candidate_evaluation import (
    LocalSplitCandidateEvaluationConfig,
    LocalSplitCandidateEvaluator,
)


class KinematicEvaluationTests(unittest.TestCase):
    def test_type_mismatch_is_excluded_from_axis_and_pivot_summary(self) -> None:
        summary = _object_mask_summary(
            predicted_joints=[{}, {}],
            gt_priors={1: {}, 2: {}},
            directed_matches=[
                {
                    "ground_truth_joint_type": "revolute",
                    "joint_type_correct": True,
                    "axis_angle_error_deg": 4.0,
                    "pivot_error_m": 0.02,
                },
                {
                    "ground_truth_joint_type": "revolute",
                    "joint_type_correct": False,
                    "axis_angle_error_deg": 80.0,
                    "pivot_error_m": 1.0,
                },
            ],
            undirected_matches=[],
            reverse_matches=[],
            overlap={},
        )

        self.assertEqual(summary["joint_type_accuracy"], 0.5)
        self.assertEqual(summary["axis_evaluable_joint_count"], 1)
        self.assertEqual(summary["axis_angle_error_deg_mean"], 4.0)
        self.assertEqual(summary["axis_angle_error_deg_median"], 4.0)
        self.assertEqual(summary["pivot_error_m_mean"], 0.02)
        revolute = summary["by_ground_truth_joint_type"]["revolute"]
        self.assertEqual(revolute["joint_type_accuracy"], 0.5)
        self.assertEqual(revolute["axis_evaluable_joint_count"], 1)

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

    def test_parser_accepts_object_mask_kinematic_eval_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate-object-mask-kinematics",
                "joint_inference.json",
                "--part-poses",
                "part_poses.json",
                "--output-json",
                "eval.json",
                "--output-csv",
                "eval.csv",
                "--matching-metric",
                "overlap",
            ]
        )
        self.assertEqual(args.command, "evaluate-object-mask-kinematics")
        self.assertEqual(args.part_poses, Path("part_poses.json"))
        self.assertEqual(args.output_csv, Path("eval.csv"))
        self.assertEqual(args.matching_metric, "overlap")

    def test_parser_accepts_local_split_candidate_eval_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate-local-split-candidates",
                "local_split_summary.json",
                "--output-dir",
                "candidate_eval",
                "--output-json",
                "candidate_eval.json",
                "--output-csv",
                "candidate_eval.csv",
                "--mujoco-prior",
                "off",
                "--robust-track-model-trim-ratio",
                "0.2",
            ]
        )
        self.assertEqual(args.command, "evaluate-local-split-candidates")
        self.assertEqual(args.local_split_summary, Path("local_split_summary.json"))
        self.assertEqual(args.output_dir, Path("candidate_eval"))
        self.assertEqual(args.output_csv, Path("candidate_eval.csv"))
        self.assertAlmostEqual(args.robust_track_model_trim_ratio, 0.2)

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

    def test_prismatic_joint_has_no_pivot_error_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            episode_path = root / "episode.json"
            manifest_path = root / "fusion_manifest.json"
            part_pose_path = root / "part_poses.json"
            joint_path = root / "joint_inference.json"

            model_path.write_text(
                """
<mujoco model="eval_prismatic_test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="drawer" pos="0 0 0">
        <joint name="slide" type="slide" axis="1 0 0" pos="0.1 0.2 0.3" range="0 1" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
                + "\n",
                encoding="utf-8",
            )
            episode_path.write_text(json.dumps({"frames": [], "metadata": {"model_path": str(model_path)}}) + "\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "episode_path": str(episode_path),
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base", "joint_names": []},
                                {"part_id": 2, "name": "drawer", "role": "articulated", "joint_names": ["slide"]},
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
                            {"part_id": 2, "name": "drawer", "samples": []},
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
                                "name": "drawer_joint",
                                "child_part_id": 2,
                                "child_name": "drawer",
                                "joint_type": "prismatic",
                                "axis": [-1.0, 0.0, 0.0],
                                "pivot": [10.0, 20.0, 30.0],
                                "limits": [0.0, 1.0],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = KinematicModelEvaluator().evaluate(KinematicModelEvaluationConfig(joint_inference_path=joint_path))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["matched_joint_count"], 1)
            self.assertAlmostEqual(payload["summary"]["axis_angle_error_deg_mean"], 0.0)
            self.assertIsNone(payload["summary"]["pivot_error_m_mean"])
            self.assertIsNone(payload["per_joint"][0]["pivot_error_m"])

    def test_object_mask_eval_matches_clusters_to_gt_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            episode_path = root / "episode.json"
            track_path = root / "motion_part_tracks.json"
            part_pose_path = root / "part_poses.json"
            joint_path = root / "joint_inference.json"
            csv_path = root / "summary.csv"

            model_path.write_text(
                """
<mujoco model="object_mask_eval_test">
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
                            for value in [-0.1, -0.2]
                        ],
                        "metadata": {"model_path": str(model_path)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tracks = []
            for track_id in range(4):
                tracks.append({"track_id": track_id, "part_id": 10, "original_part_id": 1, "samples": []})
            for track_id in range(4, 10):
                tracks.append({"track_id": track_id, "part_id": 20, "original_part_id": 2, "samples": []})
            track_path.write_text(
                json.dumps(
                    {
                        "input_episode_path": str(episode_path),
                        "original_part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base", "joint_names": []},
                                {
                                    "part_id": 2,
                                    "name": "door",
                                    "role": "articulated",
                                    "parent_part_id": 1,
                                    "joint_names": ["hinge"],
                                },
                            ]
                        },
                        "tracks": tracks,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            part_pose_path.write_text(
                json.dumps(
                    {
                        "input_path": str(track_path),
                        "anchor_part_id": 10,
                        "anchor_part_name": "cluster_base",
                        "parts": [
                            {
                                "part_id": 10,
                                "name": "cluster_base",
                                "canonical_frame": {
                                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    "translation": [0.0, 0.0, 0.0],
                                },
                                "samples": [],
                            },
                            {"part_id": 20, "name": "cluster_door", "samples": []},
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
                                "name": "cluster_door_joint",
                                "parent_part_id": 10,
                                "child_part_id": 20,
                                "child_name": "cluster_door",
                                "joint_type": "revolute",
                                "axis": [0.0, 0.0, -1.0],
                                "pivot": [0.1, 0.2, 0.3],
                                "limits": [-1.0, 0.0],
                                "q_samples": [
                                    {"frame_index": index, "timestamp_s": index * 0.1, "q": value}
                                    for index, value in enumerate([-0.1, -0.2])
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = ObjectMaskKinematicEvaluator().evaluate(
                ObjectMaskKinematicEvaluationConfig(
                    joint_inference_path=joint_path,
                    output_csv=csv_path,
                )
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["diagnostic_only"])
            self.assertEqual(payload["matching"]["pred_to_gt"], {"10": 1, "20": 2})
            self.assertAlmostEqual(payload["overlap"]["mean_purity"], 1.0)
            self.assertAlmostEqual(payload["overlap"]["mean_gt_coverage"], 1.0)
            self.assertEqual(payload["summary"]["directed_matched_joint_count"], 1)
            self.assertEqual(payload["summary"]["undirected_matched_joint_count"], 1)
            self.assertAlmostEqual(payload["summary"]["directed_joint_coverage"], 1.0)
            self.assertAlmostEqual(payload["summary"]["joint_type_accuracy"], 1.0)
            self.assertAlmostEqual(payload["summary"]["axis_angle_error_deg_mean"], 0.0)
            self.assertAlmostEqual(payload["summary"]["pivot_error_m_mean"], 0.0)
            self.assertEqual(payload["per_joint"][0]["parent_cluster_debug"]["gt_purity"], 1.0)
            self.assertEqual(payload["per_joint"][0]["child_cluster_debug"]["gt_purity"], 1.0)
            self.assertIsNone(payload["per_joint"][0]["failure_reason_guess"])
            self.assertTrue(csv_path.exists())

    def test_object_mask_failure_reason_flags_mixed_nonrigid_child_cluster(self) -> None:
        joint = {
            "directed_matched": True,
            "axis_angle_error_deg": 3.0,
            "pivot_error_m": 0.7,
            "child_cluster_debug": {
                "gt_purity": 0.48,
                "rigid_rmse_m": 0.059,
                "inlier_ratio": 0.25,
            },
        }
        reason = _joint_failure_reason_guess(joint)
        self.assertEqual(reason, "child_cluster_mixed_or_nonrigid")
        joint["failure_reason_guess"] = reason
        self.assertEqual(_joint_cleanup_recommendation(joint), "local_split_child_cluster_then_refit_part_pose")

    def test_local_split_candidate_eval_builds_downstream_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_path = root / "candidate_tracks.json"
            summary_path = root / "local_split_summary.json"
            output_json = root / "candidate_eval.json"
            output_csv = root / "candidate_eval.csv"

            def track(track_id: int, part_id: int, reference: list[float], displacement: list[float]) -> dict:
                return {
                    "track_id": track_id,
                    "part_id": part_id,
                    "reference_xyz_world": reference,
                    "samples": [
                        {
                            "frame_index": 0,
                            "timestamp_s": 0.0,
                            "xyz_world": reference,
                            "visible": True,
                            "depth_valid": True,
                            "confidence": 1.0,
                        },
                        {
                            "frame_index": 1,
                            "timestamp_s": 0.1,
                            "xyz_world": [reference[i] + displacement[i] for i in range(3)],
                            "visible": True,
                            "depth_valid": True,
                            "confidence": 1.0,
                        },
                    ],
                }

            tracks = []
            for idx, point in enumerate([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]):
                tracks.append(track(idx, 1, point, [0.0, 0.0, 0.0]))
            for idx, point in enumerate([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]], start=10):
                tracks.append(track(idx, 4, point, [0.2, 0.0, 0.0]))
            track_path.write_text(
                json.dumps(
                    {
                        "frame_count": 2,
                        "source_frame_count": 2,
                        "sampled_frame_indices": [0, 1],
                        "part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base"},
                                {"part_id": 4, "name": "candidate_child", "role": "moving"},
                            ]
                        },
                        "tracks": tracks,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "split_cluster_id": 4,
                                "status": "written",
                                "ablation": "B",
                                "local_k": 2,
                                "output_json": str(track_path),
                                "after": {
                                    "clusters": [
                                        {
                                            "part_id": 4,
                                            "track_count": 4,
                                            "bbox_diag_m": 0.173,
                                            "mean_motion_m": 0.2,
                                            "median_motion_m": 0.2,
                                            "rigid_rmse_m": 0.0,
                                            "inlier_ratio": 1.0,
                                            "visible_frame_ratio": 1.0,
                                            "candidate_selection": {
                                                "selection_score_no_gt": 0.9,
                                                "diagnostic_score_with_gt": 0.9,
                                                "track_count_score": 1.0,
                                                "bbox_score": 1.0,
                                                "motion_score": 1.0,
                                                "inlier_score": 1.0,
                                                "visibility_score": 1.0,
                                                "base_like_low_motion": False,
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result_path = LocalSplitCandidateEvaluator().evaluate(
                LocalSplitCandidateEvaluationConfig(
                    local_split_summary=summary_path,
                    output_json=output_json,
                    output_csv=output_csv,
                    min_tracks_per_part=3,
                    mujoco_prior="off",
                )
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 1)
            row = payload["rows"][0]
            self.assertEqual(row["part_id"], 4)
            self.assertEqual(row["selection_score_no_gt"], 0.9)
            self.assertEqual(row["joint_type"], "prismatic")
            self.assertIsNotNone(row["joint_replay_error_m"])
            self.assertIsNotNone(row["joint_selection_score_no_gt"])
            self.assertIsNotNone(row["joint_replay_score"])
            self.assertIsNotNone(row["motion_normalized_replay_error"])
            self.assertLess(float(row["degeneracy_penalty"]), 0.5)
            self.assertTrue(output_csv.exists())


if __name__ == "__main__":
    unittest.main()
