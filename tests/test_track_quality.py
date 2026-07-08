from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.kinematics.joint_inference import _weighted_rmse
from rgbd_urdf_mvp.perception.quality_weights import (
    cluster_quality_summary,
    compute_articulation_trace_diagnostics,
    observation_weight,
)
from rgbd_urdf_mvp.perception.track_quality import TrackQualityAnalyzer, TrackQualityConfig, compute_track_quality
from scripts.run_quality_weighting_ablation import _coverage_breakdown


def _track(track_id: int, points: list[list[float]], *, visible: list[bool] | None = None) -> dict:
    visible = visible or [True] * len(points)
    return {
        "track_id": track_id,
        "part_id": 1,
        "original_part_id": 10,
        "samples": [
            {
                "frame_index": idx,
                "timestamp_s": 0.1 * idx,
                "xyz_world": point,
                "visible": visible[idx],
                "depth_valid": visible[idx],
                "mask_consistent": True,
                "tracker_visibility": 1.0 if visible[idx] else 0.0,
                "confidence": 1.0 if visible[idx] else 0.0,
            }
            for idx, point in enumerate(points)
        ],
    }


class TrackQualityTests(unittest.TestCase):
    def test_smooth_circular_arc_keeps_high_timestep_quality(self) -> None:
        points = [[math.cos(theta), math.sin(theta), 0.0] for theta in [idx * 0.08 for idx in range(30)]]
        summary, timesteps = compute_track_quality(_track(1, points))
        qualities = [row["timestep_quality_score"] for row in timesteps[2:-2]]
        self.assertGreater(min(qualities), 0.75)
        self.assertGreater(sum(qualities) / len(qualities), 0.9)
        self.assertEqual(summary["best_motion_type"], "revolute")
        self.assertGreater(summary["articulation_score"], 0.95)

    def test_prismatic_trajectory_prefers_prismatic_model(self) -> None:
        points = [[0.02 * idx, 0.01 * idx, 0.0] for idx in range(30)]
        summary, timesteps = compute_track_quality(_track(1, points))
        self.assertEqual(summary["best_motion_type"], "prismatic")
        self.assertLess(summary["trajectory_residual_m"], 1e-6)
        self.assertTrue(all(row["motion_model_type"] == "prismatic" for row in timesteps))

    def test_articulation_diagnostic_marks_random_jitter_low_quality(self) -> None:
        smooth_points = [[math.cos(theta), math.sin(theta), 0.0] for theta in [idx * 0.08 for idx in range(30)]]
        jitter_points = [
            [0.05 * idx, 0.3 * math.sin(idx * 2.3), 0.2 * math.cos(idx * 1.7)]
            for idx in range(30)
        ]
        smooth_summary, _ = compute_articulation_trace_diagnostics(_track(1, smooth_points))
        jitter_summary, _ = compute_articulation_trace_diagnostics(_track(2, jitter_points))
        self.assertGreater(smooth_summary["articulation_score"], jitter_summary["articulation_score"])
        self.assertLess(jitter_summary["articulation_score"], 0.75)

    def test_large_jump_marks_bad_timestep(self) -> None:
        summary, timesteps = compute_track_quality(
            _track(1, [[0, 0, 0], [0.01, 0, 0], [2.0, 0, 0], [2.01, 0, 0]])
        )
        jump_row = next(row for row in timesteps if row["frame_index"] == 2)
        smooth_row = next(row for row in timesteps if row["frame_index"] == 1)
        self.assertLess(jump_row["timestep_quality_score"], smooth_row["timestep_quality_score"])
        self.assertLess(summary["temporal_smoothness_score"], 1.0)

    def test_jagged_trajectory_has_lower_median_quality_than_smooth_arc(self) -> None:
        arc_points = [[math.cos(theta), math.sin(theta), 0.0] for theta in [idx * 0.08 for idx in range(30)]]
        jagged_points = [
            [0.05 * idx, 0.15 if idx % 2 else -0.15, 0.0]
            for idx in range(30)
        ]
        smooth_summary, _ = compute_track_quality(_track(1, arc_points))
        jagged_summary, _ = compute_track_quality(_track(2, jagged_points))
        self.assertLess(jagged_summary["median_timestep_quality"], smooth_summary["median_timestep_quality"])

    def test_track_quality_decreases_with_many_low_quality_timesteps(self) -> None:
        smooth_points = [[0.03 * idx, 0.0, 0.0] for idx in range(30)]
        jumpy_points = [[0.03 * idx, 0.5 if idx % 4 == 0 else 0.0, 0.0] for idx in range(30)]
        smooth_summary, _ = compute_track_quality(_track(1, smooth_points))
        jumpy_summary, _ = compute_track_quality(_track(2, jumpy_points))
        self.assertGreater(jumpy_summary["low_quality_timestep_ratio"], smooth_summary["low_quality_timestep_ratio"])
        self.assertLess(jumpy_summary["track_quality_score"], smooth_summary["track_quality_score"])

    def test_visibility_affects_track_quality(self) -> None:
        good_summary, _ = compute_track_quality(_track(1, [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]))
        poor_summary, _ = compute_track_quality(
            _track(2, [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], visible=[True, False, False])
        )
        self.assertGreater(good_summary["visible_ratio"], poor_summary["visible_ratio"])
        self.assertGreater(good_summary["track_quality_score"], poor_summary["track_quality_score"])

    def test_analyzer_writes_quality_artifacts_and_enriched_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks_path = root / "motion_part_tracks.json"
            tracks_path.write_text(json.dumps({"tracks": [_track(1, [[0, 0, 0], [0.1, 0, 0]])]}) + "\n")
            outputs = TrackQualityAnalyzer().analyze(
                TrackQualityConfig(input_tracks=tracks_path, output_dir=root / "quality")
            )
            self.assertTrue(outputs["track_quality_summary_csv"].exists())
            self.assertTrue(outputs["timestep_quality_mask_npz"].exists())
            enriched = json.loads(outputs["motion_tracks_with_quality"].read_text())
            self.assertIn("track_quality", enriched["tracks"][0])
            self.assertIn("timestep_quality_score", enriched["tracks"][0]["samples"][0])
            self.assertIn("smooth_residual_m", enriched["tracks"][0]["samples"][0])
            self.assertIn("step_outlier_score", enriched["tracks"][0]["samples"][0])
            self.assertIn("articulation_residual_m", enriched["tracks"][0]["samples"][0])
            self.assertIn("best_motion_type", enriched["tracks"][0]["track_quality"])

    def test_observation_weight_uses_track_and_timestep_quality_with_floor(self) -> None:
        track = {"track_quality": {"track_quality_score": 0.25}}
        sample = {"timestep_quality_score": 0.5}
        self.assertAlmostEqual(observation_weight(track, sample, min_weight=0.1), 0.25)
        self.assertEqual(observation_weight(track, {"timestep_quality_score": 0.0}, min_weight=0.2), 0.2)

    def test_cluster_quality_summary_reports_effective_observations(self) -> None:
        track = _track(1, [[0, 0, 0], [0.1, 0, 0]])
        track["track_quality"] = {"track_quality_score": 0.8}
        track["samples"][0]["timestep_quality_score"] = 1.0
        track["samples"][1]["timestep_quality_score"] = 0.5
        summary = cluster_quality_summary([track])
        self.assertAlmostEqual(summary["cluster_track_quality_mean"], 0.8)
        self.assertAlmostEqual(summary["cluster_effective_observation_count"], 1.5)
        self.assertAlmostEqual(summary["cluster_effective_observation_ratio"], 0.75)

    def test_weighted_rmse_downweights_bad_residuals(self) -> None:
        unweighted = math.sqrt((0.1 * 0.1 + 1.0 * 1.0) / 2.0)
        weighted = _weighted_rmse([(0.1, 1.0), (1.0, 0.1)])
        self.assertLess(weighted, unweighted)

    def test_coverage_breakdown_reports_drop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before = root / "before.json"
            after = root / "after.json"
            before.write_text(
                json.dumps(
                    {
                        "object_summaries": [
                            {
                                "object_id": "obj",
                                "baseline_overlap_per_gt": [
                                    {"gt_part_id": 2, "coverage": 0.8, "best_pred_cluster_id": 5}
                                ],
                                "baseline_overlap_per_cluster": [
                                    {"pred_cluster_id": 5, "purity": 0.9}
                                ],
                                "baseline_overlap_matrix": {"5": {"2": 8}, "6": {"2": 2}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            after.write_text(
                json.dumps(
                    {
                        "object_summaries": [
                            {
                                "object_id": "obj",
                                "baseline_overlap_per_gt": [
                                    {"gt_part_id": 2, "coverage": 0.5, "best_pred_cluster_id": 7}
                                ],
                                "baseline_overlap_per_cluster": [
                                    {"pred_cluster_id": 7, "purity": 0.7}
                                ],
                                "baseline_overlap_matrix": {"7": {"2": 5}, "8": {"2": 3}, "9": {"2": 2}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = _coverage_breakdown(before, after, "after")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["coverage_delta"], -0.3)
            self.assertEqual(rows[0]["fragmentation_delta"], 1)
            self.assertAlmostEqual(rows[0]["dominant_cluster_purity_after"], 0.7)


if __name__ == "__main__":
    unittest.main()
