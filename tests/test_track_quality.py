from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.kinematics.joint_inference import _weighted_rmse
from rgbd_urdf_mvp.perception.quality_weights import (
    articulation_pair_compatibility,
    articulation_pair_compatibility_metrics,
    cluster_quality_summary,
    compute_articulation_trace_diagnostics,
    observation_weight,
)
from rgbd_urdf_mvp.perception.track_quality import TrackQualityAnalyzer, TrackQualityConfig, compute_track_quality
from scripts.run_articulation_affinity_quick_ablation import _case_config, _should_rerun
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

    def test_static_revolute_mismatch_is_not_strongly_penalized_for_low_motion_static_track(self) -> None:
        static_track = _track(1, [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]])
        revolute_track = _track(
            2,
            [[math.cos(theta), math.sin(theta), 0.0] for theta in [0.0, 0.2, 0.4, 0.6, 0.8]],
        )
        static_track["track_quality"] = {"best_motion_type": "static", "articulation_score": 0.95}
        revolute_track["track_quality"] = {"best_motion_type": "revolute", "articulation_score": 0.95}
        metrics = articulation_pair_compatibility_metrics(static_track, revolute_track)
        self.assertEqual(metrics["penalty_reason"], "static_mismatch_neutralized")
        self.assertAlmostEqual(metrics["penalty"], 1.0)
        self.assertGreater(articulation_pair_compatibility(static_track, revolute_track), 0.9)

    def test_confident_revolute_prismatic_mismatch_is_penalized(self) -> None:
        revolute_track = _track(
            1,
            [[math.cos(theta), math.sin(theta), 0.0] for theta in [0.0, 0.2, 0.4, 0.6, 0.8]],
        )
        prismatic_track = _track(2, [[0.1 * idx, 0.0, 0.0] for idx in range(5)])
        revolute_track["track_quality"] = {"best_motion_type": "revolute", "articulation_score": 0.95}
        prismatic_track["track_quality"] = {"best_motion_type": "prismatic", "articulation_score": 0.95}
        metrics = articulation_pair_compatibility_metrics(revolute_track, prismatic_track)
        self.assertEqual(metrics["penalty_reason"], "confident_nonstatic_type_mismatch")
        self.assertAlmostEqual(metrics["penalty"], 0.65)

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

    def test_analyzer_can_mask_large_3d_steps_without_dropping_track_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks_path = root / "tracks.json"
            tracks_path.write_text(
                json.dumps({"tracks": [_track(7, [[0, 0, 0], [0.01, 0, 0], [0.20, 0, 0]])]}) + "\n"
            )
            outputs = TrackQualityAnalyzer().analyze(
                TrackQualityConfig(
                    input_tracks=tracks_path,
                    output_dir=root / "quality",
                    bad_timestep_threshold=0.0,
                    mask_bad_timesteps=True,
                    max_step_m=0.05,
                )
            )
            enriched = json.loads(outputs["motion_tracks_with_quality"].read_text())
            samples = enriched["tracks"][0]["samples"]
            self.assertEqual(enriched["tracks"][0]["track_id"], 7)
            self.assertTrue(samples[1]["visible"])
            self.assertFalse(samples[2]["visible"])
            self.assertEqual(samples[2]["quality_rejection_reasons"], ["step_too_large"])

    def test_query_connected_segment_drops_post_jump_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track = _track(7, [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.30, 0, 0], [0.31, 0, 0]])
            track["query_frame_index"] = 1
            tracks_path = root / "tracks.json"
            tracks_path.write_text(json.dumps({"tracks": [track]}) + "\n")
            outputs = TrackQualityAnalyzer().analyze(
                TrackQualityConfig(
                    input_tracks=tracks_path,
                    output_dir=root / "quality",
                    bad_timestep_threshold=0.0,
                    mask_bad_timesteps=True,
                    max_step_m=0.05,
                    keep_query_connected_segment=True,
                    min_query_connected_frames=2,
                )
            )
            enriched = json.loads(outputs["motion_tracks_with_quality"].read_text())
            samples = enriched["tracks"][0]["samples"]
            self.assertEqual([sample["visible"] for sample in samples], [True, True, True, False, False])
            self.assertIn("outside_query_connected_segment", samples[4]["quality_rejection_reasons"])

    def test_query_connected_segment_spans_short_missing_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track = _track(7, [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
            track["query_frame_index"] = 0
            track["samples"][1]["visible"] = False
            tracks_path = root / "tracks.json"
            tracks_path.write_text(json.dumps({"tracks": [track]}) + "\n")
            outputs = TrackQualityAnalyzer().analyze(
                TrackQualityConfig(
                    input_tracks=tracks_path,
                    output_dir=root / "quality",
                    bad_timestep_threshold=0.0,
                    mask_bad_timesteps=True,
                    keep_query_connected_segment=True,
                    query_connected_max_gap_frames=1,
                )
            )
            enriched = json.loads(outputs["motion_tracks_with_quality"].read_text())
            samples = enriched["tracks"][0]["samples"]
            self.assertEqual([sample["visible"] for sample in samples], [True, False, True, True])
            self.assertEqual(
                enriched["track_quality"]["query_connected_max_gap_frames"], 1
            )

    def test_spatial_dbscan_masks_isolated_timestep_without_dropping_dense_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks = [
                _track(index, [[0.005 * index, 0, 0], [0.005 * index, 0.01, 0]])
                for index in range(5)
            ]
            tracks.append(_track(99, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
            for track in tracks:
                track["part_id"] = 1
            tracks_path = root / "tracks.json"
            tracks_path.write_text(json.dumps({"tracks": tracks}) + "\n")
            outputs = TrackQualityAnalyzer().analyze(
                TrackQualityConfig(
                    input_tracks=tracks_path,
                    output_dir=root / "quality",
                    bad_timestep_threshold=0.0,
                    mask_bad_timesteps=True,
                    spatial_dbscan_eps_m=0.03,
                    spatial_dbscan_min_samples=3,
                )
            )
            enriched = json.loads(outputs["motion_tracks_with_quality"].read_text())
            self.assertTrue(all(sample["visible"] for sample in enriched["tracks"][0]["samples"]))
            self.assertTrue(all(not sample["visible"] for sample in enriched["tracks"][-1]["samples"]))
            self.assertIn(
                "spatial_dbscan_outlier",
                enriched["tracks"][-1]["samples"][0]["quality_rejection_reasons"],
            )

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

    def test_quick_ablation_reruns_on_manifest_mismatch_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs = [root / name for name in ["tracks.json", "diag.json", "poses.json", "joints.json", "eval.json"]]
            for path in outputs:
                path.write_text("{}", encoding="utf-8")
            manifest = root / "config_manifest.json"
            config = _case_config(Path.cwd(), "baseline", 5, "B")
            manifest.write_text(json.dumps({"config": config}), encoding="utf-8")
            rerun, reason = _should_rerun(
                required_outputs=outputs,
                manifest_path=manifest,
                expected_config=config,
                force=False,
                reuse_without_manifest=False,
            )
            self.assertFalse(rerun)
            self.assertEqual(reason, "reuse_manifest_match")
            changed = dict(config)
            changed["spectral_k"] = 4
            rerun, reason = _should_rerun(
                required_outputs=outputs,
                manifest_path=manifest,
                expected_config=changed,
                force=False,
                reuse_without_manifest=False,
            )
            self.assertTrue(rerun)
            self.assertEqual(reason, "config_mismatch")
            rerun, reason = _should_rerun(
                required_outputs=outputs,
                manifest_path=manifest,
                expected_config=config,
                force=True,
                reuse_without_manifest=False,
            )
            self.assertTrue(rerun)
            self.assertEqual(reason, "force")


if __name__ == "__main__":
    unittest.main()
