from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.motion_segmentation import (
    LocalMotionClusterSplitter,
    LocalSplitDiagnosticsConfig,
    MotionPartSegmentationConfig,
    MotionPartSegmenter,
    _cluster_diagnostics,
    _pair_metrics,
)
from rgbd_urdf_mvp.perception.object_mask_diagnostics_viz import (
    ObjectMaskDiagnosticsVisualizationConfig,
    ObjectMaskDiagnosticsVisualizer,
)
from rgbd_urdf_mvp.perception.object_mask_flow_html import ObjectMaskFlowHtmlBuilder, ObjectMaskFlowHtmlConfig
from rgbd_urdf_mvp.perception.part_tracking import TrackPartPoseEstimationConfig, TrackPartPoseEstimator


def _rotate_about_z(point: list[float], pivot: list[float], angle_rad: float) -> list[float]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    x_coord = point[0] - pivot[0]
    y_coord = point[1] - pivot[1]
    return [
        pivot[0] + cosine * x_coord - sine * y_coord,
        pivot[1] + sine * x_coord + cosine * y_coord,
        point[2],
    ]


def _track(track_id: int, reference: list[float], points: list[list[float]]) -> dict:
    return {
        "track_id": track_id,
        "part_id": 1,
        "part_name": "object",
        "view_index": 0,
        "query_frame_index": 0,
        "query_uv": [float(track_id), 0.0],
        "reference_xyz_world": reference,
        "samples": [
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index * 0.1,
                "uv": [float(track_id), float(frame_index)],
                "xyz_world": point,
                "visible": True,
                "depth_valid": True,
                "mask_consistent": True,
                "confidence": 1.0,
            }
            for frame_index, point in enumerate(points)
        ],
    }


class MotionSegmentationTests(unittest.TestCase):
    def test_parser_accepts_segment_motion_parts(self) -> None:
        args = build_parser().parse_args(
            [
                "segment-motion-parts",
                "part_tracks.json",
                "--output-json",
                "motion_part_tracks.json",
                "--rigidity-threshold-m",
                "0.02",
                "--max-neighbor-distance-m",
                "0.5",
                "--min-common-frames",
                "3",
                "--mode",
                "knn-spectral",
                "--k-min",
                "2",
                "--k-max",
                "4",
                "--edge-ablation",
                "all",
                "--quality-weighted-affinity",
                "--quality-weighted-affinity-time-only",
                "--quality-weighted-affinity-edge-prior",
                "--quality-affinity-min-pair-weight",
                "0.25",
                "--articulation-compatible-affinity",
            ]
        )
        self.assertEqual(args.command, "segment-motion-parts")
        self.assertEqual(args.rigidity_threshold_m, 0.02)
        self.assertEqual(args.max_neighbor_distance_m, 0.5)
        self.assertEqual(args.min_common_frames, 3)
        self.assertEqual(args.mode, "knn-spectral")
        self.assertEqual(args.k_min, 2)
        self.assertEqual(args.k_max, 4)
        self.assertEqual(args.edge_ablation, "all")
        self.assertTrue(args.quality_weighted_affinity)
        self.assertTrue(args.quality_weighted_affinity_time_only)
        self.assertTrue(args.quality_weighted_affinity_edge_prior)
        self.assertAlmostEqual(args.quality_affinity_min_pair_weight, 0.25)
        self.assertTrue(args.articulation_compatible_affinity)

    def test_quality_weighted_pair_rigidity_downweights_bad_timestep(self) -> None:
        a_by_frame = {
            0: [0.0, 0.0, 0.0],
            1: [0.1, 0.0, 0.0],
            2: [0.2, 0.0, 0.0],
        }
        b_by_frame = {
            0: [1.0, 0.0, 0.0],
            1: [5.0, 0.0, 0.0],
            2: [1.2, 0.0, 0.0],
        }
        unweighted = _pair_metrics(a_by_frame, b_by_frame, 3)
        weighted = _pair_metrics(
            a_by_frame,
            b_by_frame,
            3,
            {0: 1.0, 1: 0.05, 2: 1.0},
            {0: 1.0, 1: 0.05, 2: 1.0},
            1.0,
        )
        assert unweighted is not None
        assert weighted is not None
        self.assertLess(weighted["rigidity_rmse_m"], unweighted["rigidity_rmse_m"])
        self.assertLess(weighted["effective_pair_observation_ratio"], 1.0)

    def test_quality_weighted_pair_edge_prior_clamps_weight(self) -> None:
        a_by_frame = {0: [0.0, 0.0, 0.0], 1: [0.1, 0.0, 0.0], 2: [0.2, 0.0, 0.0]}
        b_by_frame = {0: [1.0, 0.0, 0.0], 1: [1.1, 0.0, 0.0], 2: [1.2, 0.0, 0.0]}
        metrics = _pair_metrics(
            a_by_frame,
            b_by_frame,
            3,
            {0: 1.0, 1: 1.0, 2: 1.0},
            {0: 1.0, 1: 1.0, 2: 1.0},
            0.25,
        )
        assert metrics is not None
        self.assertTrue(metrics["quality_edge_prior_enabled"])
        self.assertGreaterEqual(metrics["pair_quality_prior"], 0.25)

    def test_time_only_quality_weighted_affinity_keeps_valid_edge_count(self) -> None:
        tracks = [
            _track(0, [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            _track(1, [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ]
        for track in tracks:
            track["track_quality"] = {"track_quality_score": 1.0}
            for sample in track["samples"]:
                sample["timestep_quality_score"] = 1.0
        unweighted = MotionPartSegmenter(
            MotionPartSegmentationConfig(input_tracks="unused", min_common_frames=3)
        )
        time_only = MotionPartSegmenter(
            MotionPartSegmentationConfig(
                input_tracks="unused",
                min_common_frames=3,
                quality_weighted_affinity_time_only=True,
            )
        )
        _, unweighted_edges = unweighted._build_connected_graph(tracks)
        _, time_only_edges = time_only._build_connected_graph(tracks)
        self.assertEqual(len(time_only_edges), len(unweighted_edges))
        self.assertFalse(time_only_edges[0]["quality_edge_prior_enabled"])

    def test_articulation_compatible_affinity_adds_edge_diagnostic(self) -> None:
        tracks = [
            _track(0, [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            _track(1, [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ]
        for track in tracks:
            track["track_quality"] = {"track_quality_score": 1.0, "articulation_score": 0.5}
        segmenter = MotionPartSegmenter(
            MotionPartSegmentationConfig(
                input_tracks="unused",
                min_common_frames=3,
                articulation_compatible_affinity=True,
            )
        )
        _, edges = segmenter._build_connected_graph(tracks)
        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0]["articulation_affinity_enabled"])
        self.assertAlmostEqual(edges[0]["articulation_compatibility"], 0.5)

    def test_mixed_cluster_has_higher_articulation_residual_than_clean_cluster(self) -> None:
        base_tracks = [
            _track(idx, point, [point, point, point])
            for idx, point in enumerate([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
        ]
        moving_tracks = [
            _track(
                idx + 10,
                point,
                [
                    point,
                    [point[0] + 0.2, point[1], point[2]],
                    [point[0] + 0.4, point[1], point[2]],
                ],
            )
            for idx, point in enumerate([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, -0.2, 0.0]])
        ]
        tracks = base_tracks + moving_tracks
        clean = _cluster_diagnostics(1, [3, 4, 5], tracks, {})
        mixed = _cluster_diagnostics(1, [0, 1, 2, 3, 4, 5], tracks, {})
        self.assertLess(clean["cluster_articulation_rmse_m"], mixed["cluster_articulation_rmse_m"])

    def test_parser_accepts_local_split_motion_cluster(self) -> None:
        args = build_parser().parse_args(
            [
                "local-split-motion-cluster",
                "motion_part_tracks.json",
                "--evaluation-json",
                "eval.json",
                "--output-dir",
                "split",
                "--split-cluster-id",
                "4",
                "--local-k-min",
                "2",
                "--local-k-max",
                "3",
                "--edge-ablation",
                "all",
            ]
        )
        self.assertEqual(args.command, "local-split-motion-cluster")
        self.assertEqual(args.split_cluster_id, [4])
        self.assertEqual(args.local_k_min, 2)
        self.assertEqual(args.local_k_max, 3)
        self.assertEqual(args.edge_ablation, "all")

    def test_parser_accepts_object_mask_diagnostics_visualization(self) -> None:
        args = build_parser().parse_args(
            [
                "visualize-object-mask-diagnostics",
                "motion_part_tracks.json",
                "--output-dir",
                "viz",
                "--joint-inference",
                "joint_inference.json",
                "--evaluation-json",
                "eval.json",
                "--frame",
                "median",
                "--top-n-candidates",
                "3",
                "--axis-remap",
                "x,z,-y",
                "--flow-min-motion",
                "0.01",
                "--make-matplotlib",
                "--plot-projections",
                "xz",
                "--make-animation",
                "--animation-fps",
                "8",
                "--animation-max-tracks",
                "250",
                "--animation-color-by",
                "gt_part",
            ]
        )
        self.assertEqual(args.command, "visualize-object-mask-diagnostics")
        self.assertEqual(args.motion_tracks, Path("motion_part_tracks.json"))
        self.assertEqual(args.output_dir, Path("viz"))
        self.assertEqual(args.frame, "median")
        self.assertEqual(args.top_n_candidates, 3)
        self.assertEqual(args.axis_remap, "x,z,-y")
        self.assertAlmostEqual(args.flow_min_motion, 0.01)
        self.assertTrue(args.make_matplotlib)
        self.assertEqual(args.plot_projections, "xz")
        self.assertTrue(args.make_animation)
        self.assertEqual(args.animation_fps, 8)
        self.assertEqual(args.animation_max_tracks, 250)
        self.assertEqual(args.animation_color_by, "gt_part")

    def test_parser_accepts_object_mask_flow_html_visualization(self) -> None:
        args = build_parser().parse_args(
            [
                "visualize-object-mask-flow-html",
                "motion_part_tracks.json",
                "--output-html",
                "flow.html",
                "--joint-inference",
                "joint_inference.json",
                "--evaluation-json",
                "eval.json",
                "--max-tracks",
                "500",
                "--frame-stride",
                "3",
                "--trail-length",
                "8",
                "--axis-remap",
                "x,z,-y",
                "--color-by",
                "motion_magnitude",
            ]
        )
        self.assertEqual(args.command, "visualize-object-mask-flow-html")
        self.assertEqual(args.motion_tracks, Path("motion_part_tracks.json"))
        self.assertEqual(args.output_html, Path("flow.html"))
        self.assertEqual(args.max_tracks, 500)
        self.assertEqual(args.frame_stride, 3)
        self.assertEqual(args.trail_length, 8)
        self.assertEqual(args.axis_remap, "x,z,-y")
        self.assertEqual(args.color_by, "motion_magnitude")

    def test_motion_segmentation_relabels_object_tracks_into_rigid_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_points = [
                [-0.2, -0.1, 0.0],
                [-0.2, 0.2, 0.0],
                [0.1, -0.1, 0.0],
                [0.1, 0.2, 0.2],
            ]
            door_reference = [
                [1.0, -0.2, 0.0],
                [1.0, 0.2, 0.0],
                [1.3, -0.2, 0.0],
                [1.3, 0.2, 0.2],
            ]
            pivot = [0.8, 0.0, 0.0]
            angles = [0.0, math.radians(25.0), math.radians(50.0)]
            tracks = []
            for index, point in enumerate(base_points):
                tracks.append(_track(index, point, [point, point, point]))
            for index, point in enumerate(door_reference, start=10):
                tracks.append(_track(index, point, [_rotate_about_z(point, pivot, angle) for angle in angles]))

            input_path = root / "object_tracks.json"
            input_path.write_text(
                json.dumps(
                    {
                        "input_episode_path": str(root / "episode.json"),
                        "estimator": "cotracker-depth-backprojection",
                        "frame_count": 3,
                        "source_frame_count": 3,
                        "sampled_frame_indices": [0, 1, 2],
                        "part_segmentation": {
                            "source": "object-mask",
                            "parts": [{"part_id": 1, "name": "object", "role": "object"}],
                        },
                        "tracks": tracks,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = MotionPartSegmenter(
                MotionPartSegmentationConfig(
                    input_tracks=input_path,
                    rigidity_threshold_m=0.01,
                    max_neighbor_distance_m=0.75,
                    min_common_frames=3,
                    min_tracks_per_part=3,
                    static_motion_threshold_m=0.01,
                )
            ).segment()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["estimator"], "motion-rigidity-cotracker-clustering")
            self.assertEqual(payload["motion_segmentation"]["part_count"], 2)
            part_ids = sorted({int(track["part_id"]) for track in payload["tracks"]})
            self.assertEqual(part_ids, [1, 2])
            counts = {int(part_id): item["count"] for part_id, item in payload["part_track_counts"].items()}
            self.assertEqual(counts, {1: 4, 2: 4})
            self.assertEqual(payload["part_segmentation"]["parts"][0]["role"], "base")

            pose_path = TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(input_path=output_path, min_tracks_per_part=3)
            ).estimate()
            pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
            self.assertEqual(len(pose_payload["parts"]), 2)
            moving = next(part for part in pose_payload["parts"] if part["part_id"] == 2)
            self.assertGreater(moving["relative_motion_summary"]["rotation_angle_range_rad"], 0.5)

    def test_knn_spectral_sweeps_fixed_k_and_writes_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks = []
            for index, point in enumerate([[-0.2, 0.0, 0.0], [-0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]]):
                tracks.append(_track(index, point, [point, point, point]))
            for index, point in enumerate([[0.8, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, -0.1, 0.0]], start=10):
                tracks.append(
                    _track(
                        index,
                        point,
                        [
                            point,
                            [point[0] + 0.08, point[1], point[2]],
                            [point[0] + 0.16, point[1], point[2]],
                        ],
                    )
                )
            for index, point in enumerate([[0.0, 0.8, 0.0], [0.1, 0.9, 0.0], [-0.1, 0.9, 0.0]], start=20):
                tracks.append(
                    _track(
                        index,
                        point,
                        [
                            point,
                            [point[0], point[1] + 0.08, point[2]],
                            [point[0], point[1] + 0.16, point[2]],
                        ],
                    )
                )

            input_path = root / "object_tracks.json"
            input_path.write_text(
                json.dumps(
                    {
                        "input_episode_path": str(root / "episode.json"),
                        "estimator": "cotracker-depth-backprojection",
                        "frame_count": 3,
                        "source_frame_count": 3,
                        "sampled_frame_indices": [0, 1, 2],
                        "part_segmentation": {
                            "source": "object-mask",
                            "parts": [{"part_id": 1, "name": "object", "role": "object"}],
                        },
                        "tracks": tracks,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            diagnostics_path = root / "diagnostics.json"
            sweep_dir = root / "sweep"
            output_path = MotionPartSegmenter(
                MotionPartSegmentationConfig(
                    input_tracks=input_path,
                    mode="knn-spectral",
                    diagnostics_json=diagnostics_path,
                    sweep_output_dir=sweep_dir,
                    static_motion_threshold_m=0.01,
                    moving_motion_threshold_m=0.03,
                    knn_k=2,
                    k_min=2,
                    k_max=3,
                    spectral_k=2,
                    edge_ablation="all",
                )
            ).segment()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["motion_segmentation"]["mode"], "knn-spectral")
            self.assertEqual(payload["motion_segmentation"]["selected_k"], 2)
            self.assertEqual(payload["motion_segmentation"]["part_count"], 3)
            self.assertEqual(len(diagnostics["sweep"]), 6)
            self.assertTrue((sweep_dir / "motion_part_tracks_A_K2.json").exists())
            part_ids = sorted({int(track["part_id"]) for track in payload["tracks"]})
            self.assertEqual(part_ids, [1, 2, 3])

    def test_local_split_uses_evaluator_recommendation_and_writes_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks = []
            for index, point in enumerate([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], start=0):
                item = _track(index, point, [point, point, point])
                item["part_id"] = 1
                item["original_part_id"] = 1
                tracks.append(item)
            for index, point in enumerate([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.1, 0.0]], start=10):
                item = _track(index, point, [point, [point[0] + 0.2, point[1], point[2]], [point[0] + 0.4, point[1], point[2]]])
                item["part_id"] = 4
                item["original_part_id"] = 2
                tracks.append(item)
            for index, point in enumerate([[3.0, 0.0, 0.0], [3.1, 0.0, 0.0], [3.0, 0.1, 0.0]], start=20):
                item = _track(index, point, [point, [point[0], point[1] + 0.2, point[2]], [point[0], point[1] + 0.4, point[2]]])
                item["part_id"] = 4
                item["original_part_id"] = 3
                tracks.append(item)

            input_path = root / "motion_part_tracks.json"
            input_path.write_text(
                json.dumps(
                    {
                        "estimator": "motion-knn-spectral-cotracker-clustering",
                        "tracks": tracks,
                        "original_part_segmentation": {
                            "parts": [
                                {"part_id": 1, "name": "base", "role": "base"},
                                {"part_id": 2, "name": "door_a", "role": "door"},
                                {"part_id": 3, "name": "door_b", "role": "door"},
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_path = root / "eval.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "per_joint": [
                            {
                                "child_part_id": 4,
                                "failure_reason_guess": "child_cluster_mixed_or_nonrigid",
                                "cleanup_recommendation": "local_split_child_cluster_then_refit_part_pose",
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary_path = LocalMotionClusterSplitter(
                LocalSplitDiagnosticsConfig(
                    input_tracks=input_path,
                    evaluation_json=eval_path,
                    output_dir=root / "split",
                    local_k_min=2,
                    local_k_max=2,
                    knn_k=2,
                    edge_ablation="B",
                )
            ).split()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["target_cluster_ids"], [4])
            self.assertEqual(summary["candidate_count"], 1)
            candidate_path = Path(summary["candidates"][0]["output_json"])
            self.assertTrue(candidate_path.exists())
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            part_ids = sorted({int(track["part_id"]) for track in candidate["tracks"]})
            self.assertEqual(part_ids, [1, 4, 5])
            affected = summary["candidates"][0]["after"]["clusters"]
            self.assertEqual(sorted(item["track_count"] for item in affected), [3, 3])
            self.assertIn("best_cluster_by_no_gt_score", summary["candidates"][0]["after"])
            for cluster in affected:
                self.assertIn("bbox_diag_m", cluster)
                self.assertIn("visible_frame_ratio", cluster)
                self.assertIn("candidate_selection", cluster)
                self.assertIn("selection_score_no_gt", cluster["candidate_selection"])
                self.assertFalse(cluster["candidate_selection"]["selection_score_uses_gt"])

    def test_object_mask_diagnostics_visualization_writes_ply_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks_path = root / "motion_part_tracks.json"
            joint_path = root / "joint_inference.json"
            eval_path = root / "object_mask_eval.json"
            tracks = []
            for index, point in enumerate([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]):
                item = _track(index, point, [point, point])
                item["part_id"] = 1
                item["original_part_id"] = 1
                tracks.append(item)
            for index, point in enumerate([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.1, 0.0]], start=10):
                item = _track(index, point, [point, [point[0] + 0.2, point[1], point[2]]])
                item["part_id"] = 2
                item["original_part_id"] = 2
                tracks.append(item)
            tracks_path.write_text(json.dumps({"tracks": tracks}) + "\n", encoding="utf-8")
            joint_path.write_text(
                json.dumps(
                    {
                        "joints": [
                            {
                                "name": "door_joint",
                                "parent_part_id": 1,
                                "child_part_id": 2,
                                "axis": [0.0, 0.0, 1.0],
                                "pivot": [0.9, 0.0, 0.0],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps(
                    {
                        "overlap": {
                            "per_cluster": [
                                {"pred_cluster_id": 1, "dominant_gt_part_id": 1},
                                {"pred_cluster_id": 2, "dominant_gt_part_id": 2},
                            ]
                        },
                        "per_joint": [
                            {
                                "child_part_id": 2,
                                "ground_truth_axis": [0.0, 0.0, 1.0],
                                "ground_truth_pivot": [1.0, 0.0, 0.0],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = ObjectMaskDiagnosticsVisualizer().build(
                ObjectMaskDiagnosticsVisualizationConfig(
                    motion_tracks=tracks_path,
                    output_dir=root / "viz",
                    joint_inference=joint_path,
                    evaluation_json=eval_path,
                    make_matplotlib=True,
                    plot_projections="xz",
                )
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(Path(payload["outputs"]["clusters_pred"]).exists())
            self.assertTrue(Path(payload["outputs"]["clusters_gt"]).exists())
            self.assertTrue(Path(payload["outputs"]["flow_arrows_pred_cluster"]).exists())
            if "overview_xz" in payload["outputs"]:
                self.assertTrue(Path(payload["outputs"]["overview_xz"]).exists())
            self.assertTrue(Path(payload["outputs"]["joints_pred_vs_gt"]).exists())
            joint_ply = Path(payload["outputs"]["joints_pred_vs_gt"]).read_text(encoding="utf-8")
            self.assertIn("element edge", joint_ply)
            self.assertIn("axis_remap", payload)

    def test_object_mask_flow_html_writes_interactive_slider_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks_path = root / "motion_part_tracks.json"
            tracks_path.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {**_track(1, [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]), "part_id": 2, "original_part_id": 10},
                            {**_track(2, [0.0, 0.2, 0.0], [[0.0, 0.2, 0.0], [0.0, 0.3, 0.0], [0.0, 0.4, 0.0]]), "part_id": 3, "original_part_id": 11},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            html_path = ObjectMaskFlowHtmlBuilder().build(
                ObjectMaskFlowHtmlConfig(
                    motion_tracks=tracks_path,
                    output_html=root / "flow.html",
                    max_tracks=2,
                    frame_stride=1,
                    trail_length=2,
                    color_by="gt_part",
                )
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Object-Mask Flow Viewer", html)
            self.assertIn("trackCountSlider", html)
            self.assertIn("frameSlider", html)
            self.assertIn("trailSlider", html)
            self.assertIn("Plotly.react", html)
            self.assertIn("function jointColor", html)
            self.assertIn("function clippedAxisSegment", html)
            self.assertIn("Track quality", html)
            self.assertIn("hideLowQualityTimesteps", html)
            self.assertIn("...buildJointTraces(activeTracks, colorBy)", html)
            self.assertIn("\"track_count_embedded\":2", html)
            self.assertIn("\"default_color_by\":\"gt_part\"", html)

    def test_object_mask_flow_html_defaults_to_sibling_viewers_dir_for_tracks_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks_dir = root / "tracks"
            tracks_dir.mkdir()
            tracks_path = tracks_dir / "motion_part_tracks.json"
            tracks_path.write_text(
                json.dumps({"tracks": [_track(1, [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])]}) + "\n",
                encoding="utf-8",
            )
            html_path = ObjectMaskFlowHtmlBuilder().build(ObjectMaskFlowHtmlConfig(motion_tracks=tracks_path))
            self.assertEqual(html_path, (root / "viewers" / "object_mask_flow_viewer.html").resolve())
            self.assertTrue(html_path.exists())


if __name__ == "__main__":
    unittest.main()
