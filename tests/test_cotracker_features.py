from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.cotracker_features import CoTrackerFeatureProbeConfig, CoTrackerFeatureProber
from rgbd_urdf_mvp.perception.motion_segmentation import (
    MotionPartSegmentationConfig,
    MotionPartSegmenter,
    _edge_summary,
)
from rgbd_urdf_mvp.perception.pairwise_affinity import (
    PairwiseAffinityEvaluator,
    PairwiseAffinityEvaluationConfig,
    PairwiseAffinityTrainer,
    PairwiseAffinityTrainingConfig,
    pair_feature_vector,
)


class CoTrackerFeatureTests(unittest.TestCase):
    def test_cli_accepts_feature_export_probe_and_soft_affinity(self) -> None:
        parser = build_parser()
        tracking = parser.parse_args(
            [
                "track-part-pixels",
                "episode.json",
                "--export-cotracker-features",
                "--cotracker-features-output",
                "features.npz",
            ]
        )
        self.assertTrue(tracking.export_cotracker_features)
        self.assertEqual(tracking.cotracker_features_output, Path("features.npz"))

        probe = parser.parse_args(
            [
                "probe-cotracker-features",
                "tracks.json",
                "features.npz",
                "--cluster-k",
                "3",
            ]
        )
        self.assertEqual(probe.command, "probe-cotracker-features")
        self.assertEqual(probe.cluster_k, 3)

        segmentation = parser.parse_args(
            [
                "segment-motion-parts",
                "tracks.json",
                "--cotracker-features-npz",
                "features.npz",
                "--learned-affinity-floor",
                "0.8",
            ]
        )
        self.assertEqual(segmentation.cotracker_features_npz, Path("features.npz"))
        self.assertAlmostEqual(segmentation.learned_affinity_floor, 0.8)

        training = parser.parse_args(
            [
                "train-pairwise-affinity",
                "manifest.tsv",
                "--output-dir",
                "model",
                "--epochs",
                "4",
            ]
        )
        self.assertEqual(training.command, "train-pairwise-affinity")
        self.assertEqual(training.epochs, 4)

    def test_probe_separates_synthetic_part_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rng = np.random.default_rng(4)
            labels = np.repeat(np.asarray([1, 2, 3], dtype=np.int64), 12)
            centers = np.eye(3, 8, dtype=np.float64)
            embeddings = np.stack(
                [centers[label - 1] + 0.015 * rng.normal(size=8) for label in labels]
            ).astype(np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            track_ids = np.arange(labels.size, dtype=np.int64)
            tracks_path = root / "tracks.json"
            tracks_path.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {"track_id": int(track_id), "original_part_id": int(label)}
                            for track_id, label in zip(track_ids, labels)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            features_path = root / "features.npz"
            np.savez_compressed(features_path, track_ids=track_ids, embeddings=embeddings)
            output_path = CoTrackerFeatureProber(
                CoTrackerFeatureProbeConfig(
                    tracks_path=tracks_path,
                    features_npz=features_path,
                    output_embedding_csv=root / "embedding.csv",
                    cluster_k=3,
                )
            ).probe()
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertGreater(result["pairwise"]["same_vs_cross_auc"], 0.99)
            self.assertGreater(result["hidden_only_clustering"]["adjusted_rand_index"], 0.99)
            self.assertGreater(result["hidden_only_clustering"]["purity"], 0.99)
            self.assertTrue((root / "embedding.csv").exists())

    def test_learned_affinity_is_soft_and_missing_features_are_neutral(self) -> None:
        segmenter = MotionPartSegmenter(
            MotionPartSegmentationConfig(
                input_tracks="tracks.json",
                learned_affinity_floor=0.7,
            )
        )
        segmenter._learned_feature_map = {
            1: np.asarray([1.0, 0.0]),
            2: np.asarray([-1.0, 0.0]),
        }
        metrics: dict = {}
        segmenter._add_learned_feature_metrics(metrics, {"track_id": 1}, {"track_id": 2})
        self.assertAlmostEqual(metrics["learned_affinity_multiplier"], 0.7)

        missing_metrics: dict = {}
        segmenter._add_learned_feature_metrics(missing_metrics, {"track_id": 1}, {"track_id": 99})
        self.assertAlmostEqual(missing_metrics["learned_affinity_multiplier"], 1.0)

    def test_edge_summary_reports_learned_feature_strength(self) -> None:
        summary = _edge_summary(
            [
                {
                    "i": 0,
                    "j": 1,
                    "weight": 0.4,
                    "learned_feature_pair_available": True,
                    "learned_feature_cosine": 0.6,
                    "learned_affinity_multiplier": 0.9,
                }
            ],
            [{"original_part_id": 1}, {"original_part_id": 1}],
            {},
        )
        self.assertEqual(summary["learned_feature_pair_count"], 1)
        self.assertAlmostEqual(summary["learned_feature_pair_coverage"], 1.0)
        self.assertAlmostEqual(summary["mean_learned_feature_cosine"], 0.6)
        self.assertAlmostEqual(summary["mean_learned_affinity_multiplier"], 0.9)
        self.assertAlmostEqual(summary["mean_same_gt_learned_feature_cosine"], 0.6)
        self.assertIsNone(summary["mean_cross_gt_learned_feature_cosine"])
        self.assertIsNone(summary["mean_same_gt_pairwise_probability"])

    def test_pair_feature_vector_is_order_invariant(self) -> None:
        metrics = {
            "rigidity_rmse_m": 0.01,
            "motion_disagreement_m": 0.02,
            "displacement_curve_rmse_m": 0.01,
            "velocity_cosine": 0.8,
            "common_frames": 5,
        }
        track_a = _synthetic_track(1, 1, [0.0, 0.0, 0.0], 0.1)
        track_b = _synthetic_track(2, 1, [0.1, 0.0, 0.0], 0.2)
        a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        forward = pair_feature_vector(a, b, track_a, track_b, metrics, reference_distance_m=0.1)
        reverse = pair_feature_vector(b, a, track_b, track_a, metrics, reference_distance_m=0.1)
        np.testing.assert_allclose(forward, reverse)

    def test_pairwise_affinity_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for object_index, split in enumerate(["train", "val"]):
                tracks = []
                embeddings = []
                for part_id in [1, 2]:
                    for local_index in range(6):
                        track_id = (object_index * 100) + (part_id * 10) + local_index
                        tracks.append(
                            _synthetic_track(
                                track_id,
                                part_id,
                                [0.04 * local_index, 0.08 * part_id, 0.0],
                                0.02 if part_id == 1 else 0.18,
                            )
                        )
                        center = np.asarray([1.0, 0.0, 0.0, 0.0] if part_id == 1 else [0.0, 1.0, 0.0, 0.0])
                        embeddings.append(center + 0.01 * local_index)
                tracks_path = root / f"tracks_{object_index}.json"
                tracks_path.write_text(json.dumps({"tracks": tracks}), encoding="utf-8")
                features_path = root / f"features_{object_index}.npz"
                np.savez_compressed(
                    features_path,
                    track_ids=np.asarray([track["track_id"] for track in tracks], dtype=np.int64),
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                )
                rows.append(f"object_{object_index}\t{tracks_path}\t{features_path}\t{split}")
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "object_id\ttracks_path\tfeatures_npz\tsplit\n" + "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            model_path = PairwiseAffinityTrainer(
                PairwiseAffinityTrainingConfig(
                    manifest_path=manifest,
                    output_dir=root / "model",
                    knn_k=8,
                    epochs=3,
                    batch_size=32,
                    hidden_dim=32,
                    device="cpu",
                )
            ).train()
            self.assertTrue(model_path.exists())
            evaluation_path = PairwiseAffinityEvaluator(
                PairwiseAffinityEvaluationConfig(
                    manifest_path=manifest,
                    model_path=model_path,
                    output_json=root / "evaluation.json",
                    split="val",
                    knn_k=8,
                    device="cpu",
                )
            ).evaluate()
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(evaluation["metrics"]["auc"], 0.5)


def _synthetic_track(track_id: int, part_id: int, reference: list[float], displacement: float) -> dict:
    return {
        "track_id": track_id,
        "part_id": 1,
        "original_part_id": part_id,
        "reference_xyz_world": reference,
        "samples": [
            {
                "frame_index": frame_index,
                "xyz_world": [reference[0] + displacement * frame_index, reference[1], reference[2]],
                "visible": True,
                "depth_valid": True,
            }
            for frame_index in range(5)
        ],
    }


if __name__ == "__main__":
    unittest.main()
