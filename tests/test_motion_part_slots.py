from __future__ import annotations

import unittest
import random

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.motion_part_slots import (
    MotionPartSlotInferenceConfig,
    _build_slot_model,
    _filter_inference_tracks,
    _hungarian_slot_targets,
    _matched_slot_dice_loss,
    _part_balanced_cross_entropy,
    _sample_from_artifact,
    _topology_balanced_object_order,
    _training_track_indices,
    _weighted_rigid_replay_loss,
    evaluate_slot_assignments,
)


class MotionPartSlotTests(unittest.TestCase):
    def test_inference_quality_filter_removes_short_and_jumping_tracks(self) -> None:
        def track(track_id: int, points: list[list[float]]) -> dict:
            return {
                "track_id": track_id,
                "samples": [
                    {"frame_index": index, "xyz_world": point, "visible": True}
                    for index, point in enumerate(points)
                ],
            }

        artifact = {
            "frame_count": 5,
            "tracks": [
                track(1, [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]]),
                track(2, [[0, 0, 0], [0.01, 0, 0]]),
                track(3, [[0, 0, 0], [0.5, 0, 0], [0.51, 0, 0], [0.52, 0, 0]]),
            ],
        }
        config = MotionPartSlotInferenceConfig(
            tracks_path="tracks.json", features_npz="features.npz",
            model_path="model.pt", output_json="output.json",
            min_visible_frames=3, min_visible_ratio=0.5, max_trajectory_jump_m=0.1,
        )
        filtered, report = _filter_inference_tracks(artifact, config)
        self.assertEqual([row["track_id"] for row in filtered["tracks"]], [1])
        self.assertEqual(report["dropped_by_reason"]["too_few_visible_frames"], 1)
        self.assertEqual(report["dropped_by_reason"]["trajectory_jump"], 1)

    def test_training_sample_accepts_simulator_part_id_as_label(self) -> None:
        from rgbd_urdf_mvp.perception.motion_part_slots import _sample_from_artifact

        def track(track_id: int, part_id: int, offset: float) -> dict:
            return {
                "track_id": track_id,
                "part_id": part_id,
                "reference_xyz_world": [offset, 0.0, 0.0],
                "samples": [
                    {"frame_index": 0, "xyz_world": [offset, 0.0, 0.0], "visible": True},
                    {"frame_index": 1, "xyz_world": [offset, 0.1, 0.0], "visible": True},
                ],
            }

        artifact = {
            "tracks": [
                track(1, 7, 0.0),
                track(2, 9, 0.2),
            ]
        }
        sample = _sample_from_artifact(
            artifact,
            {1: np.ones(4), 2: -np.ones(4)},
            object_id="sim-object",
            require_labels=True,
        )
        self.assertEqual(sample["labels"].tolist(), [0, 1])

    def test_training_sample_collapses_fixed_connected_labels(self) -> None:
        from rgbd_urdf_mvp.perception.motion_part_slots import _sample_from_artifact

        def track(track_id: int, part_id: int, offset: float) -> dict:
            return {
                "track_id": track_id,
                "part_id": part_id,
                "reference_xyz_world": [offset, 0.0, 0.0],
                "samples": [
                    {"frame_index": 0, "xyz_world": [offset, 0.0, 0.0], "visible": True},
                    {"frame_index": 1, "xyz_world": [offset, 0.1, 0.0], "visible": True},
                ],
            }

        artifact = {
            "tracks": [track(1, 7, 0.0), track(2, 9, 0.2), track(3, 11, 0.4)],
            "original_part_segmentation": {
                "parts": [
                    {"part_id": 7, "body_id": 1, "body_name": "base", "role": "base"},
                    {
                        "part_id": 9,
                        "body_id": 2,
                        "body_name": "fixed_trim",
                        "role": "fixed_child",
                        "parent_body_id": 1,
                    },
                    {"part_id": 11, "body_id": 3, "body_name": "door", "role": "articulated"},
                ]
            },
        }
        sample = _sample_from_artifact(
            artifact,
            {1: np.ones(4), 2: -np.ones(4), 3: np.full(4, 2.0)},
            object_id="sim-object",
            require_labels=True,
            collapse_fixed_connected_labels=True,
        )
        self.assertEqual(sample["labels"].tolist(), [0, 0, 1])
        self.assertEqual(sample["label_ontology"], "maximal-fixed-joint-connected-components")
        self.assertEqual(sample["raw_part_to_label"], {7: 0, 9: 0, 11: 1})

    def test_parser_accepts_slot_training_and_post_ransac_inference(self) -> None:
        parser = build_parser()
        training = parser.parse_args(
            [
                "train-motion-part-slots",
                "manifest.tsv",
                "--output-dir",
                "model",
                "--object-batch-size",
                "4",
                "--collapse-fixed-connected-labels",
            ]
        )
        inference = parser.parse_args(
            [
                "infer-motion-part-slots",
                "tracks.json",
                "features.npz",
                "model.pt",
                "--output-json",
                "slots.json",
                "--post-ransac-refine",
                "--slot-existence-threshold",
                "0.6",
            ]
        )
        self.assertEqual(training.command, "train-motion-part-slots")
        self.assertEqual(training.object_batch_size, 4)
        self.assertTrue(training.collapse_fixed_connected_labels)
        self.assertEqual(training.slot_feature_schema, "legacy_v1")
        self.assertEqual(inference.command, "infer-motion-part-slots")
        self.assertTrue(inference.post_ransac_refine)
        self.assertAlmostEqual(inference.slot_existence_threshold, 0.6)

    def test_quality_temporal_schema_appends_reliability_features(self) -> None:
        track = {
            "track_id": 1,
            "part_id": 3,
            "reference_xyz_world": [0.0, 0.0, 0.0],
            "track_quality_score": 0.8,
            "samples": [
                {"frame_index": 1, "xyz_world": [0.0, 0.0, 0.0], "visible": True,
                 "timestep_quality_score": 0.9},
                {"frame_index": 2, "xyz_world": [0.1, 0.0, 0.0], "visible": True,
                 "timestep_quality_score": 0.7},
                {"frame_index": 5, "xyz_world": [0.2, 0.0, 0.0], "visible": True,
                 "timestep_quality_score": 0.5},
            ],
        }
        artifact = {"tracks": [track]}
        legacy = _sample_from_artifact(
            artifact, {1: np.ones(4)}, object_id="quality", require_labels=True,
        )
        enriched = _sample_from_artifact(
            artifact, {1: np.ones(4)}, object_id="quality", require_labels=True,
            feature_schema="quality_temporal_v2",
        )
        self.assertEqual(enriched["features"].shape[1], legacy["features"].shape[1] + 8)
        self.assertEqual(enriched["feature_schema"], "quality_temporal_v2")
        self.assertTrue(np.all(np.isfinite(enriched["features"])))
        quality_features = enriched["features"][0, -8:]
        self.assertGreater(float(quality_features[0]), 0.0)
        self.assertGreater(float(quality_features[4]), 0.0)

    def test_slot_model_supports_padded_object_batches(self) -> None:
        import torch

        model = _build_slot_model(
            torch, input_dim=12, hidden_dim=16, max_slots=4,
            encoder_layers=1, decoder_layers=1, attention_heads=4,
        )
        features = torch.randn(2, 7, 12)
        padding = torch.tensor(
            [[False] * 7, [False] * 4 + [True] * 3], dtype=torch.bool
        )
        logits, existence = model(features, padding_mask=padding)
        self.assertEqual(tuple(logits.shape), (2, 7, 4))
        self.assertEqual(tuple(existence.shape), (2, 4))

    def test_batched_rigid_replay_is_zero_for_shared_translation(self) -> None:
        import torch

        references = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        points = np.stack([references, references + [0.2, -0.1, 0.0], references + [0.4, -0.2, 0.0]], axis=1)
        probabilities = torch.full((4, 2), 0.5, requires_grad=True)
        loss = _weighted_rigid_replay_loss(
            probabilities, references, points, np.ones((4, 3), dtype=np.float32), torch, "cpu"
        )
        self.assertLess(float(loss.detach()), 1e-6)
        loss.backward()
        self.assertIsNotNone(probabilities.grad)
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_batched_rigid_replay_has_finite_gradient_for_degenerate_points(self) -> None:
        import torch

        references = np.zeros((6, 3), dtype=np.float32)
        points = np.zeros((6, 4, 3), dtype=np.float32)
        probabilities = torch.full((6, 3), 1.0 / 3.0, requires_grad=True)
        loss = _weighted_rigid_replay_loss(
            probabilities, references, points, np.ones((6, 4), dtype=np.float32), torch, "cpu"
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_hungarian_rejects_nonfinite_logits(self) -> None:
        import torch

        logits = torch.tensor([[0.0, float("nan")], [1.0, 0.0]])
        labels = np.asarray([0, 1], dtype=np.int64)
        with self.assertRaisesRegex(FloatingPointError, "non-finite logits"):
            _hungarian_slot_targets(logits, labels, 2, torch)

    def test_hungarian_targets_are_permutation_invariant(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 4.0, 0.0],
                [5.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        labels = np.asarray([10, 10, 20, 20], dtype=np.int64)
        targets = _hungarian_slot_targets(logits, labels, 3, torch)
        self.assertEqual(targets[:2], [1, 1])
        self.assertEqual(targets[2:], [0, 0])

    def test_one_to_one_metrics_expose_merged_gt_parts(self) -> None:
        metrics = evaluate_slot_assignments(
            np.asarray([0, 0, 0, 0, 1, 2]),
            np.asarray([0, 0, 1, 1, 2, 2]),
        )
        self.assertEqual(metrics["gt_collision_count"], 1)
        self.assertLess(metrics["one_to_one_mean_recall"], 1.0)
        self.assertLess(metrics["pairwise_same_part_precision"], 1.0)
        self.assertLess(metrics["mean_cluster_purity"], 1.0)
        self.assertLess(metrics["mean_gt_coverage"], 1.0)
        # Exact cluster count can coexist with one merged GT part and one split GT part.
        self.assertTrue(metrics["part_count_exact"])
        self.assertIn("adjusted_rand_index", metrics)
        self.assertIn("rand_index", metrics)
        self.assertIn("normalized_mutual_information", metrics)

    def test_rand_index_counts_agreeing_same_and_different_pairs(self) -> None:
        metrics = evaluate_slot_assignments(
            np.asarray([0, 0, 0, 0]),
            np.asarray([0, 0, 1, 1]),
        )
        self.assertAlmostEqual(metrics["rand_index"], 1.0 / 3.0)

    def test_nmi_reports_constant_prediction_as_uninformative(self) -> None:
        metrics = evaluate_slot_assignments(
            np.asarray([0, 0, 0, 0]),
            np.asarray([0, 0, 1, 1]),
        )
        self.assertEqual(metrics["normalized_mutual_information"], 0.0)

    def test_matching_scales_to_many_parts(self) -> None:
        labels = np.repeat(np.arange(12, dtype=np.int64), 2)
        permutation = np.asarray([7, 2, 10, 0, 11, 5, 1, 9, 4, 8, 3, 6])
        predicted = permutation[labels]
        metrics = evaluate_slot_assignments(predicted, labels)
        self.assertAlmostEqual(metrics["one_to_one_mean_iou"], 1.0)
        self.assertAlmostEqual(metrics["adjusted_rand_index"], 1.0)

    def test_balanced_assignment_and_dice_are_finite(self) -> None:
        import torch

        logits = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 3.0]], requires_grad=True)
        target = torch.tensor([0, 0, 1])
        balanced = _part_balanced_cross_entropy(logits, target, torch, enabled=True)
        dice = _matched_slot_dice_loss(torch.softmax(logits, -1), target, torch)
        (balanced + dice).backward()
        self.assertTrue(torch.isfinite(balanced))
        self.assertTrue(torch.isfinite(dice))
        self.assertIsNotNone(logits.grad)

    def test_topology_balancing_repeats_complex_objects(self) -> None:
        samples = [
            {"labels": np.asarray([0, 1, 2])},
            {"labels": np.asarray([0, 1, 2, 3, 4, 5])},
        ]
        self.assertEqual(_topology_balanced_object_order(samples, enabled=True), [0, 1, 1])

    def test_two_view_dropout_never_keeps_only_minimum_motion_view(self) -> None:
        tracks = []
        labels = []
        views = []
        for view, displacement in enumerate((0.01, 0.10, 0.20)):
            for label in (0, 1):
                for _ in range(3):
                    tracks.append({
                        "samples": [
                            {"frame_index": 0, "xyz_world": [0.0, 0.0, 0.0], "visible": True},
                            {"frame_index": 1, "xyz_world": [displacement, 0.0, 0.0], "visible": True},
                        ]
                    })
                    labels.append(label)
                    views.append(view)
        sample = {
            "tracks": tracks,
            "labels": np.asarray(labels, dtype=np.int64),
            "view_indices": np.asarray(views, dtype=np.int64),
        }
        for seed in range(20):
            selected = _training_track_indices(
                sample, 0.0, random.Random(seed), np,
                view_dropout_probability=1.0, max_dropped_views=2,
            )
            kept_views = set(sample["view_indices"][selected].tolist())
            if len(kept_views) == 1:
                self.assertNotEqual(kept_views, {0})
            self.assertEqual(set(sample["labels"][selected].tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
