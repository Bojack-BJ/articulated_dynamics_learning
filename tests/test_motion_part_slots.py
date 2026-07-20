from __future__ import annotations

import unittest

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.motion_part_slots import (
    _hungarian_slot_targets,
    _matched_slot_dice_loss,
    _part_balanced_cross_entropy,
    _topology_balanced_object_order,
    evaluate_slot_assignments,
)


class MotionPartSlotTests(unittest.TestCase):
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

    def test_parser_accepts_slot_training_and_post_ransac_inference(self) -> None:
        parser = build_parser()
        training = parser.parse_args(["train-motion-part-slots", "manifest.tsv", "--output-dir", "model"])
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
        self.assertEqual(inference.command, "infer-motion-part-slots")
        self.assertTrue(inference.post_ransac_refine)
        self.assertAlmostEqual(inference.slot_existence_threshold, 0.6)

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
        self.assertIn("normalized_mutual_information", metrics)

    def test_nmi_reports_constant_prediction_as_uninformative(self) -> None:
        metrics = evaluate_slot_assignments(
            np.asarray([0, 0, 0, 0]),
            np.asarray([0, 0, 1, 1]),
        )
        self.assertEqual(metrics["normalized_mutual_information"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
