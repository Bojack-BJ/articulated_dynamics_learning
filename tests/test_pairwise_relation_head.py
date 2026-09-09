from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli_parser import build_parser
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    extract_manual_gt_relations,
    JOINT_TYPES,
    SlotRelationTrainingConfig,
    _augment_relation_sample,
    _augment_temporal_occlusion,
    _augment_trajectory_corruption,
    _augment_slot_probabilities,
    _apply_slot_geometry_representation,
    _build_relation_model,
    _configure_slot_trainability,
    decode_legal_relation_tree,
    _joint_model_replay_loss,
    _permutation_aligned_slot_consistency_loss,
    _relation_targets,
    _target_axis_excitation,
    _undirected_axis_equivariance_loss,
    _relation_batch_loss,
    _slot_motion_invariants,
    _structured_parent_selection_loss,
    _trajectory_tensors,
    extract_gt_relations,
    graph_legality_diagnostics,
)
from rgbd_urdf_mvp.perception.motion_part_slots import _build_slot_model


class PairwiseRelationHeadTests(unittest.TestCase):
    def test_manual_gt_relations_are_canonicalized(self) -> None:
        annotation = {
            "annotation_source": "manual-html-viewer",
            "joints": [{
                "name": "door",
                "joint_type": "revolute",
                "parent_part_id": 1,
                "child_part_id": 2,
                "axis": [0.0, 0.0, 2.0],
                "pivot": [3.0, 4.0, 5.0],
            }],
        }
        sample = {
            "raw_part_to_label": {1: 0, 2: 1},
            "canonical_center_m": [1.0, 2.0, 3.0],
            "canonical_scale_m": 2.0,
        }
        relations = extract_manual_gt_relations(annotation, sample)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["parent_label"], 0)
        self.assertEqual(relations[0]["child_label"], 1)
        self.assertEqual(relations[0]["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(relations[0]["pivot"], [1.0, 1.0, 1.0])

    def test_slot_contamination_preserves_probability_simplex(self) -> None:
        import random
        import torch

        random.seed(5)
        probabilities = torch.tensor([
            [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8],
        ])
        config = SlotRelationTrainingConfig(
            manifest_path="m", slot_model_path="s", output_dir="o",
            slot_contamination_probability=1.0,
            slot_contamination_fraction=0.5,
        )
        result = _augment_slot_probabilities(probabilities, torch, config)
        self.assertFalse(torch.allclose(result, probabilities))
        self.assertTrue(torch.allclose(result.sum(dim=-1), torch.ones(4)))

    def test_slot_contamination_supports_backpropagation(self) -> None:
        import random
        import torch

        random.seed(5)
        logits = torch.randn(2, 4, 3, requires_grad=True)
        probabilities = torch.softmax(logits, dim=-1)
        config = SlotRelationTrainingConfig(
            manifest_path="m", slot_model_path="s", output_dir="o",
            slot_contamination_probability=1.0,
            slot_contamination_fraction=0.5,
        )
        result = _augment_slot_probabilities(probabilities, torch, config)
        (result.square().sum()).backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_axis_excitation_uses_full_path_and_rejects_static_child(self) -> None:
        import numpy as np
        import torch

        points = np.zeros((4, 5, 3), dtype=np.float32)
        points[2:, :, 0] = np.asarray([0.0, 0.2, 0.4, 0.2, 0.0])[None, :]
        sample = {
            "points": points,
            "references": np.zeros((4, 3), dtype=np.float32),
            "visibility": np.ones((4, 5), dtype=bool),
            "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
        }
        positive = torch.zeros((2, 2), dtype=torch.bool)
        positive[0, 1] = True
        target = {"relations": [{
            "parent_slot": 0, "child_slot": 1, "child_label": 1,
        }]}
        excitation = _target_axis_excitation(
            sample, target, positive, torch, "cpu"
        )
        self.assertGreater(float(excitation[0]), 0.3)

        sample["labels"] = np.asarray([1, 1, 0, 0], dtype=np.int64)
        static = _target_axis_excitation(sample, target, positive, torch, "cpu")
        self.assertAlmostEqual(float(static[0]), 0.0, places=6)

    def test_trajectory_corruption_changes_only_selected_motion_evidence(self) -> None:
        import random
        import numpy as np

        sample = {
            "points": np.zeros((4, 8, 3), dtype=np.float32),
            "visibility": np.ones((4, 8), dtype=bool),
            "canonical_scale_m": 2.0,
            "gt_relations": [{"joint_type": "prismatic"}],
        }
        result = _augment_trajectory_corruption(
            sample, rng=random.Random(7), np=np, track_fraction=0.5,
            drift_scale_fraction=0.05, spike_probability=1.0,
        )
        self.assertIsNot(result, sample)
        self.assertEqual(result["_trajectory_corruption"]["selected_tracks"], 2)
        self.assertTrue(np.any(result["points"] != sample["points"]))
        np.testing.assert_array_equal(result["visibility"], sample["visibility"])
        self.assertEqual(result["gt_relations"], sample["gt_relations"])

    def test_structured_corruption_adds_recovery_and_id_switch(self) -> None:
        import random
        import numpy as np

        points = np.zeros((4, 8, 3), dtype=np.float32)
        points[1, :, 0] = 1.0
        sample = {
            "points": points,
            "references": points[:, 0].copy(),
            "visibility": np.ones((4, 8), dtype=bool),
            "canonical_scale_m": 1.0,
        }
        result = _augment_trajectory_corruption(
            sample, rng=random.Random(13), np=np, track_fraction=1.0,
            drift_scale_fraction=0.05, spike_probability=0.0,
            coherent_drift_probability=1.0,
            recovery_offset_probability=1.0,
            id_switch_probability=1.0,
        )
        metadata = result["_trajectory_corruption"]
        self.assertEqual(metadata["recovery_offset_count"], 4)
        self.assertEqual(metadata["id_switch_count"], 1)
        self.assertTrue(np.all(np.isfinite(result["points"])))

    def test_temporal_occlusion_creates_contiguous_internal_gaps(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        import random

        sample = {
            "points": np.zeros((8, 20, 3), dtype=np.float32),
            "visibility": np.ones((8, 20), dtype=bool),
        }
        augmented = _augment_temporal_occlusion(
            sample, rng=random.Random(11), np=np,
            min_fraction=0.3, max_fraction=0.3, track_fraction=0.5,
        )
        changed = np.flatnonzero(
            np.any(augmented["visibility"] != sample["visibility"], axis=1)
        )
        self.assertEqual(len(changed), 4)
        for track_index in changed:
            hidden = np.flatnonzero(~augmented["visibility"][track_index])
            self.assertEqual(len(hidden), 6)
            self.assertTrue(np.all(np.diff(hidden) == 1))
            self.assertGreater(hidden[0], 0)
            self.assertLess(hidden[-1], 19)
        self.assertTrue(np.all(sample["visibility"]))

    def test_invariant_slot_geometry_is_so3_invariant(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        rng = np.random.default_rng(7)
        embedding_dim = 4
        features = rng.normal(size=(6, embedding_dim + 32)).astype(np.float32)
        sample = {
            "features": features,
            "embedding_dim": embedding_dim,
            "gt_relations": [],
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
            "references": np.zeros((6, 3), dtype=np.float32),
            "points": np.zeros((6, 2, 3), dtype=np.float32),
        }
        invariant = _apply_slot_geometry_representation(
            [sample], "invariant_v1", np
        )[0]
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rotated = _augment_relation_sample(
            invariant, rng=rng, np=np, rotation=rotation, rotate_slot_geometry=True
        )
        np.testing.assert_allclose(
            rotated["slot_features"], invariant["slot_features"], atol=1e-6
        )
        self.assertFalse(np.allclose(rotated["features"], sample["features"]))

    def test_invariant_slot_geometry_preserves_embedding_and_shape(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        features = np.arange(2 * 37, dtype=np.float32).reshape(2, 37)
        sample = {"features": features, "embedding_dim": 5}
        adapted = _apply_slot_geometry_representation([sample], "invariant_v1", np)[0]
        self.assertEqual(adapted["slot_features"].shape, features.shape)
        np.testing.assert_array_equal(adapted["slot_features"][:, :5], features[:, :5])
        np.testing.assert_array_equal(adapted["features"], features)

    def test_clustering_pair_metrics_are_permutation_invariant(self) -> None:
        from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
            _clustering_pair_metrics,
        )

        ari, ri = _clustering_pair_metrics([4, 4, 2, 2], [0, 0, 1, 1])
        self.assertAlmostEqual(ari, 1.0)
        self.assertAlmostEqual(ri, 1.0)

    def test_legal_tree_decoder_removes_cycles_and_multiple_parents(self) -> None:
        pairs = [
            {"parent_slot_id": 0, "child_slot_id": 1, "edge_probability": 0.95},
            {"parent_slot_id": 2, "child_slot_id": 1, "edge_probability": 0.90},
            {"parent_slot_id": 1, "child_slot_id": 0, "edge_probability": 0.85},
            {"parent_slot_id": 1, "child_slot_id": 2, "edge_probability": 0.80},
            {"parent_slot_id": 0, "child_slot_id": 2, "edge_probability": 0.70},
            {"parent_slot_id": 2, "child_slot_id": 0, "edge_probability": 0.60},
        ]
        selected = decode_legal_relation_tree([0, 1, 2], pairs)
        self.assertEqual(
            [(row["parent_slot_id"], row["child_slot_id"]) for row in selected],
            [(0, 1), (1, 2)],
        )
        self.assertTrue(
            graph_legality_diagnostics([0, 1, 2], selected)["is_legal_tree"]
        )

    def test_legal_tree_decoder_handles_single_active_slot(self) -> None:
        self.assertEqual(decode_legal_relation_tree([3], []), [])

    def test_legal_tree_decoder_allows_sparse_forest(self) -> None:
        pairs = [{
            "parent_slot_id": 0,
            "child_slot_id": 1,
            "edge_probability": 0.9,
        }]
        selected = decode_legal_relation_tree([0, 1, 2], pairs)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["child_slot_id"], 1)

    def test_relation_targets_exclude_unmatched_slot_pairs(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        logits = torch.tensor([
            [8.0, -8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0],
            [-8.0, -8.0, 8.0, -8.0],
        ])
        sample = {
            "labels": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "gt_relations": [{
                "parent_label": 0,
                "child_label": 1,
                "joint_type": "prismatic",
                "axis": [1.0, 0.0, 0.0],
                "pivot": [0.0, 0.0, 0.0],
            }],
        }
        target = _relation_targets(sample, logits, 4, torch, "cpu")
        active = torch.nonzero(target["active_slot"], as_tuple=False).flatten().tolist()
        self.assertEqual(active, [0, 2])
        self.assertEqual(int(target["valid_pair"].sum()), 2)
        self.assertTrue(bool(target["positive"][0, 2]))
        self.assertFalse(bool(target["valid_pair"][0, 1]))

    def test_axis_equivariance_loss_is_undirected(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        source = torch.tensor([[1.0, 0.0, 0.0]])
        rotated_with_opposite_sign = torch.tensor([[0.0, -1.0, 0.0]])
        loss = _undirected_axis_equivariance_loss(
            source, rotated_with_opposite_sign, rotation, torch
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_slot_consistency_ignores_slot_permutation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        source = torch.tensor([
            [0.90, 0.08, 0.02],
            [0.10, 0.85, 0.05],
            [0.02, 0.03, 0.95],
        ])
        rotated = source[:, [2, 0, 1]]
        loss = _permutation_aligned_slot_consistency_loss(source, rotated, torch)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_slot_consistency_penalizes_assignment_change(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        source = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
        collapsed = torch.full_like(source, 0.5)
        loss = _permutation_aligned_slot_consistency_loss(source, collapsed, torch)
        self.assertGreater(float(loss), 0.1)

    def test_decoder_unfreeze_keeps_slot_encoder_frozen(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_slot_model(
            torch,
            input_dim=8,
            hidden_dim=16,
            max_slots=4,
            encoder_layers=1,
            decoder_layers=1,
            attention_heads=4,
        )
        config = SlotRelationTrainingConfig(
            manifest_path="manifest.tsv",
            slot_model_path="slots.pt",
            output_dir="out",
            unfreeze_slot_backbone=True,
            slot_unfreeze_scope="decoder",
        )
        _configure_slot_trainability(model, config)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertIn("queries", trainable)
        self.assertTrue(any(name.startswith("decoder.") for name in trainable))
        self.assertFalse(any(name.startswith("encoder.") for name in trainable))

    def test_cli_registers_train_and_infer_commands(self) -> None:
        parser = build_parser()
        train = parser.parse_args([
            "train-slot-relation-head", "manifest.tsv", "slots.pt",
            "--output-dir", "relations",
        ])
        infer = parser.parse_args([
            "infer-slot-relation-head", "tracks.json", "features.npz",
            "slots.pt", "relations.pt", "--output-json", "predictions.json",
        ])
        self.assertEqual(train.command, "train-slot-relation-head")
        self.assertEqual(infer.command, "infer-slot-relation-head")
        self.assertIsNone(infer.trajectory_assignment_override)
        self.assertFalse(train.axis_geometry_branch)
        self.assertEqual(train.geometry_encoder_type, "track_gru_average")
        self.assertEqual(train.trajectory_samples, 32)
        self.assertEqual(train.joint_type_head_type, "pair_context")
        self.assertEqual(train.edge_head_type, "pair_context")
        self.assertEqual(train.structured_parent_loss_weight, 0.0)

        motion_heads = parser.parse_args([
            "train-slot-relation-head", "manifest.tsv", "slots.pt",
            "--output-dir", "relations", "--joint-type-head-type", "child_motion",
            "--edge-head-type", "motion_residual",
            "--structured-parent-loss-weight", "0.5",
        ])
        self.assertEqual(motion_heads.joint_type_head_type, "child_motion")
        self.assertEqual(motion_heads.edge_head_type, "motion_residual")
        self.assertEqual(motion_heads.structured_parent_loss_weight, 0.5)

        override = parser.parse_args([
            "infer-slot-relation-head", "tracks.json", "features.npz",
            "slots.pt", "relations.pt", "--output-json", "predictions.json",
            "--trajectory-assignment-override", "override.json",
        ])
        self.assertEqual(override.trajectory_assignment_override, Path("override.json"))

    def test_relation_model_outputs_ordered_pair_tensors(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_relation_model(torch, slot_dim=16, hidden_dim=32)
        result = model(torch.randn(4, 16))
        self.assertEqual(tuple(result["edge_logits"].shape), (4, 4))
        self.assertEqual(tuple(result["type_logits"].shape), (4, 4, 3))
        self.assertEqual(tuple(result["axes"].shape), (4, 4, 3))
        self.assertEqual(tuple(result["pivots"].shape), (4, 4, 3))

    def test_slot_motion_invariants_are_so3_invariant(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        generator = torch.Generator().manual_seed(17)
        tokens = torch.randn(4, 7, 9, 11, generator=generator)
        tokens[..., 9] = 1.0
        tokens[..., 10] = torch.linspace(0.0, 1.0, 9)
        visibility = torch.ones(4, 7, 9)
        probabilities = torch.softmax(
            torch.randn(4, 7, 3, generator=generator), dim=-1
        )
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated = tokens.clone()
        for start in (0, 3, 6):
            rotated[..., start:start + 3] = tokens[..., start:start + 3] @ rotation.T
        first = _slot_motion_invariants(tokens, visibility, probabilities, torch)
        second = _slot_motion_invariants(rotated, visibility, probabilities, torch)
        self.assertEqual(tuple(first.shape), (4, 3, 10))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))

    def test_child_motion_type_is_independent_of_parent_candidate(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_relation_model(
            torch, slot_dim=16, hidden_dim=32,
            axis_geometry_branch=True, trajectory_hidden_dim=12,
            joint_type_head_type="child_motion", edge_head_type="motion_residual",
        )
        result = model(
            torch.randn(3, 16),
            trajectory_tokens=torch.randn(8, 6, 11),
            trajectory_visibility=torch.ones(8, 6),
            slot_probabilities=torch.softmax(torch.randn(8, 3), dim=-1),
        )
        self.assertEqual(tuple(result["edge_logits"].shape), (3, 3))
        self.assertEqual(tuple(result["type_logits"].shape), (3, 3, 3))
        self.assertTrue(torch.isfinite(result["edge_logits"]).all())
        for child in range(3):
            expected = result["type_logits"][0, child]
            self.assertTrue(torch.allclose(result["type_logits"][:, child], expected[None]))

    def test_structured_parent_loss_prefers_true_parent(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        valid_pair = ~torch.eye(3, dtype=torch.bool)
        positive = torch.zeros((3, 3), dtype=torch.bool)
        positive[0, 2] = True
        target = {"valid_pair": valid_pair, "positive": positive}
        correct = torch.zeros((3, 3))
        correct[0, 2] = 4.0
        wrong = torch.zeros((3, 3))
        wrong[1, 2] = 4.0
        correct_loss = _structured_parent_selection_loss(correct, target, torch)
        wrong_loss = _structured_parent_selection_loss(wrong, target, torch)
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_relation_model_supports_padded_object_batches(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_relation_model(torch, slot_dim=16, hidden_dim=32, motion_summary_dim=5)
        result = model(torch.randn(3, 4, 16), torch.randn(3, 4, 5))
        self.assertEqual(tuple(result["edge_logits"].shape), (3, 4, 4))
        self.assertEqual(tuple(result["type_logits"].shape), (3, 4, 4, 3))
        self.assertEqual(tuple(result["axes"].shape), (3, 4, 4, 3))

    def test_axis_geometry_branch_backpropagates_through_soft_assignments(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_relation_model(
            torch, slot_dim=16, hidden_dim=32, motion_summary_dim=5,
            axis_geometry_branch=True, trajectory_hidden_dim=12,
        )
        probabilities = torch.softmax(
            torch.randn(6, 4, requires_grad=True), dim=-1
        )
        result = model(
            torch.randn(4, 16),
            torch.randn(4, 5),
            trajectory_tokens=torch.randn(6, 8, 11),
            trajectory_visibility=torch.ones(6, 8, dtype=torch.bool),
            slot_probabilities=probabilities,
        )
        (result["axes"][0, 1, 0] + result["pivots"][0, 1, 0]).backward()
        self.assertIsNotNone(probabilities.grad_fn)
        self.assertIsNotNone(next(model.trajectory_encoder.parameters()).grad)

    def test_cross_track_transformer_is_track_permutation_invariant(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        torch.manual_seed(3)
        model = _build_relation_model(
            torch, slot_dim=16, hidden_dim=32, motion_summary_dim=5,
            axis_geometry_branch=True,
            geometry_encoder_type="track_gru_transformer",
            trajectory_hidden_dim=16,
            geometry_max_tracks=8,
            geometry_attention_heads=4,
        )
        model.eval()
        slots = torch.randn(4, 16)
        summary = torch.randn(4, 5)
        trajectories = torch.randn(7, 6, 11)
        visibility = torch.ones(7, 6, dtype=torch.bool)
        probabilities = torch.softmax(torch.randn(7, 4), dim=-1)
        permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
        with torch.no_grad():
            original = model(
                slots, summary,
                trajectory_tokens=trajectories,
                trajectory_visibility=visibility,
                slot_probabilities=probabilities,
            )
            permuted = model(
                slots, summary,
                trajectory_tokens=trajectories[permutation],
                trajectory_visibility=visibility[permutation],
                slot_probabilities=probabilities[permutation],
            )
        self.assertTrue(torch.allclose(
            original["axes"], permuted["axes"], atol=1e-6, rtol=1e-6
        ))
        self.assertTrue(torch.allclose(
            original["pivots"], permuted["pivots"], atol=1e-6, rtol=1e-6
        ))

    def test_cross_track_transformer_handles_visibility_and_padding(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = _build_relation_model(
            torch, slot_dim=16, hidden_dim=32,
            axis_geometry_branch=True,
            geometry_encoder_type="track_gru_transformer",
            trajectory_hidden_dim=16,
            geometry_max_tracks=4,
            geometry_attention_heads=4,
        )
        trajectories = torch.randn(2, 6, 5, 11)
        visibility = torch.zeros(2, 6, 5, dtype=torch.bool)
        visibility[0, :, :] = True
        visibility[1, :3, :3] = True
        probabilities = torch.softmax(torch.randn(2, 6, 4), dim=-1)
        probabilities[1, 3:] = 0.0
        result = model(
            torch.randn(2, 4, 16),
            trajectory_tokens=trajectories,
            trajectory_visibility=visibility,
            slot_probabilities=probabilities,
        )
        self.assertTrue(torch.isfinite(result["axes"]).all())
        self.assertTrue(torch.isfinite(result["pivots"]).all())
        self.assertTrue(torch.isfinite(result["geometry_pooling_entropy"]).all())
        self.assertEqual(tuple(result["geometry_valid_track_count"].shape), (2, 4))
        self.assertEqual(tuple(result["geometry_effective_track_count"].shape), (2, 4))
        self.assertLessEqual(
            float(result["geometry_effective_track_count"].max()), 4.0 + 1e-6
        )

    def test_trajectory_tokens_preserve_time_visibility_and_velocity(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        points = np.asarray([[[0, 0, 0], [1, 0, 0], [3, 0, 0]]], dtype=np.float32)
        sample = {
            "points": points,
            "references": np.asarray([[0, 0, 0]], dtype=np.float32),
            "visibility": np.asarray([[True, True, False]]),
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
        }
        tokens, visibility = _trajectory_tensors(
            sample, torch, "cpu", sample_count=3
        )
        self.assertEqual(tuple(tokens.shape), (1, 3, 11))
        self.assertEqual(visibility.tolist(), [[True, True, False]])
        self.assertAlmostEqual(float(tokens[0, 1, 6]), 1.0)
        self.assertAlmostEqual(float(tokens[0, 2, 9]), 0.0)

    def test_quality_weighted_trajectory_uses_observation_scores(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        sample = {
            "points": np.zeros((1, 3, 3), dtype=np.float32),
            "references": np.zeros((1, 3), dtype=np.float32),
            "visibility": np.asarray([[True, True, False]]),
            "observation_quality": np.asarray([[0.8, 0.25, 0.9]], dtype=np.float32),
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
        }
        _, weights = _trajectory_tensors(
            sample, torch, "cpu", sample_count=3, quality_weighted=True
        )
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.allclose(
            weights, torch.tensor([[0.8, 0.25, 0.0]], dtype=torch.float32)
        ))

    def test_quality_weighted_trajectory_hard_rejects_low_scores(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        sample = {
            "points": np.zeros((1, 3, 3), dtype=np.float32),
            "references": np.zeros((1, 3), dtype=np.float32),
            "visibility": np.asarray([[True, True, True]]),
            "observation_quality": np.asarray([[0.8, 0.25, 0.35]], dtype=np.float32),
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
        }
        _, weights = _trajectory_tensors(
            sample,
            torch,
            "cpu",
            sample_count=3,
            quality_weighted=True,
            min_quality=0.35,
        )
        self.assertTrue(torch.allclose(
            weights, torch.tensor([[0.8, 0.0, 0.35]], dtype=torch.float32)
        ))

    def test_robust_segment_weights_reject_jump_and_reset_velocity(self) -> None:
        import numpy as np
        import torch
        sample = {
            "points": np.asarray([[[0, 0, 0], [0.01, 0, 0], [0.40, 0, 0], [0.41, 0, 0]]], dtype=np.float32),
            "references": np.zeros((1, 3), dtype=np.float32),
            "visibility": np.ones((1, 4), dtype=bool),
            "canonical_center_m": np.zeros(3, dtype=np.float32),
            "canonical_scale_m": 1.0,
        }
        tokens, weights = _trajectory_tensors(
            sample, torch, "cpu", sample_count=4, robust_segment_weights=True
        )
        self.assertEqual(float(weights[0, 2]), 0.0)
        self.assertTrue(torch.allclose(tokens[0, 2, 6:9], torch.zeros(3)))

    def test_robust_segment_weights_are_so3_invariant(self) -> None:
        import numpy as np
        import torch
        points = np.asarray([[[0, 0, 0], [0.01, 0.02, 0], [0.4, -0.2, 0.1]]], dtype=np.float32)
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        base = {"points": points, "references": np.zeros((1, 3), dtype=np.float32), "visibility": np.ones((1, 3), bool), "canonical_center_m": np.zeros(3), "canonical_scale_m": 1.0}
        rotated = {**base, "points": points @ rotation.T}
        _, first = _trajectory_tensors(base, torch, "cpu", sample_count=3, robust_segment_weights=True)
        _, second = _trajectory_tensors(rotated, torch, "cpu", sample_count=3, robust_segment_weights=True)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_relation_batch_loss_backpropagates_through_padded_objects(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        slot_model = _build_slot_model(
            torch, input_dim=9, hidden_dim=16, max_slots=3,
            encoder_layers=1, decoder_layers=1, attention_heads=4,
        )
        relation_model = _build_relation_model(torch, slot_dim=16, hidden_dim=32, motion_summary_dim=8)
        config = SlotRelationTrainingConfig(
            manifest_path="manifest.tsv", slot_model_path="slots.pt", output_dir="out",
        )
        samples = []
        for count in (3, 5):
            samples.append({
                "features": np.random.randn(count, 9).astype(np.float32),
                "labels": np.asarray([0] * (count // 2) + [1] * (count - count // 2), dtype=np.int64),
                "gt_relations": [],
                "embedding_dim": 1,
            })
        losses = _relation_batch_loss(
            samples, slot_model, relation_model,
            torch.zeros(9), torch.ones(9), torch, "cpu", config, torch.ones(3),
        )
        losses["total"].backward()
        self.assertIsNotNone(next(relation_model.parameters()).grad)

    def test_graph_diagnostics_exposes_unconstrained_failures(self) -> None:
        legal = graph_legality_diagnostics(
            [1, 2, 3],
            [
                {"parent_slot_id": 1, "child_slot_id": 2},
                {"parent_slot_id": 1, "child_slot_id": 3},
            ],
        )
        self.assertTrue(legal["is_legal_tree"])
        illegal = graph_legality_diagnostics(
            [1, 2, 3],
            [
                {"parent_slot_id": 1, "child_slot_id": 2},
                {"parent_slot_id": 3, "child_slot_id": 2},
                {"parent_slot_id": 2, "child_slot_id": 1},
            ],
        )
        self.assertFalse(illegal["is_legal_tree"])
        self.assertEqual(illegal["multiple_parent_slots"], [2])
        self.assertTrue(illegal["has_cycle"])

    def test_extracts_directed_joint_and_canonical_axis_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.xml"
            model_path.write_text(
                """<mujoco><worldbody><body name="base" pos="1 0 0">
                <geom name="base_geom" type="box" size=".1 .1 .1"/>
                <body name="door" pos="0 2 0"><joint name="hinge" type="hinge" pos="0 0 1" axis="0 0 1"/>
                <geom name="door_geom" type="box" size=".1 .1 .1"/></body>
                </body></worldbody></mujoco>""",
                encoding="utf-8",
            )
            episode_path = root / "episode.json"
            episode_path.write_text(
                json.dumps({"metadata": {"model_path": str(model_path)}}), encoding="utf-8"
            )
            artifact = {
                "input_episode_path": str(episode_path),
                "original_part_segmentation": {
                    "parts": [
                        {"part_id": 10, "body_id": 1, "parent_body_id": None, "joint_names": []},
                        {"part_id": 20, "body_id": 2, "parent_body_id": 1, "joint_names": ["hinge"]},
                    ]
                },
            }
            sample = {
                "raw_part_to_label": {10: 0, 20: 1},
                "canonical_center_m": [1.0, 1.0, 0.0],
                "canonical_scale_m": 2.0,
            }
            relations = extract_gt_relations(artifact, sample)
            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0]["parent_label"], 0)
            self.assertEqual(relations[0]["child_label"], 1)
            self.assertEqual(relations[0]["joint_type"], "revolute")
            self.assertEqual(relations[0]["axis"], [0.0, 0.0, 1.0])
            self.assertEqual(relations[0]["pivot"], [0.0, 0.5, 0.5])

    def test_rotation_augmentation_rotates_geometry_and_joint_together(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")

        class FixedRng:
            def uniform(self, _low: float, _high: float) -> float:
                return 0.5

        features = np.zeros((2, 12), dtype=np.float32)
        features[:, 1:4] = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        features[:, 4:7] = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
        features[:, 9:12] = features[:, 1:4]
        sample = {
            "features": features,
            "embedding_dim": 1,
            "gt_relations": [{"axis": [0, 0, 1], "pivot": [1, 0, 0]}],
            "canonical_center_m": [0, 0, 0],
            "canonical_scale_m": 1.0,
            "references": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            "points": np.asarray([[[1, 0, 0]], [[0, 1, 0]]], dtype=np.float32),
        }
        augmented = _augment_relation_sample(sample, rng=FixedRng(), np=np)
        axis = np.asarray(augmented["gt_relations"][0]["axis"])
        pivot = np.asarray(augmented["gt_relations"][0]["pivot"])
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(pivot)), 1.0, places=5)
        np.testing.assert_allclose(
            augmented["features"][:, 1:4], augmented["replay_references"], atol=1e-6
        )

    def test_joint_replay_prefers_correct_prismatic_axis(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is not installed")
        references = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        points = np.stack([references, references + np.asarray([0.5, 0, 0])], axis=1)
        sample = {
            "replay_references": references,
            "replay_points": points,
            "visibility": np.ones((3, 2), dtype=bool),
            "labels": np.ones(3, dtype=np.int64),
            "reference_frames": np.zeros(3, dtype=np.int64),
        }
        target = {"relations": [{
            "parent_slot": 0, "child_slot": 1, "child_label": 1,
            "joint_type": "prismatic",
        }]}

        def prediction(axis: list[float]) -> dict[str, torch.Tensor]:
            axes = torch.zeros((2, 2, 3), dtype=torch.float32)
            axes[0, 1] = torch.tensor(axis)
            return {
                "axes": axes,
                "pivots": torch.zeros((2, 2, 3), dtype=torch.float32),
                "edge_logits": torch.zeros((2, 2), dtype=torch.float32),
            }

        correct = _joint_model_replay_loss(prediction([1, 0, 0]), target, sample, torch, "cpu")
        wrong = _joint_model_replay_loss(prediction([0, 1, 0]), target, sample, torch, "cpu")
        self.assertLess(float(correct), 1e-3)
        self.assertGreater(float(wrong), float(correct) + 0.1)
        self.assertEqual(JOINT_TYPES.index("prismatic"), 2)


if __name__ == "__main__":
    unittest.main()
