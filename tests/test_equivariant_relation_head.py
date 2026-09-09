from __future__ import annotations

import unittest

import numpy as np

from rgbd_urdf_mvp.kinematics.equivariant_relation_geometry import (
    build_equivariant_pair_geometry,
)
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import _build_relation_model


class EquivariantRelationHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        self.torch = torch

    def test_pull_return_prismatic_candidate_uses_complete_path(self) -> None:
        tokens, visible, probabilities = self._synthetic_tokens("prismatic")
        geometry = build_equivariant_pair_geometry(
            tokens, visible, probabilities, self.torch
        )
        axis = geometry["candidate_axes"][0, 0, 1, 0]
        self.assertGreater(float(axis[0].abs()), 0.99)
        self.assertTrue(bool(geometry["candidate_valid"][0, 0, 1, 0]))

    def test_equivariant_heads_rotate_axes_without_world_bias(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        rotated = tokens.clone()
        for start in (0, 3, 6):
            rotated[..., start : start + 3] = tokens[..., start : start + 3] @ rotation.T
        slots = torch.randn(2, 8)
        motion = torch.zeros(2, 4)
        for architecture in ("equivariant_proposal", "vector_neuron"):
            model = _build_relation_model(
                torch,
                slot_dim=8,
                hidden_dim=16,
                motion_summary_dim=4,
                axis_geometry_branch=True,
                axis_head_type=architecture,
                trajectory_hidden_dim=8,
            )
            model.eval()
            with torch.no_grad():
                source = model(
                    slots,
                    motion,
                    trajectory_tokens=tokens[0],
                    trajectory_visibility=visible[0],
                    slot_probabilities=probabilities[0],
                )
                target = model(
                    slots,
                    motion,
                    trajectory_tokens=rotated[0],
                    trajectory_visibility=visible[0],
                    slot_probabilities=probabilities[0],
                )
            expected = source["axes"][0, 1] @ rotation.T
            dot = torch.sum(expected * target["axes"][0, 1]).abs()
            self.assertGreater(float(dot), 0.999, architecture)
            self.assertTrue(torch.allclose(source["edge_logits"], target["edge_logits"], atol=1e-5))
            self.assertTrue(torch.allclose(source["type_logits"], target["type_logits"], atol=1e-5))

    def test_equivariant_heads_have_no_free_three_vector_output_bias(self) -> None:
        torch = self.torch
        for architecture in ("equivariant_proposal", "vector_neuron"):
            model = _build_relation_model(
                torch,
                slot_dim=8,
                hidden_dim=16,
                axis_geometry_branch=True,
                axis_head_type=architecture,
                trajectory_hidden_dim=8,
            )
            offenders = [
                name
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
                and module.out_features == 3
                and module.bias is not None
                and name != "joint_type"
            ]
            self.assertEqual(offenders, [], architecture)

    def test_low_motion_is_reported_unobservable(self) -> None:
        tokens, visible, probabilities = self._synthetic_tokens("static")
        geometry = build_equivariant_pair_geometry(
            tokens, visible, probabilities, self.torch
        )
        self.assertFalse(bool(geometry["candidate_valid"][0, 0, 1, 1]))
        self.assertLess(float(geometry["observability"][0, 0, 1]), 1e-4)

    def test_revolute_hinge_line_rotates_with_input(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        rotation = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        rotated = tokens.clone()
        for start in (0, 3, 6):
            rotated[..., start : start + 3] = tokens[..., start : start + 3] @ rotation.T
        source = build_equivariant_pair_geometry(tokens, visible, probabilities, torch)
        target = build_equivariant_pair_geometry(rotated, visible, probabilities, torch)
        expected = source["candidate_pivots"][0, 0, 1, 1] @ rotation.T
        observed = target["candidate_pivots"][0, 0, 1, 1]
        axis = target["candidate_axes"][0, 0, 1, 1]
        line_error = torch.linalg.vector_norm(torch.linalg.cross(observed - expected, axis))
        self.assertLess(float(line_error), 1e-3)

    def test_vector_neuron_pivot_line_is_strictly_equivariant(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        rotation = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        rotated = tokens.clone()
        for start in (0, 3, 6):
            rotated[..., start : start + 3] = tokens[..., start : start + 3] @ rotation.T
        model = _build_relation_model(
            torch, slot_dim=8, hidden_dim=16, axis_geometry_branch=True,
            axis_head_type="vector_neuron", trajectory_hidden_dim=8,
            vector_pivot_parameterization="analytic_plane_residual_v1",
        ).eval()
        slots = torch.randn(2, 8)
        with torch.no_grad():
            source = model(
                slots, trajectory_tokens=tokens[0], trajectory_visibility=visible[0],
                slot_probabilities=probabilities[0],
            )
            target = model(
                slots, trajectory_tokens=rotated[0], trajectory_visibility=visible[0],
                slot_probabilities=probabilities[0],
            )
        expected_pivot = source["pivots"][0, 1] @ rotation.T
        target_axis = target["axes"][0, 1]
        line_error = torch.linalg.vector_norm(
            torch.linalg.cross(target["pivots"][0, 1] - expected_pivot, target_axis)
        )
        self.assertLess(float(line_error), 1e-4)

    def test_vector_neuron_new_pivot_starts_from_analytic_hinge_line(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        model = _build_relation_model(
            torch, slot_dim=8, hidden_dim=16, axis_geometry_branch=True,
            axis_head_type="vector_neuron", trajectory_hidden_dim=8,
            vector_pivot_parameterization="analytic_plane_residual_v1",
        ).eval()
        torch.nn.init.zeros_(model.vector_pivot_weights.weight)
        torch.nn.init.zeros_(model.vector_pivot_weights.bias)
        geometry = build_equivariant_pair_geometry(tokens, visible, probabilities, torch)
        with torch.no_grad():
            output = model(
                torch.randn(2, 8), trajectory_tokens=tokens[0],
                trajectory_visibility=visible[0], slot_probabilities=probabilities[0],
            )
        self.assertTrue(torch.allclose(
            output["pivots"][0, 1], geometry["candidate_pivots"][0, 0, 1, 1], atol=1e-5
        ))

    def test_pure_vector_neuron_pivot_is_strictly_equivariant(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        rotated = tokens.clone()
        for start in (0, 3, 6):
            rotated[..., start : start + 3] = tokens[..., start : start + 3] @ rotation.T
        model = _build_relation_model(
            torch, slot_dim=8, hidden_dim=16, axis_geometry_branch=True,
            axis_head_type="vector_neuron", trajectory_hidden_dim=8,
            vector_pivot_parameterization="pure_plane_residual_v1",
        ).eval()
        slots = torch.randn(2, 8)
        with torch.no_grad():
            source = model(
                slots, trajectory_tokens=tokens[0], trajectory_visibility=visible[0],
                slot_probabilities=probabilities[0],
            )
            target = model(
                slots, trajectory_tokens=rotated[0], trajectory_visibility=visible[0],
                slot_probabilities=probabilities[0],
            )
        expected = source["pivots"][0, 1] @ rotation.T
        self.assertTrue(torch.allclose(expected, target["pivots"][0, 1], atol=1e-4))

    def test_pure_pivot_primitives_do_not_use_analytic_hinge_point(self) -> None:
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        geometry = build_equivariant_pair_geometry(
            tokens, visible, probabilities, self.torch
        )
        self.assertIn("pivot_basis_vectors", geometry)
        expected_center_delta = (
            tokens[0, 8:, 0, :3].mean(dim=0) - tokens[0, :8, 0, :3].mean(dim=0)
        )
        self.assertTrue(self.torch.allclose(
            geometry["pivot_basis_vectors"][0, 0, 1, 0], expected_center_delta, atol=1e-5
        ))

    def test_legacy_vector_pivot_parameterization_remains_available(self) -> None:
        model = _build_relation_model(
            self.torch, slot_dim=8, hidden_dim=16, axis_geometry_branch=True,
            axis_head_type="vector_neuron", trajectory_hidden_dim=8,
            vector_pivot_parameterization="legacy_center_delta",
        )
        self.assertEqual(model.vector_pivot_parameterization, "legacy_center_delta")

    def test_vector_pivot_parameters_are_a_separate_trainable_group(self) -> None:
        model = _build_relation_model(
            self.torch, slot_dim=8, hidden_dim=16, axis_geometry_branch=True,
            axis_head_type="vector_neuron", trajectory_hidden_dim=8,
        )
        pivot_ids = {id(parameter) for parameter in model.vector_pivot_weights.parameters()}
        self.assertEqual(len(pivot_ids), 2)
        self.assertTrue(pivot_ids.issubset({id(parameter) for parameter in model.parameters()}))

    def test_missing_frames_preserve_revolute_candidate(self) -> None:
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        visible[:, 8:, 2:4] = False
        geometry = build_equivariant_pair_geometry(
            tokens, visible, probabilities, self.torch
        )
        self.assertTrue(bool(geometry["candidate_valid"][0, 0, 1, 1]))
        self.assertGreater(float(geometry["candidate_axes"][0, 0, 1, 1, 2].abs()), 0.99)

    def test_equivariant_scalar_weights_receive_gradients(self) -> None:
        torch = self.torch
        tokens, visible, probabilities = self._synthetic_tokens("revolute")
        model = _build_relation_model(
            torch,
            slot_dim=8,
            hidden_dim=16,
            motion_summary_dim=4,
            axis_geometry_branch=True,
            axis_head_type="equivariant_proposal",
            trajectory_hidden_dim=8,
        )
        output = model(
            torch.randn(2, 8),
            torch.zeros(2, 4),
            trajectory_tokens=tokens[0],
            trajectory_visibility=visible[0],
            slot_probabilities=probabilities[0],
        )
        loss = output["axes"][0, 1, 2] + output["edge_logits"].sum()
        loss.backward()
        self.assertIsNotNone(model.axis_candidate_weights.weight.grad)

    def _synthetic_tokens(self, motion: str):
        torch = self.torch
        rng = np.random.default_rng(7)
        parent = rng.normal(size=(8, 3)).astype(np.float32) * 0.15
        child = rng.normal(size=(8, 3)).astype(np.float32) * 0.15
        child[:, 0] += 0.7
        references = np.concatenate([parent, child], axis=0)
        frames = 9
        positions = np.repeat(references[:, None, :], frames, axis=1)
        values = np.concatenate(
            [np.linspace(0.0, 1.0, 5), np.linspace(0.75, 0.0, 4)]
        ).astype(np.float32)
        if motion == "prismatic":
            positions[8:, :, 0] += values[None, :] * 0.4
        elif motion == "revolute":
            pivot = np.asarray([0.2, -0.1, 0.05], dtype=np.float32)
            for frame, value in enumerate(values):
                angle = float(value) * 0.8
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                positions[8:, frame] = (child - pivot) @ rotation.T + pivot
        displacement = positions - references[:, None, :]
        velocity = np.zeros_like(displacement)
        velocity[:, 1:] = displacement[:, 1:] - displacement[:, :-1]
        time = np.broadcast_to(
            np.linspace(0.0, 1.0, frames, dtype=np.float32)[None, :, None],
            (len(references), frames, 1),
        )
        visible = np.ones((len(references), frames), dtype=bool)
        tokens = np.concatenate(
            [
                np.broadcast_to(references[:, None, :], positions.shape),
                displacement,
                velocity,
                visible[..., None].astype(np.float32),
                time,
            ],
            axis=-1,
        )
        probabilities = np.zeros((1, len(references), 2), dtype=np.float32)
        probabilities[0, :8, 0] = 1.0
        probabilities[0, 8:, 1] = 1.0
        return (
            torch.tensor(tokens[None], dtype=torch.float32),
            torch.tensor(visible[None]),
            torch.tensor(probabilities),
        )


if __name__ == "__main__":
    unittest.main()
