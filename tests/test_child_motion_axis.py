from __future__ import annotations

import unittest

import numpy as np

from rgbd_urdf_mvp.kinematics.child_motion_axis import (
    build_child_motion_axis_model,
    build_child_motion_features,
    estimate_child_motion_axis,
)


class ChildMotionAxisTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        self.torch = torch

    def _tracks(self, joint_type: str):
        torch = self.torch
        generator = torch.Generator().manual_seed(7)
        reference = torch.randn(1, 48, 3, generator=generator)
        axis = torch.tensor([0.3, -0.4, 0.8660254])
        axis = axis / torch.linalg.vector_norm(axis)
        pivot = torch.tensor([0.2, -0.1, 0.3])
        frames = []
        for value in torch.linspace(0.0, 0.7, 12):
            if joint_type == "prismatic":
                frame = reference + value * axis
            else:
                skew = torch.tensor([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ])
                rotation = torch.eye(3) + torch.sin(value) * skew + (1 - torch.cos(value)) * (skew @ skew)
                frame = (reference - pivot) @ rotation.T + pivot
            frames.append(frame)
        points = torch.stack(frames, dim=2)
        visible = torch.ones(points.shape[:-1], dtype=torch.bool)
        return points, visible, axis

    def test_prismatic_motion_vector_recovers_axis(self) -> None:
        points, visible, axis = self._tracks("prismatic")
        features = build_child_motion_features(points, visible, self.torch)
        displacement = features["translation_vectors"].reshape(-1, 3)
        error = 1.0 - (displacement / displacement.norm(dim=-1, keepdim=True) @ axis).abs().mean()
        self.assertLess(float(error), 1e-5)

    def test_analytic_child_only_recovers_both_joint_axes(self) -> None:
        for joint_type in ("prismatic", "revolute"):
            points, visible, axis = self._tracks(joint_type)
            estimate = estimate_child_motion_axis(
                points[0].numpy(), visible[0].numpy(), joint_type
            )
            self.assertTrue(estimate.valid, joint_type)
            dot = abs(float(np.dot(estimate.axis, axis.numpy())))
            self.assertGreater(dot, 0.995, joint_type)

    def test_feedforward_outputs_are_se3_equivariant(self) -> None:
        torch = self.torch
        points, visible, _ = self._tracks("revolute")
        model = build_child_motion_axis_model(torch, hidden_dim=16).eval()
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = torch.tensor([0.7, -0.2, 0.4])
        with torch.no_grad():
            source = model(points, visible)
            target = model(points @ rotation.T + translation, visible)
        self.assertTrue(torch.allclose(source["type_logits"], target["type_logits"], atol=1e-5))
        for key in ("prismatic_axis", "revolute_axis"):
            expected = source[key] @ rotation.T
            self.assertGreater(float((expected * target[key]).sum(dim=-1).abs()), 0.999)
        self.assertTrue(torch.allclose(
            source["line_point"] @ rotation.T + translation,
            target["line_point"], atol=1e-5,
        ))

    def test_missing_observations_do_not_create_world_axis(self) -> None:
        torch = self.torch
        points = torch.zeros(2, 10, 8, 3)
        visible = torch.zeros(2, 10, 8, dtype=torch.bool)
        output = build_child_motion_axis_model(torch, hidden_dim=8)(points, visible)
        self.assertTrue(torch.equal(output["prismatic_axis"], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(output["revolute_axis"], torch.zeros(2, 3)))

    def test_no_linear_layer_directly_regresses_xyz(self) -> None:
        torch = self.torch
        model = build_child_motion_axis_model(torch, hidden_dim=16)
        offenders = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and module.out_features == 3
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
