from __future__ import annotations

import math
import random
import unittest

import numpy as np

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import _augment_relation_sample
from rgbd_urdf_mvp.kinematics.so3_augmentation import (
    conjugate_rotations,
    geometry_registry_report,
    nearest_canonical_axis_id,
    rotate_vectors,
    sample_limited_so3,
    sample_uniform_so3,
    sample_uniform_so3_batch,
    sample_yaw_so3,
)


def _rotation_z_90() -> np.ndarray:
    return np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _rodrigues(point: np.ndarray, axis: np.ndarray, pivot: np.ndarray, angle: float) -> np.ndarray:
    vector = point - pivot
    return pivot + vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle) + axis * (
        axis @ vector
    ) * (1.0 - math.cos(angle))


class SO3AugmentationTests(unittest.TestCase):
    def test_uniform_sampler_returns_rotation(self) -> None:
        rotation = sample_uniform_so3(random.Random(4))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)

    def test_uniform_direction_distribution_has_balanced_nearest_axes(self) -> None:
        rotations = sample_uniform_so3_batch(30_000, seed=7)
        directions = rotations @ np.asarray([1.0, 0.0, 0.0])
        counts = np.bincount(nearest_canonical_axis_id(directions), minlength=3) / len(directions)
        np.testing.assert_allclose(counts, np.ones(3) / 3.0, atol=0.015)

    def test_yaw_sampler_preserves_world_z(self) -> None:
        rotation = sample_yaw_so3(random.Random(19))
        np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)

    def test_limited_sampler_respects_angle_bound(self) -> None:
        for seed in range(100):
            rotation = sample_limited_so3(random.Random(seed), 15.0)
            angle = math.degrees(
                math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
            )
            self.assertLessEqual(angle, 15.001)

    def test_axis_and_point_rotation(self) -> None:
        rotation = _rotation_z_90()
        np.testing.assert_allclose(rotate_vectors([[1.0, 0.0, 0.0]], rotation), [[0.0, 1.0, 0.0]])

    def test_flow_rotation_matches_rotated_point_difference(self) -> None:
        rotation = sample_uniform_so3(random.Random(9))
        start = np.asarray([[0.2, -0.1, 0.4]])
        end = np.asarray([[0.9, 0.3, -0.2]])
        np.testing.assert_allclose(
            rotate_vectors(end - start, rotation),
            rotate_vectors(end, rotation) - rotate_vectors(start, rotation),
            atol=1e-6,
        )

    def test_relative_transform_uses_conjugation(self) -> None:
        q = sample_uniform_so3(random.Random(11))
        relative = sample_uniform_so3(random.Random(12))
        translation = np.asarray([0.2, -0.3, 0.5])
        point = np.asarray([0.4, 0.1, -0.2])
        lhs = q @ (relative @ point + translation)
        rhs = conjugate_rotations(relative, q) @ (q @ point) + q @ translation
        np.testing.assert_allclose(lhs, rhs, atol=1e-6)

    def test_rotation_log_direction_rotates_as_vector(self) -> None:
        q = sample_uniform_so3(random.Random(13))
        axis = np.asarray([0.2, 0.8, -0.1])
        axis /= np.linalg.norm(axis)
        angle = 0.7
        skew = np.asarray([
            [0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]
        ])
        relative = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
        rotated = conjugate_rotations(relative, q)
        recovered = np.asarray([
            rotated[2, 1] - rotated[1, 2],
            rotated[0, 2] - rotated[2, 0],
            rotated[1, 0] - rotated[0, 1],
        ]) / (2.0 * math.sin(angle))
        np.testing.assert_allclose(recovered, q @ axis, atol=1e-6)

    def test_revolute_replay_is_equivariant(self) -> None:
        q = sample_uniform_so3(random.Random(14))
        point = np.asarray([0.8, 0.2, -0.1])
        axis = np.asarray([0.0, 0.0, 1.0])
        pivot = np.asarray([0.1, -0.2, 0.0])
        replay = _rodrigues(point, axis, pivot, 0.6)
        rotated_replay = _rodrigues(q @ point, q @ axis, q @ pivot, 0.6)
        np.testing.assert_allclose(q @ replay, rotated_replay, atol=1e-6)

    def test_prismatic_replay_is_equivariant(self) -> None:
        q = sample_uniform_so3(random.Random(15))
        point = np.asarray([0.2, 0.4, -0.1])
        axis = np.asarray([1.0, 0.0, 0.0])
        np.testing.assert_allclose(q @ (point + 0.3 * axis), q @ point + 0.3 * (q @ axis), atol=1e-6)

    def test_relation_feature_transform_preserves_scalars_and_rotates_geometry(self) -> None:
        rotation = _rotation_z_90().astype(np.float32)
        features = np.zeros((1, 34), dtype=np.float32)
        embedding_dim = 2
        features[0, :embedding_dim] = [7.0, 8.0]
        features[0, embedding_dim : embedding_dim + 3] = [1.0, 0.0, 0.0]
        features[0, embedding_dim + 3 : embedding_dim + 6] = [0.0, 1.0, 0.0]
        features[0, embedding_dim + 6 : embedding_dim + 8] = [2.5, 0.75]
        sample = {
            "features": features,
            "embedding_dim": embedding_dim,
            "gt_relations": [{"axis": [1, 0, 0], "pivot": [0, 1, 0]}],
            "canonical_center_m": [0, 0, 0], "canonical_scale_m": 1.0,
            "references": np.asarray([[1, 0, 0]], dtype=np.float32),
            "points": np.asarray([[[1, 0, 0]]], dtype=np.float32),
        }
        augmented = _augment_relation_sample(
            sample, rng=random.Random(0), np=np, rotation=rotation
        )
        np.testing.assert_allclose(augmented["features"][0, :2], [7.0, 8.0])
        np.testing.assert_allclose(augmented["features"][0, 2:5], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(augmented["features"][0, 8:10], [2.5, 0.75])
        np.testing.assert_allclose(augmented["gt_relations"][0]["axis"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(augmented["slot_features"], sample["features"])
        full_geometry = _augment_relation_sample(
            sample, rng=random.Random(0), np=np, rotation=rotation,
            rotate_slot_geometry=True,
        )
        np.testing.assert_allclose(full_geometry["slot_features"], full_geometry["features"])

    def test_registry_exposes_camera_frame_and_cached_embedding_risks(self) -> None:
        registry = {row["name"]: row for row in geometry_registry_report()}
        self.assertEqual(registry["camera_rays"]["status"], "unknown_frame")
        self.assertEqual(registry["cached_track_embedding"]["status"], "not_recomputed")


if __name__ == "__main__":
    unittest.main()
