from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GeometryFieldSpec:
    name: str
    semantic_type: str
    frame: str
    augmentation_rule: str
    status: str = "supported"


# This registry documents both fields currently consumed by the Relation Head
# and geometry fields that may be added later. Unknown cached embeddings are
# intentionally not treated as equivariant.
RELATION_GEOMETRY_FIELDS = (
    GeometryFieldSpec("features.reference_xyz", "point", "object_canonical", "Q @ x"),
    GeometryFieldSpec("features.endpoint_displacement", "vector", "object_canonical", "Q @ v"),
    GeometryFieldSpec("features.motion_magnitude", "scalar", "invariant", "unchanged"),
    GeometryFieldSpec("features.visibility_ratio", "scalar", "invariant", "unchanged"),
    GeometryFieldSpec("features.sampled_displacements", "vector", "object_canonical", "Q @ v"),
    GeometryFieldSpec("replay_references", "point", "object_canonical", "Q @ x"),
    GeometryFieldSpec("replay_points", "point", "object_canonical", "Q @ x"),
    GeometryFieldSpec("joint_axis", "vector", "object_canonical", "Q @ v"),
    GeometryFieldSpec("joint_pivot", "point", "object_canonical", "Q @ x"),
    GeometryFieldSpec("relative_rotation", "rotation_matrix", "world", "Q @ R @ Q.T"),
    GeometryFieldSpec("relative_translation", "vector", "world", "Q @ t"),
    GeometryFieldSpec("rotation_log", "vector", "world", "Q @ omega"),
    GeometryFieldSpec("flow", "vector", "world", "Q @ v"),
    GeometryFieldSpec("velocity", "vector", "world", "Q @ v"),
    GeometryFieldSpec("normals", "normal", "world", "Q @ n"),
    GeometryFieldSpec("camera_rays", "vector", "world_or_camera", "frame-dependent", "unknown_frame"),
    GeometryFieldSpec(
        "cached_track_embedding", "cached_descriptor", "unknown", "recompute from rotated input",
        "not_recomputed",
    ),
)


def sample_uniform_so3(rng: Any, np_module: Any = np) -> np.ndarray:
    """Sample one Haar-uniform SO(3) matrix using a unit quaternion."""
    u1, u2, u3 = (_random_unit(rng) for _ in range(3))
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return quaternion_wxyz_to_matrix([qw, qx, qy, qz], np_module=np_module)


def sample_uniform_so3_batch(count: int, seed: int = 0) -> np.ndarray:
    """Vectorized Haar-uniform sampler used by distribution audits."""
    generator = np.random.default_rng(seed)
    u1, u2, u3 = generator.random((3, int(count)))
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    matrices = np.empty((int(count), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    matrices[:, 0, 1] = 2 * (qx * qy - qz * qw)
    matrices[:, 0, 2] = 2 * (qx * qz + qy * qw)
    matrices[:, 1, 0] = 2 * (qx * qy + qz * qw)
    matrices[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    matrices[:, 1, 2] = 2 * (qy * qz - qx * qw)
    matrices[:, 2, 0] = 2 * (qx * qz - qy * qw)
    matrices[:, 2, 1] = 2 * (qy * qz + qx * qw)
    matrices[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return matrices


def sample_euler_so3(rng: Any, np_module: Any = np) -> np.ndarray:
    """Legacy independent-Euler sampler retained only for controlled ablations."""
    angles = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
    cx, cy, cz = [math.cos(value) for value in angles]
    sx, sy, sz = [math.sin(value) for value in angles]
    rotation_x = np_module.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np_module.float32)
    rotation_y = np_module.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np_module.float32)
    rotation_z = np_module.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np_module.float32)
    return rotation_z @ rotation_y @ rotation_x


def sample_yaw_so3(rng: Any, np_module: Any = np) -> np.ndarray:
    """Sample a uniform rotation around the recording world Z axis."""
    angle = float(rng.uniform(-math.pi, math.pi))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np_module.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np_module.float32,
    )


def sample_limited_so3(
    rng: Any, max_angle_deg: float, np_module: Any = np
) -> np.ndarray:
    """Sample a random-axis rotation with bounded signed angle."""
    axis = np_module.asarray(
        [rng.normalvariate(0.0, 1.0) for _ in range(3)], dtype=np_module.float32
    )
    axis /= max(float(np_module.linalg.norm(axis)), 1e-12)
    angle = math.radians(float(max_angle_deg)) * float(rng.uniform(-1.0, 1.0))
    skew = np_module.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np_module.float32,
    )
    return (
        np_module.eye(3, dtype=np_module.float32)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def sample_so3(rng: Any, mode: str = "uniform_quaternion", np_module: Any = np) -> np.ndarray:
    if mode == "uniform_quaternion":
        return sample_uniform_so3(rng, np_module=np_module)
    if mode == "euler":
        return sample_euler_so3(rng, np_module=np_module)
    if mode == "yaw":
        return sample_yaw_so3(rng, np_module=np_module)
    if mode == "limited_xyz_15":
        return sample_limited_so3(rng, 15.0, np_module=np_module)
    if mode == "limited_xyz_30":
        return sample_limited_so3(rng, 30.0, np_module=np_module)
    if mode == "identity":
        return np_module.eye(3, dtype=np_module.float32)
    raise ValueError(f"Unknown SO(3) sampling mode: {mode}")


def quaternion_wxyz_to_matrix(quaternion: Any, np_module: Any = np) -> np.ndarray:
    qw, qx, qy, qz = [float(value) for value in quaternion]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np_module.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np_module.float32)


def rotate_vectors(values: Any, rotation: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array @ np.asarray(rotation).T


def conjugate_rotations(values: Any, rotation: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    q = np.asarray(rotation)
    return q @ array @ q.T


def nearest_canonical_axis_id(axes: Any) -> np.ndarray:
    normalized = _normalize_axes(np.asarray(axes, dtype=float))
    return np.abs(normalized).argmax(axis=-1)


def nearest_canonical_axis_angle_deg(axes: Any) -> np.ndarray:
    normalized = _normalize_axes(np.asarray(axes, dtype=float))
    cosine = np.abs(normalized).max(axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def geometry_registry_report() -> list[dict[str, str]]:
    return [
        {
            "name": spec.name,
            "semantic_type": spec.semantic_type,
            "frame": spec.frame,
            "augmentation_rule": spec.augmentation_rule,
            "status": spec.status,
        }
        for spec in RELATION_GEOMETRY_FIELDS
    ]


def _random_unit(rng: Any) -> float:
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(rng.uniform(0.0, 1.0))


def _normalize_axes(axes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    return axes / np.maximum(norms, 1e-12)
