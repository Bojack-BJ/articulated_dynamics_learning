from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class JointInferenceConfig:
    input_path: str | Path
    output_json: str | Path | None = None
    rotation_threshold_rad: float = 0.20
    translation_threshold_m: float = 0.02
    delta_rotation_epsilon_rad: float = 0.01
    delta_translation_epsilon_m: float = 1e-4


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vec: list[float]) -> float:
    return math.sqrt(_dot(vec, vec))


def _normalize(vec: list[float], fallback: list[float] | None = None) -> list[float]:
    length = _norm(vec)
    if length < 1e-9:
        return list(fallback) if fallback is not None else [0.0, 0.0, 0.0]
    return [value / length for value in vec]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(vec: list[float], scalar: float) -> list[float]:
    return [scalar * value for value in vec]


def _transpose3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            out[row][col] = sum(a[row][k] * b[k][col] for k in range(3))
    return out


def _matvec3(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vec[col] for col in range(3)) for row in range(3)]


def _rotation_angle(rotation: list[list[float]]) -> float:
    trace = max(-1.0, min(3.0, rotation[0][0] + rotation[1][1] + rotation[2][2]))
    cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.acos(cosine)


def _rotation_axis(rotation: list[list[float]], angle: float) -> list[float]:
    if angle < 1e-8:
        return [0.0, 0.0, 1.0]
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        sine = 1e-8 if sine >= 0.0 else -1e-8
    axis = [
        (rotation[2][1] - rotation[1][2]) / (2.0 * sine),
        (rotation[0][2] - rotation[2][0]) / (2.0 * sine),
        (rotation[1][0] - rotation[0][1]) / (2.0 * sine),
    ]
    return _normalize(axis, fallback=[0.0, 0.0, 1.0])


def _rigid_delta(
    rotation_a: list[list[float]],
    translation_a: list[float],
    rotation_b: list[list[float]],
    translation_b: list[float],
) -> tuple[list[list[float]], list[float]]:
    delta_rotation = _matmul3(rotation_b, _transpose3(rotation_a))
    delta_translation = _subtract(translation_b, _matvec3(delta_rotation, translation_a))
    return delta_rotation, delta_translation


def _solve_3x3(ata: list[list[float]], atb: list[float], damping: float = 1e-6) -> list[float]:
    augmented = [
        [float(ata[row][col] + (damping if row == col else 0.0)) for col in range(3)] + [float(atb[row])]
        for row in range(3)
    ]
    for pivot in range(3):
        best = max(range(pivot, 3), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best][pivot]) < 1e-12:
            return [0.0, 0.0, 0.0]
        if best != pivot:
            augmented[pivot], augmented[best] = augmented[best], augmented[pivot]

        scale = augmented[pivot][pivot]
        for col in range(pivot, 4):
            augmented[pivot][col] /= scale

        for row in range(3):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            for col in range(pivot, 4):
                augmented[row][col] -= factor * augmented[pivot][col]

    return [augmented[row][3] for row in range(3)]


def _signed_angle_about_axis(rotation: list[list[float]], axis: list[float]) -> float:
    angle = _rotation_angle(rotation)
    if angle < 1e-8:
        return 0.0
    rot_axis = _rotation_axis(rotation, angle)
    sign = 1.0 if _dot(rot_axis, axis) >= 0.0 else -1.0
    return sign * angle


def _mean_point(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    inv = 1.0 / float(len(points))
    return [sum(point[axis] for point in points) * inv for axis in range(3)]


def _project_point_to_line(point: list[float], line_direction: list[float]) -> list[float]:
    direction = _normalize(line_direction, fallback=[1.0, 0.0, 0.0])
    return _scale(direction, _dot(point, direction))


class JointInferencer:
    def __init__(self, config: JointInferenceConfig) -> None:
        self.config = config

    def infer(self) -> Path:
        input_path = Path(self.config.input_path).resolve()
        artifact = load_json(input_path)
        parts = artifact.get("parts", [])
        anchor_part_id = int(artifact.get("anchor_part_id", 0))
        anchor_part_name = str(artifact.get("anchor_part_name", "anchor"))

        joints: list[dict[str, Any]] = []
        for part in parts:
            part_id = int(part.get("part_id", 0))
            if part_id == anchor_part_id:
                continue
            inferred = self._infer_part_joint(part, anchor_part_id, anchor_part_name)
            if inferred is not None:
                joints.append(inferred)

        output_json = (
            Path(self.config.output_json).resolve()
            if self.config.output_json is not None
            else input_path.with_name("joint_inference.json")
        )
        save_json(
            {
                "input_path": str(input_path),
                "estimator": "relative-se3-geometry",
                "anchor_part_id": anchor_part_id,
                "anchor_part_name": anchor_part_name,
                "joints": joints,
            },
            output_json,
        )
        return output_json

    def _infer_part_joint(
        self,
        part: dict[str, Any],
        anchor_part_id: int,
        anchor_part_name: str,
    ) -> dict[str, Any] | None:
        valid_samples = [
            sample
            for sample in part.get("samples", [])
            if bool(sample.get("valid", False)) and isinstance(sample.get("relative_to_anchor"), dict)
        ]
        if len(valid_samples) < 2:
            return None

        relative_poses = [
            {
                "frame_index": int(sample["frame_index"]),
                "timestamp_s": float(sample["timestamp_s"]),
                "rotation": [[float(value) for value in row] for row in sample["relative_to_anchor"]["rotation_matrix"]],
                "translation": [float(value) for value in sample["relative_to_anchor"]["translation"]],
            }
            for sample in valid_samples
        ]

        delta_rotations: list[list[list[float]]] = []
        delta_translations: list[list[float]] = []
        delta_angles: list[float] = []
        delta_translation_magnitudes: list[float] = []
        for previous, current in zip(relative_poses[:-1], relative_poses[1:]):
            delta_rotation, delta_translation = _rigid_delta(
                previous["rotation"],
                previous["translation"],
                current["rotation"],
                current["translation"],
            )
            angle = _rotation_angle(delta_rotation)
            translation_mag = _norm(delta_translation)
            delta_rotations.append(delta_rotation)
            delta_translations.append(delta_translation)
            delta_angles.append(angle)
            delta_translation_magnitudes.append(translation_mag)

        rotation_range = float(part.get("relative_motion_summary", {}).get("rotation_angle_range_rad", 0.0))
        translation_range = [float(value) for value in part.get("relative_motion_summary", {}).get("translation_range", [0.0, 0.0, 0.0])]
        translation_range_norm = _norm(translation_range)
        mean_delta_angle = fmean(delta_angles) if delta_angles else 0.0
        mean_delta_translation = fmean(delta_translation_magnitudes) if delta_translation_magnitudes else 0.0

        if rotation_range >= self.config.rotation_threshold_rad:
            joint_type = "revolute"
            axis, pivot, q_values, confidence = self._infer_revolute(relative_poses, delta_rotations, delta_translations)
        elif translation_range_norm >= self.config.translation_threshold_m:
            joint_type = "prismatic"
            axis, pivot, q_values, confidence = self._infer_prismatic(relative_poses, delta_translations, translation_range)
        else:
            joint_type = "fixed"
            axis = [0.0, 0.0, 1.0]
            pivot = _mean_point([pose["translation"] for pose in relative_poses])
            q_values = [0.0 for _ in relative_poses]
            confidence = 0.25

        limits = [min(q_values), max(q_values)] if q_values else [0.0, 0.0]
        return {
            "name": f"{part.get('name', f'part_{part.get('part_id', 0)}')}_joint",
            "parent_part_id": anchor_part_id,
            "parent_name": anchor_part_name,
            "child_part_id": int(part.get("part_id", 0)),
            "child_name": str(part.get("name", f"part_{part.get('part_id', 0)}")),
            "joint_type": joint_type,
            "axis": [float(value) for value in axis],
            "pivot": [float(value) for value in pivot],
            "limits": [float(value) for value in limits],
            "confidence": float(confidence),
            "metrics": {
                "rotation_range_rad": rotation_range,
                "translation_range": translation_range,
                "translation_range_norm": translation_range_norm,
                "mean_delta_angle_rad": mean_delta_angle,
                "mean_delta_translation_m": mean_delta_translation,
                "valid_sample_count": len(relative_poses),
            },
            "q_samples": [
                {
                    "frame_index": int(pose["frame_index"]),
                    "timestamp_s": float(pose["timestamp_s"]),
                    "q": float(q_value),
                }
                for pose, q_value in zip(relative_poses, q_values)
            ],
        }

    def _infer_revolute(
        self,
        relative_poses: list[dict[str, Any]],
        delta_rotations: list[list[list[float]]],
        delta_translations: list[list[float]],
    ) -> tuple[list[float], list[float], list[float], float]:
        axis_votes: list[list[float]] = []
        weights: list[float] = []
        for delta_rotation in delta_rotations:
            angle = _rotation_angle(delta_rotation)
            if angle < self.config.delta_rotation_epsilon_rad:
                continue
            axis = _rotation_axis(delta_rotation, angle)
            if axis_votes and _dot(axis, axis_votes[0]) < 0.0:
                axis = _scale(axis, -1.0)
            axis_votes.append(axis)
            weights.append(angle)

        if axis_votes:
            weighted = [0.0, 0.0, 0.0]
            for axis, weight in zip(axis_votes, weights):
                weighted = _add(weighted, _scale(axis, weight))
            axis = _normalize(weighted, fallback=axis_votes[0])
            alignment = fmean(abs(_dot(axis, vote)) for vote in axis_votes)
        else:
            axis = [0.0, 0.0, 1.0]
            alignment = 0.0

        ata = [[0.0, 0.0, 0.0] for _ in range(3)]
        atb = [0.0, 0.0, 0.0]
        for delta_rotation, delta_translation in zip(delta_rotations, delta_translations):
            angle = _rotation_angle(delta_rotation)
            if angle < self.config.delta_rotation_epsilon_rad:
                continue
            block = [
                [1.0 - delta_rotation[row][col] if row == col else -delta_rotation[row][col] for col in range(3)]
                for row in range(3)
            ]
            block_t = _transpose3(block)
            block_ata = _matmul3(block_t, block)
            block_atb = _matvec3(block_t, delta_translation)
            for row in range(3):
                for col in range(3):
                    ata[row][col] += block_ata[row][col]
                atb[row] += block_atb[row]

        pivot = _solve_3x3(ata, atb, damping=1e-5)
        pivot_perp = _subtract(pivot, _scale(axis, _dot(pivot, axis)))
        mean_translation = _mean_point([pose["translation"] for pose in relative_poses])
        mean_parallel = _dot(mean_translation, axis)
        pivot = _add(pivot_perp, _scale(axis, mean_parallel))

        reference_rotation = relative_poses[0]["rotation"]
        q_values = [
            _signed_angle_about_axis(
                _matmul3(pose["rotation"], _transpose3(reference_rotation)),
                axis,
            )
            for pose in relative_poses
        ]
        confidence = min(1.0, 0.45 + 0.45 * alignment + 0.10 * min(1.0, len(relative_poses) / 20.0))
        return axis, pivot, q_values, confidence

    def _infer_prismatic(
        self,
        relative_poses: list[dict[str, Any]],
        delta_translations: list[list[float]],
        translation_range: list[float],
    ) -> tuple[list[float], list[float], list[float], float]:
        direction_votes: list[list[float]] = []
        weights: list[float] = []
        for delta_translation in delta_translations:
            magnitude = _norm(delta_translation)
            if magnitude < self.config.delta_translation_epsilon_m:
                continue
            direction = _scale(delta_translation, 1.0 / magnitude)
            if direction_votes and _dot(direction, direction_votes[0]) < 0.0:
                direction = _scale(direction, -1.0)
            direction_votes.append(direction)
            weights.append(magnitude)

        if direction_votes:
            weighted = [0.0, 0.0, 0.0]
            for direction, weight in zip(direction_votes, weights):
                weighted = _add(weighted, _scale(direction, weight))
            axis = _normalize(weighted, fallback=direction_votes[0])
            alignment = fmean(abs(_dot(axis, vote)) for vote in direction_votes)
        else:
            axis = _normalize(translation_range, fallback=[1.0, 0.0, 0.0])
            alignment = 0.0

        translations = [pose["translation"] for pose in relative_poses]
        pivot = _mean_point(translations)
        reference_translation = translations[0]
        q_values = [_dot(_subtract(translation, reference_translation), axis) for translation in translations]
        confidence = min(1.0, 0.45 + 0.45 * alignment + 0.10 * min(1.0, len(relative_poses) / 20.0))
        return axis, pivot, q_values, confidence
