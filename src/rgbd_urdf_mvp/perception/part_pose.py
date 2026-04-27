from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

from .part_segmentation import part_name_lookup
from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class PartPoseEstimationConfig:
    input_path: str | Path
    output_json: str | Path | None = None
    min_points_per_part: int = 24
    anchor_part_id: int | None = None
    jacobi_iterations: int = 32


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


def _identity3() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


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


def _columns_to_matrix(columns: list[list[float]]) -> list[list[float]]:
    return [[columns[col][row] for col in range(3)] for row in range(3)]


def _matrix_to_columns(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(vec: list[float], scalar: float) -> list[float]:
    return [scalar * value for value in vec]


def _centroid(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    inv = 1.0 / float(len(points))
    return [
        sum(point[axis] for point in points) * inv
        for axis in range(3)
    ]


def _covariance(points: list[list[float]], center: list[float]) -> list[list[float]]:
    if not points:
        return _identity3()
    inv = 1.0 / float(len(points))
    cov = [[0.0, 0.0, 0.0] for _ in range(3)]
    for point in points:
        delta = [point[axis] - center[axis] for axis in range(3)]
        for row in range(3):
            for col in range(3):
                cov[row][col] += delta[row] * delta[col] * inv
    return cov


def _jacobi_eigendecomposition_symmetric(
    matrix: list[list[float]],
    max_iterations: int = 32,
) -> tuple[list[float], list[list[float]]]:
    a = [[float(value) for value in row] for row in matrix]
    v = _identity3()
    for _ in range(max_iterations):
        p, q = 0, 1
        max_offdiag = abs(a[p][q])
        for row, col in ((0, 1), (0, 2), (1, 2)):
            value = abs(a[row][col])
            if value > max_offdiag:
                max_offdiag = value
                p, q = row, col
        if max_offdiag < 1e-10:
            break

        if abs(a[q][q] - a[p][p]) < 1e-12:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        a[p][p] = cos_a * cos_a * app - 2.0 * sin_a * cos_a * apq + sin_a * sin_a * aqq
        a[q][q] = sin_a * sin_a * app + 2.0 * sin_a * cos_a * apq + cos_a * cos_a * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(3):
            if k in {p, q}:
                continue
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = cos_a * akp - sin_a * akq
            a[p][k] = a[k][p]
            a[k][q] = sin_a * akp + cos_a * akq
            a[q][k] = a[k][q]

        for k in range(3):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = cos_a * vkp - sin_a * vkq
            v[k][q] = sin_a * vkp + cos_a * vkq

    eigenvalues = [a[index][index] for index in range(3)]
    eigenvectors = _matrix_to_columns(v)
    return eigenvalues, eigenvectors


def _permutation_parity(perm: tuple[int, int, int]) -> int:
    inversions = 0
    for index in range(len(perm)):
        for next_index in range(index + 1, len(perm)):
            if perm[index] > perm[next_index]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _estimate_cloud_frame(
    points: list[list[float]],
    ref_basis: list[list[float]] | None = None,
    jacobi_iterations: int = 32,
) -> tuple[list[list[float]], list[float], list[float]]:
    center = _centroid(points)
    cov = _covariance(points, center)
    eigenvalues, eigenvectors = _jacobi_eigendecomposition_symmetric(cov, jacobi_iterations)
    sorted_items = sorted(
        zip(eigenvalues, eigenvectors),
        key=lambda item: item[0],
        reverse=True,
    )
    columns = [_normalize(list(item[1]), fallback=axis) for item, axis in zip(sorted_items, _identity3())]
    first = _normalize(columns[0], fallback=[1.0, 0.0, 0.0])
    second = _normalize(columns[1], fallback=[0.0, 1.0, 0.0])
    third = _normalize(_cross(first, second), fallback=[0.0, 0.0, 1.0])
    second = _normalize(_cross(third, first), fallback=[0.0, 1.0, 0.0])
    columns = [first, second, third]

    if ref_basis is not None:
        columns = _align_basis_to_reference(columns, ref_basis)

    return _columns_to_matrix(columns), center, [float(item[0]) for item in sorted_items]


def _align_basis_to_reference(
    basis_columns: list[list[float]],
    ref_columns: list[list[float]],
) -> list[list[float]]:
    best_score = -math.inf
    best_columns = basis_columns
    for perm in permutations(range(3)):
        parity = _permutation_parity(perm)
        for signs in product((-1.0, 1.0), repeat=3):
            if parity * int(signs[0] * signs[1] * signs[2]) != 1:
                continue
            candidate = [
                _scale(basis_columns[perm[index]], signs[index])
                for index in range(3)
            ]
            score = sum(_dot(candidate[index], ref_columns[index]) for index in range(3))
            if score > best_score:
                best_score = score
                best_columns = candidate
    first = _normalize(best_columns[0], fallback=[1.0, 0.0, 0.0])
    second = _normalize(best_columns[1], fallback=[0.0, 1.0, 0.0])
    third = _normalize(_cross(first, second), fallback=[0.0, 0.0, 1.0])
    second = _normalize(_cross(third, first), fallback=[0.0, 1.0, 0.0])
    return [first, second, third]


def _rotation_matrix_to_quaternion_xyzw(rotation: list[list[float]]) -> list[float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2][1] - rotation[1][2]) / s
        qy = (rotation[0][2] - rotation[2][0]) / s
        qz = (rotation[1][0] - rotation[0][1]) / s
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / s
        qx = 0.25 * s
        qy = (rotation[0][1] + rotation[1][0]) / s
        qz = (rotation[0][2] + rotation[2][0]) / s
    elif rotation[1][1] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / s
        qx = (rotation[0][1] + rotation[1][0]) / s
        qy = 0.25 * s
        qz = (rotation[1][2] + rotation[2][1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / s
        qx = (rotation[0][2] + rotation[2][0]) / s
        qy = (rotation[1][2] + rotation[2][1]) / s
        qz = 0.25 * s
    return [qx, qy, qz, qw]


def _rotation_matrix_to_rpy(rotation: list[list[float]]) -> list[float]:
    sy = math.sqrt(rotation[0][0] * rotation[0][0] + rotation[1][0] * rotation[1][0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        pitch = math.atan2(-rotation[2][0], sy)
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        pitch = math.atan2(-rotation[2][0], sy)
        yaw = 0.0
    return [roll, pitch, yaw]


def _matrix4(rotation: list[list[float]], translation: list[float]) -> list[list[float]]:
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _pose_payload(rotation: list[list[float]], translation: list[float]) -> dict[str, Any]:
    return {
        "translation": [float(value) for value in translation],
        "rotation_matrix": [[float(value) for value in row] for row in rotation],
        "quaternion_xyzw": [float(value) for value in _rotation_matrix_to_quaternion_xyzw(rotation)],
        "rotation_rpy": [float(value) for value in _rotation_matrix_to_rpy(rotation)],
        "matrix4": _matrix4(rotation, translation),
    }


def _relative_pose(
    anchor_rotation: list[list[float]],
    anchor_translation: list[float],
    child_rotation: list[list[float]],
    child_translation: list[float],
) -> tuple[list[list[float]], list[float]]:
    anchor_inv = _transpose3(anchor_rotation)
    relative_rotation = _matmul3(anchor_inv, child_rotation)
    relative_translation = _matvec3(anchor_inv, _subtract(child_translation, anchor_translation))
    return relative_rotation, relative_translation


def _rotation_angle_from_matrix(rotation: list[list[float]]) -> float:
    trace = max(-1.0, min(3.0, rotation[0][0] + rotation[1][1] + rotation[2][2]))
    cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.acos(cosine)


def _local_bounds(
    points: list[list[float]],
    rotation: list[list[float]],
    translation: list[float],
) -> dict[str, list[float]]:
    if not points:
        return {"lower": [0.0, 0.0, 0.0], "upper": [0.0, 0.0, 0.0]}
    inv = _transpose3(rotation)
    locals_ = [
        _matvec3(inv, _subtract(point[:3], translation))
        for point in points
    ]
    lower = [min(point[axis] for point in locals_) for axis in range(3)]
    upper = [max(point[axis] for point in locals_) for axis in range(3)]
    return {"lower": lower, "upper": upper}


def _estimate_confidence(point_count: int, reference_count: int, eigenvalues: list[float]) -> float:
    if point_count <= 0 or reference_count <= 0:
        return 0.0
    visibility = min(1.0, float(point_count) / float(reference_count))
    if not eigenvalues or max(eigenvalues) <= 1e-9:
        anisotropy = 0.5
    else:
        anisotropy = max(0.2, min(1.0, (max(eigenvalues) - min(eigenvalues)) / max(eigenvalues)))
    return min(1.0, visibility * anisotropy + 0.15)


def _load_part_labeled_pointcloud(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    if input_path.suffix.lower() == ".json":
        manifest = load_json(input_path)
        pointcloud_path = Path(manifest["pointcloud_4d_path"])
        meta = dict(manifest)
        output_dir = input_path.parent
    else:
        pointcloud_path = input_path
        meta = {"pointcloud_4d_path": str(pointcloud_path), "frame_count": 0, "part_ids_present": []}
        output_dir = input_path.parent

    lines = pointcloud_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Unsupported pointcloud file: {pointcloud_path}")

    header_end = None
    properties: list[str] = []
    vertex_count = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
        elif stripped.startswith("property"):
            properties.append(stripped.split()[-1])
        elif stripped == "end_header":
            header_end = index
            break
    if header_end is None:
        raise ValueError(f"Missing PLY header terminator: {pointcloud_path}")

    property_indices = {name: properties.index(name) for name in properties}
    required = {"x", "y", "z", "frame_index", "time"}
    if not required.issubset(property_indices):
        raise ValueError(f"Pointcloud missing required properties: {sorted(required - set(property_indices))}")
    if "part_id" not in property_indices:
        raise ValueError("Pointcloud does not contain part_id labels. Re-run fusion with part segmentation enabled.")

    frames: dict[int, dict[str, Any]] = {}
    for raw in lines[header_end + 1 : header_end + 1 + vertex_count]:
        if not raw.strip():
            continue
        items = raw.split()
        frame_index = int(items[property_indices["frame_index"]])
        time_s = float(items[property_indices["time"]])
        point = [
            float(items[property_indices["x"]]),
            float(items[property_indices["y"]]),
            float(items[property_indices["z"]]),
            int(items[property_indices["part_id"]]),
        ]
        frame = frames.setdefault(frame_index, {"frame_index": frame_index, "time_s": time_s, "points": []})
        frame["points"].append(point)
        frame["time_s"] = time_s

    sorted_frames = [frames[index] for index in sorted(frames)]
    meta["frame_count"] = int(meta.get("frame_count", len(sorted_frames)) or len(sorted_frames))
    return meta, sorted_frames, output_dir


def _choose_anchor_part_id(meta: dict[str, Any], per_part_counts: dict[int, int], override: int | None) -> int:
    if override is not None and override in per_part_counts:
        return int(override)
    part_segmentation = meta.get("part_segmentation")
    if isinstance(part_segmentation, dict):
        for raw_part in part_segmentation.get("parts", []):
            if isinstance(raw_part, dict) and str(raw_part.get("role")) in {"base", "static"}:
                part_id = int(raw_part.get("part_id", 0))
                if part_id in per_part_counts:
                    return part_id
    if not per_part_counts:
        return 0
    return max(sorted(per_part_counts), key=lambda part_id: per_part_counts[part_id])


class PartPoseEstimator:
    def __init__(self, config: PartPoseEstimationConfig) -> None:
        self.config = config

    def estimate(self) -> Path:
        input_path = Path(self.config.input_path).resolve()
        meta, frames, output_dir = _load_part_labeled_pointcloud(input_path)
        part_names = part_name_lookup(meta.get("part_segmentation"))

        part_counts: dict[int, int] = {}
        per_part_frames: dict[int, dict[int, list[list[float]]]] = {}
        for frame in frames:
            for point in frame["points"]:
                part_id = int(point[3])
                if part_id <= 0:
                    continue
                part_counts[part_id] = part_counts.get(part_id, 0) + 1
                per_part_frames.setdefault(part_id, {}).setdefault(int(frame["frame_index"]), []).append(point[:3])

        if not per_part_frames:
            raise ValueError("No labeled part points found in the input pointcloud.")

        anchor_part_id = _choose_anchor_part_id(meta, part_counts, self.config.anchor_part_id)
        valid_part_ids = sorted(per_part_frames)
        frame_times = {int(frame["frame_index"]): float(frame["time_s"]) for frame in frames}

        part_tracks: dict[int, dict[str, Any]] = {}
        for part_id in valid_part_ids:
            frame_clouds = per_part_frames[part_id]
            reference_frame_index = max(
                sorted(frame_clouds),
                key=lambda index: len(frame_clouds[index]),
            )
            reference_cloud = frame_clouds[reference_frame_index]
            if len(reference_cloud) < self.config.min_points_per_part:
                continue
            reference_rotation, reference_translation, reference_eigenvalues = _estimate_cloud_frame(
                reference_cloud,
                ref_basis=None,
                jacobi_iterations=self.config.jacobi_iterations,
            )
            reference_basis = _matrix_to_columns(reference_rotation)
            samples: list[dict[str, Any]] = []
            missing_frame_indices: list[int] = []
            for frame in frames:
                frame_index = int(frame["frame_index"])
                time_s = float(frame["time_s"])
                cloud = frame_clouds.get(frame_index, [])
                if len(cloud) < self.config.min_points_per_part:
                    missing_frame_indices.append(frame_index)
                    samples.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": time_s,
                            "valid": False,
                            "point_count": len(cloud),
                            "visibility_ratio": 0.0,
                            "confidence": 0.0,
                        }
                    )
                    continue
                rotation, translation, eigenvalues = _estimate_cloud_frame(
                    cloud,
                    ref_basis=reference_basis,
                    jacobi_iterations=self.config.jacobi_iterations,
                )
                confidence = _estimate_confidence(len(cloud), len(reference_cloud), eigenvalues)
                pose = _pose_payload(rotation, translation)
                pose.update(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": time_s,
                        "valid": True,
                        "point_count": len(cloud),
                        "visibility_ratio": min(1.0, len(cloud) / max(1, len(reference_cloud))),
                        "confidence": confidence,
                        "centroid_world": [float(value) for value in translation],
                    }
                )
                samples.append(pose)

            track = {
                "part_id": part_id,
                "name": part_names.get(part_id, f"part_{part_id}"),
                "role": self._part_role(meta, part_id, anchor_part_id),
                "reference_frame_index": reference_frame_index,
                "reference_timestamp_s": frame_times.get(reference_frame_index, 0.0),
                "reference_point_count": len(reference_cloud),
                "reference_centroid_world": [float(value) for value in reference_translation],
                "reference_eigenvalues": [float(value) for value in reference_eigenvalues],
                "canonical_frame": _pose_payload(reference_rotation, reference_translation),
                "reference_local_bounds": _local_bounds(reference_cloud, reference_rotation, reference_translation),
                "samples": samples,
                "missing_frame_indices": missing_frame_indices,
            }
            part_tracks[part_id] = track

        if not part_tracks:
            raise ValueError(
                "No part tracks met the minimum point threshold. Lower --min-points-per-part or regenerate a denser pointcloud."
            )

        if anchor_part_id not in part_tracks:
            anchor_part_id = max(sorted(part_tracks), key=lambda part_id: part_tracks[part_id]["reference_point_count"])

        anchor_track = part_tracks[anchor_part_id]
        anchor_samples = {
            int(sample["frame_index"]): sample
            for sample in anchor_track["samples"]
            if bool(sample.get("valid", False))
        }

        for part_id, track in part_tracks.items():
            relative_translations: list[list[float]] = []
            relative_rotation_angles: list[float] = []
            for sample in track["samples"]:
                frame_index = int(sample["frame_index"])
                if not bool(sample.get("valid", False)):
                    sample["relative_to_anchor"] = None
                    continue
                anchor_sample = anchor_samples.get(frame_index)
                if anchor_sample is None:
                    sample["relative_to_anchor"] = None
                    continue
                relative_rotation, relative_translation = _relative_pose(
                    anchor_rotation=anchor_sample["rotation_matrix"],
                    anchor_translation=anchor_sample["translation"],
                    child_rotation=sample["rotation_matrix"],
                    child_translation=sample["translation"],
                )
                relative_pose = _pose_payload(relative_rotation, relative_translation)
                sample["relative_to_anchor"] = relative_pose
                relative_translations.append(relative_translation)
                relative_rotation_angles.append(_rotation_angle_from_matrix(relative_rotation))

            track["relative_motion_summary"] = self._motion_summary(relative_translations, relative_rotation_angles)

        artifact = {
            "input_path": str(input_path),
            "pointcloud_path": str(meta.get("pointcloud_4d_path", input_path)),
            "estimator": "pca-reference-alignment",
            "frame_count": len(frames),
            "anchor_part_id": anchor_part_id,
            "anchor_part_name": part_tracks[anchor_part_id]["name"],
            "parts": [part_tracks[part_id] for part_id in sorted(part_tracks)],
        }

        output_json = (
            Path(self.config.output_json).resolve()
            if self.config.output_json is not None
            else output_dir / "part_poses.json"
        )
        save_json(artifact, output_json)
        return output_json

    def _part_role(self, meta: dict[str, Any], part_id: int, anchor_part_id: int) -> str:
        part_segmentation = meta.get("part_segmentation")
        if isinstance(part_segmentation, dict):
            for raw_part in part_segmentation.get("parts", []):
                if isinstance(raw_part, dict) and int(raw_part.get("part_id", 0)) == part_id:
                    return str(raw_part.get("role", "unknown"))
        return "base" if part_id == anchor_part_id else "moving"

    def _motion_summary(
        self,
        relative_translations: list[list[float]],
        relative_rotation_angles: list[float],
    ) -> dict[str, Any]:
        if not relative_translations:
            return {
                "translation_range": [0.0, 0.0, 0.0],
                "rotation_angle_range_rad": 0.0,
            }
        translation_range = [
            max(item[axis] for item in relative_translations) - min(item[axis] for item in relative_translations)
            for axis in range(3)
        ]
        rotation_range = (
            max(relative_rotation_angles) - min(relative_rotation_angles)
            if relative_rotation_angles
            else 0.0
        )
        return {
            "translation_range": [float(value) for value in translation_range],
            "rotation_angle_range_rad": float(rotation_range),
        }
