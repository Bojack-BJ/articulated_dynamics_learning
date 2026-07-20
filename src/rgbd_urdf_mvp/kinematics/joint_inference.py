from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.serialization import load_json, save_json
from ..perception.quality_weights import observation_weight


@dataclass(slots=True)
class JointInferenceConfig:
    input_path: str | Path
    output_json: str | Path | None = None
    rotation_threshold_rad: float = 0.20
    translation_threshold_m: float = 0.02
    delta_rotation_epsilon_rad: float = 0.01
    delta_translation_epsilon_m: float = 1e-4
    use_track_translation_axis: bool = True
    use_track_residual_type: bool = True
    track_residual_requires_pose_candidate: bool = False
    track_residual_decision_ratio: float = 0.85
    min_track_residual_samples: int = 20
    min_track_residual_tracks: int = 12
    robust_track_model_trim_ratio: float = 0.0
    quality_weighted_replay: bool = False
    min_replay_weight: float = 0.2
    mujoco_prior: str = "auto"
    orient_parent_by_motion: bool = False
    parent_orientation_motion_margin_m: float = 0.005


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


def _distance(a: list[float], b: list[float]) -> float:
    return _norm(_subtract(a, b))


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


def _identity3() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


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


def _rotate_vector_about_axis(vector: list[float], axis: list[float], angle: float) -> list[float]:
    unit_axis = _normalize(axis, fallback=[0.0, 0.0, 1.0])
    cos_q = math.cos(angle)
    sin_q = math.sin(angle)
    return _add(
        _add(_scale(vector, cos_q), _scale(_cross(unit_axis, vector), sin_q)),
        _scale(unit_axis, _dot(unit_axis, vector) * (1.0 - cos_q)),
    )


def _signed_angle_between_about_axis(a: list[float], b: list[float], axis: list[float]) -> float | None:
    unit_axis = _normalize(axis, fallback=[0.0, 0.0, 1.0])
    a_perp = _subtract(a, _scale(unit_axis, _dot(a, unit_axis)))
    b_perp = _subtract(b, _scale(unit_axis, _dot(b, unit_axis)))
    a_norm = _norm(a_perp)
    b_norm = _norm(b_perp)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return None
    a_unit = _scale(a_perp, 1.0 / a_norm)
    b_unit = _scale(b_perp, 1.0 / b_norm)
    return math.atan2(_dot(unit_axis, _cross(a_unit, b_unit)), max(-1.0, min(1.0, _dot(a_unit, b_unit))))


def _parse_vec3(raw: str | None, fallback: list[float]) -> list[float]:
    if raw is None:
        return list(fallback)
    values = [float(item) for item in raw.split()]
    if len(values) != 3:
        return list(fallback)
    return values


def _parse_joint_range(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [float(item) for item in raw.split()]
    if len(values) != 2:
        return None
    return values


def _quat_wxyz_to_matrix(raw: str | None) -> list[list[float]]:
    if raw is None:
        return _identity3()
    values = [float(item) for item in raw.split()]
    if len(values) != 4:
        return _identity3()
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return _identity3()
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _world_body_transforms(model_path: Path) -> dict[str, tuple[list[list[float]], list[float]]]:
    root = ET.parse(model_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return {}

    transforms: dict[str, tuple[list[list[float]], list[float]]] = {}

    def visit(body: ET.Element, parent_rotation: list[list[float]], parent_translation: list[float]) -> None:
        local_rotation = _quat_wxyz_to_matrix(body.get("quat"))
        local_translation = _parse_vec3(body.get("pos"), [0.0, 0.0, 0.0])
        world_rotation = _matmul3(parent_rotation, local_rotation)
        world_translation = _add(parent_translation, _matvec3(parent_rotation, local_translation))
        body_name = body.get("name")
        if body_name:
            transforms[body_name] = (world_rotation, world_translation)
        for child in body.findall("body"):
            visit(child, world_rotation, world_translation)

    for body in worldbody.findall("body"):
        visit(body, _identity3(), [0.0, 0.0, 0.0])
    return transforms


def _joint_defs_from_mjcf(model_path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(model_path).getroot()
    body_transforms = _world_body_transforms(model_path)
    joints: dict[str, dict[str, Any]] = {}
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if not body_name or body_name not in body_transforms:
            continue
        body_rotation, body_translation = body_transforms[body_name]
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if not joint_name:
                continue
            local_axis = _parse_vec3(joint.get("axis"), [0.0, 0.0, 1.0])
            local_pos = _parse_vec3(joint.get("pos"), [0.0, 0.0, 0.0])
            axis_world = _normalize(_matvec3(body_rotation, local_axis), fallback=[0.0, 0.0, 1.0])
            pivot_world = _add(_matvec3(body_rotation, local_pos), body_translation)
            joint_type = joint.get("type", "hinge")
            joints[joint_name] = {
                "name": joint_name,
                "body_name": body_name,
                "joint_type": "prismatic" if joint_type == "slide" else "revolute" if joint_type == "hinge" else joint_type,
                "axis_world": axis_world,
                "pivot_world": pivot_world,
                "limits": _parse_joint_range(joint.get("range")),
            }
    return joints


def _part_pose_by_id(artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(part.get("part_id", 0)): part
        for part in artifact.get("parts", [])
        if isinstance(part, dict)
    }


def _canonical_parent_transform(part: dict[str, Any]) -> tuple[list[list[float]], list[float]] | None:
    canonical = part.get("canonical_frame")
    if not isinstance(canonical, dict):
        return None
    rotation = canonical.get("rotation_matrix")
    translation = canonical.get("translation")
    if not isinstance(rotation, list) or not isinstance(translation, list):
        return None
    return (
        [[float(value) for value in row] for row in rotation],
        [float(value) for value in translation],
    )


def _load_mujoco_joint_priors(artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    input_path = artifact.get("input_path")
    if not isinstance(input_path, str):
        return {}
    manifest_path = Path(input_path)
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    episode_path_raw = manifest.get("episode_path") or manifest.get("input_episode_path")
    if not isinstance(episode_path_raw, str):
        return {}
    episode_path = Path(episode_path_raw)
    if not episode_path.is_absolute():
        episode_path = manifest_path.parent / episode_path
    if not episode_path.exists():
        return {}
    episode = load_json(episode_path)
    model_path_raw = episode.get("metadata", {}).get("model_path")
    if not isinstance(model_path_raw, str):
        return {}
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = episode_path.parent / model_path
    if not model_path.exists():
        return {}

    joint_defs = _joint_defs_from_mjcf(model_path)
    part_segmentation = manifest.get("part_segmentation")
    if not isinstance(part_segmentation, dict):
        return {}
    parts_by_id = _part_pose_by_id(artifact)
    frames = episode.get("frames", [])
    priors: dict[int, dict[str, Any]] = {}

    for raw_part in part_segmentation.get("parts", []):
        if not isinstance(raw_part, dict):
            continue
        part_id = int(raw_part.get("part_id", 0))
        joint_names = [str(name) for name in raw_part.get("joint_names", []) if isinstance(name, str)]
        if not joint_names:
            continue
        parent_part_id = int(raw_part.get("parent_part_id") or artifact.get("anchor_part_id", 0))
        parent_part = parts_by_id.get(parent_part_id)
        if parent_part is None:
            continue
        parent_transform = _canonical_parent_transform(parent_part)
        if parent_transform is None:
            continue
        parent_rotation, parent_translation = parent_transform
        parent_inv = _transpose3(parent_rotation)
        for joint_name in joint_names:
            joint_def = joint_defs.get(joint_name)
            if joint_def is None:
                continue
            axis_parent = _normalize(_matvec3(parent_inv, joint_def["axis_world"]), fallback=[0.0, 0.0, 1.0])
            pivot_parent = _matvec3(parent_inv, _subtract(joint_def["pivot_world"], parent_translation))
            q_by_frame: dict[int, float] = {}
            for frame_index, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    continue
                sample_frame_index = int(frame.get("frame_index", frame_index))
                joint_positions = frame.get("action_log", {}).get("joint_positions", {})
                if isinstance(joint_positions, dict) and joint_name in joint_positions:
                    q_by_frame[sample_frame_index] = float(joint_positions[joint_name])
            priors[part_id] = {
                **joint_def,
                "joint_name": joint_name,
                "parent_part_id": parent_part_id,
                "axis": axis_parent,
                "pivot": pivot_parent,
                "q_by_frame": q_by_frame,
                "source": "mujoco-mjcf",
                "model_path": str(model_path),
            }
            break
    return priors


def _load_track_artifact(part_pose_artifact: dict[str, Any]) -> dict[str, Any] | None:
    input_path = part_pose_artifact.get("input_path")
    if not isinstance(input_path, str):
        return None
    track_path = Path(input_path)
    if not track_path.exists():
        return None
    try:
        payload = load_json(track_path)
    except Exception:
        return None
    return payload if isinstance(payload.get("tracks"), list) else None


def _tracks_by_part(track_artifact: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    if track_artifact is None:
        return {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for track in track_artifact.get("tracks", []):
        if not isinstance(track, dict):
            continue
        part_id = int(track.get("part_id", 0))
        if part_id <= 0:
            continue
        grouped.setdefault(part_id, []).append(track)
    return grouped


class JointInferencer:
    def __init__(self, config: JointInferenceConfig) -> None:
        self.config = config
        self._tracks_by_part: dict[int, list[dict[str, Any]]] = {}

    def infer(self) -> Path:
        input_path = Path(self.config.input_path).resolve()
        artifact = load_json(input_path)
        parts = artifact.get("parts", [])
        self._tracks_by_part = _tracks_by_part(_load_track_artifact(artifact))
        anchor_part_id = int(artifact.get("anchor_part_id", 0))
        anchor_part_name = str(artifact.get("anchor_part_name", "anchor"))
        mujoco_priors = {}
        if self.config.mujoco_prior != "off":
            mujoco_priors = _load_mujoco_joint_priors(artifact)
            if self.config.mujoco_prior == "required" and not mujoco_priors:
                raise ValueError("No MuJoCo joint priors found for this part pose artifact.")

        joints: list[dict[str, Any]] = []
        for part in parts:
            part_id = int(part.get("part_id", 0))
            if part_id == anchor_part_id:
                continue
            inferred = self._infer_part_joint(part, anchor_part_id, anchor_part_name, mujoco_priors.get(part_id))
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
                "mujoco_prior_mode": self.config.mujoco_prior,
                "mujoco_prior_count": len(mujoco_priors),
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
        mujoco_prior: dict[str, Any] | None = None,
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

        rev_axis, rev_pivot, rev_q_values, rev_confidence = self._infer_revolute(
            relative_poses,
            delta_rotations,
            delta_translations,
        )
        prism_axis, prism_pivot, prism_q_values, prism_confidence = self._infer_prismatic(
            relative_poses,
            delta_translations,
            translation_range,
        )
        track_translation_axis = self._track_translation_axis(
            int(part.get("part_id", 0)),
            {int(pose["frame_index"]) for pose in relative_poses},
        )
        if self.config.use_track_translation_axis and track_translation_axis is not None:
            prism_axis = track_translation_axis["axis"]
            prism_q_values = self._prismatic_q_values_for_axis(relative_poses, prism_axis)
            prism_confidence = max(prism_confidence, min(1.0, 0.65 + 0.35 * min(1.0, track_translation_axis["track_count"] / 32.0)))
        track_model_comparison = self._compare_track_motion_models(
            part_id=int(part.get("part_id", 0)),
            revolute_axis=rev_axis,
            revolute_pivot=rev_pivot,
            prismatic_axis=prism_axis,
            frame_indices=[int(pose["frame_index"]) for pose in relative_poses],
            rotation_candidate=rotation_range >= self.config.rotation_threshold_rad,
            translation_candidate=translation_range_norm >= self.config.translation_threshold_m,
        )

        if mujoco_prior is not None:
            joint_type = str(mujoco_prior.get("joint_type", "fixed"))
            axis = [float(value) for value in mujoco_prior.get("axis", [0.0, 0.0, 1.0])]
            pivot = [float(value) for value in mujoco_prior.get("pivot", [0.0, 0.0, 0.0])]
            q_by_frame = mujoco_prior.get("q_by_frame", {})
            if isinstance(q_by_frame, dict) and q_by_frame:
                q_values = [
                    float(q_by_frame.get(int(pose["frame_index"]), 0.0))
                    for pose in relative_poses
                ]
            elif joint_type == "prismatic":
                q_values = prism_q_values
            elif joint_type == "revolute":
                q_values = rev_q_values
            else:
                q_values = [0.0 for _ in relative_poses]
            confidence = 1.0
        else:
            joint_type = self._select_joint_type(rotation_range, translation_range_norm, track_model_comparison)
            if joint_type == "prismatic":
                axis, pivot, q_values, confidence = prism_axis, prism_pivot, prism_q_values, prism_confidence
            elif joint_type == "revolute":
                axis, pivot, q_values, confidence = rev_axis, rev_pivot, rev_q_values, rev_confidence
            else:
                axis = [0.0, 0.0, 1.0]
                pivot = _mean_point([pose["translation"] for pose in relative_poses])
                q_values = [0.0 for _ in relative_poses]
                confidence = 0.25

        limits = [min(q_values), max(q_values)] if q_values else [0.0, 0.0]
        if mujoco_prior is not None and mujoco_prior.get("limits") is not None:
            limits = [float(value) for value in mujoco_prior["limits"]]
        parent_motion = self._part_motion_score(anchor_part_id)
        child_part_id = int(part.get("part_id", 0))
        child_motion = self._part_motion_score(child_part_id)
        parent_orientation = {
            "mode": "motion_heuristic" if self.config.orient_parent_by_motion else "anchor_parent",
            "applied": False,
            "parent_motion_score_m": parent_motion,
            "child_motion_score_m": child_motion,
            "motion_margin_m": float(self.config.parent_orientation_motion_margin_m),
        }
        parent_part_id = anchor_part_id
        parent_name = anchor_part_name
        child_name = str(part.get("name", f"part_{child_part_id}"))
        if (
            self.config.orient_parent_by_motion
            and parent_motion is not None
            and child_motion is not None
            and parent_motion > child_motion + self.config.parent_orientation_motion_margin_m
        ):
            parent_part_id = child_part_id
            parent_name = child_name
            child_part_id = anchor_part_id
            child_name = anchor_part_name
            q_values = [-value for value in q_values]
            limits = [-limits[1], -limits[0]]
            parent_orientation["applied"] = True
            parent_orientation["reason"] = "anchor_cluster_has_larger_motion_than_child_cluster"

        return {
            "name": f"{part.get('name', f'part_{part.get('part_id', 0)}')}_joint",
            "parent_part_id": parent_part_id,
            "parent_name": parent_name,
            "child_part_id": child_part_id,
            "child_name": child_name,
            "joint_type": joint_type,
            "axis": [float(value) for value in axis],
            "pivot": [float(value) for value in pivot],
            "limits": [float(value) for value in limits],
            "confidence": float(confidence),
            "prior": (
                {
                    "source": str(mujoco_prior.get("source", "unknown")),
                    "joint_name": str(mujoco_prior.get("joint_name", "")),
                    "model_path": str(mujoco_prior.get("model_path", "")),
                }
                if mujoco_prior is not None
                else None
            ),
            "metrics": {
                "rotation_range_rad": rotation_range,
                "translation_range": translation_range,
                "translation_range_norm": translation_range_norm,
                "mean_delta_angle_rad": mean_delta_angle,
                "mean_delta_translation_m": mean_delta_translation,
                "valid_sample_count": len(relative_poses),
                "track_translation_axis": track_translation_axis,
                "track_model_comparison": track_model_comparison,
                "parent_orientation": parent_orientation,
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

    def _part_motion_score(self, part_id: int) -> float | None:
        tracks = self._tracks_by_part.get(part_id, [])
        motions = []
        for track in tracks:
            samples = [
                sample
                for sample in track.get("samples", [])
                if _valid_track_sample(sample)
            ]
            if len(samples) < 2:
                continue
            samples = sorted(samples, key=lambda item: int(item.get("frame_index", 0)))
            start = [float(value) for value in samples[0]["xyz_world"]]
            end = [float(value) for value in samples[-1]["xyz_world"]]
            motions.append(_distance(start, end))
        if not motions:
            return None
        ordered = sorted(motions)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return 0.5 * (ordered[mid - 1] + ordered[mid])

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

    def _prismatic_q_values_for_axis(self, relative_poses: list[dict[str, Any]], axis: list[float]) -> list[float]:
        unit_axis = _normalize(axis, fallback=[1.0, 0.0, 0.0])
        translations = [pose["translation"] for pose in relative_poses]
        reference_translation = translations[0]
        return [_dot(_subtract(translation, reference_translation), unit_axis) for translation in translations]

    def _track_translation_axis(self, part_id: int, allowed_frames: set[int]) -> dict[str, Any] | None:
        tracks = self._tracks_by_part.get(part_id, [])
        if len(tracks) < self.config.min_track_residual_tracks:
            return None
        displacements = []
        for track in tracks:
            valid_samples = [
                sample
                for sample in track.get("samples", [])
                if _valid_track_sample(sample) and int(sample["frame_index"]) in allowed_frames
            ]
            if len(valid_samples) < 2:
                continue
            start = [float(value) for value in valid_samples[0]["xyz_world"]]
            end = [float(value) for value in valid_samples[-1]["xyz_world"]]
            displacement = _subtract(end, start)
            if _norm(displacement) < self.config.delta_translation_epsilon_m:
                continue
            displacements.append(displacement)
        if len(displacements) < self.config.min_track_residual_tracks:
            return None
        mean_displacement = _mean_point(displacements)
        if _norm(mean_displacement) < self.config.translation_threshold_m:
            return None
        axis = _normalize(mean_displacement, fallback=[1.0, 0.0, 0.0])
        return {
            "source": "track_endpoint_centroid_displacement",
            "axis": axis,
            "track_count": len(displacements),
            "displacement": mean_displacement,
            "displacement_norm_m": _norm(mean_displacement),
        }

    def _select_joint_type(
        self,
        rotation_range: float,
        translation_range_norm: float,
        track_model_comparison: dict[str, Any] | None,
    ) -> str:
        rotation_candidate = rotation_range >= self.config.rotation_threshold_rad
        translation_candidate = translation_range_norm >= self.config.translation_threshold_m
        residual_type = str(track_model_comparison.get("selected_type", "")) if track_model_comparison else ""
        residual_decisive = bool(track_model_comparison.get("decision_applied", False)) if track_model_comparison else False
        if residual_decisive and residual_type in {"revolute", "prismatic"}:
            return residual_type
        if rotation_candidate:
            return "revolute"
        if translation_candidate:
            return "prismatic"
        return "fixed"

    def _compare_track_motion_models(
        self,
        part_id: int,
        revolute_axis: list[float],
        revolute_pivot: list[float],
        prismatic_axis: list[float],
        frame_indices: list[int],
        rotation_candidate: bool,
        translation_candidate: bool,
    ) -> dict[str, Any] | None:
        if not self.config.use_track_residual_type:
            return None
        tracks = self._tracks_by_part.get(part_id, [])
        if not tracks:
            return None
        allowed_frames = set(frame_indices)
        trim_ratio = max(0.0, min(0.8, float(self.config.robust_track_model_trim_ratio)))
        prismatic = self._prismatic_track_residual(tracks, prismatic_axis, allowed_frames, trim_ratio)
        revolute = self._revolute_track_residual(tracks, revolute_axis, revolute_pivot, allowed_frames, trim_ratio)
        if prismatic is None or revolute is None:
            return None
        sample_count = min(int(prismatic["sample_count"]), int(revolute["sample_count"]))
        selected_type = "prismatic" if prismatic["rmse_m"] <= revolute["rmse_m"] else "revolute"
        best = min(float(prismatic["rmse_m"]), float(revolute["rmse_m"]))
        other = max(float(prismatic["rmse_m"]), float(revolute["rmse_m"]))
        ratio = best / other if other > 1e-12 else 1.0
        residual_decisive = (
            sample_count >= self.config.min_track_residual_samples
            and len(tracks) >= self.config.min_track_residual_tracks
            and ratio <= self.config.track_residual_decision_ratio
        )
        pose_candidate_allows_selected_type = (
            (selected_type == "revolute" and rotation_candidate)
            or (selected_type == "prismatic" and translation_candidate)
        )
        type_override_applied = residual_decisive and (
            pose_candidate_allows_selected_type
            or not self.config.track_residual_requires_pose_candidate
        )
        return {
            "source": "part_tracks_3d_replay",
            "quality_weighted_replay_enabled": bool(self.config.quality_weighted_replay),
            "min_replay_weight": float(self.config.min_replay_weight),
            "selected_type": selected_type,
            "residual_decisive": bool(residual_decisive),
            "type_override_applied": bool(type_override_applied),
            "decision_applied": bool(type_override_applied),
            "selection_mode": (
                "pose_candidate_gated"
                if self.config.track_residual_requires_pose_candidate
                else "track_model_first"
            ),
            "pose_candidate_allows_selected_type": bool(pose_candidate_allows_selected_type),
            "rotation_candidate": bool(rotation_candidate),
            "translation_candidate": bool(translation_candidate),
            "decision_ratio": float(ratio),
            "decision_threshold": float(self.config.track_residual_decision_ratio),
            "min_samples": int(self.config.min_track_residual_samples),
            "min_tracks": int(self.config.min_track_residual_tracks),
            "robust_trim_ratio": trim_ratio,
            "sample_count": int(sample_count),
            "track_count": len(tracks),
            "prismatic": prismatic,
            "revolute": revolute,
        }

    def _prismatic_track_residual(
        self,
        tracks: list[dict[str, Any]],
        axis: list[float],
        allowed_frames: set[int],
        trim_ratio: float = 0.0,
    ) -> dict[str, Any] | None:
        unit_axis = _normalize(axis, fallback=[1.0, 0.0, 0.0])
        reference_by_track: dict[int, list[float]] = {}
        frame_samples: dict[int, list[tuple[int, list[float], float]]] = {}
        for track in tracks:
            track_id = int(track.get("track_id", len(reference_by_track)))
            reference = track.get("reference_xyz_world")
            if not isinstance(reference, list) or len(reference) != 3:
                continue
            reference_by_track[track_id] = [float(value) for value in reference]
            for sample in track.get("samples", []):
                if not _valid_track_sample(sample):
                    continue
                frame_index = int(sample["frame_index"])
                if frame_index not in allowed_frames:
                    continue
                weight = observation_weight(track, sample, min_weight=float(self.config.min_replay_weight))
                frame_samples.setdefault(frame_index, []).append((track_id, [float(value) for value in sample["xyz_world"]], weight))
        residuals: list[float] = []
        weighted_residuals: list[tuple[float, float]] = []
        q_values: list[float] = []
        for samples in frame_samples.values():
            frame_q_votes = []
            valid_samples = []
            for track_id, point, weight in samples:
                reference = reference_by_track.get(track_id)
                if reference is None:
                    continue
                frame_q_votes.append(_dot(_subtract(point, reference), unit_axis))
                valid_samples.append((reference, point, weight))
            if not frame_q_votes:
                continue
            q = fmean(frame_q_votes)
            q_values.append(q)
            for reference, point, weight in valid_samples:
                predicted = _add(reference, _scale(unit_axis, q))
                residual = _distance(point, predicted)
                residuals.append(residual)
                weighted_residuals.append((residual, weight))
        if not residuals:
            return None
        kept_residuals, trim_stats = _trim_residuals(residuals, trim_ratio)
        kept_weighted_residuals, weighted_trim_stats = _trim_weighted_residuals(weighted_residuals, trim_ratio)
        raw_weighted_rmse = _weighted_rmse(weighted_residuals)
        trimmed_weighted_rmse = _weighted_rmse(kept_weighted_residuals)
        rmse = trimmed_weighted_rmse if self.config.quality_weighted_replay else math.sqrt(fmean(value * value for value in kept_residuals))
        return {
            "rmse_m": rmse,
            "rmse_unweighted_m": math.sqrt(fmean(value * value for value in kept_residuals)),
            "rmse_weighted_m": trimmed_weighted_rmse,
            "joint_replay_unweighted_m": math.sqrt(fmean(value * value for value in kept_residuals)),
            "joint_replay_weighted_m": trimmed_weighted_rmse,
            "weighted_replay_rmse_raw": raw_weighted_rmse,
            "weighted_replay_rmse_trimmed": trimmed_weighted_rmse,
            "weighted_replay_sample_count_raw": len(weighted_residuals),
            "weighted_replay_sample_count_trimmed": len(kept_weighted_residuals),
            "mae_m": fmean(kept_residuals),
            "sample_count": len(kept_residuals),
            "raw_sample_count": len(residuals),
            **trim_stats,
            **weighted_trim_stats,
            "q_range": [min(q_values), max(q_values)] if q_values else [0.0, 0.0],
        }

    def _revolute_track_residual(
        self,
        tracks: list[dict[str, Any]],
        axis: list[float],
        pivot: list[float],
        allowed_frames: set[int],
        trim_ratio: float = 0.0,
    ) -> dict[str, Any] | None:
        unit_axis = _normalize(axis, fallback=[0.0, 0.0, 1.0])
        reference_by_track: dict[int, list[float]] = {}
        frame_samples: dict[int, list[tuple[int, list[float], float]]] = {}
        for track in tracks:
            track_id = int(track.get("track_id", len(reference_by_track)))
            reference = track.get("reference_xyz_world")
            if not isinstance(reference, list) or len(reference) != 3:
                continue
            reference_by_track[track_id] = [float(value) for value in reference]
            for sample in track.get("samples", []):
                if not _valid_track_sample(sample):
                    continue
                frame_index = int(sample["frame_index"])
                if frame_index not in allowed_frames:
                    continue
                sample_weight = observation_weight(track, sample, min_weight=float(self.config.min_replay_weight))
                frame_samples.setdefault(frame_index, []).append(
                    (track_id, [float(value) for value in sample["xyz_world"]], sample_weight)
                )
        residuals: list[float] = []
        weighted_residuals: list[tuple[float, float]] = []
        q_values: list[float] = []
        for samples in frame_samples.values():
            sin_sum = 0.0
            cos_sum = 0.0
            valid_samples = []
            for track_id, point, sample_weight in samples:
                reference = reference_by_track.get(track_id)
                if reference is None:
                    continue
                angle = _signed_angle_between_about_axis(_subtract(reference, pivot), _subtract(point, pivot), unit_axis)
                if angle is None:
                    continue
                radius = _norm(_subtract(_subtract(reference, pivot), _scale(unit_axis, _dot(_subtract(reference, pivot), unit_axis))))
                angle_weight = max(1e-6, radius) * sample_weight
                sin_sum += angle_weight * math.sin(angle)
                cos_sum += angle_weight * math.cos(angle)
                valid_samples.append((reference, point, sample_weight))
            if not valid_samples:
                continue
            q = math.atan2(sin_sum, cos_sum)
            q_values.append(q)
            for reference, point, weight in valid_samples:
                predicted = _add(pivot, _rotate_vector_about_axis(_subtract(reference, pivot), unit_axis, q))
                residual = _distance(point, predicted)
                residuals.append(residual)
                weighted_residuals.append((residual, weight))
        if not residuals:
            return None
        kept_residuals, trim_stats = _trim_residuals(residuals, trim_ratio)
        kept_weighted_residuals, weighted_trim_stats = _trim_weighted_residuals(weighted_residuals, trim_ratio)
        raw_weighted_rmse = _weighted_rmse(weighted_residuals)
        trimmed_weighted_rmse = _weighted_rmse(kept_weighted_residuals)
        rmse = trimmed_weighted_rmse if self.config.quality_weighted_replay else math.sqrt(fmean(value * value for value in kept_residuals))
        return {
            "rmse_m": rmse,
            "rmse_unweighted_m": math.sqrt(fmean(value * value for value in kept_residuals)),
            "rmse_weighted_m": trimmed_weighted_rmse,
            "joint_replay_unweighted_m": math.sqrt(fmean(value * value for value in kept_residuals)),
            "joint_replay_weighted_m": trimmed_weighted_rmse,
            "weighted_replay_rmse_raw": raw_weighted_rmse,
            "weighted_replay_rmse_trimmed": trimmed_weighted_rmse,
            "weighted_replay_sample_count_raw": len(weighted_residuals),
            "weighted_replay_sample_count_trimmed": len(kept_weighted_residuals),
            "mae_m": fmean(kept_residuals),
            "sample_count": len(kept_residuals),
            "raw_sample_count": len(residuals),
            **trim_stats,
            **weighted_trim_stats,
            "q_range": [min(q_values), max(q_values)] if q_values else [0.0, 0.0],
        }


def _trim_residuals(residuals: list[float], trim_ratio: float) -> tuple[list[float], dict[str, Any]]:
    ratio = max(0.0, min(0.8, float(trim_ratio)))
    if ratio <= 0.0 or len(residuals) < 4:
        return residuals, {
            "trimmed_sample_count": 0,
            "trim_ratio": ratio,
            "raw_rmse_m": math.sqrt(fmean(value * value for value in residuals)),
            "raw_mae_m": fmean(residuals),
        }
    trim_count = min(len(residuals) - 1, int(round(len(residuals) * ratio)))
    kept = sorted(residuals)[: len(residuals) - trim_count]
    return kept, {
        "trimmed_sample_count": trim_count,
        "trim_ratio": ratio,
        "raw_rmse_m": math.sqrt(fmean(value * value for value in residuals)),
        "raw_mae_m": fmean(residuals),
    }


def _trim_weighted_residuals(
    weighted_residuals: list[tuple[float, float]],
    trim_ratio: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    ratio = max(0.0, min(0.8, float(trim_ratio)))
    if ratio <= 0.0 or len(weighted_residuals) < 4:
        return weighted_residuals, {"weighted_trimmed_sample_count": 0}
    trim_count = min(len(weighted_residuals) - 1, int(round(len(weighted_residuals) * ratio)))
    kept = sorted(weighted_residuals, key=lambda item: float(item[0]))[: len(weighted_residuals) - trim_count]
    return kept, {"weighted_trimmed_sample_count": trim_count}


def _weighted_rmse(weighted_residuals: list[tuple[float, float]]) -> float:
    if not weighted_residuals:
        return float("inf")
    total_weight = sum(max(1e-9, float(weight)) for _, weight in weighted_residuals)
    if total_weight <= 1e-12:
        return float("inf")
    weighted_square_sum = sum(
        max(1e-9, float(weight)) * float(residual) * float(residual)
        for residual, weight in weighted_residuals
    )
    return math.sqrt(weighted_square_sum / total_weight)


def _valid_track_sample(sample: Any) -> bool:
    return (
        isinstance(sample, dict)
        and bool(sample.get("visible", False))
        and bool(sample.get("depth_valid", False))
        and bool(sample.get("mask_consistent", True))
        and isinstance(sample.get("xyz_world"), list)
        and len(sample.get("xyz_world", [])) == 3
    )
