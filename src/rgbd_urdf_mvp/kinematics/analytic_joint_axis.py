from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class AnalyticAxisConfig:
    min_points_per_frame: int = 3
    min_valid_frames: int = 3
    min_rotation_rad: float = math.radians(2.0)
    min_total_rotation_rad: float = math.radians(8.0)
    min_total_translation: float = 0.01
    rigid_trim_quantile: float = 0.8
    rigid_trim_iterations: int = 2
    axis_trim_deg: float = 25.0
    robust_segment_fitting: bool = False
    max_segment_gap_frames: int = 2
    max_segment_translation_jump: float = 0.08
    huber_delta_m: float = 0.02
    irls_iterations: int = 4


@dataclass(slots=True)
class AnalyticAxisEstimate:
    joint_type: str
    valid: bool
    axis: list[float] | None
    line_point: list[float] | None
    valid_frame_count: int
    total_motion: float
    residual: float
    confidence: float
    track_count_parent: int
    track_count_child: int
    eigenvalue_ratio: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalyticJointModelSelection:
    selected_type: str | None
    selected_estimate: AnalyticAxisEstimate | None
    revolute_estimate: AnalyticAxisEstimate
    prismatic_estimate: AnalyticAxisEstimate
    revolute_replay_rmse: float
    prismatic_replay_rmse: float
    decision_ratio: float
    valid: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_finite(asdict(self))


def axis_angle_error_deg(axis_a: list[float] | np.ndarray, axis_b: list[float] | np.ndarray) -> float:
    a = _normalize(np.asarray(axis_a, dtype=float))
    b = _normalize(np.asarray(axis_b, dtype=float))
    return math.degrees(math.acos(float(np.clip(abs(a @ b), 0.0, 1.0))))


def axis_line_distance(
    point_a: list[float] | np.ndarray,
    axis_a: list[float] | np.ndarray,
    point_b: list[float] | np.ndarray,
    axis_b: list[float] | np.ndarray,
) -> float:
    """Shortest distance between two undirected 3D lines."""
    p_a = np.asarray(point_a, dtype=float)
    p_b = np.asarray(point_b, dtype=float)
    a = _normalize(np.asarray(axis_a, dtype=float))
    b = _normalize(np.asarray(axis_b, dtype=float))
    cross = np.cross(a, b)
    cross_norm = float(np.linalg.norm(cross))
    delta = p_b - p_a
    if cross_norm < 1e-8:
        return float(np.linalg.norm(np.cross(delta, a)))
    return float(abs(delta @ cross) / cross_norm)


def estimate_analytic_joint_axis(
    parent_points: np.ndarray,
    parent_visibility: np.ndarray,
    child_points: np.ndarray,
    child_visibility: np.ndarray,
    joint_type: str,
    config: AnalyticAxisConfig | None = None,
) -> AnalyticAxisEstimate:
    config = config or AnalyticAxisConfig()
    parent_points = np.asarray(parent_points, dtype=float)
    child_points = np.asarray(child_points, dtype=float)
    parent_visibility = np.asarray(parent_visibility, dtype=bool)
    child_visibility = np.asarray(child_visibility, dtype=bool)
    _validate_tracks(parent_points, parent_visibility, "parent")
    _validate_tracks(child_points, child_visibility, "child")
    frame_count = min(parent_points.shape[1], child_points.shape[1])
    parent_poses, parent_valid = _accumulate_part_poses(
        parent_points[:, :frame_count], parent_visibility[:, :frame_count], config
    )
    child_poses, child_valid = _accumulate_part_poses(
        child_points[:, :frame_count], child_visibility[:, :frame_count], config
    )
    relative = []
    for frame in range(frame_count):
        if not parent_valid[frame] or not child_valid[frame]:
            continue
        relative.append((frame, np.linalg.inv(parent_poses[frame]) @ child_poses[frame]))
    if config.robust_segment_fitting:
        relative = _rebase_relative_segments(relative, config)
    common = {
        "joint_type": joint_type,
        "track_count_parent": int(parent_points.shape[0]),
        "track_count_child": int(child_points.shape[0]),
    }
    if len(relative) < config.min_valid_frames:
        return AnalyticAxisEstimate(
            **common, valid=False, axis=None, line_point=None,
            valid_frame_count=len(relative), total_motion=0.0, residual=math.inf,
            confidence=0.0, reason="insufficient_common_pose_frames",
        )
    if joint_type == "revolute":
        return _estimate_revolute(relative, config, common)
    if joint_type == "prismatic":
        return _estimate_prismatic(relative, config, common)
    return AnalyticAxisEstimate(
        **common, valid=False, axis=None, line_point=None,
        valid_frame_count=len(relative), total_motion=0.0, residual=math.inf,
        confidence=0.0, reason=f"unsupported_joint_type:{joint_type}",
    )


def select_analytic_joint_model(
    parent_points: np.ndarray,
    parent_visibility: np.ndarray,
    child_points: np.ndarray,
    child_visibility: np.ndarray,
    config: AnalyticAxisConfig | None = None,
) -> AnalyticJointModelSelection:
    """Select revolute or prismatic using a common point-replay objective.

    Both hypotheses are fit from the same relative part transforms. Their
    native estimator residuals use different units, so they are not compared
    directly. Instead, each fitted transform is replayed on the same canonical
    child points and scored with one normalized 3D RMSE.
    """
    config = config or AnalyticAxisConfig()
    parent_points = np.asarray(parent_points, dtype=float)
    child_points = np.asarray(child_points, dtype=float)
    parent_visibility = np.asarray(parent_visibility, dtype=bool)
    child_visibility = np.asarray(child_visibility, dtype=bool)
    revolute = estimate_analytic_joint_axis(
        parent_points, parent_visibility, child_points, child_visibility,
        "revolute", config,
    )
    prismatic = estimate_analytic_joint_axis(
        parent_points, parent_visibility, child_points, child_visibility,
        "prismatic", config,
    )
    relative = _relative_part_transforms(
        parent_points, parent_visibility, child_points, child_visibility, config
    )
    reference = _reference_child_points(child_points, child_visibility, config)
    revolute_rmse = _model_replay_rmse(reference, relative, revolute)
    prismatic_rmse = _model_replay_rmse(reference, relative, prismatic)
    candidates = [
        ("revolute", revolute, revolute_rmse),
        ("prismatic", prismatic, prismatic_rmse),
    ]
    valid = [row for row in candidates if row[1].valid and math.isfinite(row[2])]
    if not valid:
        return AnalyticJointModelSelection(
            selected_type=None, selected_estimate=None,
            revolute_estimate=revolute, prismatic_estimate=prismatic,
            revolute_replay_rmse=revolute_rmse,
            prismatic_replay_rmse=prismatic_rmse,
            decision_ratio=1.0, valid=False, reason="no_valid_motion_model",
        )
    selected_type, selected, best = min(valid, key=lambda row: row[2])
    other_values = [row[2] for row in valid if row[0] != selected_type]
    other = min(other_values) if other_values else math.inf
    ratio = best / other if math.isfinite(other) and other > 1e-12 else 0.0
    return AnalyticJointModelSelection(
        selected_type=selected_type, selected_estimate=selected,
        revolute_estimate=revolute, prismatic_estimate=prismatic,
        revolute_replay_rmse=revolute_rmse,
        prismatic_replay_rmse=prismatic_rmse,
        decision_ratio=float(ratio), valid=True,
    )


def _relative_part_transforms(
    parent_points: np.ndarray, parent_visibility: np.ndarray,
    child_points: np.ndarray, child_visibility: np.ndarray,
    config: AnalyticAxisConfig,
) -> list[tuple[int, np.ndarray]]:
    frame_count = min(parent_points.shape[1], child_points.shape[1])
    parent_poses, parent_valid = _accumulate_part_poses(
        parent_points[:, :frame_count], parent_visibility[:, :frame_count], config
    )
    child_poses, child_valid = _accumulate_part_poses(
        child_points[:, :frame_count], child_visibility[:, :frame_count], config
    )
    relative = [
        (frame, np.linalg.inv(parent_poses[frame]) @ child_poses[frame])
        for frame in range(frame_count)
        if parent_valid[frame] and child_valid[frame]
    ]
    return _rebase_relative_segments(relative, config) if config.robust_segment_fitting else relative


def _rebase_relative_segments(
    relative: list[tuple[int, np.ndarray]], config: AnalyticAxisConfig,
) -> list[tuple[int, np.ndarray]]:
    """Split discontinuous observations and express every segment from its own origin."""
    if not relative:
        return []
    segments: list[list[tuple[int, np.ndarray]]] = [[relative[0]]]
    for row in relative[1:]:
        previous = segments[-1][-1]
        delta = np.linalg.inv(previous[1]) @ row[1]
        gap = row[0] - previous[0]
        jump = float(np.linalg.norm(delta[:3, 3]))
        if gap > config.max_segment_gap_frames or jump > config.max_segment_translation_jump:
            segments.append([row])
        else:
            segments[-1].append(row)
    rebased = []
    for segment in segments:
        if len(segment) < config.min_valid_frames:
            continue
        origin_inverse = np.linalg.inv(segment[0][1])
        rebased.extend((frame, origin_inverse @ transform) for frame, transform in segment)
    return rebased


def _reference_child_points(
    points: np.ndarray, visibility: np.ndarray, config: AnalyticAxisConfig,
) -> np.ndarray:
    frame = next(
        (index for index in range(points.shape[1])
         if int(visibility[:, index].sum()) >= config.min_points_per_frame),
        None,
    )
    return points[visibility[:, frame], frame] if frame is not None else np.empty((0, 3))


def _model_replay_rmse(
    reference: np.ndarray,
    relative: list[tuple[int, np.ndarray]],
    estimate: AnalyticAxisEstimate,
) -> float:
    if not estimate.valid or estimate.axis is None or not len(reference) or not relative:
        return math.inf
    axis = _normalize(np.asarray(estimate.axis, dtype=float))
    errors = []
    for _, transform in relative:
        observed = reference @ transform[:3, :3].T + transform[:3, 3]
        if estimate.joint_type == "prismatic":
            q_value = float(axis @ transform[:3, 3])
            predicted = reference + q_value * axis
        else:
            pivot = np.asarray(estimate.line_point, dtype=float)
            observed_vectors = observed - pivot
            source_vectors = reference - pivot
            source_perp = source_vectors - (source_vectors @ axis)[:, None] * axis
            observed_perp = observed_vectors - (observed_vectors @ axis)[:, None] * axis
            cosine = float(np.sum(source_perp * observed_perp))
            sine = float(np.sum(np.cross(source_perp, observed_perp) * axis))
            angle = math.atan2(sine, cosine)
            rotation = _axis_angle_rotation(axis, angle)
            predicted = source_vectors @ rotation.T + pivot
        errors.extend(np.sum((predicted - observed) ** 2, axis=1).tolist())
    return float(math.sqrt(np.mean(errors))) if errors else math.inf


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_tracks(points: np.ndarray, visibility: np.ndarray, name: str) -> None:
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"{name}_points must have shape [tracks, frames, 3]")
    if visibility.shape != points.shape[:2]:
        raise ValueError(f"{name}_visibility must match the first two point dimensions")


def _accumulate_part_poses(
    points: np.ndarray, visibility: np.ndarray, config: AnalyticAxisConfig
) -> tuple[list[np.ndarray], list[bool]]:
    poses = [np.eye(4, dtype=float) for _ in range(points.shape[1])]
    valid = [False] * points.shape[1]
    anchor = next(
        (
            frame
            for frame in range(points.shape[1])
            if int(visibility[:, frame].sum()) >= config.min_points_per_frame
        ),
        None,
    )
    if anchor is None:
        return poses, valid
    valid[anchor] = True
    for frame in range(anchor + 1, points.shape[1]):
        # Recover after short visibility gaps by linking to the most recent
        # valid frame that still has enough shared tracks.
        for previous in range(frame - 1, -1, -1):
            if not valid[previous]:
                continue
            mask = visibility[:, previous] & visibility[:, frame]
            if int(mask.sum()) < config.min_points_per_frame:
                continue
            transform = _robust_rigid_transform(
                points[mask, previous], points[mask, frame], config
            )
            if transform is None:
                continue
            poses[frame] = transform @ poses[previous]
            valid[frame] = True
            break
    return poses, valid


def _robust_rigid_transform(
    source: np.ndarray, target: np.ndarray, config: AnalyticAxisConfig
) -> np.ndarray | None:
    keep = np.arange(len(source))
    transform = None
    for _ in range(max(1, config.rigid_trim_iterations + 1)):
        if len(keep) < config.min_points_per_frame:
            return None
        src = source[keep]
        dst = target[keep]
        weights = np.ones(len(src), dtype=float)
        for _irls in range(max(1, config.irls_iterations if config.robust_segment_fitting else 1)):
            weights /= max(float(weights.sum()), 1e-12)
            src_center = np.sum(src * weights[:, None], axis=0)
            dst_center = np.sum(dst * weights[:, None], axis=0)
            centered = src - src_center
            if np.linalg.matrix_rank(centered) < 2:
                return None
            u_mat, _, vt_mat = np.linalg.svd((centered * weights[:, None]).T @ (dst - dst_center))
            rotation = vt_mat.T @ u_mat.T
            if np.linalg.det(rotation) < 0.0:
                vt_mat[-1] *= -1.0
                rotation = vt_mat.T @ u_mat.T
            translation = dst_center - rotation @ src_center
            if not config.robust_segment_fitting:
                break
            errors = np.linalg.norm(src @ rotation.T + translation - dst, axis=1)
            delta = max(config.huber_delta_m, 1e-8)
            weights = np.where(errors <= delta, 1.0, delta / np.maximum(errors, 1e-12))
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        residuals = np.linalg.norm((source @ rotation.T + translation) - target, axis=1)
        count = max(config.min_points_per_frame, int(math.ceil(len(source) * config.rigid_trim_quantile)))
        next_keep = np.argsort(residuals)[:count]
        if np.array_equal(np.sort(next_keep), np.sort(keep)):
            break
        keep = next_keep
    return transform


def _rotation_vector(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=float), 0.0
    if abs(math.pi - angle) < 1e-5:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    else:
        axis = np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]) / (2.0 * math.sin(angle))
    return _normalize(axis), angle


def _estimate_revolute(
    relative: list[tuple[int, np.ndarray]], config: AnalyticAxisConfig, common: dict[str, Any]
) -> AnalyticAxisEstimate:
    rows = []
    for frame, transform in relative:
        axis, angle = _rotation_vector(transform[:3, :3])
        if angle >= config.min_rotation_rad:
            rows.append((frame, transform, axis, angle))
    total_rotation = max((row[3] for row in rows), default=0.0)
    if len(rows) < config.min_valid_frames or total_rotation < config.min_total_rotation_rad:
        return AnalyticAxisEstimate(
            **common, valid=False, axis=None, line_point=None,
            valid_frame_count=len(rows), total_motion=total_rotation, residual=math.inf,
            confidence=0.0, reason="insufficient_rotation",
        )
    axes = np.stack([row[2] for row in rows])
    weights = np.asarray([row[3] for row in rows], dtype=float)
    affinity = np.abs(axes @ axes.T) @ weights
    reference = axes[int(np.argmax(affinity))]
    aligned = axes * np.where(axes @ reference >= 0.0, 1.0, -1.0)[:, None]
    preliminary = _normalize(np.sum(aligned * weights[:, None], axis=0))
    deviations = np.degrees(np.arccos(np.clip(np.abs(aligned @ preliminary), 0.0, 1.0)))
    keep = deviations <= max(config.axis_trim_deg, float(np.median(deviations) * 2.5))
    if int(keep.sum()) >= config.min_valid_frames:
        axis = _normalize(np.sum(aligned[keep] * weights[keep, None], axis=0))
        used_rows = [row for row, use in zip(rows, keep) if use]
    else:
        axis = preliminary
        used_rows = rows
    angular = [math.acos(float(np.clip(abs(row[2] @ axis), 0.0, 1.0))) for row in used_rows]
    residual = float(np.average(angular, weights=[row[3] for row in used_rows]))
    line_point, line_residual = _fit_hinge_line(axis, used_rows)
    residual_combined = residual + line_residual
    observation = min(1.0, len(used_rows) / max(config.min_valid_frames * 2.0, 1.0))
    excitation = min(1.0, total_rotation / math.radians(30.0))
    confidence = observation * excitation * math.exp(-residual / math.radians(15.0))
    return AnalyticAxisEstimate(
        **common, valid=True, axis=axis.tolist(), line_point=line_point.tolist(),
        valid_frame_count=len(used_rows), total_motion=total_rotation,
        residual=residual_combined, confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _fit_hinge_line(
    axis: np.ndarray, rows: list[tuple[int, np.ndarray, np.ndarray, float]]
) -> tuple[np.ndarray, float]:
    matrices = []
    targets = []
    for _, transform, _, _ in rows:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        matrices.append(np.eye(3) - rotation)
        targets.append(translation - axis * float(axis @ translation))
    # The point closest to the origin makes the otherwise underdetermined line unique.
    matrices.append(axis[None, :])
    targets.append(np.zeros(1, dtype=float))
    matrix = np.concatenate(matrices, axis=0)
    target = np.concatenate(targets, axis=0)
    point, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = float(np.sqrt(np.mean((matrix @ point - target) ** 2)))
    return point, residual


def _estimate_prismatic(
    relative: list[tuple[int, np.ndarray]], config: AnalyticAxisConfig, common: dict[str, Any]
) -> AnalyticAxisEstimate:
    translations = np.stack([transform[:3, 3] for _, transform in relative])
    translations = translations - translations[0]
    centered = translations - translations.mean(axis=0)
    _, singular, vt_mat = np.linalg.svd(centered, full_matrices=False)
    axis = _normalize(vt_mat[0])
    projections = translations @ axis
    total_translation = float(np.ptp(projections))
    orthogonal = translations - projections[:, None] * axis
    residual = float(np.sqrt(np.mean(np.sum(orthogonal * orthogonal, axis=1))))
    variance = singular * singular
    eigenvalue_ratio = float(variance[0] / max(float(variance.sum()), 1e-12))
    if total_translation < config.min_total_translation:
        return AnalyticAxisEstimate(
            **common, valid=False, axis=None, line_point=None,
            valid_frame_count=len(relative), total_motion=total_translation,
            residual=residual, confidence=0.0, eigenvalue_ratio=eigenvalue_ratio,
            reason="insufficient_translation",
        )
    observation = min(1.0, len(relative) / max(config.min_valid_frames * 2.0, 1.0))
    excitation = min(1.0, total_translation / max(4.0 * config.min_total_translation, 1e-8))
    confidence = observation * excitation * eigenvalue_ratio * math.exp(
        -residual / max(total_translation * 0.25, 1e-6)
    )
    return AnalyticAxisEstimate(
        **common, valid=True, axis=axis.tolist(), line_point=None,
        valid_frame_count=len(relative), total_motion=total_translation,
        residual=residual, confidence=float(np.clip(confidence, 0.0, 1.0)),
        eigenvalue_ratio=eigenvalue_ratio,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return vector / norm
