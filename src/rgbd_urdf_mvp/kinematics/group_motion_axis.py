from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class GroupMotionProposal:
    start_frame: int
    end_frame: int
    rotation: list[list[float]]
    translation: list[float]
    rotation_vector: list[float]
    axis: list[float] | None
    translation_axis: list[float] | None
    translation_magnitude: float
    line_point: list[float] | None
    rigid_residual: float
    effective_tracks: float
    mean_quality: float
    condition_number: float
    valid: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_group_motion_proposals(
    points: np.ndarray,
    visibility: np.ndarray,
    *,
    quality: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    strides: Iterable[int] = (1, 2, 4, 8),
    min_tracks: int = 6,
    huber_delta: float = 0.02,
    irls_iterations: int = 3,
) -> list[GroupMotionProposal]:
    """Create multi-scale local-rigid proposals without recovering a full part pose."""
    points = np.asarray(points, dtype=float)
    visibility = np.asarray(visibility, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (tracks, frames, 3)")
    if visibility.shape != points.shape[:2]:
        raise ValueError("visibility must match the track and frame dimensions")
    quality = np.ones(visibility.shape, float) if quality is None else np.asarray(quality, float)
    membership = np.ones(points.shape[0], float) if membership is None else np.asarray(membership, float)
    if quality.shape != visibility.shape:
        raise ValueError("quality must match visibility")
    if membership.shape != (points.shape[0],):
        raise ValueError("membership must have one value per track")
    proposals = []
    for stride in strides:
        for start in range(0, points.shape[1] - int(stride)):
            end = start + int(stride)
            base = (
                visibility[:, start].astype(float)
                * visibility[:, end].astype(float)
                * quality[:, start]
                * quality[:, end]
                * membership
            )
            selected = base > 0
            if int(selected.sum()) < min_tracks:
                continue
            source, target, weights = points[selected, start], points[selected, end], base[selected]
            rotation = np.eye(3)
            translation = np.zeros(3)
            residual = np.full(len(source), np.inf)
            for _ in range(max(1, irls_iterations)):
                rotation, translation, condition = _weighted_kabsch(source, target, weights)
                residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
                huber = np.where(residual <= huber_delta, 1.0, huber_delta / np.maximum(residual, 1e-12))
                weights = base[selected] * huber
            rotvec = _rotation_vector(rotation)
            angle = float(np.linalg.norm(rotvec))
            axis = rotvec / angle if angle > np.deg2rad(0.5) else None
            translation_magnitude = float(np.linalg.norm(translation))
            translation_axis = (
                translation / translation_magnitude
                if translation_magnitude > 1e-4
                else None
            )
            line_point = _axis_line_point(rotation, translation, axis) if axis is not None else None
            effective = float(weights.sum() ** 2 / max(float(np.sum(weights * weights)), 1e-12))
            proposals.append(GroupMotionProposal(
                start, end, rotation.tolist(), translation.tolist(), rotvec.tolist(),
                axis.tolist() if axis is not None else None,
                translation_axis.tolist() if translation_axis is not None else None,
                translation_magnitude,
                line_point.tolist() if line_point is not None else None,
                float(np.average(residual, weights=np.maximum(weights, 1e-12))),
                effective, float(np.average(quality[selected, start] * quality[selected, end], weights=membership[selected])),
                condition, True,
            ))
    return proposals


def consensus_group_axis(
    proposals: list[GroupMotionProposal], joint_type: str = "revolute", *,
    learned_logits: np.ndarray | None = None,
    residual_floor: float = 0.01,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fuse proposal axes; learned logits may calibrate but cannot create a world vector."""
    if joint_type == "revolute":
        axis_field = "axis"
    elif joint_type == "prismatic":
        axis_field = "translation_axis"
    else:
        raise ValueError(f"unsupported joint type: {joint_type}")
    valid = [
        proposal for proposal in proposals
        if proposal.valid and getattr(proposal, axis_field) is not None
    ]
    if not valid:
        return None, None
    base = np.asarray([
        proposal.effective_tracks * proposal.mean_quality
        / max(proposal.rigid_residual, residual_floor) for proposal in valid
    ])
    if learned_logits is not None:
        logits = np.asarray(learned_logits, float)
        if logits.shape != base.shape:
            raise ValueError("learned_logits must match the valid proposal count")
        base *= np.exp(logits - np.max(logits))
    base /= max(float(base.sum()), 1e-12)
    axes = np.asarray([getattr(proposal, axis_field) for proposal in valid])
    moment = np.einsum("n,ni,nj->ij", base, axes, axes)
    axis = np.linalg.eigh(moment)[1][:, -1]
    if joint_type == "prismatic":
        return axis, None
    systems, targets = [], []
    for weight, proposal in zip(base, valid):
        rotation = np.asarray(proposal.rotation)
        systems.append(np.sqrt(weight) * (np.eye(3) - rotation))
        targets.append(np.sqrt(weight) * np.asarray(proposal.translation))
    systems.append(axis[None])
    targets.append(np.zeros(1))
    point = np.linalg.lstsq(np.concatenate(systems), np.concatenate(targets), rcond=None)[0]
    return axis, point


def _weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weights = weights / max(float(weights.sum()), 1e-12)
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    covariance = (source - source_center).T @ ((target - target_center) * weights[:, None])
    u, singular, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    condition = float(singular[0] / max(float(singular[-1]), 1e-12))
    return rotation, target_center - rotation @ source_center, condition


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    vector = np.asarray([
        rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    return vector * (angle / max(2.0 * np.sin(angle), 1e-8))


def _axis_line_point(rotation: np.ndarray, translation: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(
        np.concatenate([np.eye(3) - rotation, axis[None]]),
        np.concatenate([translation, [0.0]]), rcond=None,
    )[0]
