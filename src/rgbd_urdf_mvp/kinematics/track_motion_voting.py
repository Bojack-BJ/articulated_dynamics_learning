from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrackMotionProposal:
    track_index: int
    start_frame: int
    end_frame: int
    prismatic_axis: list[float]
    prismatic_residual: float
    revolute_axis: list[float]
    circle_center: list[float] | None
    circle_radius: float | None
    circle_residual: float
    angular_coverage_rad: float
    mean_quality: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_track_motion_proposals(
    points: np.ndarray,
    visibility: np.ndarray,
    *,
    quality: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    min_segment_frames: int = 5,
    max_gap: int = 1,
) -> list[TrackMotionProposal]:
    """Fit independent line/circle hypotheses to each visible track segment."""
    points = np.asarray(points, float)
    visibility = np.asarray(visibility, bool)
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
    result = []
    for track in range(points.shape[0]):
        if membership[track] <= 0:
            continue
        for frames in _segments(np.flatnonzero(visibility[track] & (quality[track] > 0)), max_gap):
            if len(frames) < min_segment_frames:
                continue
            trajectory = points[track, frames]
            weights = quality[track, frames] * membership[track]
            center = np.average(trajectory, axis=0, weights=weights)
            centered = trajectory - center
            covariance = centered.T @ (centered * weights[:, None]) / max(float(weights.sum()), 1e-12)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            line_axis = eigenvectors[:, -1]
            line_distance = np.linalg.norm(centered - (centered @ line_axis)[:, None] * line_axis, axis=1)
            plane_axis = eigenvectors[:, 0]
            basis_u, basis_v = eigenvectors[:, 2], eigenvectors[:, 1]
            projected = np.stack([centered @ basis_u, centered @ basis_v], axis=1)
            circle_center_2d, radius, circle_residual = _fit_circle(projected, weights)
            circle_center = center + circle_center_2d[0] * basis_u + circle_center_2d[1] * basis_v
            angles = np.unwrap(np.arctan2(
                projected[:, 1] - circle_center_2d[1], projected[:, 0] - circle_center_2d[0]
            ))
            result.append(TrackMotionProposal(
                track, int(frames[0]), int(frames[-1]), line_axis.tolist(),
                float(np.average(line_distance, weights=weights)), plane_axis.tolist(),
                circle_center.tolist(), float(radius), float(circle_residual),
                float(np.max(angles) - np.min(angles)), float(np.average(quality[track, frames])),
            ))
    return result


def consensus_track_axes(
    proposals: list[TrackMotionProposal], joint_type: str, *, learned_logits: np.ndarray | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Vote over independent track hypotheses without recovering a rigid part pose."""
    if not proposals:
        return None, None
    if joint_type == "prismatic":
        axes = np.asarray([row.prismatic_axis for row in proposals])
        residual = np.asarray([row.prismatic_residual for row in proposals])
        centers = None
        coverage = np.ones(len(proposals))
    elif joint_type == "revolute":
        selected = [row for row in proposals if row.circle_center is not None]
        if not selected:
            return None, None
        proposals = selected
        axes = np.asarray([row.revolute_axis for row in proposals])
        residual = np.asarray([row.circle_residual for row in proposals])
        centers = np.asarray([row.circle_center for row in proposals])
        coverage = np.asarray([row.angular_coverage_rad for row in proposals])
    else:
        raise ValueError(f"unsupported joint type: {joint_type}")
    weights = np.asarray([row.mean_quality for row in proposals])
    weights *= coverage / np.maximum(residual, 1e-4)
    if learned_logits is not None:
        logits = np.asarray(learned_logits, float)
        if logits.shape != weights.shape:
            raise ValueError("learned_logits must match proposal count")
        weights *= np.exp(logits - np.max(logits))
    weights /= max(float(weights.sum()), 1e-12)
    moment = np.einsum("n,ni,nj->ij", weights, axes, axes)
    axis = np.linalg.eigh(moment)[1][:, -1]
    if centers is None:
        return axis, None
    # Circle centers should lie on the common joint axis. Their robust weighted
    # mean gives one representative point; remove its unobservable axial offset.
    point = np.average(centers, axis=0, weights=weights)
    point = point - float(point @ axis) * axis
    return axis, point


def _segments(frames: np.ndarray, max_gap: int) -> list[np.ndarray]:
    if not len(frames):
        return []
    split = np.flatnonzero(np.diff(frames) > max_gap) + 1
    return [row for row in np.split(frames, split) if len(row)]


def _fit_circle(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    root = np.sqrt(weights / max(float(weights.sum()), 1e-12))
    solution = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    center = solution[:2]
    radius = float(np.sqrt(max(float(solution[2] + center @ center), 0.0)))
    radial = np.linalg.norm(points - center, axis=1)
    residual = float(np.sqrt(np.average((radial - radius) ** 2, weights=weights)))
    return center, radius, residual
