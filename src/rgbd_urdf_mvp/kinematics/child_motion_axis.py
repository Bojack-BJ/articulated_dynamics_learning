from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ChildMotionAxisEstimate:
    joint_type: str
    valid: bool
    axis: list[float] | None
    line_point: list[float] | None
    confidence: float
    residual: float
    evidence_count: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_child_motion_axis(
    points: np.ndarray,
    visibility: np.ndarray,
    joint_type: str,
    *,
    quality: np.ndarray | None = None,
    min_correspondences: int = 6,
    min_motion: float = 1e-3,
) -> ChildMotionAxisEstimate:
    """Robust child-only analytic baseline using matched track motion."""
    points = np.asarray(points, dtype=float)
    visibility = np.asarray(visibility, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3 or visibility.shape != points.shape[:-1]:
        raise ValueError("points/visibility must have shape [tracks, frames, 3]/[tracks, frames]")
    quality = np.ones_like(visibility, dtype=float) if quality is None else np.asarray(quality, dtype=float)
    candidates: list[np.ndarray] = []
    pivots: list[np.ndarray] = []
    residuals: list[float] = []
    for frame in range(1, points.shape[1]):
        valid = visibility[:, frame - 1] & visibility[:, frame]
        valid &= quality[:, frame - 1] > 0
        valid &= quality[:, frame] > 0
        if int(valid.sum()) < min_correspondences:
            continue
        source, target = points[valid, frame - 1], points[valid, frame]
        displacement = target - source
        if joint_type == "prismatic":
            median = np.median(displacement, axis=0)
            magnitude = float(np.linalg.norm(median))
            if magnitude < min_motion:
                continue
            candidates.append(median / magnitude)
            residuals.append(float(np.median(np.linalg.norm(displacement - median, axis=1))))
            continue
        if joint_type != "revolute":
            return ChildMotionAxisEstimate(
                joint_type, False, None, None, 0.0, math.inf, 0,
                f"unsupported_joint_type:{joint_type}",
            )
        source_center, target_center = source.mean(0), target.mean(0)
        u, _, vh = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = vh.T @ u.T
        if np.linalg.det(rotation) < 0:
            vh[-1] *= -1
            rotation = vh.T @ u.T
        trace = float(np.trace(rotation))
        angle = math.acos(float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        if angle < math.radians(0.5):
            continue
        vector = np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            continue
        axis = vector / norm
        translation = target_center - rotation @ source_center
        system = np.eye(3) - rotation
        pivot = np.linalg.lstsq(
            np.concatenate([system, axis[None]], axis=0),
            np.concatenate([translation, [0.0]], axis=0), rcond=None,
        )[0]
        replay = source @ rotation.T + translation
        candidates.append(axis)
        pivots.append(pivot)
        residuals.append(float(np.median(np.linalg.norm(replay - target, axis=1))))
    if not candidates:
        return ChildMotionAxisEstimate(
            joint_type, False, None, None, 0.0, math.inf, 0, "insufficient_motion_evidence"
        )
    candidate_array = np.asarray(candidates)
    residual_array = np.asarray(residuals)
    median_residual = float(np.median(residual_array))
    keep = residual_array <= max(3.0 * median_residual, 1e-5)
    candidate_array = candidate_array[keep]
    moment = candidate_array.T @ candidate_array
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    axis = eigenvectors[:, -1]
    confidence = float((eigenvalues[-1] - eigenvalues[-2]) / max(eigenvalues[-1], 1e-8))
    line_point = None
    if joint_type == "revolute" and pivots:
        kept_pivots = np.asarray(pivots)[keep]
        line_point = np.median(kept_pivots, axis=0)
        line_point = line_point - float(line_point @ axis) * axis
    return ChildMotionAxisEstimate(
        joint_type=joint_type,
        valid=True,
        axis=axis.tolist(),
        line_point=line_point.tolist() if line_point is not None else None,
        confidence=confidence,
        residual=median_residual,
        evidence_count=int(keep.sum()),
    )


def _normalize(vector: Any, torch: Any) -> Any:
    return torch.nn.functional.normalize(vector, dim=-1, eps=1e-6)


def build_child_motion_features(
    points: Any,
    visibility: Any,
    torch: Any,
    *,
    quality: Any | None = None,
) -> dict[str, Any]:
    """Build SE(3)-equivariant child-only features from track correspondences.

    `points` has shape ``[B, N, T, 3]``. Vector outputs rotate with the input;
    scalar outputs are rotation and translation invariant. Missing observations
    never contribute to the pooled representation.
    """
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [batch, tracks, frames, 3]")
    if visibility.shape != points.shape[:-1]:
        raise ValueError("visibility must have shape [batch, tracks, frames]")
    visible = visibility.to(points.dtype)
    if quality is None:
        quality = torch.ones_like(visible)
    if quality.shape != visibility.shape:
        raise ValueError("quality must have shape [batch, tracks, frames]")
    weights = visible * quality.to(points.dtype).clamp_min(0.0)

    # Each displacement uses the closest earlier visible observation. This is
    # robust to a missing first frame and preserves exact track correspondence.
    previous = points[:, :, :-1]
    current = points[:, :, 1:]
    pair_weights = weights[:, :, :-1] * weights[:, :, 1:]
    displacement = current - previous

    reference_weight = weights.sum(dim=2).clamp_min(1e-6)
    reference = (points * weights[..., None]).sum(dim=2) / reference_weight[..., None]
    object_center = (
        reference * reference_weight[..., None]
    ).sum(dim=1) / reference_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    radius = reference - object_center[:, None, :]
    radius_time = radius[:, :, None, :].expand_as(displacement)
    rotational_motion = torch.linalg.cross(radius_time, displacement, dim=-1)

    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    radius_norm = torch.linalg.vector_norm(radius_time, dim=-1)
    rotational_norm = torch.linalg.vector_norm(rotational_motion, dim=-1)
    radial_motion = (radius_time * displacement).sum(dim=-1)
    scalar = torch.stack(
        [displacement_norm, radius_norm, rotational_norm, radial_motion.abs()], dim=-1
    )
    valid = pair_weights > 0
    return {
        "scalar": scalar,
        "translation_vectors": displacement,
        "rotation_vectors": rotational_motion,
        "reference_points": reference,
        "pair_weights": pair_weights,
        "track_weights": reference_weight,
        "valid": valid,
    }


def _weighted_undirected_axis(vectors: Any, logits: Any, valid: Any, torch: Any) -> tuple[Any, Any]:
    vectors = _normalize(vectors, torch)
    weights = torch.softmax(logits.masked_fill(~valid, -1e4).flatten(1), dim=1)
    weights = weights.reshape_as(logits) * valid.to(logits.dtype)
    weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    moment = torch.einsum("bnt,bnti,bntj->bij", weights, vectors, vectors)
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    observable = valid.any(dim=(1, 2))
    axis = torch.where(observable[:, None], eigenvectors[..., -1], torch.zeros_like(eigenvectors[..., -1]))
    confidence = torch.where(
        observable,
        (eigenvalues[..., -1] - eigenvalues[..., -2]).clamp_min(0.0),
        torch.zeros_like(eigenvalues[..., -1]),
    )
    return axis, confidence


def build_child_motion_axis_model(torch: Any, *, hidden_dim: int = 64) -> Any:
    """Create a feedforward child-motion head with exact SE(3) output laws.

    The network predicts only invariant scalar weights. Axis vectors and the
    axis-line point are assembled from input vectors/positions, preventing a
    learned world-coordinate bias.
    """

    class ChildMotionAxisHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = torch.nn.Sequential(
                torch.nn.Linear(4, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
            )
            self.translation_weight = torch.nn.Linear(hidden_dim, 1)
            self.rotation_weight = torch.nn.Linear(hidden_dim, 1)
            self.track_line_weight = torch.nn.Linear(hidden_dim, 1)
            self.type_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2 + 4, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 2),
            )

        def forward(self, points: Any, visibility: Any, quality: Any | None = None) -> dict[str, Any]:
            features = build_child_motion_features(
                points, visibility, torch, quality=quality
            )
            hidden = self.token(features["scalar"])
            valid = features["valid"]
            translation_axis, translation_confidence = _weighted_undirected_axis(
                features["translation_vectors"],
                self.translation_weight(hidden).squeeze(-1), valid, torch,
            )
            rotation_axis, rotation_confidence = _weighted_undirected_axis(
                features["rotation_vectors"],
                self.rotation_weight(hidden).squeeze(-1), valid, torch,
            )
            pair_weight = features["pair_weights"]
            denominator = pair_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            pooled_mean = (hidden * pair_weight[..., None]).sum(dim=(1, 2)) / denominator.squeeze(2)
            centered = hidden - pooled_mean[:, None, None, :]
            pooled_scale = torch.sqrt(
                (centered.square() * pair_weight[..., None]).sum(dim=(1, 2))
                / denominator.squeeze(2)
            )
            scalar_mean = (
                features["scalar"] * pair_weight[..., None]
            ).sum(dim=(1, 2)) / denominator.squeeze(2)
            type_logits = self.type_head(torch.cat([pooled_mean, pooled_scale, scalar_mean], dim=-1))

            track_hidden = (hidden * pair_weight[..., None]).sum(dim=2)
            track_hidden = track_hidden / pair_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
            line_logits = self.track_line_weight(track_hidden).squeeze(-1)
            track_valid = pair_weight.any(dim=2)
            line_weights = torch.softmax(line_logits.masked_fill(~track_valid, -1e4), dim=1)
            line_weights = line_weights * track_valid.to(line_weights.dtype)
            line_weights = line_weights / line_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            line_point = torch.einsum(
                "bn,bnd->bd", line_weights, features["reference_points"]
            )
            return {
                "type_logits": type_logits,
                "prismatic_axis": translation_axis,
                "revolute_axis": rotation_axis,
                "line_point": line_point,
                "prismatic_confidence": translation_confidence,
                "revolute_confidence": rotation_confidence,
            }

    return ChildMotionAxisHead()
