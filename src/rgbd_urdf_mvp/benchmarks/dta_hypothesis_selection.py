"""No-GT replay selection for DTA's released dual joint hypotheses."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def replay_hypothesis(points: np.ndarray, hypothesis: dict[str, Any], joint_type: str) -> np.ndarray:
    """Apply the motion after enforcing either a pure revolute or prismatic model."""
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if joint_type == "prismatic":
        translation = np.asarray(hypothesis["translation"], dtype=np.float64)
        return source + translation
    if joint_type == "revolute":
        rotation = np.asarray(hypothesis["rotation"], dtype=np.float64)
        pivot = np.asarray(hypothesis["axis_position"], dtype=np.float64)
        return (rotation @ (source - pivot).T).T + pivot
    raise ValueError(f"Unsupported joint type: {joint_type}")


def trimmed_surface_residual(
    replayed_points: np.ndarray,
    target_points: np.ndarray,
    *,
    trim_quantile: float = 0.9,
) -> float:
    """Return a robust one-way surface residual in target-cloud units."""
    if not 0.5 <= trim_quantile <= 1.0:
        raise ValueError("trim_quantile must be in [0.5, 1.0]")
    replayed = np.asarray(replayed_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if len(replayed) == 0 or len(target) == 0:
        raise ValueError("replayed_points and target_points must be non-empty")
    distances, _ = cKDTree(target).query(replayed, k=1, workers=-1)
    cutoff = float(np.quantile(distances, trim_quantile))
    inliers = distances[distances <= cutoff]
    return float(np.sqrt(np.mean(np.square(inliers))))


def select_joint_hypothesis(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    prismatic: dict[str, Any],
    revolute: dict[str, Any],
    trim_quantile: float = 0.9,
    ambiguity_relative_margin: float = 0.08,
    minimum_motion_fraction: float = 0.005,
) -> dict[str, Any]:
    """Select a constrained model by target-state replay without GT kinematics."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    bbox_extent = np.ptp(np.concatenate((source, target), axis=0), axis=0)
    bbox_diagonal = max(float(np.linalg.norm(bbox_extent)), 1e-12)

    replayed = {
        "prismatic": replay_hypothesis(source, prismatic, "prismatic"),
        "revolute": replay_hypothesis(source, revolute, "revolute"),
    }
    residuals = {
        name: trimmed_surface_residual(
            points, target, trim_quantile=trim_quantile
        )
        for name, points in replayed.items()
    }
    motion = {
        name: float(np.median(np.linalg.norm(points - source, axis=1)))
        for name, points in replayed.items()
    }
    ranked = sorted(residuals, key=residuals.get)
    best, second = ranked
    absolute_margin = residuals[second] - residuals[best]
    relative_margin = absolute_margin / max(residuals[second], 1e-12)
    low_motion = max(motion.values()) / bbox_diagonal < minimum_motion_fraction
    ambiguous = low_motion or relative_margin < ambiguity_relative_margin
    return {
        "selected_type": "ambiguous" if ambiguous else best,
        "best_hypothesis": best,
        "ambiguous": ambiguous,
        "ambiguity_reasons": [
            reason
            for reason, active in (
                ("low_motion", low_motion),
                ("small_replay_margin", relative_margin < ambiguity_relative_margin),
            )
            if active
        ],
        "residual_m": residuals,
        "residual_normalized_by_bbox": {
            name: value / bbox_diagonal for name, value in residuals.items()
        },
        "motion_m": motion,
        "selection_margin_m": absolute_margin,
        "selection_relative_margin": relative_margin,
        "bbox_diagonal_m": bbox_diagonal,
        "trim_quantile": trim_quantile,
        "selection_method": "constrained_target_surface_replay",
        "provenance": "project_adapter_not_official_dta",
    }
