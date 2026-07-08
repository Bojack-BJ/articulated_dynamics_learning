from __future__ import annotations

import math
from statistics import median
from typing import Any

import numpy as np


def track_quality_score(track: dict[str, Any], field: str = "track_quality_score") -> float:
    quality = track.get("track_quality")
    if isinstance(quality, dict):
        value = _safe_float(quality.get(field))
        if value is not None:
            return _clip01(value)
    value = _safe_float(track.get(field))
    return _clip01(value) if value is not None else 1.0


def timestep_quality_score(sample: dict[str, Any], field: str = "timestep_quality_score") -> float:
    value = _safe_float(sample.get(field))
    return _clip01(value) if value is not None else 1.0


def observation_weight(
    track: dict[str, Any],
    sample: dict[str, Any],
    *,
    timestep_field: str = "timestep_quality_score",
    track_field: str = "track_quality_score",
    min_weight: float = 0.2,
) -> float:
    raw = math.sqrt(track_quality_score(track, track_field)) * timestep_quality_score(sample, timestep_field)
    return max(float(min_weight), min(1.0, raw))


def cluster_quality_summary(tracks: list[dict[str, Any]], *, low_quality_threshold: float = 0.35) -> dict[str, Any]:
    track_scores = [track_quality_score(track) for track in tracks]
    timestep_scores = [
        timestep_quality_score(sample)
        for track in tracks
        for sample in track.get("samples", [])
        if isinstance(sample, dict)
    ]
    visible_timesteps = [
        sample
        for track in tracks
        for sample in track.get("samples", [])
        if isinstance(sample, dict) and bool(sample.get("visible", False)) and bool(sample.get("depth_valid", True))
    ]
    effective_observation_count = sum(timestep_scores)
    return {
        "cluster_track_quality_mean": _mean(track_scores),
        "cluster_track_quality_median": float(median(track_scores)) if track_scores else None,
        "cluster_low_quality_track_ratio": _ratio_below(track_scores, low_quality_threshold),
        "cluster_low_quality_timestep_ratio": _ratio_below(timestep_scores, low_quality_threshold),
        "cluster_effective_observation_count": effective_observation_count,
        "cluster_effective_observation_ratio": (
            effective_observation_count / float(len(visible_timesteps)) if visible_timesteps else None
        ),
    }


def compute_articulation_trace_diagnostics(track: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Fit simple static/prismatic/revolute trajectory models to one lifted 3D track.

    This is a diagnostic, not a full joint optimizer. It intentionally scores whether
    a trajectory is explainable by a simple articulated path instead of penalizing
    direction change or acceleration, which can be normal for revolute motion.
    """

    frames, points = _valid_track_points(track)
    if len(points) < 2:
        summary = {
            "best_motion_type": "unknown",
            "trajectory_residual_m": None,
            "axis_error_deg": None,
            "line_distance_error_m": None,
            "articulation_score": 0.0,
        }
        return summary, {}

    models = {
        "static": _fit_static(points),
        "prismatic": _fit_prismatic(points),
        "revolute": _fit_revolute(points),
    }
    best_type = _select_best_motion_model(models, _trajectory_extent(points))
    best = models[best_type]
    scale = max(0.02, 0.10 * _trajectory_extent(points))
    articulation_score = math.exp(-float(best["rmse_m"]) / scale)
    summary = {
        "best_motion_type": best_type,
        "trajectory_residual": float(best["rmse_m"]),
        "trajectory_residual_m": float(best["rmse_m"]),
        "static_residual_m": float(models["static"]["rmse_m"]),
        "prismatic_residual_m": float(models["prismatic"]["rmse_m"]),
        "revolute_residual_m": float(models["revolute"]["rmse_m"]),
        "axis": best.get("axis"),
        "pivot": best.get("pivot"),
        "axis_error_deg": best.get("axis_error_deg"),
        "line_distance_error": best.get("line_distance_error_m"),
        "line_distance_error_m": best.get("line_distance_error_m"),
        "articulation_score": float(_clip01(articulation_score)),
    }
    timestep = {}
    residuals = best.get("residuals_m") or [0.0] * len(frames)
    for frame, residual in zip(frames, residuals):
        timestep[int(frame)] = {
            "articulation_residual_m": float(residual),
            "motion_model_type": best_type,
            "residual_score": float(_clip01(math.exp(-float(residual) / scale))),
            "articulation_residual_score": float(_clip01(math.exp(-float(residual) / scale))),
        }
    return summary, timestep


def articulation_track_score(track: dict[str, Any], field: str = "articulation_score") -> float:
    quality = track.get("track_quality")
    if isinstance(quality, dict):
        value = _safe_float(quality.get(field))
        if value is not None:
            return _clip01(value)
    value = _safe_float(track.get(field))
    return _clip01(value) if value is not None else 1.0


def articulation_pair_compatibility(a_track: dict[str, Any], b_track: dict[str, Any]) -> float:
    a_quality = a_track.get("track_quality") if isinstance(a_track.get("track_quality"), dict) else {}
    b_quality = b_track.get("track_quality") if isinstance(b_track.get("track_quality"), dict) else {}
    a_type = str(a_quality.get("best_motion_type") or a_track.get("best_motion_type") or "unknown")
    b_type = str(b_quality.get("best_motion_type") or b_track.get("best_motion_type") or "unknown")
    score = math.sqrt(articulation_track_score(a_track) * articulation_track_score(b_track))
    if a_type != "unknown" and b_type != "unknown" and a_type != b_type:
        score *= 0.65
    return float(_clip01(score))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float | None:
    return sum(values) / float(len(values)) if values else None


def _ratio_below(values: list[float], threshold: float) -> float | None:
    return sum(value < threshold for value in values) / float(len(values)) if values else None


def _valid_track_points(track: dict[str, Any]) -> tuple[list[int], list[list[float]]]:
    rows = []
    for sample in track.get("samples", []) or []:
        if not isinstance(sample, dict):
            continue
        if sample.get("visible", False) is not True or sample.get("depth_valid", True) is False:
            continue
        xyz = sample.get("xyz_world")
        if not isinstance(xyz, list) or len(xyz) != 3:
            continue
        rows.append((int(sample.get("frame_index", len(rows))), [float(xyz[0]), float(xyz[1]), float(xyz[2])]))
    rows = sorted(rows, key=lambda item: item[0])
    return [item[0] for item in rows], [item[1] for item in rows]


def _fit_static(points: list[list[float]]) -> dict[str, Any]:
    p0 = np.asarray(points[0], dtype=float)
    pts = np.asarray(points, dtype=float)
    residuals = np.linalg.norm(pts - p0[None, :], axis=1)
    return {
        "rmse_m": _rmse(residuals),
        "residuals_m": residuals.tolist(),
        "axis": None,
        "pivot": p0.tolist(),
        "axis_error_deg": None,
        "line_distance_error_m": None,
    }


def _fit_prismatic(points: list[list[float]]) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    p0 = pts[0]
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = pts[-1] - pts[0]
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    q = (pts - p0[None, :]) @ axis
    pred = p0[None, :] + q[:, None] * axis[None, :]
    residuals = np.linalg.norm(pts - pred, axis=1)
    step_angles = []
    for a, b in zip(pts[:-1], pts[1:]):
        step = b - a
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-9:
            cosine = abs(float(step @ axis) / step_norm)
            step_angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return {
        "rmse_m": _rmse(residuals),
        "residuals_m": residuals.tolist(),
        "axis": axis.tolist(),
        "pivot": p0.tolist(),
        "axis_error_deg": float(sum(step_angles) / len(step_angles)) if step_angles else None,
        "line_distance_error_m": _rmse(residuals),
    }


def _fit_revolute(points: list[list[float]]) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return {"rmse_m": float("inf"), "residuals_m": [float("inf")] * len(pts), "axis": None, "pivot": None}
    mean = pts.mean(axis=0)
    centered = pts - mean[None, :]
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"rmse_m": float("inf"), "residuals_m": [float("inf")] * len(pts), "axis": None, "pivot": None}
    basis_u = vh[0]
    basis_v = vh[1]
    normal = vh[2] if len(vh) >= 3 else np.cross(basis_u, basis_v)
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm if normal_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    coords = np.stack([centered @ basis_u, centered @ basis_v], axis=1)
    x = coords[:, 0]
    y = coords[:, 1]
    a = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"rmse_m": float("inf"), "residuals_m": [float("inf")] * len(pts), "axis": None, "pivot": None}
    radius_sq = max(0.0, float(c + cx * cx + cy * cy))
    radius = math.sqrt(radius_sq)
    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    plane_residual = np.abs(centered @ normal)
    residuals = np.sqrt((radial - radius) ** 2 + plane_residual**2)
    pivot = mean + cx * basis_u + cy * basis_v
    return {
        "rmse_m": _rmse(residuals),
        "residuals_m": residuals.tolist(),
        "axis": normal.tolist(),
        "pivot": pivot.tolist(),
        "axis_error_deg": None,
        "line_distance_error_m": None,
    }


def _select_best_motion_model(models: dict[str, dict[str, Any]], extent: float) -> str:
    """Prefer simpler models when residuals are effectively tied.

    A short straight trajectory can be overfit by a very large-radius circle.
    This tie-break keeps the diagnostic interpretable: static < prismatic <
    revolute unless the more complex model is meaningfully better.
    """

    static = float(models["static"]["rmse_m"])
    prismatic = float(models["prismatic"]["rmse_m"])
    revolute = float(models["revolute"]["rmse_m"])
    tolerance = max(1e-5, 0.03 * max(float(extent), 1e-9))
    if static <= tolerance:
        return "static"
    if prismatic <= revolute + tolerance:
        return "prismatic"
    return "revolute"


def _trajectory_extent(points: list[list[float]]) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    lower = pts.min(axis=0)
    upper = pts.max(axis=0)
    return float(np.linalg.norm(upper - lower))


def _rmse(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("inf")
    return float(math.sqrt(float(np.mean(arr * arr))))
