from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..core.serialization import load_json, save_json
from .quality_weights import compute_articulation_trace_diagnostics


@dataclass(slots=True)
class TrackQualityConfig:
    input_tracks: str | Path
    output_dir: str | Path
    bad_track_threshold: float = 0.35
    bad_timestep_threshold: float = 0.35
    mask_bad_timesteps: bool = False
    max_step_m: float | None = None
    keep_query_connected_segment: bool = False
    min_query_connected_frames: int = 2
    query_connected_max_gap_frames: int = 0
    spatial_dbscan_eps_m: float | None = None
    spatial_dbscan_min_samples: int = 5


class TrackQualityAnalyzer:
    """Compute diagnostic per-track and per-timestep quality for lifted 3D tracks."""

    def analyze(self, config: TrackQualityConfig) -> dict[str, Path]:
        input_path = Path(config.input_tracks).expanduser().resolve()
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = load_json(input_path)
        tracks = [track for track in payload.get("tracks", []) if isinstance(track, dict)]

        track_rows: list[dict[str, Any]] = []
        timestep_rows: list[dict[str, Any]] = []
        bad_trace_ids: list[int] = []
        bad_timestep_mask: dict[str, list[int]] = {}
        enriched_tracks: list[dict[str, Any]] = []

        for track in tracks:
            summary, timestep_quality = compute_track_quality(track)
            track_rows.append(summary)
            if float(summary["track_quality_score"]) < float(config.bad_track_threshold):
                bad_trace_ids.append(int(summary["track_id"]))
            bad_frames = []
            enriched_track = dict(track)
            enriched_samples = []
            by_frame = {int(row["frame_index"]): row for row in timestep_quality}
            for sample in track.get("samples", []) or []:
                enriched_sample = dict(sample)
                try:
                    frame_index = int(sample.get("frame_index", 0))
                except (TypeError, ValueError):
                    frame_index = 0
                row = by_frame.get(frame_index)
                if row is not None:
                    enriched_sample["timestep_quality_score"] = row["timestep_quality_score"]
                    enriched_sample["step_length_m"] = row["step_length_m"]
                    enriched_sample["step_outlier_score"] = row["step_outlier_score"]
                    enriched_sample["acceleration_m"] = row["acceleration_m"]
                    enriched_sample["acceleration_outlier_score"] = row["acceleration_outlier_score"]
                    enriched_sample["jerk_m"] = row["jerk_m"]
                    enriched_sample["jerk_outlier_score"] = row["jerk_outlier_score"]
                    enriched_sample["smooth_residual_m"] = row["smooth_residual_m"]
                    enriched_sample["smooth_residual_score"] = row["smooth_residual_score"]
                    enriched_sample["direction_change_deg"] = row["direction_change_deg"]
                    enriched_sample["direction_score"] = row["direction_score"]
                    enriched_sample["local_smoothness_score"] = row["local_smoothness_score"]
                    enriched_sample["step_consistency_score"] = row["step_consistency_score"]
                    enriched_sample["articulation_residual_m"] = row["articulation_residual_m"]
                    enriched_sample["motion_model_type"] = row["motion_model_type"]
                    enriched_sample["residual_score"] = row["residual_score"]
                    enriched_sample["articulation_residual_score"] = row["articulation_residual_score"]
                    quality_rejected = float(row["timestep_quality_score"]) < float(
                        config.bad_timestep_threshold
                    )
                    step = row.get("step_length_m")
                    step_rejected = (
                        config.max_step_m is not None
                        and step is not None
                        and float(step) > float(config.max_step_m)
                    )
                    if quality_rejected or step_rejected:
                        bad_frames.append(frame_index)
                    if config.mask_bad_timesteps and (quality_rejected or step_rejected):
                        enriched_sample["quality_original_visible"] = bool(
                            enriched_sample.get("visible", False)
                        )
                        enriched_sample["visible"] = False
                        enriched_sample["confidence"] = 0.0
                        enriched_sample["quality_rejected"] = True
                        enriched_sample["quality_rejection_reasons"] = [
                            reason
                            for reason, rejected in (
                                ("low_timestep_quality", quality_rejected),
                                ("step_too_large", step_rejected),
                            )
                            if rejected
                        ]
                enriched_samples.append(enriched_sample)
            if config.keep_query_connected_segment:
                connected_frames = _keep_query_connected_segment(
                    enriched_samples,
                    query_frame_index=int(track.get("query_frame_index", 0)),
                    min_frames=max(1, int(config.min_query_connected_frames)),
                    max_gap_frames=max(0, int(config.query_connected_max_gap_frames)),
                )
                for enriched_sample in enriched_samples:
                    frame_index = int(enriched_sample.get("frame_index", 0))
                    if frame_index in connected_frames:
                        continue
                    if bool(enriched_sample.get("visible", False)):
                        enriched_sample["quality_original_visible"] = True
                    enriched_sample["visible"] = False
                    enriched_sample["confidence"] = 0.0
                    enriched_sample["quality_rejected"] = True
                    reasons = list(enriched_sample.get("quality_rejection_reasons", []))
                    if "outside_query_connected_segment" not in reasons:
                        reasons.append("outside_query_connected_segment")
                    enriched_sample["quality_rejection_reasons"] = reasons
                    bad_frames.append(frame_index)
            enriched_track["samples"] = enriched_samples
            enriched_track["track_quality"] = {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "track_id",
                    "part_id",
                    "original_part_id",
                }
            }
            enriched_tracks.append(enriched_track)
            timestep_rows.extend(timestep_quality)
            if bad_frames:
                bad_timestep_mask[str(summary["track_id"])] = sorted(set(bad_frames))

        spatial_outlier_count = 0
        if config.spatial_dbscan_eps_m is not None:
            spatial_outliers = _spatial_dbscan_outliers(
                enriched_tracks,
                eps_m=max(1e-9, float(config.spatial_dbscan_eps_m)),
                min_samples=max(2, int(config.spatial_dbscan_min_samples)),
            )
            spatial_outlier_count = len(spatial_outliers)
            for track_index, sample_index in spatial_outliers:
                track = enriched_tracks[track_index]
                sample = track["samples"][sample_index]
                frame_index = int(sample.get("frame_index", 0))
                if config.mask_bad_timesteps:
                    sample["quality_original_visible"] = bool(sample.get("visible", False))
                    sample["visible"] = False
                    sample["confidence"] = 0.0
                    sample["quality_rejected"] = True
                    reasons = list(sample.get("quality_rejection_reasons", []))
                    if "spatial_dbscan_outlier" not in reasons:
                        reasons.append("spatial_dbscan_outlier")
                    sample["quality_rejection_reasons"] = reasons
                track_id = str(track.get("track_id", track_index))
                bad_timestep_mask.setdefault(track_id, []).append(frame_index)
            for track_id, frames in bad_timestep_mask.items():
                bad_timestep_mask[track_id] = sorted(set(frames))

        enriched_payload = dict(payload)
        enriched_payload["tracks"] = enriched_tracks
        enriched_payload["track_quality"] = {
            "source_tracks": str(input_path),
            "track_count": len(tracks),
            "bad_track_threshold": float(config.bad_track_threshold),
            "bad_timestep_threshold": float(config.bad_timestep_threshold),
            "mask_bad_timesteps": bool(config.mask_bad_timesteps),
            "max_step_m": config.max_step_m,
            "keep_query_connected_segment": bool(config.keep_query_connected_segment),
            "min_query_connected_frames": max(1, int(config.min_query_connected_frames)),
            "query_connected_max_gap_frames": max(
                0, int(config.query_connected_max_gap_frames)
            ),
            "spatial_dbscan_eps_m": config.spatial_dbscan_eps_m,
            "spatial_dbscan_min_samples": max(2, int(config.spatial_dbscan_min_samples)),
            "notes": {
                "track_quality_score": "Whole-trajectory reliability aggregated from visibility, evidence, median timestep quality, and low-quality timestep ratio.",
                "timestep_quality_score": "Local per-frame contribution weight; it penalizes invalid samples, outlier steps/accelerations, and local smooth residuals.",
                "articulation_score": "Whole-trajectory diagnostic score from the best static/prismatic/revolute model residual. It is separate from the default track quality score.",
                "articulation_residual_m": "Per-frame residual to the best simple articulated motion model.",
                "direction_change_deg": "Diagnostic only. It can be high for valid revolute arcs and is not a main quality penalty.",
                "quality_rejected": "Rejected samples remain in the artifact but are invisible to downstream geometry and replay.",
            },
        }

        track_csv = output_dir / "track_quality_summary.csv"
        timestep_csv = output_dir / "timestep_quality_summary.csv"
        track_json = output_dir / "track_quality_summary.json"
        timestep_json = output_dir / "timestep_quality_summary.json"
        mask_npz = output_dir / "timestep_quality_mask.npz"
        bad_trace_json = output_dir / "bad_trace_ids.json"
        bad_timestep_json = output_dir / "bad_timestep_mask.json"
        enriched_json = output_dir / "motion_part_tracks_with_quality.json"

        _write_csv(track_csv, track_rows)
        _write_csv(timestep_csv, timestep_rows)
        save_json({"tracks": track_rows}, track_json)
        save_json({"timesteps": timestep_rows}, timestep_json)
        save_json({"bad_trace_ids": bad_trace_ids}, bad_trace_json)
        save_json({"bad_timestep_mask": bad_timestep_mask}, bad_timestep_json)
        save_json(enriched_payload, enriched_json)
        _write_quality_npz(mask_npz, timestep_rows, bad_timestep_threshold=float(config.bad_timestep_threshold))

        manifest = {
            "source_tracks": str(input_path),
            "outputs": {
                "track_quality_summary_csv": str(track_csv),
                "track_quality_summary_json": str(track_json),
                "timestep_quality_summary_csv": str(timestep_csv),
                "timestep_quality_summary_json": str(timestep_json),
                "timestep_quality_mask_npz": str(mask_npz),
                "bad_trace_ids_json": str(bad_trace_json),
                "bad_timestep_mask_json": str(bad_timestep_json),
                "motion_tracks_with_quality": str(enriched_json),
            },
            "summary": summarize_quality(track_rows, timestep_rows),
            "filtering": {
                "mask_bad_timesteps": bool(config.mask_bad_timesteps),
                "bad_timestep_threshold": float(config.bad_timestep_threshold),
                "max_step_m": config.max_step_m,
                "keep_query_connected_segment": bool(config.keep_query_connected_segment),
                "min_query_connected_frames": max(1, int(config.min_query_connected_frames)),
                "query_connected_max_gap_frames": max(
                    0, int(config.query_connected_max_gap_frames)
                ),
                "spatial_dbscan_eps_m": config.spatial_dbscan_eps_m,
                "spatial_dbscan_min_samples": max(2, int(config.spatial_dbscan_min_samples)),
                "spatial_dbscan_outlier_count": spatial_outlier_count,
                "masked_timestep_count": sum(len(frames) for frames in bad_timestep_mask.values())
                if config.mask_bad_timesteps
                else 0,
            },
            "notes": {
                "track_quality_score": "Whole-trajectory reliability.",
                "timestep_quality_score": "Local per-frame contribution weight.",
                "articulation_score": "Diagnostic score from best static/prismatic/revolute fit; not enabled as a default quality penalty.",
                "direction_change_deg": "Diagnostic only; high direction change can be normal for revolute motion.",
            },
        }
        manifest_path = output_dir / "track_quality_manifest.json"
        save_json(manifest, manifest_path)
        return {key: Path(value) for key, value in manifest["outputs"].items()} | {"manifest": manifest_path}


def compute_track_quality(track: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = _ordered_samples(track)
    valid_samples = [sample for sample in samples if _valid_sample(sample)]
    total_frames = len(samples)
    visible_frames = len(valid_samples)
    visible_ratio = visible_frames / float(max(1, total_frames))
    points = [_xyz(sample) for sample in valid_samples]
    frames = [int(sample.get("frame_index", idx)) for idx, sample in enumerate(valid_samples)]
    vectors = [_sub(points[idx], points[idx - 1]) for idx in range(1, len(points))]
    steps = [_norm(vector) for vector in vectors]
    accel_vectors = [_sub(vectors[idx], vectors[idx - 1]) for idx in range(1, len(vectors))]
    accelerations = [_norm(vector) for vector in accel_vectors]
    jerk_vectors = [_sub(accel_vectors[idx], accel_vectors[idx - 1]) for idx in range(1, len(accel_vectors))]
    jerks = [_norm(vector) for vector in jerk_vectors]
    smooth_residuals = [_local_linear_residual(points, idx) for idx in range(len(points))]
    valid_smooth_residuals = [value for value in smooth_residuals if value is not None]
    direction_changes = [_direction_change_deg(vectors[idx - 1], vectors[idx]) for idx in range(1, len(vectors))]
    valid_direction_changes = [value for value in direction_changes if value is not None]
    median_step = _median_or_none(steps)
    max_step = max(steps) if steps else None
    median_accel = _median_or_none(accelerations)
    max_accel = max(accelerations) if accelerations else None
    median_jerk = _median_or_none(jerks)
    max_jerk = max(jerks) if jerks else None
    median_smooth_residual = _median_or_none(valid_smooth_residuals)
    max_smooth_residual = max(valid_smooth_residuals) if valid_smooth_residuals else None
    step_median, step_mad = _median_and_mad(steps)
    accel_median, accel_mad = _median_and_mad(accelerations)
    jerk_median, jerk_mad = _median_and_mad(jerks)
    smooth_median, smooth_mad = _median_and_mad(valid_smooth_residuals)
    tracker_vis = [_safe_float(sample.get("tracker_visibility"), 1.0) for sample in samples]
    confidences = [_safe_float(sample.get("confidence"), 1.0) for sample in samples]
    mask_consistency = [1.0 if sample.get("mask_consistent", True) else 0.0 for sample in samples]
    tracker_visibility_mean = _mean([value for value in tracker_vis if value is not None])
    confidence_mean = _mean([value for value in confidences if value is not None])
    mask_consistency_ratio = _mean(mask_consistency)
    direction_scale_deg = 45.0
    depth_jump_score = _robust_outlier_score(max_accel, accel_median, accel_mad, floor=0.005)
    evidence_score = _mean([tracker_visibility_mean, confidence_mean, mask_consistency_ratio])
    articulation_summary, articulation_by_frame = compute_articulation_trace_diagnostics(track)

    timestep_rows = []
    step_by_frame = {frames[idx]: steps[idx - 1] for idx in range(1, len(frames))}
    accel_by_frame = {frames[idx]: accelerations[idx - 2] for idx in range(2, len(frames))}
    jerk_by_frame = {frames[idx]: jerks[idx - 3] for idx in range(3, len(frames))}
    smooth_by_frame = {frames[idx]: smooth_residuals[idx] for idx in range(len(frames))}
    direction_by_frame = {frames[idx]: direction_changes[idx - 2] for idx in range(2, len(frames))}
    for sample in samples:
        try:
            frame_index = int(sample.get("frame_index", 0))
        except (TypeError, ValueError):
            frame_index = 0
        valid = _valid_sample(sample)
        step = step_by_frame.get(frame_index)
        accel = accel_by_frame.get(frame_index)
        jerk = jerk_by_frame.get(frame_index)
        smooth_residual = smooth_by_frame.get(frame_index)
        direction_change = direction_by_frame.get(frame_index)
        articulation_row = articulation_by_frame.get(frame_index, {})
        step_outlier_score = _robust_outlier_score(step, step_median, step_mad, floor=0.005)
        acceleration_outlier_score = min(
            _robust_outlier_score(accel, accel_median, accel_mad, floor=0.005),
            _exp_score(accel, 0.08),
        )
        jerk_outlier_score = min(
            _robust_outlier_score(jerk, jerk_median, jerk_mad, floor=0.005),
            _exp_score(jerk, 0.10),
        )
        smooth_residual_score = min(
            _robust_outlier_score(smooth_residual, smooth_median, smooth_mad, floor=0.005),
            _exp_score(smooth_residual, 0.03),
        )
        direction_score = _exp_score(direction_change, direction_scale_deg)
        local_smooth = 0.45 * acceleration_outlier_score + 0.20 * jerk_outlier_score + 0.35 * smooth_residual_score
        timestep_quality = _clip01(
            0.25 * (1.0 if valid else 0.0)
            + 0.15 * (_safe_float(sample.get("tracker_visibility"), 1.0) or 0.0)
            + 0.10 * (_safe_float(sample.get("confidence"), 1.0) or 0.0)
            + 0.10 * (1.0 if sample.get("mask_consistent", True) else 0.0)
            + 0.20 * acceleration_outlier_score
            + 0.20 * smooth_residual_score
        )
        timestep_rows.append(
            {
                "track_id": int(track.get("track_id", -1)),
                "part_id": track.get("part_id"),
                "original_part_id": track.get("original_part_id"),
                "frame_index": frame_index,
                "valid": bool(valid),
                "step_length_m": step,
                "step_outlier_score": step_outlier_score,
                "acceleration_m": accel,
                "acceleration_outlier_score": acceleration_outlier_score,
                "jerk_m": jerk,
                "jerk_outlier_score": jerk_outlier_score,
                "smooth_residual_m": smooth_residual,
                "smooth_residual_score": smooth_residual_score,
                "direction_change_deg": direction_change,
                "local_smoothness_score": local_smooth,
                "acceleration_score": acceleration_outlier_score,
                "direction_score": direction_score,
                "step_consistency_score": local_smooth,
                "timestep_quality_score": timestep_quality,
                "articulation_residual_m": articulation_row.get("articulation_residual_m"),
                "motion_model_type": articulation_row.get(
                    "motion_model_type", articulation_summary.get("best_motion_type", "unknown")
                ),
                "residual_score": articulation_row.get("residual_score"),
                "articulation_residual_score": articulation_row.get("articulation_residual_score"),
            }
        )
    timestep_quality_values = [float(row["timestep_quality_score"]) for row in timestep_rows]
    low_quality_timestep_ratio = (
        sum(value < 0.90 for value in timestep_quality_values) / float(len(timestep_quality_values))
        if timestep_quality_values
        else 1.0
    )
    median_timestep_quality = _median_or_none(timestep_quality_values) or 0.0
    temporal_smoothness_score = median_timestep_quality
    track_quality_score = _clip01(
        0.30 * visible_ratio
        + 0.30 * median_timestep_quality
        + 0.20 * evidence_score
        + 0.20 * (1.0 - low_quality_timestep_ratio)
    )

    summary = {
        "track_id": int(track.get("track_id", -1)),
        "part_id": track.get("part_id"),
        "original_part_id": track.get("original_part_id"),
        "visible_frame_count": visible_frames,
        "total_frame_count": total_frames,
        "visible_ratio": visible_ratio,
        "mean_motion_m": _track_motion(points),
        "median_step_m": median_step,
        "max_step_m": max_step,
        "median_acceleration_m": median_accel,
        "max_acceleration_m": max_accel,
        "median_jerk_m": median_jerk,
        "max_jerk_m": max_jerk,
        "median_smooth_residual_m": median_smooth_residual,
        "max_smooth_residual_m": max_smooth_residual,
        "median_direction_change_deg": _median_or_none(valid_direction_changes),
        "max_direction_change_deg": max(valid_direction_changes) if valid_direction_changes else None,
        "step_median_m": step_median,
        "step_mad_m": step_mad,
        "acceleration_median_m": accel_median,
        "acceleration_mad_m": accel_mad,
        "smooth_residual_median_m": smooth_median,
        "smooth_residual_mad_m": smooth_mad,
        "depth_jump_score": depth_jump_score,
        "temporal_smoothness_score": temporal_smoothness_score,
        "median_timestep_quality": median_timestep_quality,
        "low_quality_timestep_ratio": low_quality_timestep_ratio,
        "tracker_visibility_mean": tracker_visibility_mean,
        "confidence_mean": confidence_mean,
        "mask_consistency_ratio": mask_consistency_ratio,
        "track_quality_score": track_quality_score,
        **articulation_summary,
    }
    return summary, timestep_rows


def summarize_quality(track_rows: list[dict[str, Any]], timestep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "track_count": len(track_rows),
        "timestep_count": len(timestep_rows),
        "mean_track_quality": _mean([float(row["track_quality_score"]) for row in track_rows]),
        "mean_timestep_quality": _mean([float(row["timestep_quality_score"]) for row in timestep_rows]),
        "low_quality_track_count_lt_0_35": sum(float(row["track_quality_score"]) < 0.35 for row in track_rows),
        "low_quality_timestep_count_lt_0_35": sum(float(row["timestep_quality_score"]) < 0.35 for row in timestep_rows),
    }


def _ordered_samples(track: dict[str, Any]) -> list[dict[str, Any]]:
    samples = [sample for sample in track.get("samples", []) if isinstance(sample, dict)]
    return sorted(samples, key=lambda sample: int(sample.get("frame_index", 0)))


def _spatial_dbscan_outliers(
    tracks: list[dict[str, Any]],
    *,
    eps_m: float,
    min_samples: int,
) -> set[tuple[int, int]]:
    """Find isolated per-timestep samples without collapsing valid dense components."""
    groups: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for track_index, track in enumerate(tracks):
        try:
            part_id = int(track.get("part_id", track.get("pred_cluster", 0)))
        except (TypeError, ValueError):
            part_id = 0
        for sample_index, sample in enumerate(track.get("samples", []) or []):
            if not _valid_sample(sample):
                continue
            frame_index = int(sample.get("frame_index", 0))
            groups.setdefault((frame_index, part_id), []).append(
                (track_index, sample_index, _xyz(sample))
            )

    outliers: set[tuple[int, int]] = set()
    min_samples = max(2, int(min_samples))
    for members in groups.values():
        if len(members) < min_samples:
            # A sparse visible part is not sufficient evidence of an outlier.
            continue
        points = np.stack([member[2] for member in members], axis=0)
        neighborhoods = cKDTree(points).query_ball_point(points, r=float(eps_m))
        core = np.asarray([len(neighbors) >= min_samples for neighbors in neighborhoods])
        clustered = core.copy()
        for index, neighbors in enumerate(neighborhoods):
            if core[index]:
                clustered[np.asarray(neighbors, dtype=int)] = True
        for index in np.flatnonzero(~clustered):
            outliers.add((members[int(index)][0], members[int(index)][1]))
    return outliers


def _keep_query_connected_segment(
    samples: list[dict[str, Any]],
    *,
    query_frame_index: int,
    min_frames: int,
    max_gap_frames: int = 0,
) -> set[int]:
    """Return the seed-anchored run, optionally spanning short visibility gaps."""
    valid_frames = sorted(
        int(sample.get("frame_index", 0))
        for sample in samples
        if _valid_sample(sample) and not bool(sample.get("quality_rejected", False))
    )
    if not valid_frames:
        return set()
    anchor = min(valid_frames, key=lambda frame: abs(frame - int(query_frame_index)))
    runs: list[list[int]] = []
    for frame in valid_frames:
        missing_frames = frame - runs[-1][-1] - 1 if runs else None
        if not runs or missing_frames is None or missing_frames > max(0, int(max_gap_frames)):
            runs.append([frame])
        else:
            runs[-1].append(frame)
    selected = next((run for run in runs if anchor in run), [])
    return set(selected) if len(selected) >= max(1, int(min_frames)) else set()


def _valid_sample(sample: dict[str, Any]) -> bool:
    return (
        sample.get("visible", False) is True
        and sample.get("depth_valid", True) is not False
        and isinstance(sample.get("xyz_world"), list)
        and len(sample["xyz_world"]) == 3
    )


def _xyz(sample: dict[str, Any]) -> list[float]:
    xyz = sample["xyz_world"]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[idx] - b[idx] for idx in range(3)]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def _direction_change_deg(a: list[float], b: list[float]) -> float | None:
    an = _norm(a)
    bn = _norm(b)
    if an < 1e-9 or bn < 1e-9:
        return None
    cosine = sum(a[idx] * b[idx] for idx in range(3)) / (an * bn)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _local_linear_residual(points: list[list[float]], idx: int) -> float | None:
    if idx <= 0 or idx >= len(points) - 1:
        return None
    predicted = [(points[idx - 1][axis] + points[idx + 1][axis]) * 0.5 for axis in range(3)]
    return _norm(_sub(points[idx], predicted))


def _median_and_mad(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    med = float(median(values))
    deviations = [abs(float(value) - med) for value in values]
    return med, float(median(deviations)) if deviations else 0.0


def _robust_outlier_score(
    value: float | None,
    median_value: float | None,
    mad_value: float | None,
    *,
    floor: float,
    z_scale: float = 3.0,
) -> float:
    if value is None:
        return 1.0
    if median_value is None:
        return 1.0
    robust_scale = max(float(floor), 1.4826 * float(mad_value or 0.0))
    z = max(0.0, float(value) - float(median_value)) / robust_scale
    return math.exp(-z / max(1e-9, float(z_scale)))


def _track_motion(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return _norm(_sub(points[-1], points[0]))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float | None]) -> float:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(filtered) / float(len(filtered)) if filtered else 0.0


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _exp_score(value: float | None, scale: float) -> float:
    if value is None:
        return 1.0
    return math.exp(-max(0.0, float(value)) / max(1e-9, float(scale)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_quality_npz(path: Path, rows: list[dict[str, Any]], *, bad_timestep_threshold: float) -> None:
    track_ids = np.array([int(row["track_id"]) for row in rows], dtype=np.int64)
    frame_indices = np.array([int(row["frame_index"]) for row in rows], dtype=np.int64)
    quality = np.array([float(row["timestep_quality_score"]) for row in rows], dtype=np.float32)
    valid = np.array([bool(row["valid"]) for row in rows], dtype=bool)
    bad = quality < float(bad_timestep_threshold)
    np.savez_compressed(path, track_ids=track_ids, frame_indices=frame_indices, quality=quality, valid=valid, bad=bad)
