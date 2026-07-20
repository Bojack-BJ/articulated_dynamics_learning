#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.perception.quality_weights import articulation_pair_compatibility


SPATIAL_CLOSE_M = 0.35
HIGH_ARTICULATION_COMPATIBILITY = 0.85
REPLAY_SCALE_M = 0.05
RIGID_SCALE_M = 0.04
PARENT_POLLUTION_SCALE_M = 0.03
SPATIAL_GAIN_SCALE_M = 0.30
LOW_MOTION_BASE_THRESHOLD_M = 0.03
MOVING_EVIDENCE_THRESHOLD = 0.35
PATCH_TRACK_COUNT_RATIO = 0.35
PATCH_BBOX_RATIO = 0.45
PATCH_ATTACH_THRESHOLD = 0.45
PATCH_ATTACH_MAX_DISTANCE_M = 0.55
STRONG_SHARED_MODEL_OVERRIDE = 0.70
GREEDY_PATCH_ATTACH_MIN_SCORE = 0.48
GREEDY_CHILD_CHILD_MIN_SCORE = 0.42
GREEDY_CHILD_BASE_MIN_SCORE = 0.55
REVOLUTE_Q_STD_SCALE_RAD = 0.18
REVOLUTE_RADIAL_STD_SCALE_M = 0.04
PRISMATIC_DIRECTION_STD_SCALE = 0.25
PRISMATIC_EQUAL_DISP_SCALE_M = 0.06


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _module_cmd(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "rgbd_urdf_mvp", *[str(arg) for arg in args]]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _exp_improvement_score(delta: float | None, scale: float) -> float:
    if delta is None:
        return 0.5
    # Positive delta means after is worse; negative delta means improvement.
    return _clip01(0.5 + math.tanh(-float(delta) / max(1e-9, scale)) * 0.5)


def _penalty_from_delta(delta: float | None, scale: float) -> float:
    if delta is None:
        return 0.0
    return _clip01(max(0.0, float(delta)) / max(1e-9, scale))


def merge_score_no_gt(features: dict[str, Any]) -> dict[str, float]:
    candidate_type = str(features.get("candidate_type") or "child_child_merge")
    shared_hinge_score = _safe_float(features.get("shared_hinge_score"), 0.0)
    shared_joint_score = _safe_float(features.get("shared_joint_score"), shared_hinge_score)
    joint_stability_score = _safe_float(features.get("joint_stability_score"), 0.5)
    replay_improvement_score = _exp_improvement_score(_maybe_float(features.get("joint_replay_delta")), REPLAY_SCALE_M)
    rigid_consistency_score = _exp_improvement_score(_maybe_float(features.get("rigid_rmse_delta")), RIGID_SCALE_M)
    coverage_gain_score = _clip01(max(0.0, _safe_float(features.get("coverage_delta"), 0.0)) / 0.25)
    complexity_reduction_bonus = _safe_float(features.get("complexity_reduction_bonus"), 0.0)
    spatial_extent_gain = _safe_float(features.get("spatial_extent_gain"), 0.0)
    parent_pollution_penalty = _penalty_from_delta(_maybe_float(features.get("parent_pollution_delta")), PARENT_POLLUTION_SCALE_M)
    instability_penalty = _safe_float(features.get("instability_penalty"), 0.0)
    purity_risk_penalty = _clip01(max(0.0, -_safe_float(features.get("purity_delta"), 0.0)) / 0.25)
    base_absorption_penalty = _safe_float(features.get("base_absorption_penalty"), 0.0)
    moving_evidence_penalty = _safe_float(features.get("moving_evidence_score"), 0.0)
    low_motion_score = 1.0 - moving_evidence_penalty
    base_rigid_consistency_score = rigid_consistency_score
    attach_score = _safe_float(features.get("attach_score_no_gt"), 0.0)
    local_prismatic_penalty = _safe_float(features.get("local_prismatic_degeneracy_penalty"), 0.0)
    attach_instability_penalty = _safe_float(features.get("attach_instability_penalty"), 0.0)
    shared_model_score = _safe_float(features.get("shared_joint_model_score"), shared_joint_score)
    revolute_over_prismatic_score = _safe_float(features.get("revolute_over_prismatic_score"), 0.0)
    joint_type_mismatch_penalty = _safe_float(features.get("joint_type_mismatch_penalty"), 0.0)
    if candidate_type == "child_base_merge":
        score = (
            0.45 * low_motion_score
            + 0.35 * base_rigid_consistency_score
            - 0.55 * base_absorption_penalty
            - 0.35 * moving_evidence_penalty
        )
    elif candidate_type == "patch_attach_to_existing_joint":
        score = (
            0.55 * attach_score
            + 0.15 * coverage_gain_score
            + 0.20 * local_prismatic_penalty
            + 0.15 * revolute_over_prismatic_score
            + 0.05 * complexity_reduction_bonus
            - 0.25 * attach_instability_penalty
        )
    else:
        score = (
            0.30 * shared_model_score
            + 0.15 * joint_stability_score
            + 0.20 * replay_improvement_score
            + 0.12 * revolute_over_prismatic_score
            + 0.15 * coverage_gain_score
            + 0.10 * complexity_reduction_bonus
            + 0.10 * spatial_extent_gain
            - 0.20 * instability_penalty
            - 0.25 * purity_risk_penalty
            - 0.45 * joint_type_mismatch_penalty
        )
    return {
        "merge_score_no_gt": score,
        "candidate_type": candidate_type,
        "shared_hinge_score_component": shared_hinge_score,
        "shared_joint_score_component": shared_joint_score,
        "shared_joint_model_score_component": shared_model_score,
        "revolute_over_prismatic_score_component": revolute_over_prismatic_score,
        "joint_stability_score_component": joint_stability_score,
        "replay_improvement_score": replay_improvement_score,
        "rigid_consistency_score": rigid_consistency_score,
        "coverage_gain_score": coverage_gain_score,
        "complexity_reduction_bonus_component": complexity_reduction_bonus,
        "spatial_extent_gain_component": spatial_extent_gain,
        "parent_pollution_penalty": parent_pollution_penalty,
        "instability_penalty_component": instability_penalty,
        "purity_risk_penalty": purity_risk_penalty,
        "base_absorption_penalty_component": base_absorption_penalty,
        "moving_evidence_penalty": moving_evidence_penalty,
        "low_motion_score": low_motion_score,
        "base_rigid_consistency_score": base_rigid_consistency_score,
        "attach_score_component": attach_score,
        "local_prismatic_degeneracy_penalty_component": local_prismatic_penalty,
        "attach_instability_penalty_component": attach_instability_penalty,
        "joint_type_mismatch_penalty_component": joint_type_mismatch_penalty,
    }


def _safe_float(value: Any, default: float) -> float:
    parsed = _maybe_float(value)
    return default if parsed is None else parsed


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _part_id(track: dict[str, Any]) -> int | None:
    value = track.get("part_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _visible_xyz_samples(track: dict[str, Any]) -> list[list[float]]:
    samples: list[list[float]] = []
    for sample in track.get("samples", []) or []:
        if sample.get("visible") is False or sample.get("depth_valid") is False:
            continue
        xyz = sample.get("xyz_world")
        if isinstance(xyz, list) and len(xyz) == 3:
            samples.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return samples


def _reference_point(track: dict[str, Any]) -> list[float] | None:
    raw = track.get("reference_xyz_world")
    if isinstance(raw, list) and len(raw) == 3:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    samples = _visible_xyz_samples(track)
    return samples[0] if samples else None


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _norm(vec: list[float] | None) -> float | None:
    if vec is None:
        return None
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _bbox_diag(points: list[list[float]]) -> float | None:
    if not points:
        return None
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return _norm([maxs[i] - mins[i] for i in range(3)])


def _track_displacement(track: dict[str, Any]) -> list[float] | None:
    samples = _visible_xyz_samples(track)
    if len(samples) < 2:
        return None
    return [samples[-1][i] - samples[0][i] for i in range(3)]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _cluster_stats(track_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for track in track_payload.get("tracks", []) or []:
        part_id = _part_id(track)
        if part_id is not None:
            grouped.setdefault(part_id, []).append(track)
    stats: dict[int, dict[str, Any]] = {}
    for part_id, tracks in grouped.items():
        points = [point for track in tracks if (point := _reference_point(track)) is not None]
        all_points = [xyz for track in tracks for xyz in _visible_xyz_samples(track)]
        motions = [norm for track in tracks if (norm := _norm(_track_displacement(track))) is not None]
        centroid = [sum(point[i] for point in points) / float(len(points)) for i in range(3)] if points else [0, 0, 0]
        stats[part_id] = {
            "part_id": part_id,
            "tracks": tracks,
            "track_count": len(tracks),
            "centroid": centroid,
            "bbox_diag_m": _bbox_diag(all_points or points),
            "mean_motion_m": sum(motions) / float(len(motions)) if motions else None,
            "median_motion_m": _median(motions),
            "is_base_like": False,
            "moving_evidence_score": 0.0,
            "cluster_rigid_rmse": None,
        }
    return stats


def _annotate_cluster_roles(
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
    before_eval: dict[str, Any],
) -> None:
    child_ids = {int(joint.get("child_part_id")) for joint in before_joints.get("joints", []) if joint.get("child_part_id") is not None}
    anchor_id = before_joints.get("anchor_part_id")
    largest_id = max(stats, key=lambda part_id: int(stats[part_id].get("track_count") or 0), default=None)
    for part_id, item in stats.items():
        motion = _safe_float(item.get("mean_motion_m"), 0.0)
        bbox = _safe_float(item.get("bbox_diag_m"), 0.0)
        track_count = _safe_float(item.get("track_count"), 0.0)
        has_joint = part_id in child_ids
        rigid = _maybe_float(_cluster_debug(before_eval, part_id).get("rigid_rmse_m"))
        item["cluster_rigid_rmse"] = rigid
        motion_score = _clip01(motion / 0.12)
        extent_score = _clip01(bbox / 0.6)
        joint_score = 0.75 if has_joint else 0.0
        item["moving_evidence_score"] = max(motion_score, joint_score) * max(0.35, extent_score)
        item["is_base_like"] = (
            (anchor_id is not None and int(anchor_id) == int(part_id))
            or (part_id == largest_id and motion <= LOW_MOTION_BASE_THRESHOLD_M * 2.0)
            or (part_id not in child_ids and part_id == largest_id)
        )


def _merge_clusters(payload: dict[str, Any], keep_id: int, merge_id: int) -> dict[str, Any]:
    merged = copy.deepcopy(payload)
    keep_name = f"motion_part_{keep_id}"
    counts: dict[int, int] = {}
    for track in merged.get("tracks", []) or []:
        part_id = _part_id(track)
        if part_id == merge_id:
            track["part_id"] = keep_id
            track["part_name"] = keep_name
            track["post_spectral_merged_from_part_id"] = merge_id
        new_id = _part_id(track)
        if new_id is not None:
            counts[new_id] = counts.get(new_id, 0) + 1
    merged["part_track_counts"] = {
        str(part_id): {"name": f"motion_part_{part_id}", "count": count} for part_id, count in sorted(counts.items())
    }
    merged.setdefault("post_spectral_merge", {})
    if isinstance(merged["post_spectral_merge"], dict):
        merged["post_spectral_merge"].setdefault("merges", []).append({"keep_part_id": keep_id, "merged_part_id": merge_id})
    return merged


def _mean_articulation_compatibility(a_tracks: list[dict[str, Any]], b_tracks: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for left in a_tracks[:24]:
        for right in b_tracks[:24]:
            values.append(articulation_pair_compatibility(left, right, static_mismatch_penalty=1.0))
    return sum(values) / float(len(values)) if values else None


def _joint_for_child(joint_payload: dict[str, Any], part_id: int) -> dict[str, Any] | None:
    for joint in joint_payload.get("joints", []) or []:
        if int(joint.get("child_part_id", -1)) == int(part_id):
            return joint
    return None


def _joint_replay(joint: dict[str, Any] | None) -> float | None:
    if not joint:
        return None
    comparison = (((joint.get("metrics") or {}).get("track_model_comparison")) or {})
    selected = comparison.get("selected_type") or joint.get("joint_type")
    metrics = comparison.get(str(selected)) if selected else None
    if isinstance(metrics, dict):
        return _maybe_float(metrics.get("rmse_m"))
    return None


def _axis_angle_deg(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    an = _norm(a)
    bn = _norm(b)
    if an is None or bn is None or an < 1e-9 or bn < 1e-9:
        return None
    cosine = abs(max(-1.0, min(1.0, sum(float(x) * float(y) for x, y in zip(a, b)) / (an * bn))))
    return math.degrees(math.acos(cosine))


def _point_line_distance(point: list[float], line_point: list[float], line_dir: list[float]) -> float | None:
    dn = _norm(line_dir)
    if dn is None or dn < 1e-9:
        return None
    v = [point[i] - line_point[i] for i in range(3)]
    cross = [
        v[1] * line_dir[2] - v[2] * line_dir[1],
        v[2] * line_dir[0] - v[0] * line_dir[2],
        v[0] * line_dir[1] - v[1] * line_dir[0],
    ]
    cn = _norm(cross)
    return None if cn is None else cn / dn


def _normalize(vec: list[float] | None, fallback: list[float] | None = None) -> list[float] | None:
    if vec is None:
        return fallback
    norm = _norm(vec)
    if norm is None or norm < 1e-9:
        return fallback
    return [float(value) / norm for value in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))


def _pearson_abs(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return None
    mean_a = sum(values_a) / float(len(values_a))
    mean_b = sum(values_b) / float(len(values_b))
    da = [value - mean_a for value in values_a]
    db = [value - mean_b for value in values_b]
    denom_a = math.sqrt(sum(value * value for value in da))
    denom_b = math.sqrt(sum(value * value for value in db))
    if denom_a < 1e-9 or denom_b < 1e-9:
        return None
    return abs(sum(x * y for x, y in zip(da, db)) / (denom_a * denom_b))


def _signed_angle_about_axis(a: list[float], b: list[float], axis: list[float]) -> float | None:
    unit_axis = _normalize(axis, fallback=[0.0, 0.0, 1.0])
    if unit_axis is None:
        return None
    a_perp = _sub(a, [unit_axis[i] * _dot(a, unit_axis) for i in range(3)])
    b_perp = _sub(b, [unit_axis[i] * _dot(b, unit_axis) for i in range(3)])
    a_norm = _norm(a_perp)
    b_norm = _norm(b_perp)
    if a_norm is None or b_norm is None or a_norm < 1e-8 or b_norm < 1e-8:
        return None
    a_unit = [value / a_norm for value in a_perp]
    b_unit = [value / b_norm for value in b_perp]
    return math.atan2(_dot(unit_axis, _cross(a_unit, b_unit)), max(-1.0, min(1.0, _dot(a_unit, b_unit))))


def _revolute_group_consistency(
    tracks: list[dict[str, Any]],
    axis: list[float] | None,
    pivot: list[float] | None,
) -> dict[str, Any]:
    unit_axis = _normalize(axis, fallback=None)
    if unit_axis is None or not isinstance(pivot, list) or len(pivot) != 3:
        return {
            "revolute_group_score": 0.0,
            "shared_q_consistency": 0.0,
            "radial_distance_consistency": 0.0,
            "tangential_direction_consistency": 0.0,
            "radius_displacement_consistency": 0.0,
            "revolute_q_std_rad": None,
            "revolute_radial_std_m": None,
        }
    frame_qs: dict[int, list[float]] = {}
    radial_stds: list[float] = []
    tangential_scores: list[float] = []
    radii: list[float] = []
    displacements: list[float] = []
    pivot_f = [float(value) for value in pivot]
    for track in tracks:
        samples = [
            sample
            for sample in track.get("samples", []) or []
            if isinstance(sample, dict)
            and sample.get("visible") is not False
            and sample.get("depth_valid", True) is not False
            and isinstance(sample.get("xyz_world"), list)
            and len(sample["xyz_world"]) == 3
        ]
        samples = sorted(samples, key=lambda sample: int(sample.get("frame_index", 0)))
        if len(samples) < 2:
            continue
        points = [[float(value) for value in sample["xyz_world"]] for sample in samples]
        radius0 = _sub(points[0], pivot_f)
        radius0_perp = _sub(radius0, [unit_axis[i] * _dot(radius0, unit_axis) for i in range(3)])
        radius0_norm = _norm(radius0_perp)
        if radius0_norm is None or radius0_norm < 1e-6:
            continue
        track_radii: list[float] = []
        for sample, point in zip(samples, points):
            radius = _sub(point, pivot_f)
            radius_perp = _sub(radius, [unit_axis[i] * _dot(radius, unit_axis) for i in range(3)])
            radius_norm = _norm(radius_perp)
            if radius_norm is not None:
                track_radii.append(radius_norm)
            q = _signed_angle_about_axis(radius0_perp, radius_perp, unit_axis)
            if q is not None:
                frame_qs.setdefault(int(sample.get("frame_index", 0)), []).append(q)
        if len(track_radii) >= 2:
            radial_std = _std(track_radii)
            if radial_std is not None:
                radial_stds.append(radial_std)
        displacement = _sub(points[-1], points[0])
        disp_norm = _norm(displacement)
        if disp_norm is not None and disp_norm > 1e-8:
            radial_motion = abs(_dot(radius0_perp, displacement) / (radius0_norm * disp_norm))
            tangential_scores.append(1.0 - _clip01(radial_motion))
            radii.append(radius0_norm)
            displacements.append(disp_norm)
    q_stds = [_std(qs) for qs in frame_qs.values() if len(qs) >= 2]
    q_stds = [value for value in q_stds if value is not None]
    mean_q_std = _mean(q_stds)
    mean_radial_std = _mean(radial_stds)
    shared_q_score = math.exp(-_safe_float(mean_q_std, 1.0) / REVOLUTE_Q_STD_SCALE_RAD) if q_stds else 0.0
    radial_score = math.exp(-_safe_float(mean_radial_std, 1.0) / REVOLUTE_RADIAL_STD_SCALE_M) if radial_stds else 0.0
    tangential_score = _mean(tangential_scores) or 0.0
    radius_disp_corr = _pearson_abs(radii, displacements)
    radius_disp_score = _safe_float(radius_disp_corr, 0.0)
    group_score = _clip01(
        0.35 * shared_q_score
        + 0.25 * radial_score
        + 0.20 * tangential_score
        + 0.20 * radius_disp_score
    )
    return {
        "revolute_group_score": group_score,
        "shared_q_consistency": shared_q_score,
        "radial_distance_consistency": radial_score,
        "tangential_direction_consistency": tangential_score,
        "radius_displacement_consistency": radius_disp_score,
        "revolute_q_std_rad": mean_q_std,
        "revolute_radial_std_m": mean_radial_std,
    }


def _prismatic_group_consistency(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    displacements = [disp for track in tracks if (disp := _track_displacement(track)) is not None and (_norm(disp) or 0.0) > 1e-8]
    if len(displacements) < 2:
        return {
            "prismatic_group_score": 0.0,
            "parallel_direction_score": 0.0,
            "equal_displacement_score": 0.0,
            "prismatic_direction_std": None,
            "prismatic_displacement_std_m": None,
        }
    mean_disp = [sum(disp[i] for disp in displacements) / float(len(displacements)) for i in range(3)]
    axis = _normalize(mean_disp, fallback=None)
    if axis is None:
        return {
            "prismatic_group_score": 0.0,
            "parallel_direction_score": 0.0,
            "equal_displacement_score": 0.0,
            "prismatic_direction_std": None,
            "prismatic_displacement_std_m": None,
        }
    direction_errors: list[float] = []
    magnitudes: list[float] = []
    for disp in displacements:
        norm = _norm(disp)
        if norm is None or norm < 1e-8:
            continue
        direction_errors.append(1.0 - abs(_dot(disp, axis) / norm))
        magnitudes.append(abs(_dot(disp, axis)))
    direction_std = _mean(direction_errors)
    displacement_std = _std(magnitudes)
    parallel_score = math.exp(-_safe_float(direction_std, 1.0) / PRISMATIC_DIRECTION_STD_SCALE)
    equal_disp_score = math.exp(-_safe_float(displacement_std, 1.0) / PRISMATIC_EQUAL_DISP_SCALE_M)
    group_score = _clip01(0.55 * parallel_score + 0.45 * equal_disp_score)
    return {
        "prismatic_group_score": group_score,
        "parallel_direction_score": parallel_score,
        "equal_displacement_score": equal_disp_score,
        "prismatic_direction_std": direction_std,
        "prismatic_displacement_std_m": displacement_std,
    }


def _best_revolute_seed_joint(
    part_a: int,
    part_b: int,
    before_joints: dict[str, Any],
    stats: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, int | None]:
    candidates: list[tuple[float, dict[str, Any], int]] = []
    for joint in before_joints.get("joints", []) or []:
        if joint.get("joint_type") != "revolute":
            continue
        child_id = int(joint.get("child_part_id", -1))
        if child_id not in stats:
            continue
        if child_id in {part_a, part_b}:
            priority = 1.0
        else:
            priority = max(
                0.0,
                SPATIAL_CLOSE_M - min(
                    _distance(stats[part_a]["centroid"], stats[child_id]["centroid"]),
                    _distance(stats[part_b]["centroid"], stats[child_id]["centroid"]),
                ),
            )
        if priority > 0.0:
            candidates.append((priority, joint, child_id))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _shared_joint_model_features(
    part_a: int,
    part_b: int,
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
) -> dict[str, Any]:
    tracks = list(stats[part_a]["tracks"]) + list(stats[part_b]["tracks"])
    seed_joint, seed_child = _best_revolute_seed_joint(part_a, part_b, before_joints, stats)
    revolute = _revolute_group_consistency(
        tracks,
        seed_joint.get("axis") if seed_joint else None,
        seed_joint.get("pivot") if seed_joint else None,
    )
    prismatic = _prismatic_group_consistency(tracks)
    revolute_score = _safe_float(revolute.get("revolute_group_score"), 0.0)
    prismatic_score = _safe_float(prismatic.get("prismatic_group_score"), 0.0)
    revolute_over_prismatic = _clip01(max(0.0, revolute_score - prismatic_score + 0.25))
    shared_model_score = max(revolute_score, prismatic_score)
    return {
        **revolute,
        **prismatic,
        "shared_revolute_seed_child": seed_child,
        "shared_joint_model_score": shared_model_score,
        "best_group_motion_model": "revolute" if revolute_score >= prismatic_score else "prismatic",
        "revolute_over_prismatic_score": revolute_over_prismatic,
    }


def _shared_hinge_score(joint_a: dict[str, Any] | None, joint_b: dict[str, Any] | None) -> float:
    if not joint_a or not joint_b:
        return 0.0
    if joint_a.get("joint_type") != joint_b.get("joint_type"):
        return 0.0
    if joint_a.get("joint_type") == "revolute":
        angle = _axis_angle_deg(joint_a.get("axis"), joint_b.get("axis"))
        pivot_a = joint_a.get("pivot")
        pivot_b = joint_b.get("pivot")
        if not isinstance(pivot_a, list) or not isinstance(pivot_b, list):
            distance = None
        else:
            distance = _point_line_distance([float(x) for x in pivot_b], [float(x) for x in pivot_a], joint_a.get("axis"))
        angle_score = math.exp(-(angle or 90.0) / 20.0)
        distance_score = math.exp(-(distance or 1.0) / 0.08)
        return _clip01(angle_score * distance_score)
    if joint_a.get("joint_type") == "prismatic":
        angle = _axis_angle_deg(joint_a.get("axis"), joint_b.get("axis"))
        return _clip01(math.exp(-(angle or 90.0) / 20.0))
    return 0.0


def _cluster_mean_displacement(tracks: list[dict[str, Any]]) -> list[float] | None:
    displacements = [disp for track in tracks if (disp := _track_displacement(track)) is not None]
    if not displacements:
        return None
    return [sum(disp[i] for disp in displacements) / float(len(displacements)) for i in range(3)]


def _cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    an = _norm(a)
    bn = _norm(b)
    if a is None or b is None or an is None or bn is None or an < 1e-9 or bn < 1e-9:
        return None
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (an * bn)


def _motion_compatibility(a_tracks: list[dict[str, Any]], b_tracks: list[dict[str, Any]]) -> float:
    cosine = _cosine(_cluster_mean_displacement(a_tracks), _cluster_mean_displacement(b_tracks))
    return 0.0 if cosine is None else _clip01((cosine + 1.0) / 2.0)


def _max_compatibility_with_moving_children(
    part_id: int,
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
) -> float:
    item = stats[part_id]
    values = []
    child_ids = {int(joint.get("child_part_id")) for joint in before_joints.get("joints", []) if joint.get("child_part_id") is not None}
    for child_id in child_ids:
        if child_id == part_id or child_id not in stats or stats[child_id].get("is_base_like"):
            continue
        articulation = _mean_articulation_compatibility(item["tracks"], stats[child_id]["tracks"]) or 0.0
        motion = _motion_compatibility(item["tracks"], stats[child_id]["tracks"])
        values.append(max(articulation, motion))
    return max(values, default=0.0)


def _base_absorption_guard(
    nonbase_id: int,
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
) -> dict[str, Any]:
    item = stats[nonbase_id]
    moving_evidence = _safe_float(item.get("moving_evidence_score"), 0.0)
    max_child_compatibility = _max_compatibility_with_moving_children(nonbase_id, stats, before_joints)
    extent = _safe_float(item.get("bbox_diag_m"), 0.0)
    motion = _safe_float(item.get("mean_motion_m"), 0.0)
    penalty = max(
        moving_evidence,
        max_child_compatibility if max_child_compatibility > 0.75 else 0.0,
        _clip01(extent / 0.5) if extent > 0.15 else 0.0,
    )
    reasons = []
    if motion > LOW_MOTION_BASE_THRESHOLD_M:
        reasons.append("nonbase_motion_above_threshold")
    if moving_evidence > MOVING_EVIDENCE_THRESHOLD:
        reasons.append("high_moving_evidence")
    if max_child_compatibility > 0.75:
        reasons.append("compatible_with_existing_moving_child")
    if extent > 0.15:
        reasons.append("nontrivial_spatial_extent")
    allowed = not reasons
    return {
        "base_absorption_penalty": penalty,
        "moving_evidence_score": moving_evidence,
        "base_merge_allowed": allowed,
        "base_merge_rejection_reason": ";".join(reasons) if reasons else "",
        "max_existing_child_compatibility": max_child_compatibility,
    }


def _joint_type(joint: dict[str, Any] | None) -> str | None:
    return str(joint.get("joint_type")) if isinstance(joint, dict) and joint.get("joint_type") else None


def _joint_type_mismatch_penalty(
    joint_a: dict[str, Any] | None,
    joint_b: dict[str, Any] | None,
    shared_model: dict[str, Any],
) -> float:
    type_a = _joint_type(joint_a)
    type_b = _joint_type(joint_b)
    if type_a is None or type_b is None or type_a == type_b:
        return 0.0
    # A local door patch can be classified as prismatic. Permit it only when
    # the combined tracks strongly support one shared revolute explanation.
    if _safe_float(shared_model.get("revolute_group_score"), 0.0) >= STRONG_SHARED_MODEL_OVERRIDE:
        return 0.0
    return 1.0


def _revolute_patch_consistency(patch_stats: dict[str, Any], child_joint: dict[str, Any] | None) -> dict[str, Any]:
    if not child_joint or child_joint.get("joint_type") != "revolute":
        return {
            "axis_consistency_error": None,
            "tangential_motion_consistency": 0.0,
            "radial_distance_consistency": 0.0,
        }
    axis = child_joint.get("axis")
    pivot = child_joint.get("pivot")
    if not isinstance(axis, list) or not isinstance(pivot, list):
        return {
            "axis_consistency_error": None,
            "tangential_motion_consistency": 0.0,
            "radial_distance_consistency": 0.0,
        }
    radial_scores = []
    tangential_scores = []
    for track in patch_stats["tracks"][:64]:
        samples = _visible_xyz_samples(track)
        if len(samples) < 2:
            continue
        radius0 = [samples[0][i] - float(pivot[i]) for i in range(3)]
        radius1 = [samples[-1][i] - float(pivot[i]) for i in range(3)]
        r0 = _norm(radius0)
        r1 = _norm(radius1)
        if r0 is not None and r1 is not None and r0 > 1e-6:
            radial_scores.append(math.exp(-abs(r1 - r0) / 0.04))
        displacement = [samples[-1][i] - samples[0][i] for i in range(3)]
        rn = _norm(radius0)
        dn = _norm(displacement)
        if rn is None or dn is None or rn < 1e-6 or dn < 1e-6:
            continue
        radial_motion = abs(sum(radius0[i] * displacement[i] for i in range(3)) / (rn * dn))
        tangential_scores.append(1.0 - _clip01(radial_motion))
    return {
        "axis_consistency_error": 0.0,
        "tangential_motion_consistency": _mean(tangential_scores) or 0.0,
        "radial_distance_consistency": _mean(radial_scores) or 0.0,
    }


def _patch_attach_features(
    patch_id: int,
    child_id: int,
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
) -> dict[str, Any]:
    patch = stats[patch_id]
    child = stats[child_id]
    child_joint = _joint_for_child(before_joints, child_id)
    articulation = _mean_articulation_compatibility(patch["tracks"], child["tracks"]) or 0.0
    motion = _motion_compatibility(patch["tracks"], child["tracks"])
    attach_residual = 1.0 - max(articulation, motion)
    attach_replay_score = max(articulation, motion)
    q_consistency_error = attach_residual
    joint_type = _joint_type(child_joint)
    patch_type = _joint_type(_joint_for_child(before_joints, patch_id))
    revolute = _revolute_patch_consistency(patch, child_joint)
    direction_error = None
    if joint_type == "prismatic":
        direction_error = 1.0 - motion
    shared_model = _shared_joint_model_features(patch_id, child_id, stats, before_joints)
    shared_revolute = _safe_float(shared_model.get("revolute_group_score"), 0.0)
    revolute_over_prismatic = _safe_float(shared_model.get("revolute_over_prismatic_score"), 0.0)
    attach_score = 0.30 * attach_replay_score + 0.20 * motion + 0.15 * (
        revolute["tangential_motion_consistency"] if joint_type == "revolute" else 0.5
    ) + 0.15 * (revolute["radial_distance_consistency"] if joint_type == "revolute" else 0.5)
    if joint_type == "revolute":
        attach_score += 0.20 * shared_revolute
    else:
        attach_score += 0.10 * _safe_float(shared_model.get("shared_joint_model_score"), 0.0)
    centroid_distance = _distance(patch["centroid"], child["centroid"])
    strong_shared_model = _safe_float(shared_model.get("shared_joint_model_score"), 0.0) >= STRONG_SHARED_MODEL_OVERRIDE
    joint_type_compatible = (
        patch_type is None
        or joint_type is None
        or patch_type == joint_type
        or (
            joint_type == "revolute"
            and _safe_float(shared_model.get("revolute_group_score"), 0.0) >= STRONG_SHARED_MODEL_OVERRIDE
        )
    )
    attach_allowed = joint_type_compatible and (
        centroid_distance <= PATCH_ATTACH_MAX_DISTANCE_M or strong_shared_model
    )
    if not attach_allowed:
        attach_score = 0.0
    return {
        "candidate_type": "patch_attach_to_existing_joint",
        "patch_cluster": patch_id,
        "target_child_cluster": child_id,
        "attach_residual_to_existing_joint": attach_residual,
        "attach_replay_score": attach_replay_score,
        "q_consistency_error": q_consistency_error,
        "axis_consistency_error": revolute["axis_consistency_error"],
        "direction_consistency_error": direction_error,
        "tangential_motion_consistency": revolute["tangential_motion_consistency"],
        "radial_distance_consistency": revolute["radial_distance_consistency"],
        "shared_joint_model_score": shared_model.get("shared_joint_model_score"),
        "best_group_motion_model": shared_model.get("best_group_motion_model"),
        "revolute_group_score": shared_model.get("revolute_group_score"),
        "prismatic_group_score": shared_model.get("prismatic_group_score"),
        "revolute_over_prismatic_score": revolute_over_prismatic,
        "shared_q_consistency": shared_model.get("shared_q_consistency"),
        "radius_displacement_consistency": shared_model.get("radius_displacement_consistency"),
        "parallel_direction_score": shared_model.get("parallel_direction_score"),
        "equal_displacement_score": shared_model.get("equal_displacement_score"),
        "attach_centroid_distance_m": centroid_distance,
        "attach_joint_type_compatible": joint_type_compatible,
        "attach_allowed": attach_allowed,
        "attach_score_no_gt": _clip01(attach_score),
    }


def _is_patch_like(patch: dict[str, Any], child: dict[str, Any]) -> bool:
    return (
        _safe_float(patch.get("track_count"), 0.0) <= max(8.0, PATCH_TRACK_COUNT_RATIO * _safe_float(child.get("track_count"), 0.0))
        or _safe_float(patch.get("bbox_diag_m"), 0.0) <= PATCH_BBOX_RATIO * max(1e-9, _safe_float(child.get("bbox_diag_m"), 0.0))
    )


def _local_prismatic_degeneracy(
    patch_id: int,
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
) -> dict[str, Any]:
    patch_joint = _joint_for_child(before_joints, patch_id)
    if _joint_type(patch_joint) != "prismatic":
        return {
            "local_prismatic_degeneracy_penalty": 0.0,
            "nearby_revolute_attach_score": 0.0,
            "prismatic_degeneracy_reason": "",
        }
    best_score = 0.0
    best_child = None
    for joint in before_joints.get("joints", []) or []:
        child_id = int(joint.get("child_part_id", -1))
        if child_id == patch_id or child_id not in stats or joint.get("joint_type") != "revolute":
            continue
        attach = _patch_attach_features(patch_id, child_id, stats, before_joints)
        score = attach["attach_score_no_gt"]
        distance = _distance(stats[patch_id]["centroid"], stats[child_id]["centroid"])
        if not attach["attach_allowed"] or (distance > SPATIAL_CLOSE_M and score < PATCH_ATTACH_THRESHOLD):
            continue
        if score > best_score:
            best_score = score
            best_child = child_id
    reason = "small_local_prismatic_explained_by_nearby_revolute" if best_score > PATCH_ATTACH_THRESHOLD else ""
    return {
        "local_prismatic_degeneracy_penalty": best_score if reason else 0.0,
        "nearby_revolute_attach_score": best_score,
        "nearby_revolute_child": best_child,
        "prismatic_degeneracy_reason": reason,
    }


def _cluster_debug(eval_payload: dict[str, Any], part_id: int) -> dict[str, Any]:
    debug = eval_payload.get("cluster_debug") or {}
    return debug.get(str(part_id)) or {}


def _overlap_info(eval_payload: dict[str, Any], part_id: int) -> dict[str, Any]:
    for item in ((eval_payload.get("overlap") or {}).get("per_cluster") or []):
        if str(item.get("pred_cluster_id")) == str(part_id):
            return item
    return {}


def _run_stack(cwd: Path, tracks_path: Path, output_dir: Path, dry_run: bool, generate_viewer: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = output_dir / "part_poses.json"
    joints = output_dir / "joint_inference.json"
    evaluation = output_dir / "object_mask_kinematic_evaluation.json"
    evaluation_csv = output_dir / "object_mask_kinematic_evaluation.csv"
    _run(_module_cmd("estimate-part-poses", tracks_path, "--method", "tracks", "--output-json", poses), cwd, dry_run)
    _run(_module_cmd("infer-joints", poses, "--output-json", joints, "--mujoco-prior", "off"), cwd, dry_run)
    _run(
        _module_cmd(
            "evaluate-object-mask-kinematics",
            joints,
            "--part-poses",
            poses,
            "--output-json",
            evaluation,
            "--output-csv",
            evaluation_csv,
        ),
        cwd,
        dry_run,
    )
    viewer = output_dir / "viewer.html"
    if generate_viewer:
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                tracks_path,
                "--output-html",
                viewer,
                "--joint-inference",
                joints,
                "--evaluation-json",
                evaluation,
                "--max-tracks",
                "1000",
                "--frame-stride",
                "1",
                "--trail-length",
                "10",
                "--color-by",
                "pred_cluster",
            ),
            cwd,
            dry_run,
        )
    return {"poses": poses, "joints": joints, "evaluation": evaluation, "viewer": viewer}


def _candidate_row(
    *,
    part_a: int,
    part_b: int,
    tracks_payload: dict[str, Any],
    stats: dict[int, dict[str, Any]],
    before_joints: dict[str, Any],
    before_eval: dict[str, Any],
    after_joints: dict[str, Any] | None = None,
    after_eval: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    a = stats[part_a]
    b = stats[part_b]
    a_base = bool(a.get("is_base_like"))
    b_base = bool(b.get("is_base_like"))
    candidate_type = "child_base_merge" if a_base or b_base else "child_child_merge"
    nonbase_id = part_b if a_base else part_a if b_base else None
    distance = _distance(a["centroid"], b["centroid"])
    compatibility = _mean_articulation_compatibility(a["tracks"], b["tracks"])
    joint_a = _joint_for_child(before_joints, part_a)
    joint_b = _joint_for_child(before_joints, part_b)
    shared_hinge = _shared_hinge_score(joint_a, joint_b)
    shared_model = _shared_joint_model_features(part_a, part_b, stats, before_joints)
    type_mismatch_penalty = _joint_type_mismatch_penalty(joint_a, joint_b, shared_model)
    before_replay = _mean([_joint_replay(joint_a), _joint_replay(joint_b)])
    before_rigid = _mean([
        _maybe_float(_cluster_debug(before_eval, part_a).get("rigid_rmse_m")),
        _maybe_float(_cluster_debug(before_eval, part_b).get("rigid_rmse_m")),
    ])
    points_a = [point for track in a["tracks"] for point in _visible_xyz_samples(track)]
    points_b = [point for track in b["tracks"] for point in _visible_xyz_samples(track)]
    merged_bbox = _bbox_diag(points_a + points_b)
    max_bbox = max(_safe_float(a.get("bbox_diag_m"), 0.0), _safe_float(b.get("bbox_diag_m"), 0.0))
    spatial_extent_gain = _clip01(max(0.0, _safe_float(merged_bbox, max_bbox) - max_bbox) / SPATIAL_GAIN_SCALE_M)

    row: dict[str, Any] = {
        "candidate_type": candidate_type,
        "cluster_a": part_a,
        "cluster_b": part_b,
        "cluster_a_is_base_like": a_base,
        "cluster_b_is_base_like": b_base,
        "cluster_a_mean_motion_m": a.get("mean_motion_m"),
        "cluster_b_mean_motion_m": b.get("mean_motion_m"),
        "cluster_a_median_motion_m": a.get("median_motion_m"),
        "cluster_b_median_motion_m": b.get("median_motion_m"),
        "cluster_a_rigid_rmse": a.get("cluster_rigid_rmse"),
        "cluster_b_rigid_rmse": b.get("cluster_rigid_rmse"),
        "cluster_a_moving_evidence_score": a.get("moving_evidence_score"),
        "cluster_b_moving_evidence_score": b.get("moving_evidence_score"),
        "centroid_distance_m": distance,
        "mean_articulation_compatibility": compatibility,
        "shared_hinge_score": shared_hinge,
        "shared_joint_score": max(
            shared_hinge,
            _safe_float(compatibility, 0.0) * 0.5,
            _safe_float(shared_model.get("shared_joint_model_score"), 0.0),
        ),
        "shared_joint_model_score": shared_model.get("shared_joint_model_score"),
        "joint_type_mismatch_penalty": type_mismatch_penalty,
        "best_group_motion_model": shared_model.get("best_group_motion_model"),
        "shared_revolute_seed_child": shared_model.get("shared_revolute_seed_child"),
        "revolute_group_score": shared_model.get("revolute_group_score"),
        "prismatic_group_score": shared_model.get("prismatic_group_score"),
        "revolute_over_prismatic_score": shared_model.get("revolute_over_prismatic_score"),
        "shared_q_consistency": shared_model.get("shared_q_consistency"),
        "radial_distance_consistency": shared_model.get("radial_distance_consistency"),
        "tangential_direction_consistency": shared_model.get("tangential_direction_consistency"),
        "radius_displacement_consistency": shared_model.get("radius_displacement_consistency"),
        "parallel_direction_score": shared_model.get("parallel_direction_score"),
        "equal_displacement_score": shared_model.get("equal_displacement_score"),
        "revolute_q_std_rad": shared_model.get("revolute_q_std_rad"),
        "revolute_radial_std_m": shared_model.get("revolute_radial_std_m"),
        "prismatic_direction_std": shared_model.get("prismatic_direction_std"),
        "prismatic_displacement_std_m": shared_model.get("prismatic_displacement_std_m"),
        "track_count_a": a["track_count"],
        "track_count_b": b["track_count"],
        "merged_track_count": a["track_count"] + b["track_count"],
        "bbox_diag_a_m": a.get("bbox_diag_m"),
        "bbox_diag_b_m": b.get("bbox_diag_m"),
        "merged_bbox_diag_m": merged_bbox,
        "spatial_extent_gain": spatial_extent_gain,
        "joint_replay_before": before_replay,
        "merged_rigid_rmse_before": before_rigid,
        "complexity_reduction_bonus": 1.0 / max(1.0, float(len(stats))),
        "axis_std_after": None,
        "pivot_std_after": None,
        "direction_std_after": None,
        "joint_type_consistency": None,
        "joint_stability_score": 0.5,
        "instability_penalty": 0.0,
        "parent_pollution_delta": 0.0,
        "base_absorption_penalty": 0.0,
        "moving_evidence_score": max(_safe_float(a.get("moving_evidence_score"), 0.0), _safe_float(b.get("moving_evidence_score"), 0.0)),
        "base_merge_allowed": True,
        "base_merge_rejection_reason": "",
        "attach_score_no_gt": 0.0,
        "attach_residual_to_existing_joint": None,
        "q_consistency_error": None,
        "axis_consistency_error": None,
        "direction_consistency_error": None,
        "tangential_motion_consistency": None,
        "radial_distance_consistency": None,
        "local_prismatic_degeneracy_penalty": 0.0,
        "nearby_revolute_attach_score": 0.0,
        "prismatic_degeneracy_reason": "",
        "same_dominant_gt_part": _overlap_info(before_eval, part_a).get("dominant_gt_part_id")
        == _overlap_info(before_eval, part_b).get("dominant_gt_part_id"),
        "coverage_before": max(
            _safe_float(_overlap_info(before_eval, part_a).get("coverage"), 0.0),
            _safe_float(_overlap_info(before_eval, part_b).get("coverage"), 0.0),
        ),
        "purity_before": _mean([
            _maybe_float(_overlap_info(before_eval, part_a).get("purity")),
            _maybe_float(_overlap_info(before_eval, part_b).get("purity")),
        ]),
    }
    if candidate_type == "child_base_merge" and nonbase_id is not None:
        row.update(_base_absorption_guard(nonbase_id, stats, before_joints))
    if candidate_type == "child_child_merge":
        # If one side is a local patch and the other already has a moving joint, evaluate attach as a safer alternative.
        patch_id, child_id = (part_a, part_b)
        if not _is_patch_like(stats[patch_id], stats[child_id]):
            patch_id, child_id = part_b, part_a
        if _is_patch_like(stats[patch_id], stats[child_id]) and _joint_for_child(before_joints, child_id):
            attach = _patch_attach_features(patch_id, child_id, stats, before_joints)
            if attach["attach_allowed"] and attach["attach_score_no_gt"] >= PATCH_ATTACH_THRESHOLD:
                row.update(attach)
        row.update(_local_prismatic_degeneracy(part_a, stats, before_joints))
        other_degen = _local_prismatic_degeneracy(part_b, stats, before_joints)
        if _safe_float(other_degen.get("local_prismatic_degeneracy_penalty"), 0.0) > _safe_float(
            row.get("local_prismatic_degeneracy_penalty"), 0.0
        ):
            row.update(other_degen)
    if after_joints is not None and after_eval is not None:
        after_joint = _joint_for_child(after_joints, part_a)
        row["joint_replay_after"] = _joint_replay(after_joint)
        row["joint_replay_delta"] = (
            None if before_replay is None or row["joint_replay_after"] is None else row["joint_replay_after"] - before_replay
        )
        row["merged_rigid_rmse"] = _maybe_float(_cluster_debug(after_eval, part_a).get("rigid_rmse_m"))
        row["rigid_rmse_delta"] = (
            None if before_rigid is None or row["merged_rigid_rmse"] is None else row["merged_rigid_rmse"] - before_rigid
        )
        after_overlap = _overlap_info(after_eval, part_a)
        row["coverage_after"] = after_overlap.get("coverage")
        row["coverage_delta"] = (
            None
            if row["coverage_before"] is None or row["coverage_after"] is None
            else float(row["coverage_after"]) - float(row["coverage_before"])
        )
        row["purity_after"] = after_overlap.get("purity")
        row["purity_delta"] = (
            None if row["purity_before"] is None or row["purity_after"] is None else float(row["purity_after"]) - float(row["purity_before"])
        )
        summary = after_eval.get("summary") or {}
        row["after_predicted_joint_count"] = summary.get("predicted_joint_count")
        row["after_directed_joint_coverage"] = summary.get("directed_joint_coverage")
        row["after_axis_mean_deg"] = summary.get("axis_angle_error_deg_mean")
        row["after_pivot_mean_m"] = summary.get("pivot_error_m_mean")
        row["after_mean_gt_coverage"] = summary.get("mean_gt_coverage")
        row["after_mean_cluster_purity"] = summary.get("mean_cluster_purity")
        row["refit_dir"] = str(output_dir) if output_dir else None
        row["viewer_after_merge"] = str(output_dir / "viewer.html") if output_dir else None
    else:
        row["joint_replay_after"] = None
        row["joint_replay_delta"] = None
        row["merged_rigid_rmse"] = None
        row["rigid_rmse_delta"] = None
    row.update(merge_score_no_gt(row))
    row["considered_for_refit"] = (
        distance <= SPATIAL_CLOSE_M
        or _safe_float(compatibility, 0.0) >= HIGH_ARTICULATION_COMPATIBILITY
        or _safe_float(shared_hinge, 0.0) > 0.2
        or _safe_float(row.get("shared_joint_model_score"), 0.0) > 0.55
        or _safe_float(row.get("revolute_over_prismatic_score"), 0.0) > 0.55
        or _safe_float(row.get("attach_score_no_gt"), 0.0) >= PATCH_ATTACH_THRESHOLD
    )
    return row


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / float(len(clean)) if clean else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_index(
    path: Path,
    baseline_viewer: Path | None,
    candidates: list[dict[str, Any]],
    postmerge_viewer: Path | None = None,
    segmentation_source: str = "generic",
) -> None:
    def table(title: str, rows_in: list[dict[str, Any]]) -> str:
        rows = []
        for row in rows_in:
            viewer = row.get("viewer_after_merge")
            link = ""
            if viewer:
                link = f'<a href="{os.path.relpath(str(viewer), path.parent)}">viewer</a>'
            rows.append(
                "<tr>"
                f"<td>{row.get('candidate_type')}</td><td>{row.get('cluster_a')}</td><td>{row.get('cluster_b')}</td>"
                f"<td>{row.get('patch_cluster') or ''}</td><td>{row.get('target_child_cluster') or ''}</td>"
                f"<td>{row.get('merge_score_no_gt')}</td><td>{row.get('attach_score_no_gt')}</td>"
                f"<td>{row.get('best_group_motion_model') or ''}</td><td>{row.get('revolute_group_score') or ''}</td>"
                f"<td>{row.get('prismatic_group_score') or ''}</td><td>{row.get('shared_q_consistency') or ''}</td>"
                f"<td>{row.get('radius_displacement_consistency') or ''}</td>"
                f"<td>{row.get('base_merge_allowed')}</td><td>{row.get('base_merge_rejection_reason') or ''}</td>"
                f"<td>{row.get('local_prismatic_degeneracy_penalty')}</td>"
                f"<td>{row.get('after_directed_joint_coverage')}</td><td>{row.get('after_axis_mean_deg')}</td>"
                f"<td>{link}</td>"
                "</tr>"
            )
        return (
            f"<h2>{title}</h2>"
            "<table border='1'><tr><th>type</th><th>A</th><th>B</th><th>patch</th><th>target</th>"
            "<th>score</th><th>attach</th><th>group model</th><th>rev score</th><th>pris score</th>"
            "<th>q consistency</th><th>radius-disp</th><th>base allowed</th><th>base rejection</th>"
            "<th>prismatic degeneracy</th><th>coverage</th><th>axis</th><th>viewer</th></tr>"
            + "\n".join(rows)
            + "</table>"
        )

    patch_rows = [row for row in candidates if row.get("candidate_type") == "patch_attach_to_existing_joint"]
    child_rows = [row for row in candidates if row.get("candidate_type") == "child_child_merge"]
    base_rows = [row for row in candidates if row.get("candidate_type") == "child_base_merge"]
    legacy_rows = []
    for row in candidates:
        viewer = row.get("viewer_after_merge")
        link = ""
        if viewer:
            link = f'<a href="{os.path.relpath(str(viewer), path.parent)}">viewer</a>'
        legacy_rows.append(
            "<tr>"
            f"<td>{row.get('cluster_a')}</td><td>{row.get('cluster_b')}</td>"
            f"<td>{row.get('merge_score_no_gt')}</td><td>{row.get('same_dominant_gt_part')}</td>"
            f"<td>{row.get('after_directed_joint_coverage')}</td><td>{row.get('after_axis_mean_deg')}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    baseline = (
        f'<p><a href="{os.path.relpath(str(baseline_viewer), path.parent)}">Baseline viewer</a></p>'
        if baseline_viewer
        else ""
    )
    postmerge = (
        f'<p><a href="{os.path.relpath(str(postmerge_viewer), path.parent)}">Greedy postmerge viewer</a></p>'
        if postmerge_viewer and postmerge_viewer.exists()
        else ""
    )
    path.write_text(
        f"<html><body><h1>Post-segmentation merge diagnostic ({segmentation_source})</h1>"
        + baseline
        + postmerge
        + table("Patch attach candidates", patch_rows[:20])
        + table("Child-child merge candidates", child_rows[:20])
        + table("Child-base merge candidates", base_rows[:20])
        + "<h2>All candidates</h2><table border='1'><tr><th>A</th><th>B</th><th>score</th><th>same GT</th><th>coverage</th><th>axis</th><th>viewer</th></tr>"
        + "\n".join(legacy_rows)
        + "</table></body></html>",
        encoding="utf-8",
    )


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    cwd = Path.cwd()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = args.motion_tracks.expanduser().resolve()
    base_tracks = _load_json(tracks_path)
    before_joints = _load_json(args.joint_inference.expanduser().resolve())
    before_eval = _load_json(args.evaluation_json.expanduser().resolve()) if args.evaluation_json else {}
    stats = _cluster_stats(base_tracks)
    _annotate_cluster_roles(stats, before_joints, before_eval)
    part_ids = sorted(stats)

    baseline_viewer = output_dir / "viewer_baseline.html"
    if args.generate_viewer:
        cmd = _module_cmd(
            "visualize-object-mask-flow-html",
            tracks_path,
            "--output-html",
            baseline_viewer,
            "--joint-inference",
            args.joint_inference,
            "--max-tracks",
            "1000",
            "--frame-stride",
            "1",
            "--trail-length",
            "10",
            "--color-by",
            "pred_cluster",
        )
        if args.evaluation_json:
            cmd.extend(["--evaluation-json", args.evaluation_json])
        _run(cmd, cwd, args.dry_run)

    candidates: list[dict[str, Any]] = []
    for idx, part_a in enumerate(part_ids[:-1]):
        for part_b in part_ids[idx + 1 :]:
            row = _candidate_row(
                part_a=part_a,
                part_b=part_b,
                tracks_payload=base_tracks,
                stats=stats,
                before_joints=before_joints,
                before_eval=before_eval,
            )
            if row["considered_for_refit"]:
                candidates.append(row)
    candidates.sort(key=lambda row: float(row.get("merge_score_no_gt") or -1e9), reverse=True)

    refit_rows: list[dict[str, Any]] = []
    for row in candidates[: max(0, int(args.max_candidates_to_refit))]:
        part_a = int(row["cluster_a"])
        part_b = int(row["cluster_b"])
        refit_dir = output_dir / f"merge_{part_a}_{part_b}"
        merged_tracks = refit_dir / "motion_part_tracks_merged.json"
        _save_json(merged_tracks, _merge_clusters(base_tracks, part_a, part_b))
        outputs = _run_stack(cwd, merged_tracks, refit_dir, args.dry_run, bool(args.generate_viewer))
        after_joints = _load_json(outputs["joints"]) if outputs["joints"].exists() else {}
        after_eval = _load_json(outputs["evaluation"]) if outputs["evaluation"].exists() else {}
        refit_rows.append(
            _candidate_row(
                part_a=part_a,
                part_b=part_b,
                tracks_payload=base_tracks,
                stats=stats,
                before_joints=before_joints,
                before_eval=before_eval,
                after_joints=after_joints,
                after_eval=after_eval,
                output_dir=refit_dir,
            )
        )
    refit_by_pair = {(int(row["cluster_a"]), int(row["cluster_b"])): row for row in refit_rows}
    final_rows = [refit_by_pair.get((int(row["cluster_a"]), int(row["cluster_b"])), row) for row in candidates]
    final_rows.sort(key=lambda row: float(row.get("merge_score_no_gt") or -1e9), reverse=True)

    postmerge_tracks: Path | None = None
    if args.enable_post_spectral_merge and args.post_merge_mode == "greedy":
        postmerge_tracks = _run_greedy(cwd, base_tracks, before_joints, before_eval, stats, output_dir, args)

    candidates_json = output_dir / "overseg_merge_candidates.json"
    candidates_csv = output_dir / "overseg_merge_candidates.csv"
    patch_json = output_dir / "patch_attach_candidates.json"
    patch_csv = output_dir / "patch_attach_candidates.csv"
    patch_rows = [row for row in final_rows if row.get("candidate_type") == "patch_attach_to_existing_joint"]
    _save_json(
        candidates_json,
        {
            "segmentation_source": args.segmentation_source,
            "input_motion_tracks": str(tracks_path),
            "candidates": final_rows,
            "postmerge_tracks": str(postmerge_tracks) if postmerge_tracks else None,
        },
    )
    _write_csv(candidates_csv, final_rows)
    _save_json(patch_json, {"candidates": patch_rows})
    _write_csv(patch_csv, patch_rows)
    postmerge_viewer = output_dir / "postmerge_stack" / "viewer.html"
    _write_index(
        output_dir / "index.html",
        baseline_viewer if args.generate_viewer else None,
        final_rows[:20],
        postmerge_viewer if postmerge_tracks else None,
        args.segmentation_source,
    )
    return {
        "segmentation_source": args.segmentation_source,
        "overseg_merge_candidates_json": str(candidates_json),
        "overseg_merge_candidates_csv": str(candidates_csv),
        "postmerge_tracks": str(postmerge_tracks) if postmerge_tracks else None,
        "index_html": str(output_dir / "index.html"),
        "patch_attach_candidates_json": str(patch_json),
        "patch_attach_candidates_csv": str(patch_csv),
        "candidate_count": len(final_rows),
        "patch_attach_candidate_count": len(patch_rows),
    }


def _run_greedy(
    cwd: Path,
    base_tracks: dict[str, Any],
    before_joints: dict[str, Any],
    before_eval: dict[str, Any],
    stats: dict[int, dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path | None:
    current = copy.deepcopy(base_tracks)
    accepted: list[dict[str, Any]] = []
    for iteration in range(max(1, int(args.merge_max_iterations))):
        iter_stats = _cluster_stats(current)
        _annotate_cluster_roles(iter_stats, before_joints, before_eval)
        part_ids = sorted(iter_stats)
        rows: list[dict[str, Any]] = []
        for idx, part_a in enumerate(part_ids[:-1]):
            for part_b in part_ids[idx + 1 :]:
                row = _candidate_row(
                    part_a=part_a,
                    part_b=part_b,
                    tracks_payload=current,
                    stats=iter_stats,
                    before_joints=before_joints,
                    before_eval=before_eval,
                )
                if not row["considered_for_refit"]:
                    continue
                if row.get("candidate_type") == "child_base_merge" and row.get("base_merge_allowed") is not True:
                    continue
                rows.append(row)
        rows.sort(key=_greedy_priority, reverse=True)
        best = rows[0] if rows else None
        if (
            best is None
            or _greedy_priority(best)[0] <= 0.0
            or float(best["merge_score_no_gt"]) < float(args.merge_score_threshold)
        ):
            break
        keep_id, merge_id = _merge_ids_for_row(best, iter_stats)
        current = _merge_clusters(current, keep_id, merge_id)
        accepted.append(best)
    if not accepted:
        return None
    postmerge = output_dir / "motion_part_tracks_postmerge.json"
    current["post_spectral_merge"] = {"mode": "greedy", "accepted_merges": accepted}
    _save_json(postmerge, current)
    stack_dir = output_dir / "postmerge_stack"
    _run_stack(cwd, postmerge, stack_dir, args.dry_run, bool(args.generate_viewer))
    return postmerge


def _greedy_priority(row: dict[str, Any]) -> tuple[float, float]:
    score = float(row.get("merge_score_no_gt") or -1e9)
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type == "patch_attach_to_existing_joint" and score >= GREEDY_PATCH_ATTACH_MIN_SCORE:
        return (3.0, score)
    if candidate_type == "child_child_merge" and score >= GREEDY_CHILD_CHILD_MIN_SCORE:
        return (2.0, score)
    if (
        candidate_type == "child_base_merge"
        and row.get("base_merge_allowed") is True
        and score >= GREEDY_CHILD_BASE_MIN_SCORE
    ):
        return (1.0, score)
    return (0.0, score)


def _merge_ids_for_row(row: dict[str, Any], stats: dict[int, dict[str, Any]]) -> tuple[int, int]:
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type == "patch_attach_to_existing_joint":
        return int(row["target_child_cluster"]), int(row["patch_cluster"])
    cluster_a = int(row["cluster_a"])
    cluster_b = int(row["cluster_b"])
    if candidate_type == "child_base_merge":
        if stats[cluster_a].get("is_base_like"):
            return cluster_a, cluster_b
        if stats[cluster_b].get("is_base_like"):
            return cluster_b, cluster_a
    return cluster_a, cluster_b


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic post-segmentation merge stage for spectral, RANSAC, or other motion-part proposals."
    )
    parser.add_argument("motion_tracks", type=Path)
    parser.add_argument("--part-poses", type=Path, default=None, help="Reserved for compatibility; refits are recomputed.")
    parser.add_argument("--joint-inference", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--segmentation-source",
        choices=["generic", "knn-spectral", "sequential-ransac"],
        default="generic",
        help="Metadata only. The diagnostic consumes generic motion-part artifacts.",
    )
    parser.add_argument("--enable-post-spectral-merge", action="store_true")
    parser.add_argument("--post-merge-mode", choices=["diagnostic", "greedy"], default="diagnostic")
    parser.add_argument("--merge-score-threshold", type=float, default=0.2)
    parser.add_argument("--merge-max-iterations", type=int, default=3)
    parser.add_argument("--max-candidates-to-refit", type=int, default=8)
    parser.add_argument("--generate-viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.perf_counter()
    result = run_diagnostic(args)
    result["runtime_s"] = time.perf_counter() - start
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
