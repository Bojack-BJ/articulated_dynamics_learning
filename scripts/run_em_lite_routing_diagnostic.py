#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.perception.quality_weights import cluster_quality_summary

RIGID_DELTA_SCALE_M = 0.02
JOINT_REPLAY_DELTA_SCALE_M = 0.03
PARENT_POLLUTION_SCALE_M = 0.02
MOTION_USEFULNESS_SCALE_M = 0.08
BBOX_USEFULNESS_SCALE_M = 0.25
TRACK_COUNT_USEFULNESS_SCALE = 64.0
JOINT_REPLAY_WORSENING_SCALE_M = 0.06
MIN_MOTION_VECTOR_NORM_M = 1e-4
STABILITY_AXIS_SCALE_DEG = 20.0
STABILITY_PIVOT_SCALE_M = 0.08
STABILITY_DIRECTION_SCALE_DEG = 20.0
DIAGNOSTIC_STABILITY_WEIGHT = 0.30
DIAGNOSTIC_EFFECTIVE_COVERAGE_WEIGHT = 0.20
DIAGNOSTIC_INSTABILITY_PENALTY_WEIGHT = 0.30
SEVERE_STABILITY_THRESHOLD = 0.10
MODERATE_STABILITY_THRESHOLD = 0.35
SEVERE_AXIS_STD_DEG = 15.0
SEVERE_PIVOT_STD_M = 0.15
SEVERE_INSTABILITY_PENALTY = 0.90
MODERATE_INSTABILITY_PENALTY = 0.50


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _module_cmd(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "rgbd_urdf_mvp", *[str(arg) for arg in args]]


def _ensure_quality_tracks(cwd: Path, tracks: Path, dry_run: bool) -> Path:
    try:
        payload = _load_json(tracks)
    except Exception:
        return tracks
    first_track = next((track for track in payload.get("tracks", []) if isinstance(track, dict)), None)
    if isinstance(first_track, dict) and isinstance(first_track.get("track_quality"), dict):
        return tracks
    quality_dir = tracks.parent / "track_quality_for_weighting" / tracks.stem
    enriched = quality_dir / "motion_part_tracks_with_quality.json"
    if enriched.exists():
        return enriched
    _run(_module_cmd("compute-track-quality", tracks, "--output-dir", quality_dir), cwd, dry_run)
    return enriched if enriched.exists() else tracks


def _part_id(track: dict[str, Any]) -> int | None:
    value = track.get("part_id")
    return None if value is None else int(value)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _exp_decay_score(delta: float | None, scale: float) -> float:
    if delta is None:
        return 0.5
    return math.exp(-max(0.0, float(delta)) / max(1e-9, float(scale)))


def _parent_pollution_penalty(delta_rigid_rmse: float | None) -> float:
    if delta_rigid_rmse is None:
        return 0.0
    return _clip01(max(0.0, float(delta_rigid_rmse)) / PARENT_POLLUTION_SCALE_M)


def _motion_compatibility_score(cosine: float | None) -> float:
    if cosine is None:
        return 0.0
    return _clip01((float(cosine) + 1.0) / 2.0)


def _usefulness_score(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    return _clip01(float(value) / max(1e-9, float(scale)))


def _part_names(tracks_payload: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for part_id, info in (tracks_payload.get("part_track_counts") or {}).items():
        names[int(part_id)] = str(info.get("name") or f"motion_part_{part_id}")
    for track in tracks_payload.get("tracks", []):
        part_id = _part_id(track)
        if part_id is not None and part_id not in names:
            names[part_id] = str(track.get("part_name") or f"motion_part_{part_id}")
    return names


def _refresh_counts(payload: dict[str, Any], names: dict[int, str]) -> None:
    counts: Counter[int] = Counter()
    for track in payload.get("tracks", []):
        part_id = _part_id(track)
        if part_id is not None:
            counts[part_id] += 1
    payload["part_track_counts"] = {
        str(part_id): {"name": names.get(part_id, f"motion_part_{part_id}"), "count": count}
        for part_id, count in sorted(counts.items())
    }


def _visible_xyz_samples(track: dict[str, Any]) -> list[list[float]]:
    samples = []
    for sample in track.get("samples") or []:
        xyz = sample.get("xyz_world")
        if not xyz or len(xyz) != 3:
            continue
        if sample.get("visible") is False or sample.get("depth_valid") is False:
            continue
        samples.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return samples


def _track_displacement(track: dict[str, Any]) -> list[float] | None:
    samples = _visible_xyz_samples(track)
    if len(samples) < 2:
        return None
    first = samples[0]
    last = samples[-1]
    return [last[i] - first[i] for i in range(3)]


def _norm3(vec: list[float] | tuple[float, float, float] | None) -> float | None:
    if vec is None:
        return None
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _cosine3(a: list[float] | None, b: list[float] | None) -> float | None:
    an = _norm3(a)
    bn = _norm3(b)
    if an is None or bn is None or an < MIN_MOTION_VECTOR_NORM_M or bn < MIN_MOTION_VECTOR_NORM_M:
        return None
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (an * bn)


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _axis_angle_deg(a: list[float] | None, b: list[float] | None) -> float | None:
    an = _norm3(a)
    bn = _norm3(b)
    if a is None or b is None or an is None or bn is None or an < 1e-9 or bn < 1e-9:
        return None
    cosine = sum(float(x) * float(y) for x, y in zip(a, b)) / (an * bn)
    cosine = abs(max(-1.0, min(1.0, cosine)))
    return math.degrees(math.acos(cosine))


def _mean_pairwise_axis_std_deg(axes: list[list[float]]) -> float | None:
    if len(axes) < 2:
        return None
    values = []
    for idx, axis_a in enumerate(axes[:-1]):
        for axis_b in axes[idx + 1 :]:
            value = _axis_angle_deg(axis_a, axis_b)
            if value is not None:
                values.append(value)
    return sum(values) / len(values) if values else None


def _point_rms_std(points: list[list[float]]) -> float | None:
    if len(points) < 2:
        return None
    centroid = [sum(point[i] for point in points) / float(len(points)) for i in range(3)]
    distances = [_norm3(_sub3(point, centroid)) or 0.0 for point in points]
    return math.sqrt(sum(distance * distance for distance in distances) / float(len(distances)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _tracks_by_part(track_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for track in track_payload.get("tracks", []):
        part_id = _part_id(track)
        if part_id is not None:
            grouped.setdefault(part_id, []).append(track)
    return grouped


def _cluster_track_stats(track_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for part_id, tracks in _tracks_by_part(track_payload).items():
        points = [xyz for track in tracks for xyz in _visible_xyz_samples(track)]
        displacements = [disp for track in tracks if (disp := _track_displacement(track)) is not None]
        motion_norms = [norm for disp in displacements if (norm := _norm3(disp)) is not None]
        bbox_diag = None
        if points:
            mins = [min(point[i] for point in points) for i in range(3)]
            maxs = [max(point[i] for point in points) for i in range(3)]
            bbox_diag = _norm3([maxs[i] - mins[i] for i in range(3)])
        mean_displacement = None
        if displacements:
            mean_displacement = [
                sum(disp[i] for disp in displacements) / float(len(displacements))
                for i in range(3)
            ]
        sorted_motions = sorted(motion_norms)
        median_motion = None
        if sorted_motions:
            mid = len(sorted_motions) // 2
            median_motion = (
                sorted_motions[mid]
                if len(sorted_motions) % 2
                else 0.5 * (sorted_motions[mid - 1] + sorted_motions[mid])
            )
        stats[str(part_id)] = {
            "track_count": len(tracks),
            "bbox_diag_m": bbox_diag,
            "mean_motion_m": sum(motion_norms) / len(motion_norms) if motion_norms else None,
            "median_motion_m": median_motion,
            "mean_displacement": mean_displacement,
            **cluster_quality_summary(tracks),
        }
    return stats


def _track_reference_point(track: dict[str, Any]) -> list[float] | None:
    xyz = track.get("reference_xyz_world")
    if xyz and len(xyz) == 3:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    samples = _visible_xyz_samples(track)
    return samples[0] if samples else None


def _spatial_bin_labels(tracks: list[dict[str, Any]], bin_count: int, min_tracks_per_bin: int) -> dict[int, int]:
    points: list[tuple[int, list[float]]] = []
    for idx, track in enumerate(tracks):
        point = _track_reference_point(track)
        if point is not None:
            points.append((idx, point))
    if len(points) < max(2, int(bin_count) * max(1, int(min_tracks_per_bin))):
        return {}
    spreads = []
    for axis in range(3):
        values = [point[axis] for _, point in points]
        spreads.append(max(values) - min(values))
    split_axis = max(range(3), key=lambda axis: spreads[axis])
    if spreads[split_axis] <= 1e-9:
        return {}
    ordered = sorted(points, key=lambda item: item[1][split_axis])
    labels: dict[int, int] = {}
    for rank, (track_idx, _) in enumerate(ordered):
        bin_id = min(int(bin_count) - 1, int(rank * int(bin_count) / max(1, len(ordered))))
        labels[track_idx] = bin_id
    counts = Counter(labels.values())
    if not counts or min(counts.values()) < int(min_tracks_per_bin):
        return {}
    return labels


def _filter_part_bin(
    payload: dict[str, Any],
    part_id: int,
    drop_bin: int,
    labels: dict[int, int],
    names: dict[int, str],
) -> dict[str, Any]:
    filtered = dict(payload)
    new_tracks = []
    part_track_idx = 0
    for track in payload.get("tracks", []):
        if _part_id(track) != int(part_id):
            new_tracks.append(track)
            continue
        if labels.get(part_track_idx) != int(drop_bin):
            new_tracks.append(track)
        part_track_idx += 1
    filtered["tracks"] = new_tracks
    _refresh_counts(filtered, names)
    filtered["stability_lite"] = {
        "method": "leave_one_spatial_bin_out",
        "part_id": part_id,
        "drop_bin": drop_bin,
    }
    return filtered


def _baseline_paths(object_dir: Path) -> dict[str, Path]:
    baseline_dir = object_dir / "em_lite" / "baseline"
    return {
        "tracks": baseline_dir / "motion_part_tracks_knn.json",
        "poses": baseline_dir / "part_poses_knn.json",
        "joints": baseline_dir / "joint_inference_knn.json",
        "evaluation": baseline_dir / "object_mask_kinematic_evaluation_knn.json",
        "diagnostics": baseline_dir / "motion_segmentation_diagnostics_knn.json",
    }


def _ensure_baseline(
    cwd: Path,
    object_dir: Path,
    spectral_k: int,
    edge_ablation: str,
    quality_weighted_affinity: bool,
    quality_weighted_affinity_time_only: bool,
    quality_weighted_affinity_edge_prior: bool,
    articulation_compatible_affinity: bool,
    quality_weighted_part_poses: bool,
    quality_weighted_replay: bool,
    min_quality_weight: float,
    quality_affinity_min_pair_weight: float,
    rerun_baseline: bool,
    dry_run: bool,
) -> dict[str, Path]:
    paths = _baseline_paths(object_dir)
    if not rerun_baseline and all(paths[key].exists() for key in ["tracks", "poses", "joints", "evaluation"]):
        return paths

    object_tracks = object_dir / "object_tracks.json"
    if not object_tracks.exists():
        raise FileNotFoundError(f"Missing object tracks: {object_tracks}")

    needs_affinity_quality = (
        quality_weighted_affinity
        or quality_weighted_affinity_time_only
        or quality_weighted_affinity_edge_prior
        or articulation_compatible_affinity
    )
    segment_input = _ensure_quality_tracks(cwd, object_tracks, dry_run) if needs_affinity_quality else object_tracks
    segment_cmd = _module_cmd(
        "segment-motion-parts",
        segment_input,
        "--mode",
        "knn-spectral",
        "--spectral-k",
        str(spectral_k),
        "--edge-ablation",
        edge_ablation,
        "--output-json",
        paths["tracks"],
        "--diagnostics-json",
        paths["diagnostics"],
    )
    if quality_weighted_affinity:
        segment_cmd.append("--quality-weighted-affinity")
    if quality_weighted_affinity_time_only:
        segment_cmd.append("--quality-weighted-affinity-time-only")
    if quality_weighted_affinity_edge_prior:
        segment_cmd.append("--quality-weighted-affinity-edge-prior")
    if articulation_compatible_affinity:
        segment_cmd.append("--articulation-compatible-affinity")
    if quality_affinity_min_pair_weight > 0.0:
        segment_cmd.extend(["--quality-affinity-min-pair-weight", str(quality_affinity_min_pair_weight)])
    _run(segment_cmd, cwd, dry_run)
    _run_stack(
        cwd,
        paths["tracks"],
        paths["poses"],
        paths["joints"],
        paths["evaluation"],
        dry_run,
        quality_weighted_part_poses=quality_weighted_part_poses,
        quality_weighted_replay=quality_weighted_replay,
        min_quality_weight=min_quality_weight,
    )
    return paths


def _run_stack(
    cwd: Path,
    tracks: Path,
    poses: Path,
    joints: Path,
    evaluation: Path,
    dry_run: bool,
    *,
    quality_weighted_part_poses: bool = False,
    quality_weighted_replay: bool = False,
    min_quality_weight: float = 0.2,
) -> dict[str, float]:
    total_start = time.perf_counter()
    stack_tracks = (
        _ensure_quality_tracks(cwd, tracks, dry_run)
        if quality_weighted_part_poses or quality_weighted_replay
        else tracks
    )
    estimate_start = time.perf_counter()
    estimate_cmd = _module_cmd("estimate-part-poses", stack_tracks, "--method", "tracks", "--output-json", poses)
    if quality_weighted_part_poses:
        estimate_cmd.extend(["--quality-weighted", "--min-timestep-weight", str(min_quality_weight)])
    _run(estimate_cmd, cwd, dry_run)
    estimate_runtime = time.perf_counter() - estimate_start
    infer_start = time.perf_counter()
    infer_cmd = _module_cmd("infer-joints", poses, "--output-json", joints, "--mujoco-prior", "off")
    if quality_weighted_replay:
        infer_cmd.extend(["--quality-weighted-replay", "--min-replay-weight", str(min_quality_weight)])
    _run(infer_cmd, cwd, dry_run)
    infer_runtime = time.perf_counter() - infer_start
    evaluate_start = time.perf_counter()
    _run(
        _module_cmd(
            "evaluate-object-mask-kinematics",
            joints,
            "--part-poses",
            poses,
            "--output-json",
            evaluation,
            "--output-csv",
            evaluation.with_suffix(".csv"),
        ),
        cwd,
        dry_run,
    )
    evaluate_runtime = time.perf_counter() - evaluate_start
    return {
        "route_stack_runtime_s": time.perf_counter() - total_start,
        "estimate_part_poses_runtime_s": estimate_runtime,
        "infer_joints_runtime_s": infer_runtime,
        "evaluate_runtime_s": evaluate_runtime,
    }


def _unmatched_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in evaluation.get("per_joint", []):
        if bool(row.get("matched")):
            continue
        if row.get("child_part_id") is None:
            continue
        rows.append(row)
    return rows


def _matched_child_ids(evaluation: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for row in evaluation.get("per_joint", []):
        if not bool(row.get("matched")):
            continue
        child = row.get("child_part_id")
        if child is not None:
            ids.append(int(child))
    return sorted(set(ids))


def _rewrite_route(
    input_tracks: Path,
    output_tracks: Path,
    cluster_id: int,
    route: str,
    target_part_id: int | None,
) -> dict[str, Any]:
    payload = _load_json(input_tracks)
    names = _part_names(payload)
    old_tracks = payload.get("tracks", [])
    new_tracks = []
    changed = 0
    dropped = 0
    for track in old_tracks:
        part_id = _part_id(track)
        if part_id != cluster_id:
            new_tracks.append(track)
            continue
        if route == "ignore":
            dropped += 1
            continue
        if target_part_id is None:
            raise ValueError("target_part_id is required for merge routes")
        new_track = dict(track)
        new_track["em_lite_source_part_id"] = cluster_id
        new_track["em_lite_source_part_name"] = names.get(cluster_id, f"motion_part_{cluster_id}")
        new_track["part_id"] = int(target_part_id)
        new_track["part_name"] = names.get(int(target_part_id), f"motion_part_{target_part_id}")
        new_tracks.append(new_track)
        changed += 1

    payload["tracks"] = new_tracks
    _refresh_counts(payload, names)
    route_summary = {
        "method": "em_lite_routing_diagnostic",
        "source_tracks": str(input_tracks),
        "cluster_id": cluster_id,
        "route": route,
        "target_part_id": target_part_id,
        "changed_track_count": changed,
        "dropped_track_count": dropped,
    }
    payload["em_lite"] = route_summary
    _save_json(output_tracks, payload)
    return route_summary


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_metric(evaluation: dict[str, Any], key: str) -> float | None:
    return _safe_float((evaluation.get("summary") or {}).get(key))


def _overlap_count(evaluation: dict[str, Any], pred_part_id: int | None, gt_part_id: int | None) -> int | None:
    if pred_part_id is None or gt_part_id is None:
        return None
    matrix = ((evaluation.get("overlap") or {}).get("overlap_matrix") or {})
    value = (matrix.get(str(pred_part_id)) or {}).get(str(gt_part_id))
    return None if value is None else int(value)


def _gt_total(evaluation: dict[str, Any], gt_part_id: int | None) -> int | None:
    if gt_part_id is None:
        return None
    totals = ((evaluation.get("overlap") or {}).get("gt_part_totals") or {})
    value = totals.get(str(gt_part_id))
    return None if value is None else int(value)


def _coverage(evaluation: dict[str, Any], pred_part_id: int | None, gt_part_id: int | None) -> float | None:
    count = _overlap_count(evaluation, pred_part_id, gt_part_id)
    total = _gt_total(evaluation, gt_part_id)
    if count is None or not total:
        return None
    return float(count) / float(total)


def _best_gt_coverage(evaluation: dict[str, Any], gt_part_id: int | None) -> float | None:
    if gt_part_id is None:
        return None
    for row in (evaluation.get("overlap") or {}).get("per_gt") or []:
        if int(row.get("gt_part_id", -1)) == int(gt_part_id):
            return _safe_float(row.get("coverage"))
    return None


def _joint_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("name")): row for row in evaluation.get("per_joint", []) if row.get("name")}


def _mean_matched_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    key: str,
) -> float | None:
    base_rows = _joint_metrics(baseline)
    cand_rows = _joint_metrics(candidate)
    deltas = []
    for name, base_row in base_rows.items():
        if not bool(base_row.get("matched")):
            continue
        cand_row = cand_rows.get(name)
        if not cand_row or not bool(cand_row.get("matched")):
            continue
        before = _safe_float(base_row.get(key))
        after = _safe_float(cand_row.get(key))
        if before is not None and after is not None:
            deltas.append(after - before)
    return sum(deltas) / len(deltas) if deltas else None


def _joint_replay_rmse_for_child(joints_payload: dict[str, Any], child_part_id: int | None) -> float | None:
    if child_part_id is None:
        return None
    for joint in joints_payload.get("joints", []):
        if int(joint.get("child_part_id", -1)) != int(child_part_id):
            continue
        metrics = joint.get("metrics") or {}
        comparison = metrics.get("track_model_comparison") or {}
        selected = comparison.get("selected_type") or joint.get("joint_type")
        selected_payload = comparison.get(str(selected)) or {}
        rmse = _safe_float(selected_payload.get("rmse_m"))
        if rmse is not None:
            return rmse
        candidates = [
            _safe_float((comparison.get("revolute") or {}).get("rmse_m")),
            _safe_float((comparison.get("prismatic") or {}).get("rmse_m")),
        ]
        candidates = [value for value in candidates if value is not None]
        return min(candidates) if candidates else None
    return None


def _mean_matched_joint_replay(joints_payload: dict[str, Any], matched_child_ids: list[int]) -> float | None:
    values = [
        value
        for child_id in matched_child_ids
        if (value := _joint_replay_rmse_for_child(joints_payload, child_id)) is not None
    ]
    return sum(values) / len(values) if values else None


def _joint_for_child(joints_payload: dict[str, Any], child_part_id: int | None) -> dict[str, Any] | None:
    if child_part_id is None:
        return None
    for joint in joints_payload.get("joints", []):
        if int(joint.get("child_part_id", -1)) == int(child_part_id):
            return joint
    return None


def _summarize_joint_stability(joints: list[dict[str, Any]]) -> dict[str, Any]:
    if not joints:
        return {
            "stability_sample_count": 0,
            "joint_type_consistency": None,
            "axis_std_deg": None,
            "pivot_std_m": None,
            "direction_std_deg": None,
            "joint_stability_score": None,
        }
    type_counts = Counter(str(joint.get("joint_type") or "unknown") for joint in joints)
    dominant_type, dominant_count = type_counts.most_common(1)[0]
    type_consistency = float(dominant_count) / float(len(joints))
    dominant_joints = [joint for joint in joints if str(joint.get("joint_type") or "unknown") == dominant_type]
    axes = [joint.get("axis") for joint in dominant_joints if isinstance(joint.get("axis"), list)]
    pivots = [joint.get("pivot") for joint in dominant_joints if isinstance(joint.get("pivot"), list)]
    axis_std = _mean_pairwise_axis_std_deg(axes)
    pivot_std = _point_rms_std(pivots) if dominant_type == "revolute" else None
    direction_std = axis_std if dominant_type == "prismatic" else None
    axis_scale = STABILITY_DIRECTION_SCALE_DEG if dominant_type == "prismatic" else STABILITY_AXIS_SCALE_DEG
    axis_score = _exp_decay_score(axis_std, axis_scale)
    pivot_score = 1.0 if dominant_type != "revolute" else _exp_decay_score(pivot_std, STABILITY_PIVOT_SCALE_M)
    stability_score = type_consistency * axis_score * pivot_score
    return {
        "stability_sample_count": len(joints),
        "joint_type_consistency": type_consistency,
        "dominant_joint_type": dominant_type,
        "axis_std_deg": axis_std,
        "pivot_std_m": pivot_std,
        "direction_std_deg": direction_std,
        "joint_stability_score": stability_score,
    }


def _run_stability_lite(
    cwd: Path,
    source_tracks: Path,
    part_id: int | None,
    output_dir: Path,
    bin_count: int,
    min_tracks_per_bin: int,
    quality_weighted_part_poses: bool,
    quality_weighted_replay: bool,
    min_quality_weight: float,
    dry_run: bool,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    if part_id is None:
        return {"available": False, "reason": "no target part", "runtime_s": time.perf_counter() - start_time}
    payload = _load_json(source_tracks)
    names = _part_names(payload)
    part_tracks = [track for track in payload.get("tracks", []) if _part_id(track) == int(part_id)]
    labels = _spatial_bin_labels(part_tracks, bin_count, min_tracks_per_bin)
    if not labels:
        return {
            "available": False,
            "reason": "not enough spatially distributed target tracks",
            "target_part_id": int(part_id),
            "target_track_count": len(part_tracks),
            "runtime_s": time.perf_counter() - start_time,
        }
    stability_dir = output_dir / "stability_lite"
    stability_dir.mkdir(parents=True, exist_ok=True)
    joints_for_target: list[dict[str, Any]] = []
    sample_paths: list[dict[str, str]] = []
    for bin_id in sorted(set(labels.values())):
        sample_dir = stability_dir / f"drop_bin_{bin_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        filtered = _filter_part_bin(payload, int(part_id), int(bin_id), labels, names)
        sample_tracks = sample_dir / "motion_part_tracks_stability.json"
        sample_poses = sample_dir / "part_poses_stability.json"
        sample_joints = sample_dir / "joint_inference_stability.json"
        sample_eval = sample_dir / "object_mask_kinematic_evaluation_stability.json"
        _save_json(sample_tracks, filtered)
        _run_stack(
            cwd,
            sample_tracks,
            sample_poses,
            sample_joints,
            sample_eval,
            dry_run,
            quality_weighted_part_poses=quality_weighted_part_poses,
            quality_weighted_replay=quality_weighted_replay,
            min_quality_weight=min_quality_weight,
        )
        sample_joints_payload = _load_json(sample_joints)
        joint = _joint_for_child(sample_joints_payload, int(part_id))
        if joint is not None:
            joints_for_target.append(joint)
        sample_paths.append(
            {
                "drop_bin": str(bin_id),
                "tracks": str(sample_tracks),
                "poses": str(sample_poses),
                "joints": str(sample_joints),
                "evaluation": str(sample_eval),
            }
        )
    summary = _summarize_joint_stability(joints_for_target)
    summary.update(
        {
            "available": bool(joints_for_target),
            "method": "leave_one_spatial_bin_out",
            "target_part_id": int(part_id),
            "target_track_count": len(part_tracks),
            "bin_count": len(set(labels.values())),
            "min_tracks_per_bin": int(min_tracks_per_bin),
            "samples": sample_paths,
            "runtime_s": time.perf_counter() - start_time,
        }
    )
    _save_json(stability_dir / "stability_lite_summary.json", summary)
    return summary


def _score_components_v2(
    route: str,
    u_stats: dict[str, Any],
    target_before_stats: dict[str, Any] | None,
    target_after_stats: dict[str, Any] | None,
    features: dict[str, Any],
) -> dict[str, Any]:
    motion_compatibility_score = _motion_compatibility_score(features.get("motion_compatibility_raw"))
    rigid_delta_score = _exp_decay_score(features.get("rigid_rmse_delta"), RIGID_DELTA_SCALE_M)
    joint_replay_delta_score = _exp_decay_score(features.get("joint_replay_delta"), JOINT_REPLAY_DELTA_SCALE_M)
    parent_pollution_penalty = _parent_pollution_penalty(features.get("parent_rigid_rmse_delta"))
    joint_replay_worsening_penalty = _clip01(
        max(0.0, float(features.get("joint_replay_delta") or 0.0)) / JOINT_REPLAY_WORSENING_SCALE_M
    )

    track_score = _usefulness_score(u_stats.get("track_count"), TRACK_COUNT_USEFULNESS_SCALE)
    bbox_score = _usefulness_score(u_stats.get("bbox_diag_m"), BBOX_USEFULNESS_SCALE_M)
    motion_score = _usefulness_score(u_stats.get("mean_motion_m"), MOTION_USEFULNESS_SCALE_M)
    anti_degeneracy_score = (track_score + bbox_score + motion_score) / 3.0
    low_motion_base_like_score = _clip01((1.0 - motion_score) * (1.0 - 0.5 * bbox_score))
    moving_cluster_to_base_penalty = _clip01(0.6 * motion_score + 0.4 * bbox_score)

    target_count_before = (target_before_stats or {}).get("track_count")
    target_count_after = (target_after_stats or {}).get("track_count")
    target_bbox_before = (target_before_stats or {}).get("bbox_diag_m")
    target_bbox_after = (target_after_stats or {}).get("bbox_diag_m")
    track_gain = 0.0
    if target_count_before is not None and target_count_after is not None:
        track_gain = max(0.0, float(target_count_after) - float(target_count_before))
    bbox_gain = 0.0
    if target_bbox_before is not None and target_bbox_after is not None:
        bbox_gain = max(0.0, float(target_bbox_after) - float(target_bbox_before))
    coverage_gain_score = _clip01(
        0.6 * _usefulness_score(track_gain, TRACK_COUNT_USEFULNESS_SCALE)
        + 0.4 * _usefulness_score(bbox_gain, BBOX_USEFULNESS_SCALE_M)
    )

    ignore_safety_score = _clip01(
        0.45 * rigid_delta_score
        + 0.35 * joint_replay_delta_score
        + 0.20 * (1.0 - min(1.0, features.get("largest_cluster_ratio_after") or 0.0))
    )
    lost_motion_usefulness_penalty = _clip01(
        anti_degeneracy_score
        * max(motion_compatibility_score, 0.25 if motion_score > 0.5 else 0.0)
    )

    if route == "merge_sibling":
        score = (
            0.30 * motion_compatibility_score
            + 0.25 * rigid_delta_score
            + 0.25 * joint_replay_delta_score
            + 0.10 * coverage_gain_score
            + 0.10 * anti_degeneracy_score
            - 0.35 * joint_replay_worsening_penalty
        )
    elif route == "merge_parent":
        score = (
            0.20 * low_motion_base_like_score
            + 0.25 * rigid_delta_score
            + 0.20 * joint_replay_delta_score
            - 0.35 * parent_pollution_penalty
            - 0.10 * moving_cluster_to_base_penalty
        )
    elif route == "ignore":
        score = 0.50 + 0.20 * ignore_safety_score - 0.30 * lost_motion_usefulness_penalty
    else:
        score = 0.0

    return {
        "route_score_no_gt_v2": score,
        "motion_compatibility_score": motion_compatibility_score,
        "rigid_delta_score": rigid_delta_score,
        "joint_replay_delta_score": joint_replay_delta_score,
        "parent_pollution_penalty": parent_pollution_penalty,
        "joint_replay_worsening_penalty": joint_replay_worsening_penalty,
        "coverage_gain_score": coverage_gain_score,
        "anti_degeneracy_score": anti_degeneracy_score,
        "low_motion_base_like_score": low_motion_base_like_score,
        "moving_cluster_to_base_penalty": moving_cluster_to_base_penalty,
        "ignore_safety_score": ignore_safety_score,
        "lost_motion_usefulness_penalty": lost_motion_usefulness_penalty,
    }


def _score_components_v3(route: str, v2_components: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    joint_stability_score = _safe_float(stability.get("joint_stability_score_after"))
    if joint_stability_score is None:
        joint_stability_score = 0.5
    axis_std = _safe_float(stability.get("axis_std_deg_after"))
    pivot_std = _safe_float(stability.get("pivot_std_m_after"))
    direction_std = _safe_float(stability.get("direction_std_deg_after"))
    axis_instability_penalty = 0.0 if axis_std is None else 1.0 - _exp_decay_score(axis_std, STABILITY_AXIS_SCALE_DEG)
    pivot_instability_penalty = 0.0 if pivot_std is None else 1.0 - _exp_decay_score(pivot_std, STABILITY_PIVOT_SCALE_M)
    direction_instability_penalty = (
        0.0 if direction_std is None else 1.0 - _exp_decay_score(direction_std, STABILITY_DIRECTION_SCALE_DEG)
    )
    instability_penalty = max(axis_instability_penalty, pivot_instability_penalty, direction_instability_penalty)
    effective_coverage_gain_score = (
        float(v2_components["coverage_gain_score"])
        * float(v2_components["rigid_delta_score"])
        * float(joint_stability_score)
    )
    if route == "merge_sibling":
        score = (
            0.20 * float(v2_components["motion_compatibility_score"])
            + 0.15 * float(v2_components["rigid_delta_score"])
            + 0.15 * float(v2_components["joint_replay_delta_score"])
            + 0.20 * effective_coverage_gain_score
            + 0.20 * float(joint_stability_score)
            + 0.10 * float(v2_components["anti_degeneracy_score"])
            - 0.25 * float(v2_components["joint_replay_worsening_penalty"])
            - 0.25 * instability_penalty
        )
    elif route == "ignore":
        lost_stable_coverage_penalty = (
            float(v2_components["lost_motion_usefulness_penalty"]) * max(0.0, float(joint_stability_score))
        )
        score = 0.50 + 0.20 * float(v2_components["ignore_safety_score"]) - 0.30 * lost_stable_coverage_penalty
    elif route == "merge_parent":
        score = (
            0.20 * float(v2_components["low_motion_base_like_score"])
            + 0.20 * float(v2_components["rigid_delta_score"])
            + 0.15 * float(v2_components["joint_replay_delta_score"])
            - 0.35 * float(v2_components["parent_pollution_penalty"])
            - 0.10 * float(v2_components["moving_cluster_to_base_penalty"])
        )
    else:
        score = float(v2_components.get("route_score_no_gt_v2") or 0.0)
    return {
        "route_score_no_gt_v3": score,
        "joint_stability_score": joint_stability_score,
        "effective_coverage_gain_score": effective_coverage_gain_score,
        "axis_instability_penalty": axis_instability_penalty,
        "pivot_instability_penalty": pivot_instability_penalty,
        "direction_instability_penalty": direction_instability_penalty,
        "instability_penalty": instability_penalty,
    }


def _route_score_diagnostic(baseline_eval: dict[str, Any], candidate_eval: dict[str, Any]) -> float:
    """Simulation diagnostic score. It uses GT-aware evaluator fields and must not drive real inference."""
    base_summary = baseline_eval.get("summary") or {}
    cand_summary = candidate_eval.get("summary") or {}
    base_unmatched = len(_unmatched_rows(baseline_eval))
    cand_unmatched = len(_unmatched_rows(candidate_eval))
    unmatched_gain = float(base_unmatched - cand_unmatched)

    largest_delta = (_safe_float(cand_summary.get("largest_cluster_ratio")) or 0.0) - (
        _safe_float(base_summary.get("largest_cluster_ratio")) or 0.0
    )
    purity_delta = (_safe_float(cand_summary.get("mean_cluster_purity")) or 0.0) - (
        _safe_float(base_summary.get("mean_cluster_purity")) or 0.0
    )
    predicted_joint_delta = (_safe_float(cand_summary.get("predicted_joint_count")) or 0.0) - (
        _safe_float(base_summary.get("predicted_joint_count")) or 0.0
    )
    axis_delta = _mean_matched_delta(baseline_eval, candidate_eval, "axis_angle_error_deg")
    pivot_delta = _mean_matched_delta(baseline_eval, candidate_eval, "pivot_error_m")

    score = 0.75 * unmatched_gain
    score += 1.0 * purity_delta
    score -= 1.5 * max(0.0, largest_delta)
    score -= 0.25 * abs(predicted_joint_delta)
    if axis_delta is not None:
        score -= max(0.0, axis_delta) / 30.0
    if pivot_delta is not None:
        score -= max(0.0, pivot_delta) / 0.10
    return score


def _diagnostic_score_stability_aware(row: dict[str, Any]) -> float:
    raw_score = _safe_float(row.get("route_score_diagnostic")) or 0.0
    joint_stability_score = _safe_float(row.get("joint_stability_score"))
    if joint_stability_score is None:
        joint_stability_score = 0.5
    effective_coverage_gain = _safe_float(row.get("effective_coverage_gain_score")) or 0.0
    instability_penalty = _safe_float(row.get("instability_penalty")) or 0.0
    return (
        raw_score
        + DIAGNOSTIC_STABILITY_WEIGHT * joint_stability_score
        + DIAGNOSTIC_EFFECTIVE_COVERAGE_WEIGHT * effective_coverage_gain
        - DIAGNOSTIC_INSTABILITY_PENALTY_WEIGHT * instability_penalty
    )


def _is_severely_unstable(row: dict[str, Any]) -> bool:
    stability = _safe_float(row.get("joint_stability_score"))
    axis_std = _safe_float(row.get("axis_std_deg_after"))
    pivot_std = _safe_float(row.get("pivot_std_m_after"))
    instability = _safe_float(row.get("instability_penalty"))
    return (
        (stability is not None and stability < SEVERE_STABILITY_THRESHOLD)
        or (axis_std is not None and axis_std > SEVERE_AXIS_STD_DEG)
        or (pivot_std is not None and pivot_std > SEVERE_PIVOT_STD_M)
        or (instability is not None and instability > SEVERE_INSTABILITY_PENALTY)
    )


def _is_moderately_unstable(row: dict[str, Any]) -> bool:
    stability = _safe_float(row.get("joint_stability_score"))
    instability = _safe_float(row.get("instability_penalty"))
    return (
        (stability is not None and SEVERE_STABILITY_THRESHOLD <= stability < MODERATE_STABILITY_THRESHOLD)
        or (
            instability is not None
            and MODERATE_INSTABILITY_PENALTY <= instability <= SEVERE_INSTABILITY_PENALTY
        )
    )


def _classify_v3_failure(chosen_v3: dict[str, Any], raw_best: dict[str, Any], stable_best: dict[str, Any]) -> tuple[str, str]:
    if chosen_v3.get("route_option") == raw_best.get("route_option"):
        return "", ""
    if _is_severely_unstable(raw_best):
        return (
            "raw_diagnostic_route_severely_unstable",
            "Raw diagnostic best route has severe leave-one-spatial-bin-out instability.",
        )
    if _is_moderately_unstable(raw_best):
        return (
            "raw_diagnostic_route_moderately_unstable",
            "Raw diagnostic best route has moderate leave-one-spatial-bin-out instability.",
        )
    raw_is_merge_sibling = str(raw_best.get("route")) == "merge_sibling"
    stable_is_merge_sibling = str(stable_best.get("route")) == "merge_sibling"
    v3_is_ignore = str(chosen_v3.get("route")) == "ignore"
    if raw_is_merge_sibling and stable_is_merge_sibling and v3_is_ignore:
        return (
            "true_v3_miss",
            "Raw and stability-aware diagnostics prefer merge_sibling, but v3 selects ignore.",
        )
    if raw_is_merge_sibling and v3_is_ignore:
        return (
            "possible_v3_over_conservative",
            "Raw diagnostic merge_sibling appears stable enough, but v3 selects ignore.",
        )
    return "v3_differs_from_raw_diagnostic", "v3 selected a different route from raw diagnostic best."


def _route_score_no_gt_proxy(
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    route_summary: dict[str, Any],
) -> float:
    """Very conservative no-GT proxy. It only flags obvious base pollution or excessive deletion."""
    base_summary = baseline_eval.get("summary") or {}
    cand_summary = candidate_eval.get("summary") or {}
    largest_delta = (_safe_float(cand_summary.get("largest_cluster_ratio")) or 0.0) - (
        _safe_float(base_summary.get("largest_cluster_ratio")) or 0.0
    )
    predicted_joint_delta = (_safe_float(cand_summary.get("predicted_joint_count")) or 0.0) - (
        _safe_float(base_summary.get("predicted_joint_count")) or 0.0
    )
    changed = float(route_summary.get("changed_track_count") or 0)
    dropped = float(route_summary.get("dropped_track_count") or 0)
    affected = changed + dropped
    drop_penalty = 0.0 if affected <= 0.0 else dropped / affected
    score = 0.0
    score -= 2.0 * max(0.0, largest_delta)
    score -= 0.15 * abs(predicted_joint_delta)
    score -= 0.20 * drop_penalty
    return score


def _diagnostic_row(
    object_id: str,
    cluster_row: dict[str, Any],
    route: str,
    target_part_id: int | None,
    target_name: str | None,
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    baseline_joints: dict[str, Any],
    candidate_joints: dict[str, Any],
    baseline_track_stats: dict[str, dict[str, Any]],
    candidate_track_stats: dict[str, dict[str, Any]],
    route_summary: dict[str, Any],
    stability_before: dict[str, Any] | None,
    stability_after: dict[str, Any] | None,
    timing: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    base_summary = baseline_eval.get("summary") or {}
    cand_summary = candidate_eval.get("summary") or {}
    child_debug = cluster_row.get("child_cluster_debug") or {}
    matching = baseline_eval.get("matching") or {}
    pred_to_gt = {str(key): int(value) for key, value in (matching.get("pred_to_gt") or {}).items()}
    dominant_gt = child_debug.get("dominant_gt_part_id")
    dominant_gt = None if dominant_gt is None else int(dominant_gt)
    target_gt = pred_to_gt.get(str(target_part_id)) if target_part_id is not None else dominant_gt

    target_cluster_coverage_before = _coverage(baseline_eval, target_part_id, target_gt)
    target_cluster_coverage_after = _coverage(candidate_eval, target_part_id, target_gt)
    dominant_gt_best_coverage_before = _best_gt_coverage(baseline_eval, dominant_gt)
    dominant_gt_best_coverage_after = _best_gt_coverage(candidate_eval, dominant_gt)
    unmatched_cluster_dominant_gt_coverage = _coverage(baseline_eval, int(cluster_row.get("child_part_id")), dominant_gt)
    unmatched_cluster_id = int(cluster_row.get("child_part_id"))
    u_stats = baseline_track_stats.get(str(unmatched_cluster_id), {})
    target_before_stats = baseline_track_stats.get(str(target_part_id)) if target_part_id is not None else None
    target_after_stats = candidate_track_stats.get(str(target_part_id)) if target_part_id is not None else None
    motion_compatibility_raw = None
    if target_before_stats is not None:
        motion_compatibility_raw = _cosine3(
            u_stats.get("mean_displacement"),
            target_before_stats.get("mean_displacement"),
        )

    cluster_debug_before = (baseline_eval.get("cluster_debug") or {})
    cluster_debug_after = (candidate_eval.get("cluster_debug") or {})
    target_debug_before = cluster_debug_before.get(str(target_part_id)) if target_part_id is not None else None
    target_debug_after = cluster_debug_after.get(str(target_part_id)) if target_part_id is not None else None
    rigid_rmse_before = _safe_float((target_debug_before or {}).get("rigid_rmse_m"))
    rigid_rmse_after = _safe_float((target_debug_after or {}).get("rigid_rmse_m"))
    rigid_rmse_delta = None if rigid_rmse_before is None or rigid_rmse_after is None else rigid_rmse_after - rigid_rmse_before

    parent_part_id = cluster_row.get("parent_part_id")
    parent_debug_before = cluster_debug_before.get(str(parent_part_id)) if parent_part_id is not None else None
    parent_debug_after = cluster_debug_after.get(str(parent_part_id)) if parent_part_id is not None else None
    parent_rigid_rmse_before = _safe_float((parent_debug_before or {}).get("rigid_rmse_m"))
    parent_rigid_rmse_after = _safe_float((parent_debug_after or {}).get("rigid_rmse_m"))
    parent_rigid_rmse_delta = (
        None
        if parent_rigid_rmse_before is None or parent_rigid_rmse_after is None
        else parent_rigid_rmse_after - parent_rigid_rmse_before
    )

    matched_child_ids = _matched_child_ids(baseline_eval)
    matched_joint_replay_before = _mean_matched_joint_replay(baseline_joints, matched_child_ids)
    matched_joint_replay_after = _mean_matched_joint_replay(candidate_joints, matched_child_ids)
    matched_joint_replay_delta = (
        None
        if matched_joint_replay_before is None or matched_joint_replay_after is None
        else matched_joint_replay_after - matched_joint_replay_before
    )
    joint_replay_before = _joint_replay_rmse_for_child(baseline_joints, target_part_id)
    joint_replay_after = _joint_replay_rmse_for_child(candidate_joints, target_part_id)
    joint_replay_delta = None if joint_replay_before is None or joint_replay_after is None else joint_replay_after - joint_replay_before
    if route == "merge_parent":
        route_joint_delta = matched_joint_replay_delta
    else:
        route_joint_delta = joint_replay_delta
    target_effective_observation_before = (
        _safe_float(target_before_stats.get("cluster_effective_observation_count"))
        if target_before_stats is not None
        else None
    )
    target_effective_observation_after = (
        _safe_float(target_after_stats.get("cluster_effective_observation_count"))
        if target_after_stats is not None
        else None
    )
    route_effective_coverage_gain = (
        None
        if target_effective_observation_before is None or target_effective_observation_after is None
        else target_effective_observation_after - target_effective_observation_before
    )
    route_quality_delta = (
        None
        if target_before_stats is None
        or target_after_stats is None
        or target_before_stats.get("cluster_track_quality_mean") is None
        or target_after_stats.get("cluster_track_quality_mean") is None
        else float(target_after_stats["cluster_track_quality_mean"]) - float(target_before_stats["cluster_track_quality_mean"])
    )

    features = {
        "motion_compatibility_raw": motion_compatibility_raw,
        "rigid_rmse_delta": rigid_rmse_delta,
        "joint_replay_delta": route_joint_delta,
        "parent_rigid_rmse_delta": parent_rigid_rmse_delta,
        "largest_cluster_ratio_after": _summary_metric(candidate_eval, "largest_cluster_ratio"),
    }
    v2_components = _score_components_v2(route, u_stats, target_before_stats, target_after_stats, features)
    stability_fields: dict[str, Any] = {}
    if stability_before or stability_after:
        before = stability_before or {}
        after = stability_after or {}
        stability_fields = {
            "stability_available": bool(after.get("available")),
            "stability_skipped_reason": after.get("skipped_reason"),
            "stability_method": after.get("method"),
            "stability_sample_count_before": before.get("stability_sample_count"),
            "stability_sample_count_after": after.get("stability_sample_count"),
            "joint_type_consistency_before": before.get("joint_type_consistency"),
            "joint_type_consistency_after": after.get("joint_type_consistency"),
            "axis_std_deg_before": before.get("axis_std_deg"),
            "axis_std_deg_after": after.get("axis_std_deg"),
            "axis_std_deg_delta": (
                None
                if before.get("axis_std_deg") is None or after.get("axis_std_deg") is None
                else float(after["axis_std_deg"]) - float(before["axis_std_deg"])
            ),
            "pivot_std_m_before": before.get("pivot_std_m"),
            "pivot_std_m_after": after.get("pivot_std_m"),
            "pivot_std_m_delta": (
                None
                if before.get("pivot_std_m") is None or after.get("pivot_std_m") is None
                else float(after["pivot_std_m"]) - float(before["pivot_std_m"])
            ),
            "direction_std_deg_before": before.get("direction_std_deg"),
            "direction_std_deg_after": after.get("direction_std_deg"),
            "direction_std_deg_delta": (
                None
                if before.get("direction_std_deg") is None or after.get("direction_std_deg") is None
                else float(after["direction_std_deg"]) - float(before["direction_std_deg"])
            ),
            "joint_stability_score_before": before.get("joint_stability_score"),
            "joint_stability_score_after": after.get("joint_stability_score"),
            "joint_stability_score_delta": (
                None
                if before.get("joint_stability_score") is None or after.get("joint_stability_score") is None
                else float(after["joint_stability_score"]) - float(before["joint_stability_score"])
            ),
        }
    v3_components = _score_components_v3(route, v2_components, stability_fields)

    row = {
        "object_id": object_id,
        "unmatched_cluster_id": unmatched_cluster_id,
        "unmatched_joint_name": cluster_row.get("name"),
        "route": route,
        "route_option": route if target_part_id is None else f"{route}_{target_part_id}",
        "target_part_id": target_part_id,
        "target_cluster_id": target_part_id,
        "target_name": target_name,
        "target_gt_part_id_diagnostic": target_gt,
        "dominant_gt_part_id_diagnostic": dominant_gt,
        "changed_track_count": route_summary.get("changed_track_count"),
        "dropped_track_count": route_summary.get("dropped_track_count"),
        "child_track_count": child_debug.get("track_count"),
        "track_count": u_stats.get("track_count"),
        "bbox_diag_m": u_stats.get("bbox_diag_m"),
        "mean_motion_m": u_stats.get("mean_motion_m"),
        "median_motion_m": u_stats.get("median_motion_m"),
        "route_added_track_quality_mean": u_stats.get("cluster_track_quality_mean"),
        "route_added_track_quality_median": u_stats.get("cluster_track_quality_median"),
        "route_added_low_quality_track_ratio": u_stats.get("cluster_low_quality_track_ratio"),
        "route_added_low_quality_timestep_ratio": u_stats.get("cluster_low_quality_timestep_ratio"),
        "route_added_effective_observation_count": u_stats.get("cluster_effective_observation_count"),
        "route_added_effective_observation_ratio": u_stats.get("cluster_effective_observation_ratio"),
        "target_track_quality_mean_before": (target_before_stats or {}).get("cluster_track_quality_mean"),
        "target_track_quality_mean_after": (target_after_stats or {}).get("cluster_track_quality_mean"),
        "target_low_quality_timestep_ratio_before": (target_before_stats or {}).get("cluster_low_quality_timestep_ratio"),
        "target_low_quality_timestep_ratio_after": (target_after_stats or {}).get("cluster_low_quality_timestep_ratio"),
        "target_effective_observation_count_before": target_effective_observation_before,
        "target_effective_observation_count_after": target_effective_observation_after,
        "route_effective_coverage_gain": route_effective_coverage_gain,
        "route_quality_delta": route_quality_delta,
        "current_child_rigid_rmse_m": child_debug.get("rigid_rmse_m"),
        "current_child_inlier_ratio": child_debug.get("inlier_ratio"),
        "child_gt_purity_diagnostic": child_debug.get("gt_purity"),
        "child_gt_composition_diagnostic": json.dumps(child_debug.get("gt_composition") or {}, sort_keys=True),
        "unmatched_cluster_dominant_gt_coverage_diagnostic": unmatched_cluster_dominant_gt_coverage,
        "target_cluster_gt_coverage_before_diagnostic": target_cluster_coverage_before,
        "target_cluster_gt_coverage_after_diagnostic": target_cluster_coverage_after,
        "target_cluster_gt_coverage_delta_diagnostic": (
            None
            if target_cluster_coverage_before is None or target_cluster_coverage_after is None
            else target_cluster_coverage_after - target_cluster_coverage_before
        ),
        "dominant_gt_best_coverage_before_diagnostic": dominant_gt_best_coverage_before,
        "dominant_gt_best_coverage_after_diagnostic": dominant_gt_best_coverage_after,
        "dominant_gt_best_coverage_delta_diagnostic": (
            None
            if dominant_gt_best_coverage_before is None or dominant_gt_best_coverage_after is None
            else dominant_gt_best_coverage_after - dominant_gt_best_coverage_before
        ),
        "baseline_predicted_joint_count": base_summary.get("predicted_joint_count"),
        "candidate_predicted_joint_count": cand_summary.get("predicted_joint_count"),
        "predicted_joint_count_after": cand_summary.get("predicted_joint_count"),
        "largest_cluster_ratio_after": cand_summary.get("largest_cluster_ratio"),
        "baseline_unmatched_count": len(_unmatched_rows(baseline_eval)),
        "candidate_unmatched_count": len(_unmatched_rows(candidate_eval)),
        "baseline_directed_joint_coverage_diagnostic": base_summary.get("directed_joint_coverage"),
        "candidate_directed_joint_coverage_diagnostic": cand_summary.get("directed_joint_coverage"),
        "baseline_axis_mean_diagnostic": base_summary.get("axis_angle_error_deg_mean"),
        "candidate_axis_mean_diagnostic": cand_summary.get("axis_angle_error_deg_mean"),
        "baseline_pivot_mean_diagnostic": base_summary.get("pivot_error_m_mean"),
        "candidate_pivot_mean_diagnostic": cand_summary.get("pivot_error_m_mean"),
        "matched_axis_delta_diagnostic": _mean_matched_delta(baseline_eval, candidate_eval, "axis_angle_error_deg"),
        "matched_pivot_delta_diagnostic": _mean_matched_delta(baseline_eval, candidate_eval, "pivot_error_m"),
        "diagnostic_axis_mean_after": cand_summary.get("axis_angle_error_deg_mean"),
        "diagnostic_pivot_mean_after": cand_summary.get("pivot_error_m_mean"),
        "diagnostic_directed_coverage_after": cand_summary.get("directed_joint_coverage"),
        "motion_compatibility_raw": motion_compatibility_raw,
        "motion_compatibility_score": v2_components["motion_compatibility_score"],
        "rigid_rmse_before": rigid_rmse_before,
        "rigid_rmse_after": rigid_rmse_after,
        "rigid_rmse_delta": rigid_rmse_delta,
        "joint_replay_before": joint_replay_before,
        "joint_replay_after": joint_replay_after,
        "joint_replay_delta": joint_replay_delta,
        "matched_joint_replay_mean_before": matched_joint_replay_before,
        "matched_joint_replay_mean_after": matched_joint_replay_after,
        "matched_joint_replay_delta": matched_joint_replay_delta,
        "parent_rigid_rmse_before": parent_rigid_rmse_before,
        "parent_rigid_rmse_after": parent_rigid_rmse_after,
        "parent_rigid_rmse_delta": parent_rigid_rmse_delta,
        "sibling_rigid_rmse_before": rigid_rmse_before if route == "merge_sibling" else None,
        "sibling_rigid_rmse_after": rigid_rmse_after if route == "merge_sibling" else None,
        "sibling_rigid_rmse_delta": rigid_rmse_delta if route == "merge_sibling" else None,
        "sibling_track_count_gain": (
            None
            if target_before_stats is None or target_after_stats is None
            else float(target_after_stats.get("track_count") or 0.0) - float(target_before_stats.get("track_count") or 0.0)
        ),
        "sibling_bbox_gain_m": (
            None
            if target_before_stats is None
            or target_after_stats is None
            or target_before_stats.get("bbox_diag_m") is None
            or target_after_stats.get("bbox_diag_m") is None
            else float(target_after_stats["bbox_diag_m"]) - float(target_before_stats["bbox_diag_m"])
        ),
        "parent_pollution_penalty": v2_components["parent_pollution_penalty"],
        "joint_replay_worsening_penalty": v2_components["joint_replay_worsening_penalty"],
        "coverage_gain_score": v2_components["coverage_gain_score"],
        "anti_degeneracy_score": v2_components["anti_degeneracy_score"],
        "rigid_delta_score": v2_components["rigid_delta_score"],
        "joint_replay_delta_score": v2_components["joint_replay_delta_score"],
        "low_motion_base_like_score": v2_components["low_motion_base_like_score"],
        "moving_cluster_to_base_penalty": v2_components["moving_cluster_to_base_penalty"],
        "ignore_safety_score": v2_components["ignore_safety_score"],
        "lost_motion_usefulness_penalty": v2_components["lost_motion_usefulness_penalty"],
        "largest_cluster_ratio_delta": (
            (_summary_metric(candidate_eval, "largest_cluster_ratio") or 0.0)
            - (_summary_metric(baseline_eval, "largest_cluster_ratio") or 0.0)
        ),
        "mean_cluster_purity_delta": (
            (_summary_metric(candidate_eval, "mean_cluster_purity") or 0.0)
            - (_summary_metric(baseline_eval, "mean_cluster_purity") or 0.0)
        ),
        "route_score_diagnostic": _route_score_diagnostic(baseline_eval, candidate_eval),
        "route_score_no_gt_v1": _route_score_no_gt_proxy(baseline_eval, candidate_eval, route_summary),
        "route_score_no_gt_proxy": _route_score_no_gt_proxy(baseline_eval, candidate_eval, route_summary),
        "route_score_no_gt_v2": v2_components["route_score_no_gt_v2"],
        "output_dir": str(output_dir),
    }
    row.update(stability_fields)
    row.update(v3_components)
    row["diagnostic_score_stability_aware"] = _diagnostic_score_stability_aware(row)
    row.update(timing)
    return row


def _route_candidates(evaluation: dict[str, Any], cluster_row: dict[str, Any]) -> list[tuple[str, int | None, str | None]]:
    parent_id = cluster_row.get("parent_part_id")
    parent_name = cluster_row.get("parent_name")
    routes: list[tuple[str, int | None, str | None]] = []
    if parent_id is not None:
        routes.append(("merge_parent", int(parent_id), parent_name))
    for child_id in _matched_child_ids(evaluation):
        if int(child_id) == int(cluster_row.get("child_part_id")):
            continue
        routes.append(("merge_sibling", int(child_id), f"motion_part_{child_id}"))
    routes.append(("ignore", None, None))
    return routes


def _neutral_skipped_stability(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "skipped_reason": reason,
        "joint_stability_score": 0.5,
        "axis_std_deg": None,
        "pivot_std_m": None,
        "direction_std_deg": None,
        "stability_sample_count": 0,
    }


def _apply_route_stability_fields(row: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any]) -> None:
    before = before or {}
    stability_fields = {
        "stability_available": bool(after.get("available")),
        "stability_skipped_reason": after.get("skipped_reason"),
        "stability_method": after.get("method"),
        "stability_sample_count_before": before.get("stability_sample_count"),
        "stability_sample_count_after": after.get("stability_sample_count"),
        "joint_type_consistency_before": before.get("joint_type_consistency"),
        "joint_type_consistency_after": after.get("joint_type_consistency"),
        "axis_std_deg_before": before.get("axis_std_deg"),
        "axis_std_deg_after": after.get("axis_std_deg"),
        "axis_std_deg_delta": (
            None
            if before.get("axis_std_deg") is None or after.get("axis_std_deg") is None
            else float(after["axis_std_deg"]) - float(before["axis_std_deg"])
        ),
        "pivot_std_m_before": before.get("pivot_std_m"),
        "pivot_std_m_after": after.get("pivot_std_m"),
        "pivot_std_m_delta": (
            None
            if before.get("pivot_std_m") is None or after.get("pivot_std_m") is None
            else float(after["pivot_std_m"]) - float(before["pivot_std_m"])
        ),
        "direction_std_deg_before": before.get("direction_std_deg"),
        "direction_std_deg_after": after.get("direction_std_deg"),
        "direction_std_deg_delta": (
            None
            if before.get("direction_std_deg") is None or after.get("direction_std_deg") is None
            else float(after["direction_std_deg"]) - float(before["direction_std_deg"])
        ),
        "joint_stability_score_before": before.get("joint_stability_score"),
        "joint_stability_score_after": after.get("joint_stability_score"),
        "joint_stability_score_delta": (
            None
            if before.get("joint_stability_score") is None or after.get("joint_stability_score") is None
            else float(after["joint_stability_score"]) - float(before["joint_stability_score"])
        ),
    }
    row.update(stability_fields)
    v2_components = {
        key: row[key]
        for key in [
            "route_score_no_gt_v2",
            "motion_compatibility_score",
            "rigid_delta_score",
            "joint_replay_delta_score",
            "coverage_gain_score",
            "anti_degeneracy_score",
            "joint_replay_worsening_penalty",
            "ignore_safety_score",
            "lost_motion_usefulness_penalty",
            "low_motion_base_like_score",
            "parent_pollution_penalty",
            "moving_cluster_to_base_penalty",
        ]
        if key in row
    }
    row.update(_score_components_v3(str(row.get("route")), v2_components, stability_fields))
    row["diagnostic_score_stability_aware"] = _diagnostic_score_stability_aware(row)


def _select_rows_for_stability(rows: list[dict[str, Any]], mode: str, top_k: int, margin: float) -> set[int]:
    if mode == "all":
        return {idx for idx, row in enumerate(rows) if row.get("target_part_id") is not None}
    selected: set[int] = set()
    grouped: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        if row.get("target_part_id") is None:
            continue
        key = (str(row["object_id"]), int(row["unmatched_cluster_id"]))
        grouped.setdefault(key, []).append((idx, row))
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: float(item[1].get("route_score_no_gt_v2") or -1e9), reverse=True)
        if mode == "topk":
            selected.update(idx for idx, _ in ordered[: max(1, int(top_k))])
        elif mode == "borderline":
            if not ordered:
                continue
            best_score = float(ordered[0][1].get("route_score_no_gt_v2") or -1e9)
            selected.update(
                idx for idx, row in ordered if best_score - float(row.get("route_score_no_gt_v2") or -1e9) <= float(margin)
            )
    return selected


def _run_object(
    cwd: Path,
    object_dir: Path,
    spectral_k: int,
    edge_ablation: str,
    quality_weighted_affinity: bool,
    quality_weighted_affinity_time_only: bool,
    quality_weighted_affinity_edge_prior: bool,
    articulation_compatible_affinity: bool,
    quality_weighted_part_poses: bool,
    quality_weighted_replay: bool,
    min_quality_weight: float,
    quality_affinity_min_pair_weight: float,
    rerun_baseline: bool,
    enable_stability_lite: bool,
    stability_eval_mode: str,
    stability_top_k: int,
    stability_score_margin: float,
    stability_bin_count: int,
    stability_min_tracks_per_bin: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_start_time = time.perf_counter()
    object_id = object_dir.name
    baseline_start_time = time.perf_counter()
    baseline_paths = _ensure_baseline(
        cwd,
        object_dir,
        spectral_k,
        edge_ablation,
        quality_weighted_affinity,
        quality_weighted_affinity_time_only,
        quality_weighted_affinity_edge_prior,
        articulation_compatible_affinity,
        quality_weighted_part_poses,
        quality_weighted_replay,
        min_quality_weight,
        quality_affinity_min_pair_weight,
        rerun_baseline,
        dry_run,
    )
    baseline_runtime_s = time.perf_counter() - baseline_start_time
    baseline_eval = _load_json(baseline_paths["evaluation"])
    baseline_eval_summary = baseline_eval.get("summary") or {}
    baseline_segmentation_diagnostics = (
        _load_json(baseline_paths["diagnostics"]) if baseline_paths["diagnostics"].exists() else {}
    )
    baseline_joints = _load_json(baseline_paths["joints"])
    baseline_tracks = baseline_paths["tracks"]
    baseline_track_payload = _load_json(baseline_tracks)
    baseline_track_stats = _cluster_track_stats(baseline_track_payload)
    unmatched = _unmatched_rows(baseline_eval)
    rows: list[dict[str, Any]] = []
    routing_root = object_dir / "em_lite" / "routing"
    baseline_stability_cache: dict[int, dict[str, Any]] = {}
    for cluster_row in unmatched:
        cluster_id = int(cluster_row["child_part_id"])
        for route, target_id, target_name in _route_candidates(baseline_eval, cluster_row):
            route_start_time = time.perf_counter()
            suffix = f"cluster_{cluster_id}_{route}"
            if target_id is not None:
                suffix += f"_{target_id}"
            output_dir = routing_root / suffix
            output_dir.mkdir(parents=True, exist_ok=True)
            routed_tracks = output_dir / "motion_part_tracks_routed.json"
            route_rewrite_start_time = time.perf_counter()
            route_summary = _rewrite_route(baseline_tracks, routed_tracks, cluster_id, route, target_id)
            route_summary["rewrite_runtime_s"] = time.perf_counter() - route_rewrite_start_time
            _save_json(output_dir / "route_summary.json", route_summary)
            poses = output_dir / "part_poses_routed.json"
            joints = output_dir / "joint_inference_routed.json"
            eval_json = output_dir / "object_mask_kinematic_evaluation_routed.json"
            stack_timing = _run_stack(
                cwd,
                routed_tracks,
                poses,
                joints,
                eval_json,
                dry_run,
                quality_weighted_part_poses=quality_weighted_part_poses,
                quality_weighted_replay=quality_weighted_replay,
                min_quality_weight=min_quality_weight,
            )
            route_stack_runtime_s = stack_timing["route_stack_runtime_s"]
            candidate_eval = _load_json(eval_json)
            candidate_joints = _load_json(joints)
            candidate_track_stats = _cluster_track_stats(_load_json(routed_tracks))
            stability_before = None
            stability_after = _neutral_skipped_stability("stability-lite disabled")
            baseline_stability_runtime_s = None
            stability_after_runtime_s = None
            if enable_stability_lite and stability_eval_mode == "all" and target_id is not None:
                if int(target_id) not in baseline_stability_cache:
                    baseline_stability_cache[int(target_id)] = _run_stability_lite(
                        cwd,
                        baseline_tracks,
                        int(target_id),
                        routing_root / "_baseline_stability" / f"part_{target_id}",
                        stability_bin_count,
                        stability_min_tracks_per_bin,
                        quality_weighted_part_poses,
                        quality_weighted_replay,
                        min_quality_weight,
                        dry_run,
                    )
                stability_before = baseline_stability_cache[int(target_id)]
                baseline_stability_runtime_s = stability_before.get("runtime_s")
                stability_after = _run_stability_lite(
                    cwd,
                    routed_tracks,
                    int(target_id),
                    output_dir,
                    stability_bin_count,
                    stability_min_tracks_per_bin,
                    quality_weighted_part_poses,
                    quality_weighted_replay,
                    min_quality_weight,
                    dry_run,
                )
                stability_after_runtime_s = stability_after.get("runtime_s")
            elif enable_stability_lite and target_id is not None:
                stability_after = _neutral_skipped_stability(f"deferred by stability_eval_mode={stability_eval_mode}")
            timing = {
                "baseline_runtime_s": baseline_runtime_s,
                "route_runtime_s": time.perf_counter() - route_start_time,
                "route_stack_runtime_s": route_stack_runtime_s,
                "route_rewrite_runtime_s": route_summary.get("rewrite_runtime_s"),
                "estimate_part_poses_runtime_s": stack_timing.get("estimate_part_poses_runtime_s"),
                "infer_joints_runtime_s": stack_timing.get("infer_joints_runtime_s"),
                "evaluate_runtime_s": stack_timing.get("evaluate_runtime_s"),
                "debug_html_runtime_s": None,
                "baseline_stability_runtime_s": baseline_stability_runtime_s,
                "stability_after_runtime_s": stability_after_runtime_s,
            }
            row = _diagnostic_row(
                object_id,
                cluster_row,
                route,
                target_id,
                target_name,
                baseline_eval,
                candidate_eval,
                baseline_joints,
                candidate_joints,
                baseline_track_stats,
                candidate_track_stats,
                route_summary,
                stability_before,
                stability_after,
                timing,
                output_dir,
            )
            rows.append(row)
            _save_json(output_dir / "route_comparison.json", row)
    if enable_stability_lite and stability_eval_mode != "all":
        selected_stability_rows = _select_rows_for_stability(
            rows,
            stability_eval_mode,
            max(1, int(stability_top_k)),
            float(stability_score_margin),
        )
        for row_idx, row in enumerate(rows):
            target_id = row.get("target_part_id")
            if target_id is None:
                _apply_route_stability_fields(row, None, _neutral_skipped_stability("ignore route has no target part"))
                continue
            if row_idx not in selected_stability_rows:
                _apply_route_stability_fields(
                    row,
                    None,
                    _neutral_skipped_stability(f"skipped by stability_eval_mode={stability_eval_mode}"),
                )
                continue
            if int(target_id) not in baseline_stability_cache:
                baseline_stability_cache[int(target_id)] = _run_stability_lite(
                    cwd,
                    baseline_tracks,
                    int(target_id),
                    routing_root / "_baseline_stability" / f"part_{target_id}",
                    stability_bin_count,
                    stability_min_tracks_per_bin,
                    quality_weighted_part_poses,
                    quality_weighted_replay,
                    min_quality_weight,
                    dry_run,
                )
            routed_tracks = Path(str(row["output_dir"])) / "motion_part_tracks_routed.json"
            stability_after_start = time.perf_counter()
            stability_after = _run_stability_lite(
                cwd,
                routed_tracks,
                int(target_id),
                Path(str(row["output_dir"])),
                stability_bin_count,
                stability_min_tracks_per_bin,
                quality_weighted_part_poses,
                quality_weighted_replay,
                min_quality_weight,
                dry_run,
            )
            row["baseline_stability_runtime_s"] = baseline_stability_cache[int(target_id)].get("runtime_s")
            row["stability_after_runtime_s"] = time.perf_counter() - stability_after_start
            _apply_route_stability_fields(row, baseline_stability_cache[int(target_id)], stability_after)
            _save_json(Path(str(row["output_dir"])) / "route_comparison.json", row)
    object_runtime_s = time.perf_counter() - object_start_time
    available_baseline_stability = [
        value for value in baseline_stability_cache.values() if value.get("available") and value.get("joint_stability_score") is not None
    ]
    axis_noise_floor = _median(
        [float(value["axis_std_deg"]) for value in available_baseline_stability if value.get("axis_std_deg") is not None]
    )
    pivot_noise_floor = _median(
        [float(value["pivot_std_m"]) for value in available_baseline_stability if value.get("pivot_std_m") is not None]
    )
    direction_noise_floor = _median(
        [
            float(value["direction_std_deg"])
            for value in available_baseline_stability
            if value.get("direction_std_deg") is not None
        ]
    )
    for row in rows:
        row["object_runtime_s"] = object_runtime_s
        row["object_route_count"] = len(rows)
        row["object_axis_std_noise_floor_deg"] = axis_noise_floor
        row["object_pivot_std_noise_floor_m"] = pivot_noise_floor
        row["object_direction_std_noise_floor_deg"] = direction_noise_floor
        axis_after = _safe_float(row.get("axis_std_deg_after"))
        pivot_after = _safe_float(row.get("pivot_std_m_after"))
        direction_after = _safe_float(row.get("direction_std_deg_after"))
        row["excess_axis_std_deg"] = (
            None if axis_after is None or axis_noise_floor is None else max(0.0, axis_after - axis_noise_floor)
        )
        row["excess_pivot_std_m"] = (
            None if pivot_after is None or pivot_noise_floor is None else max(0.0, pivot_after - pivot_noise_floor)
        )
        row["excess_direction_std_deg"] = (
            None
            if direction_after is None or direction_noise_floor is None
            else max(0.0, direction_after - direction_noise_floor)
        )
    chosen_v2 = [row for row in rows if row.get("chosen_by_no_gt_v2")]
    chosen_v3 = [row for row in rows if row.get("chosen_by_no_gt_v3")]
    object_summary = {
        "object_id": object_id,
        "object_dir": str(object_dir),
        "object_runtime_s": object_runtime_s,
        "baseline_runtime_s": baseline_runtime_s,
        "baseline_mean_cluster_purity": baseline_eval_summary.get("mean_cluster_purity"),
        "baseline_mean_gt_coverage": baseline_eval_summary.get("mean_gt_coverage"),
        "baseline_largest_cluster_ratio": baseline_eval_summary.get("largest_cluster_ratio"),
        "baseline_axis_mean_diagnostic": baseline_eval_summary.get("axis_angle_error_deg_mean"),
        "baseline_pivot_mean_diagnostic": baseline_eval_summary.get("pivot_error_m_mean"),
        "baseline_overlap_per_gt": (baseline_eval.get("overlap") or {}).get("per_gt"),
        "baseline_overlap_per_cluster": (baseline_eval.get("overlap") or {}).get("per_cluster"),
        "baseline_overlap_matrix": (baseline_eval.get("overlap") or {}).get("overlap_matrix"),
        "baseline_segmentation_edge_summary": baseline_segmentation_diagnostics.get("edge_summary"),
        "baseline_segmentation_graph_component_summary": baseline_segmentation_diagnostics.get("graph_component_summary"),
        "unmatched_cluster_count": len(unmatched),
        "route_count": len(rows),
        "chosen_v2_count": len(chosen_v2),
        "chosen_v3_count": len(chosen_v3),
        "v2_agreement_count": sum(row.get("v2_agrees_with_diagnostic") is True for row in chosen_v2),
        "v3_agreement_count": sum(row.get("v3_agrees_with_diagnostic") is True for row in chosen_v3),
        "object_axis_std_noise_floor_deg": axis_noise_floor,
        "object_pivot_std_noise_floor_m": pivot_noise_floor,
        "object_direction_std_noise_floor_deg": direction_noise_floor,
    }
    return rows, object_summary


def _annotate_route_choices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["object_id"]), int(row["unmatched_cluster_id"]))
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        by_v2 = sorted(group, key=lambda row: float(row.get("route_score_no_gt_v2") or -1e9), reverse=True)
        by_v3 = sorted(group, key=lambda row: float(row.get("route_score_no_gt_v3") or -1e9), reverse=True)
        by_diag_raw = sorted(group, key=lambda row: float(row.get("route_score_diagnostic") or -1e9), reverse=True)
        by_diag_stable = sorted(
            group, key=lambda row: float(row.get("diagnostic_score_stability_aware") or -1e9), reverse=True
        )
        best_v2 = by_v2[0] if by_v2 else None
        best_v3 = by_v3[0] if by_v3 else None
        best_diag_raw = by_diag_raw[0] if by_diag_raw else None
        best_diag_stable = by_diag_stable[0] if by_diag_stable else None
        best_v2_label = (best_v2 or {}).get("route_option")
        best_v3_label = (best_v3 or {}).get("route_option")
        best_diag_raw_label = (best_diag_raw or {}).get("route_option")
        best_diag_stable_label = (best_diag_stable or {}).get("route_option")
        for rank, row in enumerate(by_v2, start=1):
            row["rank_by_no_gt_v2"] = rank
            row["chosen_by_no_gt_v2"] = rank == 1
            row["best_route_by_no_gt_v2"] = best_v2_label
            row["best_route_by_diagnostic_gt"] = best_diag_raw_label
            row["best_route_by_diagnostic_gt_raw"] = best_diag_raw_label
            row["diagnostic_best_route_label"] = best_diag_raw_label
            row["best_route_by_diagnostic_gt_stability_aware"] = best_diag_stable_label
            row["v2_agrees_with_diagnostic"] = best_v2_label == best_diag_raw_label
            row["v2_agrees_with_diagnostic_raw"] = best_v2_label == best_diag_raw_label
            row["v2_agrees_with_diagnostic_stability_aware"] = best_v2_label == best_diag_stable_label
        for rank, row in enumerate(by_v3, start=1):
            row["rank_by_no_gt_v3"] = rank
            row["chosen_by_no_gt_v3"] = rank == 1
            row["best_route_by_no_gt_v3"] = best_v3_label
            row["v3_agrees_with_diagnostic"] = best_v3_label == best_diag_raw_label
            row["v3_agrees_with_diagnostic_raw"] = best_v3_label == best_diag_raw_label
            row["v3_agrees_with_diagnostic_stability_aware"] = best_v3_label == best_diag_stable_label
        for rank, row in enumerate(by_diag_raw, start=1):
            row["rank_by_diagnostic_gt"] = rank
            row["rank_by_diagnostic_gt_raw"] = rank
            row["chosen_by_diagnostic_gt"] = rank == 1
            row["chosen_by_diagnostic_gt_raw"] = rank == 1
        for rank, row in enumerate(by_diag_stable, start=1):
            row["rank_by_diagnostic_gt_stability_aware"] = rank
            row["chosen_by_diagnostic_gt_stability_aware"] = rank == 1
        if best_v3 is not None and best_diag_raw is not None and best_diag_stable is not None:
            category, reason = _classify_v3_failure(best_v3, best_diag_raw, best_diag_stable)
            for row in group:
                row["v3_failure_category"] = category
                row["v3_failure_reason"] = reason
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_id",
        "unmatched_cluster_id",
        "unmatched_joint_name",
        "route",
        "route_option",
        "target_part_id",
        "target_cluster_id",
        "target_name",
        "target_gt_part_id_diagnostic",
        "dominant_gt_part_id_diagnostic",
        "changed_track_count",
        "dropped_track_count",
        "child_track_count",
        "track_count",
        "bbox_diag_m",
        "mean_motion_m",
        "median_motion_m",
        "route_added_track_quality_mean",
        "route_added_track_quality_median",
        "route_added_low_quality_track_ratio",
        "route_added_low_quality_timestep_ratio",
        "route_added_effective_observation_count",
        "route_added_effective_observation_ratio",
        "target_track_quality_mean_before",
        "target_track_quality_mean_after",
        "target_low_quality_timestep_ratio_before",
        "target_low_quality_timestep_ratio_after",
        "target_effective_observation_count_before",
        "target_effective_observation_count_after",
        "route_effective_coverage_gain",
        "route_quality_delta",
        "current_child_rigid_rmse_m",
        "current_child_inlier_ratio",
        "child_gt_purity_diagnostic",
        "child_gt_composition_diagnostic",
        "unmatched_cluster_dominant_gt_coverage_diagnostic",
        "target_cluster_gt_coverage_before_diagnostic",
        "target_cluster_gt_coverage_after_diagnostic",
        "target_cluster_gt_coverage_delta_diagnostic",
        "dominant_gt_best_coverage_before_diagnostic",
        "dominant_gt_best_coverage_after_diagnostic",
        "dominant_gt_best_coverage_delta_diagnostic",
        "baseline_predicted_joint_count",
        "candidate_predicted_joint_count",
        "predicted_joint_count_after",
        "largest_cluster_ratio_after",
        "baseline_unmatched_count",
        "candidate_unmatched_count",
        "baseline_directed_joint_coverage_diagnostic",
        "candidate_directed_joint_coverage_diagnostic",
        "baseline_axis_mean_diagnostic",
        "candidate_axis_mean_diagnostic",
        "baseline_pivot_mean_diagnostic",
        "candidate_pivot_mean_diagnostic",
        "matched_axis_delta_diagnostic",
        "matched_pivot_delta_diagnostic",
        "diagnostic_axis_mean_after",
        "diagnostic_pivot_mean_after",
        "diagnostic_directed_coverage_after",
        "motion_compatibility_raw",
        "motion_compatibility_score",
        "rigid_rmse_before",
        "rigid_rmse_after",
        "rigid_rmse_delta",
        "joint_replay_before",
        "joint_replay_after",
        "joint_replay_delta",
        "matched_joint_replay_mean_before",
        "matched_joint_replay_mean_after",
        "matched_joint_replay_delta",
        "parent_rigid_rmse_before",
        "parent_rigid_rmse_after",
        "parent_rigid_rmse_delta",
        "sibling_rigid_rmse_before",
        "sibling_rigid_rmse_after",
        "sibling_rigid_rmse_delta",
        "sibling_track_count_gain",
        "sibling_bbox_gain_m",
        "parent_pollution_penalty",
        "joint_replay_worsening_penalty",
        "coverage_gain_score",
        "anti_degeneracy_score",
        "stability_available",
        "stability_skipped_reason",
        "stability_method",
        "stability_sample_count_before",
        "stability_sample_count_after",
        "joint_type_consistency_before",
        "joint_type_consistency_after",
        "axis_std_deg_before",
        "axis_std_deg_after",
        "axis_std_deg_delta",
        "pivot_std_m_before",
        "pivot_std_m_after",
        "pivot_std_m_delta",
        "direction_std_deg_before",
        "direction_std_deg_after",
        "direction_std_deg_delta",
        "joint_stability_score_before",
        "joint_stability_score_after",
        "joint_stability_score_delta",
        "joint_stability_score",
        "effective_coverage_gain_score",
        "axis_instability_penalty",
        "pivot_instability_penalty",
        "direction_instability_penalty",
        "instability_penalty",
        "object_axis_std_noise_floor_deg",
        "object_pivot_std_noise_floor_m",
        "object_direction_std_noise_floor_deg",
        "excess_axis_std_deg",
        "excess_pivot_std_m",
        "excess_direction_std_deg",
        "rigid_delta_score",
        "joint_replay_delta_score",
        "low_motion_base_like_score",
        "moving_cluster_to_base_penalty",
        "ignore_safety_score",
        "lost_motion_usefulness_penalty",
        "largest_cluster_ratio_delta",
        "mean_cluster_purity_delta",
        "route_score_diagnostic",
        "route_score_no_gt_v1",
        "route_score_no_gt_proxy",
        "route_score_no_gt_v2",
        "route_score_no_gt_v3",
        "diagnostic_score_stability_aware",
        "baseline_runtime_s",
        "route_runtime_s",
        "route_stack_runtime_s",
        "route_rewrite_runtime_s",
        "estimate_part_poses_runtime_s",
        "infer_joints_runtime_s",
        "evaluate_runtime_s",
        "debug_html_runtime_s",
        "baseline_stability_runtime_s",
        "stability_after_runtime_s",
        "object_runtime_s",
        "object_route_count",
        "rank_by_no_gt_v2",
        "chosen_by_no_gt_v2",
        "best_route_by_no_gt_v2",
        "rank_by_no_gt_v3",
        "chosen_by_no_gt_v3",
        "best_route_by_no_gt_v3",
        "rank_by_diagnostic_gt",
        "rank_by_diagnostic_gt_raw",
        "chosen_by_diagnostic_gt",
        "chosen_by_diagnostic_gt_raw",
        "best_route_by_diagnostic_gt",
        "best_route_by_diagnostic_gt_raw",
        "diagnostic_best_route_label",
        "rank_by_diagnostic_gt_stability_aware",
        "chosen_by_diagnostic_gt_stability_aware",
        "best_route_by_diagnostic_gt_stability_aware",
        "v2_agrees_with_diagnostic",
        "v3_agrees_with_diagnostic",
        "v2_agrees_with_diagnostic_raw",
        "v3_agrees_with_diagnostic_raw",
        "v2_agrees_with_diagnostic_stability_aware",
        "v3_agrees_with_diagnostic_stability_aware",
        "v3_failure_category",
        "v3_failure_reason",
        "output_dir",
    ]
    fieldnames.extend(sorted({key for row in rows for key in row.keys()} - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_object_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_id",
        "object_dir",
        "object_runtime_s",
        "baseline_runtime_s",
        "unmatched_cluster_count",
        "route_count",
        "chosen_v2_count",
        "chosen_v3_count",
        "v2_agreement_count",
        "v3_agreement_count",
        "v2_agreement_raw_count",
        "v3_agreement_raw_count",
        "v2_agreement_stability_aware_count",
        "v3_agreement_stability_aware_count",
        "v2_agreement_ratio",
        "v3_agreement_ratio",
        "v2_agreement_stability_aware_ratio",
        "v3_agreement_stability_aware_ratio",
        "mean_route_runtime_s",
        "mean_route_stack_runtime_s",
        "mean_route_rewrite_runtime_s",
        "mean_estimate_part_poses_runtime_s",
        "mean_infer_joints_runtime_s",
        "mean_evaluate_runtime_s",
        "debug_html_runtime_s",
        "mean_stability_after_runtime_s",
        "object_axis_std_noise_floor_deg",
        "object_pivot_std_noise_floor_m",
        "object_direction_std_noise_floor_deg",
    ]
    fieldnames.extend(sorted({key for row in rows for key in row.keys()} - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_diagnostic_score_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen_v2 = [row for row in rows if row.get("chosen_by_no_gt_v2")]
    chosen_v3 = [row for row in rows if row.get("chosen_by_no_gt_v3")]
    raw_best = [row for row in rows if row.get("chosen_by_diagnostic_gt_raw")]
    stable_best = [row for row in rows if row.get("chosen_by_diagnostic_gt_stability_aware")]
    category_counts: Counter[str] = Counter(
        str(row.get("v3_failure_category"))
        for row in chosen_v3
        if row.get("v3_failure_category")
    )
    unstable_raw_best = sum(_is_severely_unstable(row) or _is_moderately_unstable(row) for row in raw_best)
    summary = {
        "total_unmatched_decisions": len(chosen_v3),
        "route_rows": len(rows),
        "v2_agreement_raw": sum(row.get("v2_agrees_with_diagnostic_raw") is True for row in chosen_v2),
        "v3_agreement_raw": sum(row.get("v3_agrees_with_diagnostic_raw") is True for row in chosen_v3),
        "v2_agreement_stability_aware": sum(
            row.get("v2_agrees_with_diagnostic_stability_aware") is True for row in chosen_v2
        ),
        "v3_agreement_stability_aware": sum(
            row.get("v3_agrees_with_diagnostic_stability_aware") is True for row in chosen_v3
        ),
        "unstable_raw_diagnostic_best_routes": unstable_raw_best,
        "stability_aware_best_differs_from_raw": sum(
            raw.get("route_option") != stable.get("route_option")
            for raw, stable in zip(raw_best, stable_best)
        ),
    }
    for category, count in sorted(category_counts.items()):
        summary[f"v3_failure_{category}"] = count
    fieldnames = list(summary.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def _write_v3_failure_analysis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    for chosen in rows:
        if chosen.get("chosen_by_no_gt_v3") is not True or chosen.get("v3_agrees_with_diagnostic_raw") is True:
            continue
        group = [
            row
            for row in rows
            if row.get("object_id") == chosen.get("object_id")
            and row.get("unmatched_cluster_id") == chosen.get("unmatched_cluster_id")
        ]
        raw_best = next((row for row in group if row.get("chosen_by_diagnostic_gt_raw")), None)
        stable_best = next((row for row in group if row.get("chosen_by_diagnostic_gt_stability_aware")), None)
        failures.append(
            {
                "object_id": chosen.get("object_id"),
                "unmatched_cluster_id": chosen.get("unmatched_cluster_id"),
                "v3_route": chosen.get("route_option"),
                "raw_diagnostic_route": (raw_best or {}).get("route_option"),
                "stability_aware_diagnostic_route": (stable_best or {}).get("route_option"),
                "v3_failure_category": chosen.get("v3_failure_category"),
                "v3_failure_reason": chosen.get("v3_failure_reason"),
                "raw_diagnostic_score": (raw_best or {}).get("route_score_diagnostic"),
                "raw_diagnostic_stability_aware_score": (raw_best or {}).get("diagnostic_score_stability_aware"),
                "raw_joint_stability_score": (raw_best or {}).get("joint_stability_score"),
                "raw_instability_penalty": (raw_best or {}).get("instability_penalty"),
                "raw_axis_std_deg": (raw_best or {}).get("axis_std_deg_after"),
                "raw_pivot_std_m": (raw_best or {}).get("pivot_std_m_after"),
                "raw_direction_std_deg": (raw_best or {}).get("direction_std_deg_after"),
                "raw_effective_coverage_gain_score": (raw_best or {}).get("effective_coverage_gain_score"),
            }
        )
    fieldnames = list(failures[0].keys()) if failures else ["object_id", "unmatched_cluster_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def _rel_link(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def _write_object_debug_index(
    cwd: Path,
    object_dir: Path,
    rows: list[dict[str, Any]],
    object_summary: dict[str, Any],
    generate_viewer: bool,
    dry_run: bool,
) -> Path:
    debug_start_time = time.perf_counter()
    debug_dir = object_dir / "em_lite" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_paths(object_dir)
    viewer_html = debug_dir / "baseline_flow_viewer.html"
    quality_dir = debug_dir / "track_quality"
    quality_manifest = quality_dir / "track_quality_manifest.json"
    quality_tracks = quality_dir / "motion_part_tracks_with_quality.json"
    quality_viewer_html = debug_dir / "baseline_quality_flow_viewer.html"
    if baseline["tracks"].exists():
        _run(
            _module_cmd(
                "compute-track-quality",
                baseline["tracks"],
                "--output-dir",
                quality_dir,
            ),
            cwd,
            dry_run,
        )
    if generate_viewer and baseline["tracks"].exists():
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                baseline["tracks"],
                "--joint-inference",
                baseline["joints"],
                "--evaluation-json",
                baseline["evaluation"],
                "--output-html",
                viewer_html,
                "--max-tracks",
                "1000",
                "--frame-stride",
                "2",
                "--trail-length",
                "10",
                "--color-by",
                "pred_cluster",
            ),
            cwd,
            dry_run,
        )
    if generate_viewer and quality_tracks.exists():
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                quality_tracks,
                "--joint-inference",
                baseline["joints"],
                "--evaluation-json",
                baseline["evaluation"],
                "--output-html",
                quality_viewer_html,
                "--max-tracks",
                "1000",
                "--frame-stride",
                "2",
                "--trail-length",
                "10",
                "--color-by",
                "timestep_quality",
            ),
            cwd,
            dry_run,
        )
    object_summary["debug_html_runtime_s"] = time.perf_counter() - debug_start_time
    chosen_rows = [row for row in rows if row.get("chosen_by_no_gt_v3")]
    table_rows = []
    for row in chosen_rows:
        route_dir = Path(str(row.get("output_dir")))
        route_link = _rel_link(route_dir, debug_dir) if route_dir.exists() else ""
        route_cell = f'<a href="{html.escape(route_link)}">route</a>' if route_link else ""
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('unmatched_cluster_id')))}</td>"
            f"<td>{html.escape(str(row.get('route_option')))}</td>"
            f"<td>{html.escape(str(row.get('best_route_by_diagnostic_gt_raw')))}</td>"
            f"<td>{html.escape(str(row.get('best_route_by_diagnostic_gt_stability_aware')))}</td>"
            f"<td>{html.escape(str(row.get('v3_failure_category') or ''))}</td>"
            f"<td>{html.escape(str(row.get('joint_stability_score') or ''))}</td>"
            f"<td>{html.escape(str(row.get('instability_penalty') or ''))}</td>"
            f"<td>{route_cell}</td>"
            "</tr>"
        )
    baseline_viewer_link = _rel_link(viewer_html, debug_dir) if viewer_html.exists() else ""
    quality_viewer_link = _rel_link(quality_viewer_html, debug_dir) if quality_viewer_html.exists() else ""
    quality_manifest_link = _rel_link(quality_manifest, debug_dir) if quality_manifest.exists() else ""
    index_html = debug_dir / "index.html"
    index_html.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>EM-lite Routing Debug</title>",
                "<style>body{font-family:Arial,sans-serif;margin:24px;background:#10151c;color:#dbe7f3}"
                "a{color:#7dd3fc}table{border-collapse:collapse;width:100%;margin-top:16px}"
                "td,th{border:1px solid #334155;padding:6px;text-align:left}th{background:#1e293b}</style>",
                "</head><body>",
                f"<h1>{html.escape(str(object_summary.get('object_id')))} EM-lite Routing Debug</h1>",
                f"<p>runtime: {float(object_summary.get('object_runtime_s') or 0.0):.2f}s; "
                f"routes: {html.escape(str(object_summary.get('route_count')))}; "
                f"unmatched clusters: {html.escape(str(object_summary.get('unmatched_cluster_count')))}</p>",
                (
                    f"<p><a href='{html.escape(baseline_viewer_link)}'>Baseline object-mask flow viewer</a></p>"
                    if baseline_viewer_link
                    else "<p>Baseline viewer was not generated.</p>"
                ),
                (
                    f"<p><a href='{html.escape(quality_viewer_link)}'>Baseline quality flow viewer</a> "
                    f"(<a href='{html.escape(quality_manifest_link)}'>quality manifest</a>)</p>"
                    if quality_viewer_link
                    else "<p>Baseline quality viewer was not generated.</p>"
                ),
                "<h2>v3 chosen routes</h2>",
                "<table><thead><tr><th>cluster</th><th>v3 route</th><th>raw GT best</th>"
                "<th>stable GT best</th><th>failure category</th><th>stability</th>"
                "<th>instability</th><th>route dir</th></tr></thead><tbody>",
                *table_rows,
                "</tbody></table>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return index_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EM-lite routes for unmatched object-mask clusters.")
    parser.add_argument("object_dirs", nargs="+", type=Path)
    parser.add_argument("--spectral-k", type=int, default=5)
    parser.add_argument("--edge-ablation", choices=["A", "B", "C"], default="B")
    parser.add_argument("--rerun-baseline", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--score-version", choices=["v1", "v2", "v3", "both"], default="both")
    parser.add_argument("--quality-weighted-affinity", action="store_true")
    parser.add_argument("--quality-weighted-affinity-time-only", action="store_true")
    parser.add_argument("--quality-weighted-affinity-edge-prior", action="store_true")
    parser.add_argument("--articulation-compatible-affinity", action="store_true")
    parser.add_argument("--quality-weighted-part-poses", action="store_true")
    parser.add_argument("--quality-weighted-replay", action="store_true")
    parser.add_argument(
        "--min-quality-weight",
        type=float,
        default=0.2,
        help="Minimum observation weight passed to quality-weighted part pose and joint replay stages.",
    )
    parser.add_argument(
        "--quality-affinity-min-pair-weight",
        type=float,
        default=None,
        help="Minimum pair quality prior for quality-weighted affinity. Defaults to --min-quality-weight.",
    )
    parser.add_argument("--enable-stability-lite", action="store_true")
    parser.add_argument(
        "--stability-eval-mode",
        choices=["all", "topk", "borderline"],
        default="all",
        help="Controls which candidate routes get expensive stability-lite refits.",
    )
    parser.add_argument("--stability-top-k", type=int, default=2)
    parser.add_argument("--stability-score-margin", type=float, default=0.15)
    parser.add_argument("--stability-bin-count", type=int, default=3)
    parser.add_argument("--stability-min-tracks-per-bin", type=int, default=8)
    parser.add_argument("--no-generate-debug-viewers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    quality_affinity_min_pair_weight = (
        max(0.0, min(1.0, float(args.quality_affinity_min_pair_weight)))
        if args.quality_affinity_min_pair_weight is not None
        else max(0.0, min(1.0, float(args.min_quality_weight)))
    )
    all_rows: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    for object_dir in args.object_dirs:
        object_rows, object_summary = _run_object(
            cwd,
            object_dir,
            args.spectral_k,
            args.edge_ablation,
            bool(args.quality_weighted_affinity),
            bool(args.quality_weighted_affinity_time_only),
            bool(args.quality_weighted_affinity_edge_prior),
            bool(args.articulation_compatible_affinity),
            bool(args.quality_weighted_part_poses),
            bool(args.quality_weighted_replay),
            max(0.0, min(1.0, float(args.min_quality_weight))),
            quality_affinity_min_pair_weight,
            args.rerun_baseline,
            bool(args.enable_stability_lite),
            str(args.stability_eval_mode),
            max(1, int(args.stability_top_k)),
            float(args.stability_score_margin),
            max(2, int(args.stability_bin_count)),
            max(1, int(args.stability_min_tracks_per_bin)),
            args.dry_run,
        )
        all_rows.extend(object_rows)
        object_summary["quality_weighted_affinity"] = bool(args.quality_weighted_affinity)
        object_summary["quality_weighted_affinity_time_only"] = bool(args.quality_weighted_affinity_time_only)
        object_summary["quality_weighted_affinity_edge_prior"] = bool(args.quality_weighted_affinity_edge_prior)
        object_summary["articulation_compatible_affinity"] = bool(args.articulation_compatible_affinity)
        object_summary["quality_weighted_part_poses"] = bool(args.quality_weighted_part_poses)
        object_summary["quality_weighted_replay"] = bool(args.quality_weighted_replay)
        object_summary["min_quality_weight"] = max(0.0, min(1.0, float(args.min_quality_weight)))
        object_summary["quality_affinity_min_pair_weight"] = quality_affinity_min_pair_weight
        object_summaries.append(object_summary)
    all_rows = _annotate_route_choices(all_rows)
    for summary in object_summaries:
        object_rows = [row for row in all_rows if row.get("object_id") == summary["object_id"]]
        chosen_v2 = [row for row in object_rows if row.get("chosen_by_no_gt_v2")]
        chosen_v3 = [row for row in object_rows if row.get("chosen_by_no_gt_v3")]
        route_times = [float(row["route_runtime_s"]) for row in object_rows if row.get("route_runtime_s") is not None]
        stack_times = [
            float(row["route_stack_runtime_s"]) for row in object_rows if row.get("route_stack_runtime_s") is not None
        ]
        rewrite_times = [
            float(row["route_rewrite_runtime_s"]) for row in object_rows if row.get("route_rewrite_runtime_s") is not None
        ]
        estimate_times = [
            float(row["estimate_part_poses_runtime_s"])
            for row in object_rows
            if row.get("estimate_part_poses_runtime_s") is not None
        ]
        infer_times = [
            float(row["infer_joints_runtime_s"]) for row in object_rows if row.get("infer_joints_runtime_s") is not None
        ]
        evaluate_times = [
            float(row["evaluate_runtime_s"]) for row in object_rows if row.get("evaluate_runtime_s") is not None
        ]
        stability_times = [
            float(row["stability_after_runtime_s"])
            for row in object_rows
            if row.get("stability_after_runtime_s") is not None
        ]
        summary.update(
            {
                "chosen_v2_count": len(chosen_v2),
                "chosen_v3_count": len(chosen_v3),
                "v2_agreement_count": sum(row.get("v2_agrees_with_diagnostic") is True for row in chosen_v2),
                "v3_agreement_count": sum(row.get("v3_agrees_with_diagnostic") is True for row in chosen_v3),
                "v2_agreement_raw_count": sum(row.get("v2_agrees_with_diagnostic_raw") is True for row in chosen_v2),
                "v3_agreement_raw_count": sum(row.get("v3_agrees_with_diagnostic_raw") is True for row in chosen_v3),
                "v2_agreement_stability_aware_count": sum(
                    row.get("v2_agrees_with_diagnostic_stability_aware") is True for row in chosen_v2
                ),
                "v3_agreement_stability_aware_count": sum(
                    row.get("v3_agrees_with_diagnostic_stability_aware") is True for row in chosen_v3
                ),
                "v2_agreement_ratio": (
                    None
                    if not chosen_v2
                    else sum(row.get("v2_agrees_with_diagnostic") is True for row in chosen_v2) / len(chosen_v2)
                ),
                "v3_agreement_ratio": (
                    None
                    if not chosen_v3
                    else sum(row.get("v3_agrees_with_diagnostic") is True for row in chosen_v3) / len(chosen_v3)
                ),
                "v2_agreement_stability_aware_ratio": (
                    None
                    if not chosen_v2
                    else sum(row.get("v2_agrees_with_diagnostic_stability_aware") is True for row in chosen_v2)
                    / len(chosen_v2)
                ),
                "v3_agreement_stability_aware_ratio": (
                    None
                    if not chosen_v3
                    else sum(row.get("v3_agrees_with_diagnostic_stability_aware") is True for row in chosen_v3)
                    / len(chosen_v3)
                ),
                "mean_route_runtime_s": None if not route_times else sum(route_times) / len(route_times),
                "mean_route_stack_runtime_s": None if not stack_times else sum(stack_times) / len(stack_times),
                "mean_route_rewrite_runtime_s": None if not rewrite_times else sum(rewrite_times) / len(rewrite_times),
                "mean_estimate_part_poses_runtime_s": None
                if not estimate_times
                else sum(estimate_times) / len(estimate_times),
                "mean_infer_joints_runtime_s": None if not infer_times else sum(infer_times) / len(infer_times),
                "mean_evaluate_runtime_s": None if not evaluate_times else sum(evaluate_times) / len(evaluate_times),
                "mean_stability_after_runtime_s": None
                if not stability_times
                else sum(stability_times) / len(stability_times),
            }
        )
    if args.score_version == "v1":
        selected_score = "route_score_no_gt_v1"
    elif args.score_version == "v2":
        selected_score = "route_score_no_gt_v2"
    elif args.score_version == "v3":
        selected_score = "route_score_no_gt_v3"
    else:
        selected_score = "both"

    summary_json = args.summary_json or Path("outputs/flow_tracking_eval/em_lite_routing_summary.json")
    summary_csv = args.summary_csv or summary_json.with_suffix(".csv")
    object_summary_csv = summary_json.with_name(f"{summary_json.stem}_object_timing.csv")
    diagnostic_audit_csv = summary_json.with_name(f"{summary_json.stem}_diagnostic_score_audit.csv")
    v3_failure_csv = summary_json.with_name(f"{summary_json.stem}_v3_failure_analysis_stability_aware.csv")
    debug_indexes = []
    for object_dir in args.object_dirs:
        object_id = object_dir.name
        object_rows = [row for row in all_rows if row.get("object_id") == object_id]
        object_summary = next((summary for summary in object_summaries if summary.get("object_id") == object_id), {})
        debug_index = _write_object_debug_index(
            cwd,
            object_dir,
            object_rows,
            object_summary,
            not bool(args.no_generate_debug_viewers),
            args.dry_run,
        )
        debug_indexes.append(str(debug_index.resolve()))
    _save_json(
        summary_json,
        {
            "score_version": args.score_version,
            "selected_score": selected_score,
            "stability_eval_mode": args.stability_eval_mode,
            "object_summaries": object_summaries,
            "debug_indexes": debug_indexes,
            "routes": all_rows,
        },
    )
    _write_csv(summary_csv, all_rows)
    _write_object_summary_csv(object_summary_csv, object_summaries)
    _write_diagnostic_score_audit_csv(diagnostic_audit_csv, all_rows)
    _write_v3_failure_analysis_csv(v3_failure_csv, all_rows)
    print(
        json.dumps(
            {
                "summary_json": str(summary_json.resolve()),
                "summary_csv": str(summary_csv.resolve()),
                "object_summary_csv": str(object_summary_csv.resolve()),
                "diagnostic_audit_csv": str(diagnostic_audit_csv.resolve()),
                "v3_failure_csv": str(v3_failure_csv.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
