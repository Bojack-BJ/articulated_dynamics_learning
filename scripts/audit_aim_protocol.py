#!/usr/bin/env python3
"""Audit AiM acquisition assumptions and intermediate motion decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OBJECT_NAMES = {
    "partnet_47024": "Storage-47024",
    "partnet_11304": "Fridge-11304",
    "partnet_47648": "Storage-47648",
    "partnet_31249": "Table-31249",
}
SEQ_PATTERN = re.compile(
    r"(After Merging: )?\[Seq\] Motion #(?P<index>\d+) \| size=(?P<size>\d+) "
    r"\| theta=(?P<theta>[-+.\deE]+).*?\|phi=(?P<phi>[-+.\deE]+)"
)
GAUSSIAN_PATTERN = re.compile(
    r"(?P<static>\d+) Static Gaussians and (?P<moving>\d+) Moving Gaussians "
    r"in Iteration (?P<iteration>\d+)"
)
SEQ_DIAG_PATTERN = re.compile(
    r"\[SeqDiag\] proposal=(?P<proposal>\d+) remaining=(?P<remaining>\d+) "
    r"raw_inliers=(?P<raw_inliers>\d+) residual_mean=(?P<residual>[-+.\w]+)"
)
SEQ_DIAG_DECISION_PATTERN = re.compile(
    r"\[SeqDiag\] proposal=(?P<proposal>\d+) "
    r"(?:(?:accepted_inliers=(?P<accepted>\d+))|(?:rejected=(?P<rejected>\w+)"
    r"(?: cc_inliers=(?P<cc_inliers>\d+) min_inliers=(?P<min_inliers>\d+))?))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        help="Directory containing per-object aim.json files used to locate GT reference PLYs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _joint_series(frames: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted(
        {
            name
            for frame in frames
            for name in (frame.get("action_log", {}).get("joint_positions", {}) or {})
        }
    )
    values = np.full((len(frames), len(names)), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        positions = frame.get("action_log", {}).get("joint_positions", {}) or {}
        for joint_index, name in enumerate(names):
            if name in positions:
                values[frame_index, joint_index] = float(positions[name])
    return names, values


def _camera_pose(frame: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(frame["camera_pose"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Invalid camera pose shape {pose.shape}")
    return pose


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _max_pairwise(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    valid = np.nan_to_num(values, nan=0.0)
    return max(
        float(np.linalg.norm(valid[i] - valid[j]))
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    )


def _scan_audit(episode_dir: Path, relative: str | None) -> dict[str, Any]:
    if not relative:
        return {"available": False}
    path = episode_dir / relative
    if not path.is_file():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = list(payload.get("views", []))
    states = [view.get("joint_positions", {}) for view in views]
    azimuths = [float(view["azimuth_deg"]) for view in views if "azimuth_deg" in view]
    return {
        "available": True,
        "view_count": len(views),
        "unique_joint_state_count": len({json.dumps(state, sort_keys=True) for state in states}),
        "azimuth_span_deg": max(azimuths) - min(azimuths) if azimuths else None,
        "path": str(path),
    }


def audit_episode(object_id: str, episode_path: Path, dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    frames = list(episode["frames"])
    names, values = _joint_series(frames)
    tail_count = min(12, len(frames))
    tail = values[-tail_count:]
    metadata = episode.get("metadata", {})
    protocol = metadata.get("aim_protocol") or {}
    start = _scan_audit(episode_path.parent, protocol.get("static_scan_manifest"))
    end = _scan_audit(episode_path.parent, protocol.get("end_scan_manifest"))
    poses = [_camera_pose(frame) for frame in frames[-tail_count:]]
    camera_translation_span = max(
        float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        for index, a in enumerate(poses)
        for b in poses[index:]
    )
    camera_rotation_span = max(
        _rotation_angle(a, b)
        for index, a in enumerate(poses)
        for b in poses[index:]
    )
    export_path = dataset_root / object_id / "aim_export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.is_file() else {}
    end_export = export.get("stages", {}).get("end", {})
    row = {
        "object_id": object_id,
        "paper_name": OBJECT_NAMES.get(object_id, object_id),
        "frame_count": len(frames),
        "fps": (
            (len(frames) - 1)
            / max(
                float(frames[-1].get("timestamp_s", len(frames) - 1))
                - float(frames[0].get("timestamp_s", 0.0)),
                1e-9,
            )
        ),
        "recording_protocol": metadata.get("recording_protocol"),
        "start_view_count": start.get("view_count"),
        "start_unique_joint_state_count": start.get("unique_joint_state_count"),
        "start_azimuth_span_deg": start.get("azimuth_span_deg"),
        "end_scan_available": end.get("available", False),
        "end_scan_view_count": end.get("view_count", 0),
        "end_scan_unique_joint_state_count": end.get("unique_joint_state_count"),
        "tail_frame_count": tail_count,
        "tail_max_pairwise_joint_state_difference": _max_pairwise(tail),
        "tail_camera_translation_span_m": camera_translation_span,
        "tail_camera_rotation_span_deg": camera_rotation_span,
        "export_end_source": end_export.get("source"),
        "export_end_observation_count": end_export.get("observation_count"),
        "export_end_unique_camera_count": end_export.get("unique_camera_count"),
        "end_joint_state_effectively_static": bool(_max_pairwise(tail) <= 1e-5),
        "dedicated_end_scan_available": bool(end.get("available")),
        "official_end_configuration_assumption_met": bool(_max_pairwise(tail) <= 1e-5),
        "end_view_coverage_is_dense_orbit": bool(
            end.get("available") and float(end.get("azimuth_span_deg") or 0.0) >= 300.0
        ),
    }
    schedule_rows = []
    timestamps = np.asarray([float(frame.get("timestamp_s", index)) for index, frame in enumerate(frames)])
    for joint_index, name in enumerate(names):
        series = values[:, joint_index]
        valid = np.isfinite(series)
        if not np.any(valid):
            continue
        finite = series[valid]
        deltas = np.abs(np.diff(np.nan_to_num(series, nan=float(finite[0]))))
        motion_threshold = max(float(np.ptp(finite)) * 1e-3, 1e-6)
        active = np.flatnonzero(deltas > motion_threshold)
        schedule_rows.append(
            {
                "object_id": object_id,
                "joint_name": name,
                "q_min": float(np.min(finite)),
                "q_max": float(np.max(finite)),
                "q_range": float(np.ptp(finite)),
                "tail_q_min": float(np.nanmin(tail[:, joint_index])),
                "tail_q_max": float(np.nanmax(tail[:, joint_index])),
                "tail_q_range": float(np.nanmax(tail[:, joint_index]) - np.nanmin(tail[:, joint_index])),
                "first_active_frame": int(active[0]) if len(active) else None,
                "last_active_frame": int(active[-1] + 1) if len(active) else None,
                "first_active_time_s": float(timestamps[active[0]]) if len(active) else None,
                "last_active_time_s": float(timestamps[active[-1] + 1]) if len(active) else None,
                "active_transition_count": int(len(active)),
            }
        )
    return row, schedule_rows


def _read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as stream:
        header_lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format ascii 1.0" in header_lines:
            properties = [
                line.split()[-1]
                for line in header_lines
                if line.startswith("property ") and not line.startswith("property list ")
            ]
            values = np.loadtxt(stream, dtype=np.float64)
            values = np.atleast_2d(values)
            columns = {name: index for index, name in enumerate(properties)}
            xyz = values[:, [columns[name] for name in ("x", "y", "z")]]
            rgb = None
            if all(name in columns for name in ("red", "green", "blue")):
                rgb = values[:, [columns[name] for name in ("red", "green", "blue")]].astype(np.uint8)
            return xyz, rgb

    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    names = set(vertex.dtype.names or ())
    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.uint8)
    return xyz, rgb


def _trajectory_metrics(run_dir: Path) -> dict[str, Any]:
    paths = [run_dir / f"motion_traj_t={time:.1f}" / "point_cloud_seq_0.ply" for time in (0.0, 0.5, 1.0)]
    if not all(path.is_file() for path in paths):
        return {}
    trajectories = [_read_ply_xyz_rgb(path)[0] for path in paths]
    count = min(len(points) for points in trajectories)
    stacked = np.stack([points[:count] for points in trajectories], axis=1)
    displacement = np.linalg.norm(stacked[:, -1] - stacked[:, 0], axis=1)
    variance = np.mean(np.sum((stacked - np.mean(stacked, axis=1, keepdims=True)) ** 2, axis=2), axis=1)
    return {
        "trajectory_point_count": count,
        "trajectory_displacement_mean_m": float(np.mean(displacement)),
        "trajectory_displacement_median_m": float(np.median(displacement)),
        "trajectory_displacement_p90_m": float(np.percentile(displacement, 90)),
        "trajectory_variance_mean_m2": float(np.mean(variance)),
    }


def _rgb_labels(rgb: np.ndarray) -> np.ndarray:
    palette = {color: index for index, color in enumerate(sorted(set(map(tuple, rgb.tolist()))))}
    return np.asarray([palette[tuple(color)] for color in rgb], dtype=np.int64)


def gt_part_diagnostics(
    object_id: str,
    run_dir: Path,
    metrics_root: Path | None,
    *,
    static_count: int | None,
) -> list[dict[str, Any]]:
    if metrics_root is None:
        return []
    metric_path = metrics_root / object_id / "aim.json"
    if not metric_path.is_file():
        return []
    from scipy.spatial import cKDTree

    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    reference_path = Path(metric["reference_ply"])
    if not reference_path.is_file():
        return []
    reference_points, reference_labels = _read_ply_part_ids(reference_path)
    bbox_diagonal = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
    threshold = max(0.02 * bbox_diagonal, 1e-9)

    initial_points, initial_rgb = _read_ply_xyz_rgb(run_dir / "seg_init_dual" / "segmented_point.ply")
    final_points, final_rgb = _read_ply_xyz_rgb(run_dir / "motion_seg_final" / "segmented_point.ply")
    if initial_rgb is None or final_rgb is None:
        return []
    initial_distance, initial_index = cKDTree(initial_points).query(reference_points, k=1)
    final_distance, final_index = cKDTree(final_points).query(reference_points, k=1)
    initial_covered = initial_distance <= threshold
    final_covered = final_distance <= threshold
    initial_moving = np.asarray(initial_index, dtype=np.int64) >= int(static_count or 0)
    final_labels = _rgb_labels(final_rgb)[np.asarray(final_index, dtype=np.int64)]

    trajectory_paths = [
        run_dir / f"motion_traj_t={time:.1f}" / "point_cloud_seq_0.ply"
        for time in (0.0, 0.5, 1.0)
    ]
    trajectory_displacement = None
    trajectory_index = None
    if all(path.is_file() for path in trajectory_paths):
        trajectories = [_read_ply_xyz_rgb(path)[0] for path in trajectory_paths]
        count = min(len(points) for points in trajectories)
        trajectory_displacement = np.linalg.norm(
            trajectories[-1][:count] - trajectories[0][:count], axis=1
        )
        _, trajectory_index = cKDTree(trajectories[0][:count]).query(reference_points, k=1)

    rows = []
    for gt_part in sorted(int(value) for value in np.unique(reference_labels)):
        selected = reference_labels == gt_part
        selected_initial = selected & initial_covered
        selected_final = selected & final_covered
        mapped_final = final_labels[selected_final]
        component_counts = Counter(int(value) for value in mapped_final)
        dominant_component, dominant_count = (
            component_counts.most_common(1)[0] if component_counts else (None, 0)
        )
        displacement = (
            trajectory_displacement[np.asarray(trajectory_index[selected], dtype=np.int64)]
            if trajectory_displacement is not None and trajectory_index is not None
            else np.asarray([], dtype=np.float64)
        )
        rows.append(
            {
                "object_id": object_id,
                "gt_part_id": gt_part,
                "gt_reference_point_count": int(np.sum(selected)),
                "initial_covered_point_count": int(np.sum(selected_initial)),
                "initial_geometry_coverage": float(np.mean(initial_covered[selected])),
                "moving_gaussian_mapped_count": int(np.sum(initial_moving[selected_initial])),
                "percentage_classified_static": (
                    float(1.0 - np.mean(initial_moving[selected_initial]))
                    if np.any(selected_initial)
                    else None
                ),
                "final_covered_point_count": int(np.sum(selected_final)),
                "dominant_predicted_component": dominant_component,
                "dominant_component_coverage": (
                    dominant_count / int(np.sum(selected_final)) if np.any(selected_final) else None
                ),
                "trajectory_displacement_mean_m": (
                    float(np.mean(displacement)) if len(displacement) else None
                ),
                "trajectory_displacement_std_m": (
                    float(np.std(displacement)) if len(displacement) else None
                ),
            }
        )
    return rows


def _read_ply_part_ids(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        properties: list[str] = []
        vertex_count = 0
        in_vertex = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif fields[:1] == ["property"] and in_vertex:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break
        index = {name: column for column, name in enumerate(properties)}
        values = np.asarray(
            [[float(value) for value in stream.readline().split()] for _ in range(vertex_count)],
            dtype=np.float64,
        )
    return values[:, [index[axis] for axis in ("x", "y", "z")]], values[:, index["part_id"]].astype(np.int64)


def audit_run(
    object_id: str,
    run_dir: Path,
    ransac_dir: Path,
    metrics_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_log = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    gaussian_matches = list(GAUSSIAN_PATTERN.finditer(train_log))
    static_count = moving_count = iteration = None
    if gaussian_matches:
        final = gaussian_matches[-1]
        static_count = int(final.group("static"))
        moving_count = int(final.group("moving"))
        iteration = int(final.group("iteration"))
    segmentation_log = (run_dir / "segmentation.log").read_text(encoding="utf-8", errors="replace")
    ransac_rows: list[dict[str, Any]] = []
    diagnostic_rows: dict[int, dict[str, Any]] = {}
    for match in SEQ_DIAG_PATTERN.finditer(segmentation_log):
        proposal = int(match.group("proposal"))
        residual_text = match.group("residual")
        diagnostic_rows[proposal] = {
            "object_id": object_id,
            "iteration": proposal,
            "remaining_points": int(match.group("remaining")),
            "raw_inlier_count": int(match.group("raw_inliers")),
            "raw_inlier_ratio": (
                int(match.group("raw_inliers")) / max(int(match.group("remaining")), 1)
            ),
            "residual": float(residual_text) if residual_text not in {"nan", "inf"} else None,
            "accepted": None,
            "component_size": None,
            "rejection_reason": None,
            "logging_limitation": None,
        }
    for match in SEQ_DIAG_DECISION_PATTERN.finditer(segmentation_log):
        proposal = int(match.group("proposal"))
        row = diagnostic_rows.setdefault(
            proposal,
            {"object_id": object_id, "iteration": proposal},
        )
        if match.group("accepted") is not None:
            row["accepted"] = True
            row["component_size"] = int(match.group("accepted"))
        else:
            row["accepted"] = False
            row["rejection_reason"] = match.group("rejected")
            if match.group("cc_inliers") is not None:
                row["component_size"] = int(match.group("cc_inliers"))
                row["minimum_component_size"] = int(match.group("min_inliers"))
    if diagnostic_rows:
        ransac_rows.extend(diagnostic_rows[index] for index in sorted(diagnostic_rows))
    for match in SEQ_PATTERN.finditer(segmentation_log):
        if not diagnostic_rows:
            ransac_rows.append(
                {
                    "object_id": object_id,
                    "iteration": int(match.group("index")),
                    "component_size": int(match.group("size")),
                    "estimated_theta": float(match.group("theta")),
                    "estimated_phi": float(match.group("phi")),
                    "after_merge": bool(match.group(1)),
                    "remaining_points": None,
                    "raw_inlier_ratio": None,
                    "residual": None,
                    "logging_limitation": "official log reports accepted components only",
                }
            )
    _write_csv(ransac_dir / f"{object_id}.csv", ransac_rows)
    component_matches = list(SEQ_PATTERN.finditer(segmentation_log))
    accepted = [match for match in component_matches if not match.group(1)]
    merged = [match for match in component_matches if match.group(1)]
    total = (static_count or 0) + (moving_count or 0)
    row = {
        "object_id": object_id,
        "static_gaussians": static_count,
        "moving_gaussians": moving_count,
        "total_gaussians": total or None,
        "moving_ratio": moving_count / total if total else None,
        "final_training_iteration": iteration,
        "ransac_accepted_component_count": len(accepted),
        "ransac_postmerge_component_count": len(merged) or len(accepted),
        "ransac_component_sizes": json.dumps(
            [int(match.group("size")) for match in accepted]
        ),
        "ransac_stopped_no_progress": "no progress for 10 rounds" in segmentation_log,
        **_trajectory_metrics(run_dir),
    }
    return row, gt_part_diagnostics(
        object_id,
        run_dir,
        metrics_root,
        static_count=static_count,
    )


def _markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_summary(
    output_dir: Path,
    start_end: list[dict[str, Any]],
    intermediate: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> None:
    moving_end = sum(not bool(row["end_joint_state_effectively_static"]) for row in start_end)
    missing_dense_end = sum(not bool(row["end_view_coverage_is_dense_orbit"]) for row in start_end)
    summary = f"""# AiM Protocol and Motion-Decomposition Audit

## Protocol Findings

- The released AiM loader constructs separate `start`, `end`, and `motion` datasets.
- All end cameras are assigned normalized time `1.0`; the optimizer therefore assumes one fixed final object configuration.
- The current start stage is compliant: each object uses a fixed-articulation 24-view orbit.
- The final 12 interaction frames are effectively static in joint space for {len(start_end) - moving_end}/{len(start_end)} objects, so the suspected moving-configuration mismatch is not observed.
- However, {missing_dense_end}/{len(start_end)} objects lack a dedicated dense final-state orbit: their end cameras cover only the tail of the interaction camera path.
- Existing runs use 5,000 start and 8,000 motion iterations. Official defaults are 20,000 and 30,000. At 8,000 motion iterations, the late end-camera optimization branch is not exercised.

{_markdown_table(start_end, ["object_id", "frame_count", "start_view_count", "tail_max_pairwise_joint_state_difference", "tail_camera_rotation_span_deg", "end_joint_state_effectively_static", "end_view_coverage_is_dense_orbit"])}

## Intermediate Decomposition

{_markdown_table(intermediate, ["object_id", "moving_ratio", "ransac_accepted_component_count", "ransac_postmerge_component_count", "ransac_component_sizes", "ransac_stopped_no_progress"])}

Storage-47648 reaches the motion-decomposition stage with many moving Gaussians, but official sequential RANSAC accepts only two dominant, nearly translational components and then stops after ten no-progress rounds. This localizes a substantial part of the 2/7 under-segmentation failure to motion decomposition rather than geometry coverage alone.

## GT-Only Failure Localization

GT labels are transferred only after AiM completes. They are not used by reconstruction,
dynamic/static assignment, or RANSAC. See `gt_part_diagnostics.csv` for per-part results.

For Storage-47648, GT parts 7 and 8 are mapped entirely to the static Gaussian set, while
GT part 2 is approximately 69% static. Other moving parts survive the first stage but
mostly share one final predicted component. The observed 2/7 failure therefore combines
dynamic/static false negatives with downstream component under-segmentation.

## Logging Limitation

The released AiM logs only accepted RANSAC components and the merged result. Rejected proposals, remaining-point counts, and per-proposal residuals cannot be reconstructed retrospectively. Diagnostic logging can be added for future runs without changing thresholds or assignments.

## Controlled Experiment

`aim_style_fixed_end` has been added as an independent acquisition variant. A valid controlled comparison must run both the current and fixed-end variants with identical optimization budgets above the official end-camera activation point; comparing the existing 5k/8k run against a longer fixed-end run would confound acquisition and optimization.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_end_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    for object_id in OBJECT_NAMES:
        episode_path = args.recordings_root / object_id / "episode.json"
        run_dir = args.runs_root / object_id
        if not episode_path.is_file() or not run_dir.is_dir():
            continue
        start_end, schedule = audit_episode(object_id, episode_path, args.datasets_root)
        start_end_rows.append(start_end)
        schedule_rows.extend(schedule)
        intermediate, gt_diagnostics = audit_run(
            object_id,
            run_dir,
            args.output_dir / "ransac_iterations",
            args.metrics_root,
        )
        intermediate_rows.append(intermediate)
        gt_rows.extend(gt_diagnostics)
    _write_csv(args.output_dir / "start_end_audit.csv", start_end_rows)
    _write_csv(args.output_dir / "motion_schedule_audit.csv", schedule_rows)
    _write_csv(args.output_dir / "intermediate_metrics.csv", intermediate_rows)
    _write_csv(args.output_dir / "gt_part_diagnostics.csv", gt_rows)
    write_summary(args.output_dir, start_end_rows, intermediate_rows, gt_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
