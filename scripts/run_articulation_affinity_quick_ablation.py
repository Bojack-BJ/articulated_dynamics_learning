#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_OBJECT_ROOT = Path("outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask")
DEFAULT_OBJECT_IDS = tuple(f"refrigerator{idx:03d}" for idx in range(38, 43))


def _run(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _module_cmd(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "rgbd_urdf_mvp", *[str(arg) for arg in args]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _case_flags(case: str) -> list[str]:
    if case == "baseline":
        return []
    if case == "articulation_compatible_affinity":
        return ["--articulation-compatible-affinity"]
    if case == "quality_weighted_affinity_plus_articulation":
        return ["--quality-weighted-affinity", "--articulation-compatible-affinity"]
    raise ValueError(f"Unsupported case: {case}")


def _git_version(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _case_config(cwd: Path, case: str, spectral_k: int, edge_ablation: str) -> dict[str, Any]:
    flags = _case_flags(case)
    return {
        "case": case,
        "spectral_k": int(spectral_k),
        "edge_ablation": str(edge_ablation),
        "case_flags": flags,
        "quality_weighted_affinity": "--quality-weighted-affinity" in flags,
        "articulation_compatible_affinity": "--articulation-compatible-affinity" in flags,
        "quality_weighted_affinity_time_only": False,
        "quality_weighted_affinity_edge_prior": False,
        "quality_affinity_min_pair_weight": None,
        "articulation_type_mismatch_penalty": 0.65,
        "articulation_static_mismatch_penalty": 1.0,
        "articulation_min_motion_for_type_penalty": 0.03,
        "articulation_min_confidence_for_type_penalty": 0.75,
        "code_version": _git_version(cwd),
    }


def _should_rerun(
    *,
    required_outputs: list[Path],
    manifest_path: Path,
    expected_config: dict[str, Any],
    force: bool,
    reuse_without_manifest: bool,
) -> tuple[bool, str]:
    if force:
        return True, "force"
    missing = [path.name for path in required_outputs if not path.exists()]
    if missing:
        return True, "missing_outputs:" + ",".join(missing)
    if not manifest_path.exists():
        return (False, "reuse_without_manifest") if reuse_without_manifest else (True, "missing_manifest")
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return True, "invalid_manifest"
    previous = manifest.get("config") if isinstance(manifest, dict) else None
    if previous != expected_config:
        return True, "config_mismatch"
    return False, "reuse_manifest_match"


def _selected_diagnostics(payload: dict[str, Any], spectral_k: int, edge_ablation: str) -> dict[str, Any]:
    for row in payload.get("sweep", []) or []:
        if str(row.get("ablation")) == str(edge_ablation) and int(row.get("k", -1)) == int(spectral_k):
            return row.get("diagnostics") or {}
    return payload


def _needs_quality(case: str) -> bool:
    return case != "baseline"


def _ensure_quality_tracks(cwd: Path, object_tracks: Path, output_dir: Path) -> Path:
    quality_dir = output_dir / "track_quality"
    enriched = quality_dir / "motion_part_tracks_with_quality.json"
    if enriched.exists():
        return enriched
    _run(_module_cmd("compute-track-quality", object_tracks, "--output-dir", quality_dir), cwd)
    return enriched


def _run_one(
    cwd: Path,
    object_dir: Path,
    case: str,
    output_root: Path,
    spectral_k: int,
    edge_ablation: str,
    force: bool,
    reuse_without_manifest: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    object_id = object_dir.name
    case_dir = output_root / case / object_id
    case_dir.mkdir(parents=True, exist_ok=True)
    object_tracks = object_dir / "object_tracks.json"
    if not object_tracks.exists():
        raise FileNotFoundError(f"Missing object tracks: {object_tracks}")

    track_input = _ensure_quality_tracks(cwd, object_tracks, case_dir) if _needs_quality(case) else object_tracks
    motion_tracks = case_dir / "motion_part_tracks_knn.json"
    diagnostics = case_dir / "motion_segmentation_diagnostics_knn.json"
    poses = case_dir / "part_poses_knn.json"
    joints = case_dir / "joint_inference_knn.json"
    evaluation = case_dir / "object_mask_kinematic_evaluation_knn.json"
    evaluation_csv = case_dir / "object_mask_kinematic_evaluation_knn.csv"
    config_manifest = case_dir / "config_manifest.json"
    expected_config = _case_config(cwd, case, spectral_k, edge_ablation)

    rerun, reuse_reason = _should_rerun(
        required_outputs=[motion_tracks, diagnostics, poses, joints, evaluation],
        manifest_path=config_manifest,
        expected_config=expected_config,
        force=force,
        reuse_without_manifest=reuse_without_manifest,
    )
    print(f"[{case}] {object_id} {'rerun' if rerun else 'reuse'} ({reuse_reason})", flush=True)
    if rerun:
        _run(
            _module_cmd(
                "segment-motion-parts",
                track_input,
                "--mode",
                "knn-spectral",
                "--spectral-k",
                str(spectral_k),
                "--edge-ablation",
                edge_ablation,
                "--output-json",
                motion_tracks,
                "--diagnostics-json",
                diagnostics,
                "--skip-base-bridge-checks",
                *_case_flags(case),
            ),
            cwd,
        )
        _run(_module_cmd("estimate-part-poses", motion_tracks, "--method", "tracks", "--output-json", poses), cwd)
        _run(_module_cmd("infer-joints", poses, "--output-json", joints, "--mujoco-prior", "off"), cwd)
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
        )
        config_manifest.write_text(
            json.dumps(
                {
                    "config": expected_config,
                    "outputs": {
                        "motion_tracks": str(motion_tracks),
                        "diagnostics": str(diagnostics),
                        "poses": str(poses),
                        "joints": str(joints),
                        "evaluation": str(evaluation),
                    },
                    "updated_at_unix_s": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    eval_payload = _load_json(evaluation)
    diag_payload = _selected_diagnostics(_load_json(diagnostics), spectral_k, edge_ablation) if diagnostics.exists() else {}
    summary = eval_payload.get("summary") or {}
    edge_summary = _safe_get(diag_payload, "edge_summary") or {}
    component_summary = _safe_get(diag_payload, "graph_component_summary") or {}
    return {
        "case": case,
        "object_id": object_id,
        "runtime_s": time.perf_counter() - start,
        "directed_joint_coverage": summary.get("directed_joint_coverage"),
        "undirected_joint_coverage": summary.get("undirected_joint_coverage"),
        "joint_type_accuracy": summary.get("joint_type_accuracy"),
        "axis_mean_deg": summary.get("axis_angle_error_deg_mean"),
        "pivot_mean_m": summary.get("pivot_error_m_mean"),
        "mean_cluster_purity": summary.get("mean_cluster_purity"),
        "mean_gt_coverage": summary.get("mean_gt_coverage"),
        "largest_cluster_ratio": summary.get("largest_cluster_ratio"),
        "predicted_joint_count": summary.get("predicted_joint_count"),
        "mean_edge_weight": edge_summary.get("mean_edge_weight"),
        "mean_articulation_compatibility": edge_summary.get("mean_articulation_compatibility"),
        "largest_component_ratio": component_summary.get("largest_component_ratio"),
        "motion_tracks": str(motion_tracks),
        "evaluation_json": str(evaluation),
        "diagnostics_json": str(diagnostics),
        "config_manifest": str(config_manifest),
        "reused": not rerun,
        "reuse_reason": reuse_reason,
    }


def _mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(clean) / float(len(clean)) if clean else None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = sorted({str(row["case"]) for row in rows})
    out = []
    keys = [
        "runtime_s",
        "directed_joint_coverage",
        "undirected_joint_coverage",
        "joint_type_accuracy",
        "axis_mean_deg",
        "pivot_mean_m",
        "mean_cluster_purity",
        "mean_gt_coverage",
        "largest_cluster_ratio",
        "predicted_joint_count",
        "mean_edge_weight",
        "mean_articulation_compatibility",
    ]
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        item = {"case": case, "object_count": len(case_rows)}
        for key in keys:
            item[key] = _mean([row.get(key) for row in case_rows])
        out.append(item)
    return out


def _overseg_merge_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        tracks_path = Path(str(row.get("motion_tracks", "")))
        eval_path = Path(str(row.get("evaluation_json", "")))
        if not tracks_path.exists() or not eval_path.exists():
            continue
        try:
            track_payload = _load_json(tracks_path)
            eval_payload = _load_json(eval_path)
        except (OSError, json.JSONDecodeError):
            continue
        clusters = _cluster_track_summary(track_payload)
        overlap_rows = _safe_get(eval_payload, "overlap", "per_cluster") or []
        for item in overlap_rows:
            cluster_id = str(item.get("pred_cluster_id"))
            if cluster_id in clusters:
                clusters[cluster_id]["dominant_gt_part_id"] = item.get("dominant_gt_part_id")
                clusters[cluster_id]["purity"] = item.get("purity")
                clusters[cluster_id]["gt_coverage"] = item.get("coverage")
        cluster_ids = sorted(clusters, key=lambda value: int(value) if str(value).isdigit() else str(value))
        for left_index, left_id in enumerate(cluster_ids):
            for right_id in cluster_ids[left_index + 1:]:
                left = clusters[left_id]
                right = clusters[right_id]
                distance = _distance(left["centroid"], right["centroid"])
                same_dominant_gt = (
                    left.get("dominant_gt_part_id") is not None
                    and left.get("dominant_gt_part_id") == right.get("dominant_gt_part_id")
                )
                compatibility = _mean_pair_articulation_compatibility(left["tracks"], right["tracks"])
                if not same_dominant_gt and distance > 0.25 and (compatibility is None or compatibility < 0.85):
                    continue
                merged_count = int(left["track_count"]) + int(right["track_count"])
                out.append(
                    {
                        "case": row.get("case"),
                        "object_id": row.get("object_id"),
                        "motion_tracks": str(tracks_path),
                        "cluster_a": left_id,
                        "cluster_b": right_id,
                        "dominant_gt_a": left.get("dominant_gt_part_id"),
                        "dominant_gt_b": right.get("dominant_gt_part_id"),
                        "same_dominant_gt": same_dominant_gt,
                        "centroid_distance_m": distance,
                        "mean_articulation_compatibility": compatibility,
                        "merged_track_count": merged_count,
                        "track_count_a": left["track_count"],
                        "track_count_b": right["track_count"],
                        "purity_delta_diagnostic": _merged_purity_delta(left, right),
                        "coverage_gain_diagnostic": _coverage_gain_proxy(left, right),
                    }
                )
    return sorted(
        out,
        key=lambda item: (
            str(item.get("case")),
            str(item.get("object_id")),
            not bool(item.get("same_dominant_gt")),
            -float(item.get("mean_articulation_compatibility") or 0.0),
            -float(item.get("coverage_gain_diagnostic") or 0.0),
            float(item.get("centroid_distance_m") or 0.0),
        ),
    )


def _enrich_overseg_merge_refits(
    cwd: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    max_refits_per_object_case: int,
) -> list[dict[str, Any]]:
    if max_refits_per_object_case <= 0:
        return candidates
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (str(candidate.get("case")), str(candidate.get("object_id")))
        grouped.setdefault(key, []).append(candidate)

    enriched: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for (case, object_id), group in grouped.items():
        refit_root = output_dir / "overseg_merge_refits" / case / object_id
        for candidate in group[:max_refits_per_object_case]:
            selected_ids.add(id(candidate))
            enriched.append(_refit_overseg_merge_candidate(cwd, refit_root, candidate))
    for candidate in candidates:
        if id(candidate) not in selected_ids:
            enriched.append(candidate)
    return sorted(
        enriched,
        key=lambda item: (
            str(item.get("case")),
            str(item.get("object_id")),
            not bool(item.get("same_dominant_gt")),
            item.get("merge_refit_status") != "ok",
            -float(item.get("mean_articulation_compatibility") or 0.0),
            -float(item.get("coverage_gain_diagnostic") or 0.0),
            float(item.get("centroid_distance_m") or 0.0),
        ),
    )


def _refit_overseg_merge_candidate(cwd: Path, refit_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    tracks_path = Path(str(candidate.get("motion_tracks", "")))
    cluster_a = str(candidate.get("cluster_a"))
    cluster_b = str(candidate.get("cluster_b"))
    refit_dir = refit_root / f"merge_{cluster_a}_{cluster_b}"
    refit_dir.mkdir(parents=True, exist_ok=True)
    merged_tracks = refit_dir / "motion_part_tracks_merged.json"
    poses = refit_dir / "part_poses_merged.json"
    joints = refit_dir / "joint_inference_merged.json"
    evaluation = refit_dir / "object_mask_kinematic_evaluation_merged.json"
    evaluation_csv = refit_dir / "object_mask_kinematic_evaluation_merged.csv"
    before_viewer = refit_dir / "viewer_before_merge.html"
    after_viewer = refit_dir / "viewer_after_merge.html"
    try:
        payload = _load_json(tracks_path)
        merged_payload = _merge_track_clusters(payload, cluster_a, cluster_b)
        merged_tracks.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
        _run(_module_cmd("estimate-part-poses", merged_tracks, "--method", "tracks", "--output-json", poses), cwd)
        _run(_module_cmd("infer-joints", poses, "--output-json", joints, "--mujoco-prior", "off"), cwd)
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
        )
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                tracks_path,
                "--output-html",
                before_viewer,
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
        )
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                merged_tracks,
                "--output-html",
                after_viewer,
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
        )
        eval_payload = _load_json(evaluation)
        joint_payload = _load_json(joints)
        summary = eval_payload.get("summary") or {}
        out.update(
            {
                "merge_refit_status": "ok",
                "merge_refit_dir": str(refit_dir),
                "viewer_before_merge": str(before_viewer),
                "viewer_after_merge": str(after_viewer),
                "merged_directed_joint_coverage": summary.get("directed_joint_coverage"),
                "merged_undirected_joint_coverage": summary.get("undirected_joint_coverage"),
                "merged_axis_mean_deg": summary.get("axis_angle_error_deg_mean"),
                "merged_pivot_mean_m": summary.get("pivot_error_m_mean"),
                "merged_mean_cluster_purity": summary.get("mean_cluster_purity"),
                "merged_mean_gt_coverage": summary.get("mean_gt_coverage"),
                "merged_largest_cluster_ratio": summary.get("largest_cluster_ratio"),
                "merged_predicted_joint_count": summary.get("predicted_joint_count"),
                "merged_mean_joint_replay_m": _mean_joint_replay(joint_payload),
            }
        )
    except Exception as exc:  # Diagnostic script: keep other candidates runnable.
        out.update({"merge_refit_status": "error", "merge_refit_error": str(exc), "merge_refit_dir": str(refit_dir)})
    return out


def _merge_track_clusters(payload: dict[str, Any], cluster_a: str, cluster_b: str) -> dict[str, Any]:
    merged = copy.deepcopy(payload)
    merged_id: int | str
    try:
        merged_id = int(cluster_a)
    except ValueError:
        merged_id = cluster_a
    merged_name = f"motion_part_{cluster_a}"
    counts: dict[str, int] = {}
    for track in merged.get("tracks", []) or []:
        if not isinstance(track, dict):
            continue
        part_id = str(track.get("part_id"))
        if part_id == cluster_b:
            track["part_id"] = merged_id
            track["part_name"] = merged_name
            track["merged_from_part_id"] = cluster_b
        counts[str(track.get("part_id"))] = counts.get(str(track.get("part_id")), 0) + 1
    merged["part_track_counts"] = counts
    merged.setdefault("motion_segmentation", {})
    if isinstance(merged["motion_segmentation"], dict):
        merged["motion_segmentation"]["overseg_merge_diagnostic"] = {
            "merged_cluster_a": cluster_a,
            "merged_cluster_b": cluster_b,
            "merged_part_id": merged_id,
        }
    return merged


def _mean_joint_replay(joint_payload: dict[str, Any]) -> float | None:
    values: list[float] = []
    for joint in joint_payload.get("joints", []) or []:
        if not isinstance(joint, dict):
            continue
        comparison = _safe_get(joint, "metrics", "track_model_comparison")
        if not isinstance(comparison, dict):
            continue
        selected = comparison.get("selected_type") or joint.get("joint_type")
        metrics = comparison.get(str(selected)) if selected else None
        if isinstance(metrics, dict):
            value = metrics.get("rmse_m")
            if isinstance(value, (int, float)):
                values.append(float(value))
    return sum(values) / float(len(values)) if values else None


def _cluster_track_summary(track_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for track in track_payload.get("tracks", []) or []:
        if not isinstance(track, dict):
            continue
        cluster_id = str(track.get("part_id", "unknown"))
        entry = clusters.setdefault(cluster_id, {"tracks": [], "points": [], "track_count": 0})
        entry["tracks"].append(track)
        entry["track_count"] += 1
        point = _reference_point(track)
        if point is not None:
            entry["points"].append(point)
    for entry in clusters.values():
        points = entry.get("points") or [[0.0, 0.0, 0.0]]
        entry["centroid"] = [sum(point[axis] for point in points) / float(len(points)) for axis in range(3)]
    return clusters


def _reference_point(track: dict[str, Any]) -> list[float] | None:
    raw = track.get("reference_xyz_world")
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    for sample in track.get("samples", []) or []:
        if not isinstance(sample, dict) or not bool(sample.get("visible", False)):
            continue
        xyz = sample.get("xyz_world")
        if isinstance(xyz, list) and len(xyz) == 3:
            return [float(value) for value in xyz]
    return None


def _mean_pair_articulation_compatibility(left_tracks: list[dict[str, Any]], right_tracks: list[dict[str, Any]]) -> float | None:
    try:
        from rgbd_urdf_mvp.perception.quality_weights import articulation_pair_compatibility
    except Exception:
        return None
    values = []
    for left in left_tracks[:20]:
        for right in right_tracks[:20]:
            values.append(
                articulation_pair_compatibility(
                    left,
                    right,
                    static_mismatch_penalty=1.0,
                )
            )
    return sum(values) / float(len(values)) if values else None


def _merged_purity_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_purity = left.get("purity")
    right_purity = right.get("purity")
    if not isinstance(left_purity, (int, float)) or not isinstance(right_purity, (int, float)):
        return None
    before = (
        float(left_purity) * int(left["track_count"]) + float(right_purity) * int(right["track_count"])
    ) / float(max(1, int(left["track_count"]) + int(right["track_count"])))
    after = before if left.get("dominant_gt_part_id") == right.get("dominant_gt_part_id") else min(float(left_purity), float(right_purity))
    return after - before


def _coverage_gain_proxy(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left.get("dominant_gt_part_id") != right.get("dominant_gt_part_id"):
        return None
    left_cov = left.get("gt_coverage")
    right_cov = right.get("gt_coverage")
    if not isinstance(left_cov, (int, float)) or not isinstance(right_cov, (int, float)):
        return None
    return min(1.0, float(left_cov) + float(right_cov)) - max(float(left_cov), float(right_cov))


def _distance(a: list[float], b: list[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick segmentation-only articulation affinity ablation.")
    parser.add_argument(
        "object_dirs",
        nargs="*",
        type=Path,
        default=[DEFAULT_OBJECT_ROOT / object_id for object_id in DEFAULT_OBJECT_IDS],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OBJECT_ROOT / "articulation_affinity_quick_ablation")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["baseline", "articulation_compatible_affinity", "quality_weighted_affinity_plus_articulation"],
        choices=["baseline", "articulation_compatible_affinity", "quality_weighted_affinity_plus_articulation"],
    )
    parser.add_argument("--spectral-k", type=int, default=5)
    parser.add_argument("--edge-ablation", choices=["A", "B", "C"], default="B")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Rerun even when all artifacts and config manifest exist")
    parser.add_argument(
        "--reuse-without-manifest",
        action="store_true",
        help="Reuse pre-existing artifacts that do not have a config_manifest.json",
    )
    parser.add_argument(
        "--max-merge-refits-per-object-case",
        type=int,
        default=2,
        help="Run merge refit/evaluation for this many top over-segmentation candidates per object/case; use 0 to disable.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(case, object_dir.expanduser().resolve()) for case in args.cases for object_dir in args.object_dirs]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = [
            executor.submit(
                _run_one,
                cwd,
                object_dir,
                case,
                output_dir,
                args.spectral_k,
                args.edge_ablation,
                bool(args.force),
                bool(args.reuse_without_manifest),
            )
            for case, object_dir in tasks
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{row['case']}] {row['object_id']} done "
                f"coverage={row.get('directed_joint_coverage')} purity={row.get('mean_cluster_purity')} "
                f"runtime={row['runtime_s']:.1f}s",
                flush=True,
            )

    rows = sorted(rows, key=lambda row: (str(row["case"]), str(row["object_id"])))
    aggregate = _aggregate(rows)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    aggregate_csv = output_dir / "aggregate.csv"
    merge_candidates = _enrich_overseg_merge_refits(
        cwd,
        output_dir,
        _overseg_merge_candidates(rows),
        int(args.max_merge_refits_per_object_case),
    )
    merge_candidates_csv = output_dir / "overseg_merge_candidates.csv"
    summary_json.write_text(
        json.dumps(
            {
                "rows": rows,
                "aggregate": aggregate,
                "overseg_merge_candidates_csv": str(merge_candidates_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(summary_csv, rows)
    _write_csv(aggregate_csv, aggregate)
    _write_csv(merge_candidates_csv, merge_candidates)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "aggregate_csv": str(aggregate_csv), "overseg_merge_candidates_csv": str(merge_candidates_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
