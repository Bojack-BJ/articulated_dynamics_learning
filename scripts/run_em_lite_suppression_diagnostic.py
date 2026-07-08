#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _module_cmd(cwd: Path, *args: str | Path) -> list[str]:
    del cwd
    return [sys.executable, "-m", "rgbd_urdf_mvp", *[str(arg) for arg in args]]


def _track_part_id(track: dict[str, Any]) -> int | None:
    value = track.get("part_id")
    if value is None:
        return None
    return int(value)


def _summarize_eval(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    summary = dict(data.get("summary") or {})
    unmatched = [
        row
        for row in data.get("per_joint", [])
        if not bool(row.get("matched")) and row.get("child_part_id") is not None
    ]
    summary["unmatched_predicted_joint_count"] = len(unmatched)
    return summary


def _build_suppression_map(evaluation_json: Path) -> tuple[dict[int, int], list[dict[str, Any]]]:
    evaluation = _load_json(evaluation_json)
    mapping: dict[int, int] = {}
    suppressed: list[dict[str, Any]] = []
    for row in evaluation.get("per_joint", []):
        if bool(row.get("matched")):
            continue
        child = row.get("child_part_id")
        parent = row.get("parent_part_id")
        if child is None or parent is None or int(child) == int(parent):
            continue
        child_id = int(child)
        parent_id = int(parent)
        mapping[child_id] = parent_id
        suppressed.append(
            {
                "joint_name": row.get("name"),
                "child_part_id": child_id,
                "child_name": row.get("child_name"),
                "parent_part_id": parent_id,
                "parent_name": row.get("parent_name"),
                "predicted_joint_type": row.get("predicted_joint_type"),
                "failure_reason_guess": row.get("failure_reason_guess"),
                "child_cluster_debug": row.get("child_cluster_debug"),
            }
        )
    return mapping, suppressed


def _resolve_mapping(mapping: dict[int, int], part_id: int) -> int:
    seen: set[int] = set()
    current = part_id
    while current in mapping and current not in seen:
        seen.add(current)
        current = int(mapping[current])
    return current


def _rewrite_tracks(
    input_tracks: Path,
    output_tracks: Path,
    evaluation_json: Path,
) -> dict[str, Any]:
    payload = _load_json(input_tracks)
    mapping, suppressed = _build_suppression_map(evaluation_json)
    part_names = {
        int(part_id): info.get("name", f"motion_part_{part_id}")
        for part_id, info in (payload.get("part_track_counts") or {}).items()
    }

    reassigned_counts: Counter[str] = Counter()
    for track in payload.get("tracks", []):
        part_id = _track_part_id(track)
        if part_id is None:
            continue
        new_id = _resolve_mapping(mapping, part_id)
        if new_id == part_id:
            continue
        reassigned_counts[f"{part_id}->{new_id}"] += 1
        track["part_id"] = new_id
        track["part_name"] = part_names.get(new_id, f"motion_part_{new_id}")
        track["em_lite_source_part_id"] = part_id
        track["em_lite_source_part_name"] = part_names.get(part_id, f"motion_part_{part_id}")

    counts: Counter[int] = Counter()
    for track in payload.get("tracks", []):
        part_id = _track_part_id(track)
        if part_id is not None:
            counts[part_id] += 1
    payload["part_track_counts"] = {
        str(part_id): {"name": part_names.get(part_id, f"motion_part_{part_id}"), "count": count}
        for part_id, count in sorted(counts.items())
    }
    payload["em_lite"] = {
        "method": "extra_part_suppression",
        "suppression_rule": "merge unmatched predicted child clusters into their predicted parent cluster",
        "source_tracks": str(input_tracks),
        "source_evaluation": str(evaluation_json),
        "suppressed_joints": suppressed,
        "reassigned_track_counts": dict(sorted(reassigned_counts.items())),
    }
    _save_json(output_tracks, payload)
    return payload["em_lite"]


def _run_articulation_stack(cwd: Path, tracks_json: Path, output_dir: Path, label: str, dry_run: bool) -> dict[str, Path]:
    poses = output_dir / f"part_poses_{label}.json"
    joints = output_dir / f"joint_inference_{label}.json"
    eval_json = output_dir / f"object_mask_kinematic_evaluation_{label}.json"
    eval_csv = output_dir / f"object_mask_kinematic_evaluation_{label}.csv"
    _run(_module_cmd(cwd, "estimate-part-poses", tracks_json, "--method", "tracks", "--output-json", poses), cwd, dry_run)
    _run(
        _module_cmd(cwd, "infer-joints", poses, "--output-json", joints, "--mujoco-prior", "off"),
        cwd,
        dry_run,
    )
    _run(
        _module_cmd(
            cwd,
            "evaluate-object-mask-kinematics",
            joints,
            "--part-poses",
            poses,
            "--output-json",
            eval_json,
            "--output-csv",
            eval_csv,
        ),
        cwd,
        dry_run,
    )
    return {"poses": poses, "joints": joints, "evaluation": eval_json, "evaluation_csv": eval_csv}


def _copy_if_same_source(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _run_one(cwd: Path, object_dir: Path, spectral_k: int, edge_ablation: str, dry_run: bool) -> dict[str, Any]:
    object_id = object_dir.name
    output_dir = object_dir / "em_lite"
    baseline_dir = output_dir / "baseline"
    suppressed_dir = output_dir / "extra_part_suppression"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    suppressed_dir.mkdir(parents=True, exist_ok=True)

    object_tracks = object_dir / "object_tracks.json"
    if not object_tracks.exists():
        raise FileNotFoundError(f"Missing object tracks for {object_id}: {object_tracks}")

    baseline_tracks = baseline_dir / "motion_part_tracks_knn.json"
    diagnostics = baseline_dir / "motion_segmentation_diagnostics_knn.json"
    _run(
        _module_cmd(
            cwd,
            "segment-motion-parts",
            object_tracks,
            "--mode",
            "knn-spectral",
            "--spectral-k",
            str(spectral_k),
            "--edge-ablation",
            edge_ablation,
            "--output-json",
            baseline_tracks,
            "--diagnostics-json",
            diagnostics,
        ),
        cwd,
        dry_run,
    )
    baseline_artifacts = _run_articulation_stack(cwd, baseline_tracks, baseline_dir, "knn", dry_run)

    suppressed_tracks = suppressed_dir / "motion_part_tracks_em_lite.json"
    em_lite_summary = _rewrite_tracks(baseline_tracks, suppressed_tracks, baseline_artifacts["evaluation"])
    _save_json(suppressed_dir / "reassignment_summary.json", em_lite_summary)
    suppressed_artifacts = _run_articulation_stack(cwd, suppressed_tracks, suppressed_dir, "em_lite", dry_run)

    baseline_summary = _summarize_eval(baseline_artifacts["evaluation"])
    suppressed_summary = _summarize_eval(suppressed_artifacts["evaluation"])
    result = {
        "object_id": object_id,
        "baseline": baseline_summary,
        "em_lite": suppressed_summary,
        "suppressed_joint_count": len(em_lite_summary.get("suppressed_joints", [])),
        "reassigned_track_counts": em_lite_summary.get("reassigned_track_counts", {}),
        "paths": {
            "baseline_evaluation": str(baseline_artifacts["evaluation"]),
            "em_lite_evaluation": str(suppressed_artifacts["evaluation"]),
            "reassignment_summary": str(suppressed_dir / "reassignment_summary.json"),
        },
    }
    _save_json(output_dir / "em_lite_comparison.json", result)
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "object_id",
        "suppressed_joint_count",
        "baseline_predicted_joint_count",
        "em_lite_predicted_joint_count",
        "baseline_directed_joint_coverage",
        "em_lite_directed_joint_coverage",
        "baseline_undirected_joint_coverage",
        "em_lite_undirected_joint_coverage",
        "baseline_joint_type_accuracy",
        "em_lite_joint_type_accuracy",
        "baseline_axis_angle_error_deg_mean",
        "em_lite_axis_angle_error_deg_mean",
        "baseline_pivot_error_m_mean",
        "em_lite_pivot_error_m_mean",
        "baseline_unmatched_predicted_joint_count",
        "em_lite_unmatched_predicted_joint_count",
        "baseline_largest_cluster_ratio",
        "em_lite_largest_cluster_ratio",
        "baseline_mean_cluster_purity",
        "em_lite_mean_cluster_purity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    flat = {"object_id": result["object_id"], "suppressed_joint_count": result["suppressed_joint_count"]}
    for prefix in ["baseline", "em_lite"]:
        summary = result.get(prefix) or {}
        for key in [
            "predicted_joint_count",
            "directed_joint_coverage",
            "undirected_joint_coverage",
            "joint_type_accuracy",
            "axis_angle_error_deg_mean",
            "pivot_error_m_mean",
            "unmatched_predicted_joint_count",
            "largest_cluster_ratio",
            "mean_cluster_purity",
        ]:
            flat[f"{prefix}_{key}"] = summary.get(key)
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-step EM-lite extra-part suppression diagnostics.")
    parser.add_argument("object_dirs", nargs="+", type=Path, help="Object evaluation directories containing object_tracks.json")
    parser.add_argument("--spectral-k", type=int, default=5, help="Fixed K for KNN spectral baseline")
    parser.add_argument("--edge-ablation", default="B", choices=["A", "B", "C"], help="KNN spectral edge ablation")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional aggregate JSON output")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional aggregate CSV output")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    cwd = Path.cwd()
    results = [_run_one(cwd, path, args.spectral_k, args.edge_ablation, args.dry_run) for path in args.object_dirs]
    flat_rows = [_flatten_result(result) for result in results]
    summary_json = args.summary_json or Path("outputs/flow_tracking_eval/em_lite_suppression_summary.json")
    summary_csv = args.summary_csv or summary_json.with_suffix(".csv")
    _save_json(summary_json, {"results": results})
    _write_summary_csv(summary_csv, flat_rows)
    print(json.dumps({"summary_json": str(summary_json.resolve()), "summary_csv": str(summary_csv.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
