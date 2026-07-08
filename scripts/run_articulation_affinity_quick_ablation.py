#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    if not all(path.exists() for path in [motion_tracks, diagnostics, poses, joints, evaluation]):
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
    args = parser.parse_args()

    cwd = Path.cwd()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(case, object_dir.expanduser().resolve()) for case in args.cases for object_dir in args.object_dirs]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = [
            executor.submit(_run_one, cwd, object_dir, case, output_dir, args.spectral_k, args.edge_ablation)
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
    summary_json.write_text(json.dumps({"rows": rows, "aggregate": aggregate}, indent=2), encoding="utf-8")
    _write_csv(summary_csv, rows)
    _write_csv(aggregate_csv, aggregate)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "aggregate_csv": str(aggregate_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
