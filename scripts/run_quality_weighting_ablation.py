#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OBJECT_ROOT = Path("outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask")
DEFAULT_OBJECT_IDS = tuple(f"refrigerator{idx:03d}" for idx in range(38, 46))


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_configs(case_set: str, min_pair_weight: float) -> list[tuple[str, list[str]]]:
    base_cases = {
        "baseline": [],
        "quality_weighted_part_poses": ["--quality-weighted-part-poses"],
        "quality_weighted_replay": ["--quality-weighted-replay"],
        "quality_weighted_affinity": ["--quality-weighted-affinity"],
        "quality_weighted_part_poses_plus_replay": ["--quality-weighted-part-poses", "--quality-weighted-replay"],
        "quality_weighted_part_poses_plus_affinity": ["--quality-weighted-part-poses", "--quality-weighted-affinity"],
        "quality_weighted_all": ["--quality-weighted-part-poses", "--quality-weighted-replay", "--quality-weighted-affinity"],
        "quality_weighted_affinity_time_only": ["--quality-weighted-affinity-time-only"],
        "quality_weighted_affinity_edge_prior": ["--quality-weighted-affinity-edge-prior"],
        "quality_weighted_affinity_edge_prior_clamped": [
            "--quality-weighted-affinity-edge-prior",
            "--quality-affinity-min-pair-weight",
            str(min_pair_weight),
        ],
        "articulation_compatible_affinity": ["--articulation-compatible-affinity"],
        "quality_weighted_affinity_plus_articulation": [
            "--quality-weighted-affinity",
            "--articulation-compatible-affinity",
        ],
    }
    if case_set == "focused":
        names = ["baseline", "quality_weighted_part_poses"]
    elif case_set == "combination":
        names = [
            "baseline",
            "quality_weighted_part_poses",
            "quality_weighted_replay",
            "quality_weighted_affinity",
            "quality_weighted_part_poses_plus_replay",
            "quality_weighted_part_poses_plus_affinity",
            "quality_weighted_all",
        ]
    elif case_set == "affinity-variants":
        names = [
            "baseline",
            "quality_weighted_affinity",
            "quality_weighted_affinity_time_only",
            "quality_weighted_affinity_edge_prior",
            "quality_weighted_affinity_edge_prior_clamped",
            "articulation_compatible_affinity",
            "quality_weighted_affinity_plus_articulation",
        ]
    elif case_set == "articulation":
        names = [
            "baseline",
            "articulation_compatible_affinity",
            "quality_weighted_affinity_plus_articulation",
        ]
    elif case_set == "all":
        names = list(base_cases)
    else:
        names = ["baseline", "quality_weighted_part_poses", "quality_weighted_replay", "quality_weighted_affinity"]
    return [(name, base_cases[name]) for name in names]


def _summary_row(case_name: str, summary_json: Path, runtime_s: float) -> dict[str, Any]:
    payload = _load_json(summary_json)
    object_summaries = payload.get("object_summaries", [])
    rows = payload.get("routes", payload.get("rows", []))
    inferred_runtime_s = sum(
        float(item.get("object_runtime_s", 0.0))
        for item in object_summaries
        if isinstance(item.get("object_runtime_s"), (int, float))
    )
    chosen_v3 = [row for row in rows if row.get("chosen_by_no_gt_v3")]
    return {
        "case": case_name,
        "runtime_s": runtime_s if runtime_s > 0.0 else inferred_runtime_s,
        "object_count": len(object_summaries),
        "route_row_count": len(rows),
        "unmatched_route_count": sum(int(item.get("unmatched_cluster_count", 0)) for item in object_summaries),
        "mean_cluster_purity": _mean_summary(object_summaries, "baseline_mean_cluster_purity"),
        "mean_gt_coverage": _mean_summary(object_summaries, "baseline_mean_gt_coverage"),
        "largest_cluster_ratio": _mean_summary(object_summaries, "baseline_largest_cluster_ratio"),
        "axis_mean": _mean_row(chosen_v3, "candidate_axis_mean_diagnostic"),
        "pivot_mean": _mean_row(chosen_v3, "candidate_pivot_mean_diagnostic"),
        "joint_stability_score": _mean_row(chosen_v3, "joint_stability_score"),
        "v3_agreement_with_stability_aware_diagnostic": _mean_summary(
            object_summaries, "v3_agreement_stability_aware_ratio"
        ),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_json.with_suffix(".csv")),
    }


def _object_rows(case_name: str, summary_json: Path) -> list[dict[str, Any]]:
    payload = _load_json(summary_json)
    rows = payload.get("routes", payload.get("rows", []))
    chosen_v3_by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("chosen_by_no_gt_v3"):
            chosen_v3_by_object.setdefault(str(row.get("object_id")), []).append(row)
    out = []
    for item in payload.get("object_summaries", []):
        object_id = str(item.get("object_id"))
        chosen = chosen_v3_by_object.get(object_id, [])
        out.append(
            {
                "case": case_name,
                "object_id": object_id,
                "runtime_s": item.get("object_runtime_s"),
                "unmatched_route_count": item.get("unmatched_cluster_count"),
                "mean_cluster_purity": item.get("baseline_mean_cluster_purity"),
                "mean_gt_coverage": item.get("baseline_mean_gt_coverage"),
                "largest_cluster_ratio": item.get("baseline_largest_cluster_ratio"),
                "axis_mean": _mean_row(chosen, "candidate_axis_mean_diagnostic"),
                "pivot_mean": _mean_row(chosen, "candidate_pivot_mean_diagnostic"),
                "joint_stability_score": _mean_row(chosen, "joint_stability_score"),
                "v3_agreement_with_stability_aware_diagnostic": item.get("v3_agreement_stability_aware_ratio"),
            }
        )
    return out


def _safe_flag(baseline: dict[str, Any] | None, weighted: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None or weighted is None:
        return {"part_pose_weighting_safe": None, "reason": "missing baseline or weighted case"}
    coverage_drop = _delta(weighted.get("mean_gt_coverage"), baseline.get("mean_gt_coverage"))
    agreement_drop = _delta(
        weighted.get("v3_agreement_with_stability_aware_diagnostic"),
        baseline.get("v3_agreement_with_stability_aware_diagnostic"),
    )
    axis_delta = _delta(weighted.get("axis_mean"), baseline.get("axis_mean"))
    pivot_delta = _delta(weighted.get("pivot_mean"), baseline.get("pivot_mean"))
    stability_delta = _delta(weighted.get("joint_stability_score"), baseline.get("joint_stability_score"))
    safe = (
        coverage_drop is not None
        and coverage_drop >= -0.01
        and (agreement_drop is None or agreement_drop >= -1e-9)
        and (axis_delta is None or axis_delta <= 0.25)
        and (pivot_delta is None or pivot_delta <= 0.005)
        and (stability_delta is None or stability_delta >= -0.01)
    )
    return {
        "part_pose_weighting_safe": bool(safe),
        "mean_gt_coverage_delta": coverage_drop,
        "v3_agreement_delta": agreement_drop,
        "axis_mean_delta": axis_delta,
        "pivot_mean_delta": pivot_delta,
        "joint_stability_score_delta": stability_delta,
    }


def _delta(after: Any, before: Any) -> float | None:
    if not isinstance(after, (int, float)) or not isinstance(before, (int, float)):
        return None
    return float(after) - float(before)


def _coverage_breakdown(before_summary: Path, after_summary: Path, after_label: str) -> list[dict[str, Any]]:
    before_payload = _load_json(before_summary)
    after_payload = _load_json(after_summary)
    before_by_object = {str(item.get("object_id")): item for item in before_payload.get("object_summaries", [])}
    after_by_object = {str(item.get("object_id")): item for item in after_payload.get("object_summaries", [])}
    out: list[dict[str, Any]] = []
    for object_id, before in sorted(before_by_object.items()):
        after = after_by_object.get(object_id)
        if after is None:
            continue
        before_per_gt = {str(row.get("gt_part_id")): row for row in before.get("baseline_overlap_per_gt") or []}
        after_per_gt = {str(row.get("gt_part_id")): row for row in after.get("baseline_overlap_per_gt") or []}
        before_clusters = before.get("baseline_overlap_per_cluster") or []
        after_clusters = after.get("baseline_overlap_per_cluster") or []
        gt_ids = sorted(set(before_per_gt) | set(after_per_gt), key=lambda value: int(value) if value.isdigit() else value)
        for gt_id in gt_ids:
            before_row = before_per_gt.get(gt_id, {})
            after_row = after_per_gt.get(gt_id, {})
            before_cov = _as_float(before_row.get("coverage"))
            after_cov = _as_float(after_row.get("coverage"))
            before_covering = _covering_cluster_count(before, gt_id)
            after_covering = _covering_cluster_count(after, gt_id)
            before_pred = before_row.get("best_pred_cluster_id")
            after_pred = after_row.get("best_pred_cluster_id")
            before_purity = _cluster_purity(before_clusters, before_pred)
            after_purity = _cluster_purity(after_clusters, after_pred)
            out.append(
                {
                    "case": after_label,
                    "object_id": object_id,
                    "gt_part_id": gt_id,
                    "gt_part_name": _gt_part_name(before, after, gt_id),
                    "best_cluster_coverage_before": before_cov,
                    "best_cluster_coverage_after": after_cov,
                    "coverage_delta": None if before_cov is None or after_cov is None else after_cov - before_cov,
                    "cluster_count_covering_gt_before": before_covering,
                    "cluster_count_covering_gt_after": after_covering,
                    "fragmentation_delta": after_covering - before_covering,
                    "dominant_cluster_purity_before": before_purity,
                    "dominant_cluster_purity_after": after_purity,
                }
            )
    return out


def _graph_diagnostic_rows(case_name: str, summary_json: Path) -> list[dict[str, Any]]:
    payload = _load_json(summary_json)
    out = []
    for item in payload.get("object_summaries", []):
        edge = item.get("baseline_segmentation_edge_summary") or {}
        component = item.get("baseline_segmentation_graph_component_summary") or {}
        out.append(
            {
                "case": case_name,
                "object_id": item.get("object_id"),
                "weighted_graph_edge_count": edge.get("weighted_graph_edge_count"),
                "edge_count": edge.get("edge_count"),
                "mean_edge_weight": edge.get("mean_edge_weight"),
                "median_edge_weight": edge.get("median_edge_weight"),
                "low_weight_edge_ratio": edge.get("low_weight_edge_ratio"),
                "effective_pair_observation_count_mean": edge.get("effective_pair_observation_count_mean"),
                "effective_pair_observation_count_median": edge.get("effective_pair_observation_count_median"),
                "component_count_before_spectral": component.get("component_count"),
                "largest_component_size_before_spectral": component.get("largest_component_size"),
                "largest_component_ratio_before_spectral": component.get("largest_component_ratio"),
            }
        )
    return out


def _covering_cluster_count(summary: dict[str, Any], gt_id: str) -> int:
    matrix = summary.get("baseline_overlap_matrix") or {}
    return sum(1 for row in matrix.values() if int(row.get(str(gt_id), 0) or 0) > 0)


def _cluster_purity(per_cluster: list[dict[str, Any]], pred_cluster_id: Any) -> float | None:
    if pred_cluster_id is None:
        return None
    for row in per_cluster:
        if str(row.get("pred_cluster_id")) == str(pred_cluster_id):
            return _as_float(row.get("purity"))
    return None


def _gt_part_name(before: dict[str, Any], after: dict[str, Any], gt_id: str) -> str:
    return str((before.get("gt_part_names") or {}).get(gt_id) or (after.get("gt_part_names") or {}).get(gt_id) or gt_id)


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _mean_summary(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _mean_row(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quality-weighting ablations for EM-lite routing diagnostics.")
    parser.add_argument(
        "object_dirs",
        nargs="*",
        type=Path,
        default=[DEFAULT_OBJECT_ROOT / object_id for object_id in DEFAULT_OBJECT_IDS],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OBJECT_ROOT / "quality_weighting_ablation")
    parser.add_argument(
        "--case-set",
        choices=["standard", "focused", "combination", "affinity-variants", "articulation", "all"],
        default="standard",
        help="Which quality-weighting cases to run.",
    )
    parser.add_argument("--spectral-k", type=int, default=5)
    parser.add_argument("--edge-ablation", choices=["A", "B", "C"], default="B")
    parser.add_argument("--min-quality-weight", type=float, default=0.2)
    parser.add_argument("--quality-affinity-min-pair-weight", type=float, default=0.2)
    parser.add_argument("--enable-stability-lite", action="store_true")
    parser.add_argument("--stability-eval-mode", choices=["all", "topk", "borderline"], default="topk")
    parser.add_argument("--stability-top-k", type=int, default=2)
    parser.add_argument("--no-generate-debug-viewers", action="store_true")
    parser.add_argument("--rerun-baseline", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run experiments; only rebuild the aggregate table from existing per-case summary.json files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_object_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    case_summary_paths: dict[str, Path] = {}
    for case_name, case_flags in _case_configs(args.case_set, args.quality_affinity_min_pair_weight):
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_json = case_dir / "summary.json"
        if args.aggregate_only:
            if summary_json.exists():
                rows.append(_summary_row(case_name, summary_json, 0.0))
                per_object_rows.extend(_object_rows(case_name, summary_json))
                graph_rows.extend(_graph_diagnostic_rows(case_name, summary_json))
                case_summary_paths[case_name] = summary_json
            continue
        cmd = [
            sys.executable,
            "scripts/run_em_lite_routing_diagnostic.py",
            *[str(path) for path in args.object_dirs],
            "--spectral-k",
            str(args.spectral_k),
            "--edge-ablation",
            str(args.edge_ablation),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_json.with_suffix(".csv")),
            "--min-quality-weight",
            str(args.min_quality_weight),
            "--stability-eval-mode",
            str(args.stability_eval_mode),
            "--stability-top-k",
            str(args.stability_top_k),
            *case_flags,
        ]
        if args.enable_stability_lite:
            cmd.append("--enable-stability-lite")
        if args.no_generate_debug_viewers:
            cmd.append("--no-generate-debug-viewers")
        if args.rerun_baseline or case_flags:
            cmd.append("--rerun-baseline")
        start = time.perf_counter()
        _run(cmd, cwd, args.dry_run)
        runtime_s = time.perf_counter() - start
        if summary_json.exists():
            rows.append(_summary_row(case_name, summary_json, runtime_s))
            per_object_rows.extend(_object_rows(case_name, summary_json))
            graph_rows.extend(_graph_diagnostic_rows(case_name, summary_json))
            case_summary_paths[case_name] = summary_json
    safe_summary = _safe_flag(
        next((row for row in rows if row.get("case") == "baseline"), None),
        next((row for row in rows if row.get("case") == "quality_weighted_part_poses"), None),
    )
    coverage_rows: list[dict[str, Any]] = []
    baseline_summary = case_summary_paths.get("baseline")
    if baseline_summary is not None:
        for case_name in [
            "quality_weighted_affinity",
            "quality_weighted_affinity_time_only",
            "quality_weighted_affinity_edge_prior",
            "quality_weighted_affinity_edge_prior_clamped",
            "quality_weighted_part_poses_plus_affinity",
            "quality_weighted_all",
            "articulation_compatible_affinity",
            "quality_weighted_affinity_plus_articulation",
        ]:
            case_path = case_summary_paths.get(case_name)
            if case_path is not None:
                coverage_rows.extend(_coverage_breakdown(baseline_summary, case_path, case_name))
    aggregate_json = output_dir / "quality_weighting_ablation_summary.json"
    aggregate_json.write_text(
        json.dumps(
            {
                "case_set": args.case_set,
                "cases": rows,
                "part_pose_weighting_safety": safe_summary,
                "coverage_breakdown_csv": str(output_dir / "quality_weighting_coverage_breakdown.csv"),
                "graph_diagnostics_csv": str(output_dir / "quality_weighting_graph_diagnostics.csv"),
                "per_object_summary_csv": str(output_dir / "quality_weighting_per_object_summary.csv"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "quality_weighting_ablation_summary.csv", rows)
    _write_csv(output_dir / "quality_weighting_per_object_summary.csv", per_object_rows)
    _write_csv(output_dir / "quality_weighting_graph_diagnostics.csv", graph_rows)
    _write_csv(output_dir / "quality_weighting_coverage_breakdown.csv", coverage_rows)
    print(json.dumps({"quality_weighting_ablation_summary": str(aggregate_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
