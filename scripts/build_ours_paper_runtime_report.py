#!/usr/bin/env python3
"""Build a paper-ready Ours runtime and implementation report from measured profiles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASES = (
    "cotracker_feature_generation_s",
    "tapip_feature_generation_s",
    "slot_inference_s",
    "learned_slot_relation_inference_s",
    "analytic_part_pose_s",
    "analytic_joint_inference_s",
)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _collect_slot_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    return {
        str(row["object_id"]): {
            "object_id": str(row["object_id"]),
            "category": str(row.get("category", "unknown")),
            "slot_inference_s": _number(
                row.get("slot_inference_runtime_s", row.get("runtime_s"))
            ),
            "slot_runtime_scope": "subprocess wall time; model load included",
            "source": str(path),
        }
        for row in data.get("per_object", [])
        if row.get("method") == "hybrid"
    }


def _merge_runtime_profiles(rows: dict[str, dict[str, Any]], root: Path) -> None:
    if not root.is_dir():
        return
    for path in root.rglob("runtime_profile.json"):
        profile = _load_json(path)
        object_id = str(profile.get("object_id", path.parent.name))
        row = rows.setdefault(object_id, {"object_id": object_id, "category": "unknown"})
        phases = profile.get("phases", {})
        row["slot_inference_s"] = _number(phases.get("slot_subprocess_wall_s")) or row.get(
            "slot_inference_s"
        )
        row["learned_slot_relation_inference_s"] = _number(
            phases.get("learned_slot_relation_subprocess_wall_s")
        )
        row["analytic_part_pose_s"] = _number(
            phases.get("analytic_part_pose_subprocess_wall_s")
        )
        row["analytic_joint_inference_s"] = _number(
            phases.get("analytic_joint_inference_subprocess_wall_s")
        )
        row["backend_profile"] = str(path)
        row["inter_object_workers"] = profile.get("execution", {}).get(
            "inter_object_workers"
        )


def _merge_tracker_profiles(rows: dict[str, dict[str, Any]], root: Path) -> None:
    if not root.is_dir():
        return
    for path in root.rglob("*.runtime.json"):
        profile = _load_json(path)
        scope = str(profile.get("scope", ""))
        if scope not in {
            "cotracker_tracking_feature_generation_and_rgbd_lifting",
            "tapip3d_frozen_encoder_feature_export",
        }:
            continue
        object_id = path.parent.name
        if object_id.startswith("view_"):
            object_id = path.parents[1].name
        row = rows.setdefault(object_id, {"object_id": object_id, "category": "unknown"})
        key = (
            "cotracker_feature_generation_s"
            if scope.startswith("cotracker")
            else "tapip_feature_generation_s"
        )
        # Multiple TAPIP views are sequential for one object unless the artifact says otherwise.
        row[key] = float(row.get(key) or 0.0) + float(profile.get("total_wall_s", 0.0))
        row[f"{key}_profile_count"] = int(row.get(f"{key}_profile_count", 0)) + 1


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for phase in PHASES:
        values = [_number(row.get(phase)) for row in rows]
        measured = [value for value in values if value is not None]
        result.append(
            {
                "phase": phase,
                "measured_objects": len(measured),
                "mean_s": statistics.fmean(measured) if measured else None,
                "median_s": statistics.median(measured) if measured else None,
                "p90_s": (
                    sorted(measured)[min(len(measured) - 1, int(0.9 * len(measured)))]
                    if measured
                    else None
                ),
                "timing_scope": _phase_scope(phase),
            }
        )
    return result


def _phase_scope(phase: str) -> str:
    return {
        "cotracker_feature_generation_s": "tracking + frozen feature extraction + RGB-D lifting",
        "tapip_feature_generation_s": "frozen TAPIP3D encoder feature export",
        "slot_inference_s": "slot segmentation subprocess; checkpoint load included",
        "learned_slot_relation_inference_s": (
            "slot backbone + relation topology/type/axis/pivot; do not add slot inference"
        ),
        "analytic_part_pose_s": "robust track SE(3) pose fitting",
        "analytic_joint_inference_s": "revolute/prismatic analytic model fitting",
    }[phase]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slot-benchmark",
        type=Path,
        default=Path(
            "outputs/partnet_core_v1_training/slot_benchmark_three_methods_test/"
            "benchmark_metrics.json"
        ),
    )
    parser.add_argument("--backend-profile-root", type=Path)
    parser.add_argument("--tracker-profile-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/runtime_comparison_v2/ours"),
    )
    args = parser.parse_args()
    rows = _collect_slot_benchmark(args.slot_benchmark.expanduser().resolve())
    if args.backend_profile_root:
        _merge_runtime_profiles(rows, args.backend_profile_root.expanduser().resolve())
    if args.tracker_profile_root:
        _merge_tracker_profiles(rows, args.tracker_profile_root.expanduser().resolve())
    per_object = sorted(rows.values(), key=lambda row: row["object_id"])
    summary = _summaries(per_object)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "ours_stage_timing_per_object.csv", per_object)
    _write_csv(output / "ours_stage_timing_summary.csv", summary)

    implementation = {
        "timing_policy": {
            "single_object_latency": "one object per process, accelerator synchronization at forward boundaries",
            "batch_throughput": "reported separately using inter-object workers",
            "model_loading": "included unless explicitly stated",
            "viewer_and_metric_evaluation": "excluded from inference latency",
            "warmup": "no warmup in legacy slot measurements",
        },
        "parallelism": {
            "cotracker": "queries batched within each view; views processed sequentially",
            "tapip": "one persistent worker per GPU; object-view jobs sharded across GPUs",
            "slot_and_relation": "one object per subprocess; ThreadPoolExecutor can parallelize objects",
            "analytic_backend": "single-object CPU implementation; objects may run concurrently",
        },
        "algorithm_parameters": {
            "recording": "120 source frames at 15 Hz in the main PartNet setting",
            "cotracker": "frozen tracker; frame stride and max_queries_per_forward stored per artifact",
            "tapip": "frozen UpdateFormer; default num_iters=6, support_grid_size=16, resolution_factor=2",
            "slot_model": "max_slots/hidden_dim/encoder_layers/decoder_layers/heads stored in checkpoint",
            "relation_model": "trajectory_samples/max_tracks/geometry encoder stored in checkpoint",
            "mesh_face_cap": "not applicable to Ours feedforward segmentation/kinematics inference",
            "iterative_optimization_steps": "none for Ours learned feedforward path",
        },
    }
    (output / "ours_runtime_report.json").write_text(
        json.dumps(
            {"per_object": per_object, "summary": summary, "implementation": implementation},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Ours Runtime and Implementation Details",
        "",
        "The legacy benchmark measured only slot segmentation. It must not be cited as "
        "end-to-end latency until tracking/feature and relation-head profiles are present.",
        "",
        "## Stage Timing",
        "",
        "| Stage | Measured N | Mean (s) | Median (s) | P90 (s) | Scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        values = [
            "n/a" if row[key] is None else f"{row[key]:.3f}"
            for key in ("mean_s", "median_s", "p90_s")
        ]
        lines.append(
            f"| {row['phase']} | {row['measured_objects']} | {values[0]} | "
            f"{values[1]} | {values[2]} | {row['timing_scope']} |"
        )
    lines.extend(
        [
            "",
            "## Reporting Rules",
            "",
            "- Learned end-to-end backend latency is the learned slot+relation call; the "
            "standalone slot call is not added because the relation inferencer reruns the slot backbone.",
            "- Feature generation is reported separately and then included in cold end-to-end latency.",
            "- Single-object latency and multi-object throughput are separate quantities.",
            "- Model load is included in current cold-start subprocess measurements.",
            "- Viewer generation and GT metric evaluation are excluded.",
            "",
            "## Parallelism",
            "",
            "- CoTracker batches seed queries within a view; camera views are sequential.",
            "- TAPIP uses one persistent process per GPU and shards object-view jobs across GPUs.",
            "- Slot/relation inference parallelizes independent objects with subprocess workers.",
            "- Ours feedforward path has no Gaussian/mesh optimization iteration count and no mesh face cap.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "objects": len(per_object)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
