#!/usr/bin/env python3
"""Build the scope-aware aligned runtime comparison used by the paper."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path("outputs/external_baseline_suite_v1")
OUT = ROOT / "runtime_comparison_v3"


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_runtime_s": statistics.fmean(values),
        "median_runtime_s": statistics.median(values),
        "p90_runtime_s": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min_runtime_s": ordered[0],
        "max_runtime_s": ordered[-1],
    }


def load_aim_successes() -> dict[str, dict]:
    """Load the final successful attempt per object from the saved AiM runs."""
    successful: dict[str, dict] = {}
    for path in (
        Path("outputs/aim_baseline_extended_v1/run.log"),
        Path("outputs/aim_baseline_extended_v1/resource_retry.log"),
    ):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "success":
                successful[row["object_id"]] = row
    return successful


def main() -> None:
    with (OUT / "ours/timing_per_object_complete.csv").open(newline="") as handle:
        ours = list(csv.DictReader(handle))
    with (OUT / "ours/timing_cotracker_only_complete.csv").open(newline="") as handle:
        cotracker_only = list(csv.DictReader(handle))
    with (ROOT / "artipoint_shared_track_backend_aligned20_v1/per_object.csv").open(newline="") as handle:
        artipoint = {row["object_id"]: row for row in csv.DictReader(handle)}
    legacy = json.loads((ROOT / "runtime_comparison_v2/all_methods/runtime_summary.json").read_text())
    aim = load_aim_successes()

    rows = []

    def append(method: str, values: list[float], scope: str, outputs: str, hardware: str) -> None:
        rows.append({
            "method": method,
            "measured_count": len(values),
            **stats(values),
            "timing_scope": scope,
            "outputs": outputs,
            "hardware": hardware,
        })

    append(
        "Track2Art CoTracker frontend",
        [float(row["cotracker_feature_s"]) for row in ours],
        "three views, 120 source frames at 15 Hz, stride 4, 30 tracked timesteps; cold per-object subprocess including model loading",
        "CoTracker trajectories + appearance features",
        "1x NVIDIA A100 80GB",
    )
    append(
        "Track2Art CoTracker-only (main)",
        [float(row["cotracker_only_total_s"]) for row in cotracker_only],
        "CoTracker extraction followed by the CoTracker-only full-slot relation checkpoint; cold per-object subprocesses including model loading",
        "part segmentation + graph + joint type/axis/pivot",
        "1x NVIDIA A100 80GB",
    )
    append(
        "Track2Art CoTracker-only learned backend",
        [float(row["cotracker_only_slot_relation_s"]) for row in cotracker_only],
        "precomputed CoTracker trajectories/features; CoTracker-only slot backbone + relation heads; model loading included",
        "part segmentation + graph + joint type/axis/pivot",
        "1x NVIDIA A100 80GB",
    )
    append(
        "Track2Art TAPIP frontend",
        [float(row["tapip_persistent_three_view_s"]) for row in ours],
        "three-view persistent-worker service time; model loading excluded; reconstructed from original generation logs",
        "TAPIP 3D trajectories/features",
        "1x NVIDIA A100 80GB",
    )
    append(
        "Track2Art Hybrid (parallel frontends)",
        [float(row["hybrid_parallel_frontends_total_s"]) for row in ours],
        "CoTracker and TAPIP run concurrently; learned slot+relation follows; model loading included except TAPIP persistent worker",
        "part segmentation + graph + joint type/axis/pivot",
        "2x NVIDIA A100 80GB per object",
    )
    append(
        "Track2Art Hybrid (sequential frontends)",
        [float(row["hybrid_sequential_total_s"]) for row in ours],
        "CoTracker + TAPIP + learned slot/relation, sequential wall-time sum",
        "part segmentation + graph + joint type/axis/pivot",
        "1x NVIDIA A100 80GB equivalent sequential schedule",
    )
    append(
        "Track2Art Hybrid learned backend only",
        [float(row["relation_s"]) for row in ours],
        "precomputed trajectories/features; slot backbone + relation heads; model loading included",
        "part segmentation + graph + joint type/axis/pivot",
        "1x NVIDIA A100 80GB",
    )
    append(
        "ArtiPoint backend only",
        [float(artipoint[row["object_id"]]["runtime_s"]) for row in ours],
        "precomputed Track2Art CoTracker trajectories; filtering + DBSCAN + single screw fitting",
        "one dominant joint only; no complete part segmentation",
        "1x NVIDIA A100 80GB plus CPU backend",
    )
    append(
        "ArtiPoint + shared CoTracker",
        [float(row["cotracker_feature_s"]) + float(artipoint[row["object_id"]]["runtime_s"]) for row in ours],
        "Track2Art CoTracker extraction + unchanged ArtiPoint backend",
        "one dominant joint only; no complete part segmentation",
        "1x NVIDIA A100 80GB",
    )

    for method in ("reart", "dta", "artgs"):
        values = [
            float(row["runtime_s"])
            for row in legacy["per_object"]
            if row["method"] == method and row.get("runtime_s") is not None
        ]
        append(
            method.upper() if method != "reart" else "ReArt",
            values,
            "official/native inference" if method == "reart" else "official end-to-end optimization run",
            "native method outputs",
            "remote CUDA GPU; see baseline implementation audit",
        )

    append(
        "AiM (aligned protocol)",
        [float(row["runtime_s"]) for row in aim.values()],
        "5k canonical/start reconstruction + 8k motion optimization + official sequential RANSAC; final successful attempt per object",
        "Gaussian reconstruction + part segmentation + joint geometry",
        "1x remote CUDA GPU; per-object GPU assignment recorded in source JSONL",
    )
    append(
        "AiM reconstruction",
        [float(row["train_runtime_s"]) for row in aim.values()],
        "5k canonical/start reconstruction + 8k motion optimization",
        "static/dynamic Gaussian representation",
        "1x remote CUDA GPU; per-object GPU assignment recorded in source JSONL",
    )
    append(
        "AiM sequential RANSAC",
        [float(row["segmentation_runtime_s"]) for row in aim.values()],
        "official motion segmentation and merge on the trained Gaussian representation",
        "motion components + joint hypotheses",
        "1x remote CUDA GPU; per-object GPU assignment recorded in source JSONL",
    )

    # These durations are recovered from complete native logs on the shared
    # development filesystem. VideoArtGS uses the first/last timestamp in each
    # 20k training log. PARIS uses the Lightning run launch timestamp and final
    # process timestamp. The intentionally small N is reported rather than
    # extrapolating from file modification times for the remaining objects.
    append(
        "VideoArtGS optimization",
        [918.0, 927.0, 841.0, 810.0],
        "native 20k optimization only; excludes VGGT/TAPIP3D preprocessing and render/export",
        "Gaussian parts + GT-inventory-assisted joint parameters",
        "1x remote CUDA GPU; recovered from four complete timestamped logs",
    )
    append(
        "PARIS",
        [870.0, 867.0, 896.0],
        "native two-state optimization; two-part/one-joint objects only; launch-to-final-process timestamp",
        "one static + one moving part and one joint",
        "1x remote CUDA GPU; recovered from three complete timestamped logs",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runtime_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "runtime_summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Scope-aware Runtime Comparison V3",
        "",
        "Runtime is measured on the aligned 20-object subset where available. Viewer generation and metric evaluation are excluded. Methods retain native input protocols, so these measurements document practical latency rather than identical-kernel benchmarking.",
        "",
        "| Method | N | Mean (s) | Median (s) | P90 (s) | Output/scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['measured_count']} | {row['mean_runtime_s']:.2f} | "
            f"{row['median_runtime_s']:.2f} | {row['p90_runtime_s']:.2f} | {row['outputs']}; {row['timing_scope']} |"
        )
    lines.extend([
        "",
        "Track2Art CoTracker-only is the current main configuration. Its total is `CoTracker extraction + CoTracker-only slot/relation`; standalone slot time is not added because relation inference reruns the slot backbone. Hybrid parallel latency uses `max(CoTracker, TAPIP) + Hybrid slot/relation` and is retained as an ablation rather than the main method.",
        "",
        "CoTracker is measured as cold per-object inference, while TAPIP is measured as warm persistent-worker service time. The Hybrid totals preserve those practical deployment measurements; they are not a hardware-normalized kernel benchmark. The observed 8-GPU CoTracker batch makespan for 20 objects was 141.45 s and is reported only as throughput, not per-object latency.",
        "",
        "ArtiPoint is easier in this adapter experiment: it returns one dominant joint and no complete part decomposition or graph. Its backend latency should therefore not be interpreted as equivalent-output latency.",
        "",
        "AiM timing is the aligned 5k-start + 8k-motion protocol, not the exact-paper 20k/30k reproduction. VideoArtGS currently reports optimization time only because its retained logs do not time VGGT/TAPIP3D preprocessing. PARIS is restricted to the native two-part/one-joint scope. VideoArtGS and PARIS use measured subsets (N=4 and N=3); no missing times are inferred from artifact modification times.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
