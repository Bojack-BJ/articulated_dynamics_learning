#!/usr/bin/env python3
"""Summarize neural Relation Head training runtime for paper reporting."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    rows = []
    for summary_path in sorted(run_dir.glob("seed_*/training_summary.json")):
        payload = json.loads(summary_path.read_text())
        runtime = payload.get("runtime", {})
        history = payload.get("history", [])
        command_path = summary_path.parent / "command.json"
        command = json.loads(command_path.read_text()) if command_path.exists() else []
        started = command_path.stat().st_mtime if command_path.exists() else summary_path.stat().st_mtime
        ended = summary_path.stat().st_mtime
        rows.append({
            "seed": summary_path.parent.name.removeprefix("seed_"),
            "epochs": len(history),
            "object_batch_size": _argument(command, "--object-batch-size"),
            "trainer_time_s": runtime.get("total_training_time_s"),
            "mean_epoch_time_s": runtime.get("mean_epoch_time_s"),
            "end_to_end_job_wall_s": ended - started,
            "preload_and_final_eval_s": (
                ended - started - float(runtime.get("total_training_time_s", 0.0))
            ),
            "peak_cuda_memory_gib": (
                float(runtime.get("peak_cuda_memory_bytes", 0)) / 2**30
            ),
        })
    if not rows:
        raise FileNotFoundError(f"No training summaries under {run_dir}")
    with (run_dir / "training_runtime_per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hardware = {
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]).splitlines()[0],
        "cpu": command_output(["bash", "-lc", "lscpu | grep 'Model name' | head -1"]),
        "platform": platform.platform(),
        "parallel_seeds": len(rows),
    }
    aggregate = {
        "schema": "track2art-relation-training-runtime-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "hardware": hardware,
        "implementation": {
            "architecture": "vector_neuron_type20",
            "slot_backbone": "frozen CoTracker-only slot checkpoint",
            "object_batch_size": rows[0]["object_batch_size"],
            "so3_augmentation": "Haar uniform quaternion, probability 1.0",
            "paired_consistency_forward": False,
            "analytic_geometry": "batched weighted Kabsch and relative-SE(3) proposals",
            "feature_generation_included": False,
            "evaluation_included_in_job_wall": True,
        },
        "aggregate": {
            "seed_count": len(rows),
            "mean_trainer_time_s": mean(float(row["trainer_time_s"]) for row in rows),
            "mean_epoch_time_s": mean(float(row["mean_epoch_time_s"]) for row in rows),
            "mean_end_to_end_job_wall_s": mean(float(row["end_to_end_job_wall_s"]) for row in rows),
            "parallel_wall_time_s": max(float(row["end_to_end_job_wall_s"]) for row in rows),
            "mean_peak_cuda_memory_gib": mean(float(row["peak_cuda_memory_gib"]) for row in rows),
        },
        "per_seed": rows,
    }
    (run_dir / "training_runtime.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    a = aggregate["aggregate"]
    lines = [
        "# Relation Head Training Runtime",
        "",
        f"Three seeds were trained in parallel on {hardware['gpu']}.",
        f"Mean trainer time: {a['mean_trainer_time_s'] / 3600:.2f} h; "
        f"parallel end-to-end wall time: {a['parallel_wall_time_s'] / 3600:.2f} h; "
        f"mean epoch time: {a['mean_epoch_time_s']:.1f} s; "
        f"mean peak CUDA memory: {a['mean_peak_cuda_memory_gib']:.2f} GiB.",
        "",
        "Timing includes Relation Head optimization and validation each epoch. "
        "End-to-end job wall additionally includes feature preload and final test evaluation. "
        "It excludes RGB-D recording and CoTracker feature extraction, which use the existing frozen input features.",
    ]
    (run_dir / "training_runtime.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


def _argument(command: list[str], flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
