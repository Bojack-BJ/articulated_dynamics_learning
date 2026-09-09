#!/usr/bin/env python3
"""Run comparable short Relation Head SO(3)-difficulty ablations in parallel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SETTINGS = (
    ("none", None),
    ("yaw", "yaw"),
    ("limited15", "limited_xyz_15"),
    ("limited30", "limited_xyz_30"),
    ("haar", "uniform_quaternion"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--object-batch-size", type=int, default=8)
    return parser.parse_args()


def command_for(
    args: argparse.Namespace, output_dir: Path, augmentation_mode: str | None
) -> list[str]:
    command = [
        sys.executable, "-m", "rgbd_urdf_mvp", "train-slot-relation-head",
        str(args.manifest), str(args.slot_model),
        "--output-dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--object-batch-size", str(args.object_batch_size),
        "--axis-geometry-branch",
        "--geometry-encoder-type", "track_gru_transformer",
        "--trajectory-samples", "32",
        "--trajectory-hidden-dim", "128",
        "--geometry-max-tracks", "32",
        "--rotation-augmentation-scope", "slot_and_relation_geometry",
        "--unfreeze-slot-backbone",
        "--slot-unfreeze-scope", "decoder",
        "--slot-learning-rate-scale", "0.1",
        "--device", "cuda",
    ]
    if augmentation_mode is not None:
        command.extend([
            "--rotation-augmentation",
            "--rotation-augmentation-probability", "1.0",
            "--rotation-augmentation-mode", augmentation_mode,
        ])
    return command


def main() -> int:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) < len(SETTINGS):
        raise ValueError(f"Need at least {len(SETTINGS)} GPU ids, received {len(gpus)}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    started_at = time.monotonic()
    for gpu, (name, mode) in zip(gpus, SETTINGS):
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_root / f"{name}.log"
        stream = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            command_for(args, output_dir, mode),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        jobs.append((name, mode, gpu, output_dir, log_path, stream, process))
        print(f"launched {name} on GPU {gpu}: {log_path}", flush=True)

    failures = []
    summaries = {}
    for name, mode, gpu, output_dir, log_path, stream, process in jobs:
        return_code = process.wait()
        stream.close()
        summary_path = output_dir / "training_summary.json"
        if return_code != 0 or not summary_path.exists():
            failures.append({
                "name": name, "gpu": gpu, "return_code": return_code,
                "log": str(log_path),
            })
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[name] = {
            "augmentation_mode": mode,
            "gpu": gpu,
            "summary": str(summary_path),
            "test": payload.get("test_metrics", payload.get("test")),
            "runtime": payload.get("runtime"),
        }
        print(f"completed {name} on GPU {gpu}", flush=True)

    report = {
        "epochs": args.epochs,
        "runtime_s": time.monotonic() - started_at,
        "settings": summaries,
        "failures": failures,
    }
    (output_root / "ablation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
