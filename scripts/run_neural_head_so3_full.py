#!/usr/bin/env python3
"""Run the selected SO(3)-equivariant Relation Head on the full main split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="20260831,20260832,20260833")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--object-batch-size", type=int, default=32)
    parser.add_argument(
        "--paired-consistency", action="store_true",
        help="Enable the redundant source/rotated paired forward for ablation only.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, seed: int, output: Path) -> list[str]:
    return [
        args.python, "-m", "rgbd_urdf_mvp", "train-slot-relation-head",
        str(args.manifest), str(args.slot_model), "--output-dir", str(output),
        "--epochs", str(args.epochs), "--object-batch-size", str(args.object_batch_size),
        "--seed", str(seed), "--device", "cuda", "--learning-rate", "3e-4",
        "--axis-geometry-branch", "--axis-head-type", "vector_neuron",
        "--joint-type-loss-weight", "2.0", "--rotation-augmentation",
        "--rotation-augmentation-probability", "1.0",
        "--rotation-augmentation-mode", "uniform_quaternion",
        "--axis-equivariance-loss-weight", "1.0" if args.paired_consistency else "0.0",
        "--axis-line-equivariance-loss-weight", "0.5" if args.paired_consistency else "0.0",
        "--edge-consistency-loss-weight", "0.1" if args.paired_consistency else "0.0",
        "--type-consistency-loss-weight", "0.1" if args.paired_consistency else "0.0",
    ]


def run(args: argparse.Namespace, seed: int, device: str) -> dict[str, object]:
    output = args.output_dir / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    cmd = command(args, seed, output)
    (output / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    if args.dry_run:
        return {"seed": seed, "device": device, "status": "dry_run"}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
    with (output / "train.log").open("w") as log:
        completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"seed": seed, "device": device, "status": "ok" if completed.returncode == 0 else "failed"}


def main() -> int:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, devices[index % len(devices)]) for index, seed in enumerate(seeds)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        results = list(executor.map(lambda job: run(args, *job), jobs))
    (args.output_dir / "run_summary.json").write_text(
        json.dumps({"jobs": results}, indent=2) + "\n"
    )
    print(json.dumps({"jobs": results}, indent=2))
    return int(any(row["status"] == "failed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
