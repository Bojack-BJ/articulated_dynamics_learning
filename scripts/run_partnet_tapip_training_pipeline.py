#!/usr/bin/env python3
"""Generate PartNet TAPIP artifacts, then train TAPIP-only and hybrid slot models."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--tapip-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-output-root", type=Path, required=True)
    parser.add_argument("--generation-gpus", default="0,2,3,4,5")
    parser.add_argument("--resolution-factor", type=float, default=2.0)
    parser.add_argument("--tapip-training-gpu", default="0")
    parser.add_argument("--hybrid-training-gpu", default="2")
    parser.add_argument("--cotracker-manifest", type=Path, default=None)
    parser.add_argument("--cotracker-slot-model", type=Path, default=None)
    parser.add_argument("--relation-output-root", type=Path, default=None)
    parser.add_argument("--cotracker-relation-gpu", default="4")
    parser.add_argument("--tapip-relation-gpu", default="5")
    parser.add_argument("--hybrid-relation-gpu", default="6")
    parser.add_argument("--relation-frozen-epochs", type=int, default=50)
    parser.add_argument("--relation-finetune-epochs", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minimum-objects", type=int, default=600)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run(command: list[str], *, log_path: Path, gpu: str | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _manifest_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream, delimiter="\t"))


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.training_output_root = args.training_output_root.resolve()
    generation = [
        args.python, "scripts/run_partnet_tapip_batch.py", str(args.catalog), str(args.recording_root),
        "--tapip-root", str(args.tapip_root), "--checkpoint", str(args.checkpoint),
        "--output-root", str(args.output_root), "--gpus", args.generation_gpus,
        "--resolution-factor", str(max(0.25, float(args.resolution_factor))),
        "--views", "3", "--cpu-workers", "16", "--python", args.python,
    ]
    if args.resume:
        generation.append("--resume")
    _run(generation, log_path=args.output_root / "pipeline_generation.log")

    manifests = {
        "tapip": args.output_root / "tapip_learning_manifest.tsv",
        "hybrid": args.output_root / "hybrid_learning_manifest.tsv",
    }
    counts = {name: _manifest_count(path) for name, path in manifests.items()}
    if min(counts.values()) < args.minimum_objects:
        raise RuntimeError(f"Refusing to train with incomplete manifests: {counts}")

    def train(name: str, gpu: str) -> None:
        output = args.training_output_root / f"{name}_slots"
        if args.resume and (output / "training_summary.json").is_file():
            return
        _run(
            [
                args.python, "-m", "rgbd_urdf_mvp", "train-motion-part-slots", str(manifests[name]),
                "--output-dir", str(output), "--max-slots", "16", "--epochs", str(args.epochs),
                "--object-batch-size", "16", "--data-loader-workers", "16",
                "--rigid-loss-weight", "0.2", "--dice-loss-weight", "0.5",
                "--pairwise-loss-weight", "0.35", "--device", "cuda",
            ],
            log_path=args.training_output_root / f"{name}_training.log",
            gpu=gpu,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train, "tapip", args.tapip_training_gpu),
            executor.submit(train, "hybrid", args.hybrid_training_gpu),
        ]
        for future in futures:
            future.result()
    if (args.cotracker_manifest is None) != (args.cotracker_slot_model is None):
        raise ValueError("Provide both --cotracker-manifest and --cotracker-slot-model to train relation heads.")
    if args.cotracker_manifest is not None:
        relation_root = (args.relation_output_root or args.training_output_root / "relation_heads").resolve()
        relation_root.mkdir(parents=True, exist_ok=True)
        methods_tsv = relation_root / "three_methods.tsv"
        with methods_tsv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "manifest", "slot_model", "gpu"], delimiter="\t")
            writer.writeheader()
            writer.writerows([
                {
                    "method": "cotracker_only",
                    "manifest": str(args.cotracker_manifest.resolve()),
                    "slot_model": str(args.cotracker_slot_model.resolve()),
                    "gpu": args.cotracker_relation_gpu,
                },
                {
                    "method": "tapip_only",
                    "manifest": str(manifests["tapip"]),
                    "slot_model": str((args.training_output_root / "tapip_slots" / "motion_part_slots.pt").resolve()),
                    "gpu": args.tapip_relation_gpu,
                },
                {
                    "method": "hybrid",
                    "manifest": str(manifests["hybrid"]),
                    "slot_model": str((args.training_output_root / "hybrid_slots" / "motion_part_slots.pt").resolve()),
                    "gpu": args.hybrid_relation_gpu,
                },
            ])
        relation_command = [
            args.python, "scripts/run_three_relation_training.py", str(methods_tsv),
            "--output-root", str(relation_root),
            "--frozen-epochs", str(max(1, args.relation_frozen_epochs)),
            "--finetune-epochs", str(max(1, args.relation_finetune_epochs)),
            "--python", args.python,
        ]
        if args.resume:
            relation_command.append("--resume")
        _run(relation_command, log_path=relation_root / "pipeline.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
