#!/usr/bin/env python3
"""Run frozen then decoder-finetuned relation-head training in parallel."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods_tsv", type=Path, help="Columns: method, manifest, slot_model, gpu")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument(
        "--full-slot-finetune",
        action="store_true",
        help="Also fine-tune the full slot backbone from the frozen relation checkpoint.",
    )
    parser.add_argument("--full-slot-epochs", type=int, default=25)
    parser.add_argument(
        "--frozen-checkpoint-root",
        type=Path,
        help="Optional existing relation-head root supplying phase1 frozen checkpoints to full-slot-only runs.",
    )
    parser.add_argument(
        "--full-slot-only",
        action="store_true",
        help="Run only the independent full-slot fine-tuning phase; frozen checkpoints must exist.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_methods(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"method", "manifest", "slot_model", "gpu"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Methods TSV requires columns: {sorted(required)}")
    return rows


def run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_method(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    method = row["method"]
    method_root = args.output_root / method
    frozen_root = method_root / "phase1_frozen"
    finetune_root = method_root / "phase2_decoder_finetune"
    frozen_source_root = (
        args.frozen_checkpoint_root / method / "phase1_frozen"
        if args.frozen_checkpoint_root is not None
        else frozen_root
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    env["CUDA_VISIBLE_DEVICES"] = row["gpu"]
    started = time.perf_counter()
    common = [
        args.python, "-m", "rgbd_urdf_mvp", "train-slot-relation-head",
        row["manifest"], row["slot_model"],
        "--device", "cuda",
        "--rotation-augmentation",
        "--rotation-augmentation-probability", "0.5",
        "--joint-replay-loss-weight", "0.1",
        "--axis-loss-weight", "2.0",
        "--object-batch-size", "4",
    ]
    frozen_model = frozen_source_root / "slot_relation_head.pt"
    if args.full_slot_only and not frozen_model.exists():
        raise FileNotFoundError(
            f"Full-slot fine-tuning requires a frozen checkpoint: {frozen_model}"
        )
    if not args.full_slot_only and not (args.resume and frozen_model.exists()):
        run_command(
            common + ["--output-dir", str(frozen_root), "--epochs", str(args.frozen_epochs)],
            env=env,
            log_path=method_root / "phase1.log",
        )
    finetuned_model = finetune_root / "slot_relation_head.pt"
    if not args.full_slot_only and not (args.resume and finetuned_model.exists()):
        run_command(
            common + [
                "--output-dir", str(finetune_root),
                "--epochs", str(args.finetune_epochs),
                "--initial-relation-model", str(frozen_model),
                "--unfreeze-slot-backbone",
                "--slot-unfreeze-scope", "decoder",
                "--slot-learning-rate-scale", "0.1",
            ],
            env=env,
            log_path=method_root / "phase2.log",
        )
    full_slot_root = method_root / "phase3_full_slot_finetune"
    full_slot_model = full_slot_root / "slot_relation_head.pt"
    if args.full_slot_finetune and not (args.resume and full_slot_model.exists()):
        run_command(
            common + [
                "--output-dir", str(full_slot_root), "--epochs", str(args.full_slot_epochs),
                "--initial-relation-model", str(frozen_model),
                "--unfreeze-slot-backbone",
                "--slot-unfreeze-scope", "all",
                "--slot-learning-rate-scale", "0.1",
            ],
            env=env,
            log_path=method_root / "phase3_full_slot.log",
        )
    summaries = {}
    phase_roots = () if args.full_slot_only else (("frozen", frozen_root), ("decoder_finetune", finetune_root))
    if args.full_slot_finetune:
        phase_roots += (("full_slot_finetune", full_slot_root),)
    for phase, root in phase_roots:
        summary_path = root / "training_summary.json"
        summaries[phase] = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "gpu": row["gpu"],
        "wall_time_s": time.perf_counter() - started,
        "phases": {
            name: {
                "best_validation_loss": value.get("best_validation_loss"),
                "test_metrics": value.get("test_metrics"),
                "runtime": value.get("runtime"),
            }
            for name, value in summaries.items()
        },
    }


def main() -> int:
    args = parse_args()
    if args.full_slot_only:
        args.full_slot_finetune = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_methods(args.methods_tsv)
    results = []
    with ThreadPoolExecutor(max_workers=len(rows)) as executor:
        futures = {executor.submit(run_method, row, args): row["method"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    payload = {"methods": sorted(results, key=lambda item: str(item["method"]))}
    (args.output_root / "three_method_training_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_reports(args.output_root, payload["methods"])
    return 0


def _write_reports(output_root: Path, methods: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for method in methods:
        phases = method.get("phases", {})
        if not isinstance(phases, dict):
            continue
        for phase_name, phase in phases.items():
            if not isinstance(phase, dict):
                continue
            metrics = phase.get("test_metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            runtime = phase.get("runtime") or {}
            if not isinstance(runtime, dict):
                runtime = {}
            rows.append({
                "method": method["method"],
                "phase": phase_name,
                "edge_f1": metrics.get("edge_f1"),
                "joint_type_accuracy": metrics.get("joint_type_accuracy"),
                "axis_error_deg_type_correct": metrics.get("axis_error_deg"),
                "axis_error_deg_all_gt_pairs": metrics.get("axis_error_deg_all_gt_pairs"),
                "axis_line_error": metrics.get("axis_line_error_normalized"),
                "revolute_axis_error_deg": metrics.get("revolute_axis_error_deg"),
                "prismatic_axis_error_deg": metrics.get("prismatic_axis_error_deg"),
                "training_time_s": runtime.get("total_training_time_s"),
                "mean_epoch_time_s": runtime.get("mean_epoch_time_s"),
                "peak_cuda_memory_bytes": runtime.get("peak_cuda_memory_bytes"),
            })
    if not rows:
        return
    with (output_root / "three_method_relation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Three-Method Relation-Head Report",
        "",
        "Axis error is reported on GT joint pairs whose predicted joint type is correct. The all-pairs value is diagnostic only.",
        "",
        "| Method | Phase | Edge F1 | Type accuracy | Axis error (type-correct) | Axis error (all GT pairs) | Axis-line error | Runtime (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['method']} | {row['phase']} | {fmt('edge_f1')} | {fmt('joint_type_accuracy')} | "
            f"{fmt('axis_error_deg_type_correct')} | {fmt('axis_error_deg_all_gt_pairs')} | "
            f"{fmt('axis_line_error')} | {fmt('training_time_s')} |"
        )
    (output_root / "three_method_relation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
