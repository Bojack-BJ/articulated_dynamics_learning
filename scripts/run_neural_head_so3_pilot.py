#!/usr/bin/env python3
"""Prepare and run the staged neural Relation Head SO(3) pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.core.serialization import load_json
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


VARIANTS = {
    "direct_no_aug": {"axis_head_type": "direct", "augmentation": False},
    "direct_haar": {"axis_head_type": "direct", "augmentation": True},
    "equivariant_proposal": {
        "axis_head_type": "equivariant_proposal", "augmentation": True,
    },
    "vector_neuron": {
        "axis_head_type": "vector_neuron", "augmentation": True,
        "joint_type_loss_weight": 1.0,
    },
    "vector_neuron_type15": {
        "axis_head_type": "vector_neuron", "augmentation": True,
        "joint_type_loss_weight": 1.5,
    },
    "vector_neuron_type20": {
        "axis_head_type": "vector_neuron", "augmentation": True,
        "joint_type_loss_weight": 2.0,
    },
    "vector_neuron_type20_no_pair": {
        "axis_head_type": "vector_neuron", "augmentation": True,
        "joint_type_loss_weight": 2.0,
        "paired_consistency": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument(
        "--catalog", type=Path,
        help="Optional PartNet catalog; defaults to catalog.json beside the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_head_so3_pilot_v1"))
    parser.add_argument("--phase", choices=("prepare", "stage1", "stage2", "summarize", "all"), default="prepare")
    parser.add_argument("--seeds", default="20260831,20260832,20260833")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated Stage-1 variants to run/summarize.",
    )
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--object-batch-size", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_variants(args: argparse.Namespace) -> list[str]:
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown pilot variants: {', '.join(unknown)}")
    if not variants:
        raise ValueError("At least one pilot variant is required")
    return variants


def _metadata(
    row: dict[str, Any], catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    object_id = str(row["object_id"])
    category = str(row.get("category") or object_id.split("_")[0]).lower()
    part_count = int(row.get("gt_part_count") or 0)
    joint_types = str(row.get("joint_types") or "unknown")
    catalog_row = (catalog or {}).get(object_id)
    if catalog_row is not None:
        category = str(
            catalog_row.get("category")
            or catalog_row.get("source_category")
            or category
        ).lower()
        joint_count = int(catalog_row.get("joint_count") or 0)
        part_count = int(catalog_row.get("gt_part_count") or joint_count + 1)
        types = catalog_row.get("joint_types")
        if types:
            joint_types = "+".join(sorted({str(value) for value in types}))
    path = Path(str(row["tracks_path"])).expanduser()
    if path.exists():
        artifact = load_json(path)
        category = str(
            artifact.get("category")
            or artifact.get("source_category")
            or artifact.get("metadata", {}).get("category")
            or category
        ).lower()
        segmentation = artifact.get("original_part_segmentation", {})
        parts = segmentation.get("parts", []) if isinstance(segmentation, dict) else []
        labels = artifact.get("original_part_id", [])
        part_count = len(parts) or len({int(value) for value in labels if int(value) >= 0}) or part_count
        types = artifact.get("gt_joint_types") or artifact.get("metadata", {}).get("joint_types")
        if types:
            joint_types = "+".join(sorted({str(value) for value in types}))
    complexity = "2" if part_count <= 2 else "3-4" if part_count <= 4 else ">=5"
    return {
        "category": category,
        "gt_part_count": part_count,
        "joint_types": joint_types,
        "complexity": complexity,
    }


def prepare_manifest(args: argparse.Namespace) -> Path:
    rows = load_pairwise_manifest(args.manifest.expanduser().resolve())
    catalog_path = getattr(args, "catalog", None)
    if catalog_path is None:
        candidate = args.manifest.expanduser().resolve().parent / "catalog.json"
        catalog_path = candidate if candidate.exists() else None
    catalog: dict[str, dict[str, Any]] = {}
    if catalog_path is not None:
        payload = json.loads(Path(catalog_path).expanduser().resolve().read_text(encoding="utf-8"))
        catalog_rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
        catalog = {str(row["object_id"]): row for row in catalog_rows}
    enriched = [{**row, **_metadata(row, catalog)} for row in rows]
    enriched.sort(
        key=lambda row: (
            str(row["category"]), str(row["joint_types"]), str(row["complexity"]),
            str(row["object_id"]),
        )
    )
    total = args.train_count + args.val_count + args.test_count
    if len(enriched) < total:
        raise ValueError(f"Pilot requires {total} objects, but manifest has {len(enriched)}")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        strata[(row["category"], row["joint_types"], row["complexity"])].append(row)
    rng = random.Random(20260831)
    for rows_in_stratum in strata.values():
        rng.shuffle(rows_in_stratum)
    selected: list[dict[str, Any]] = []
    while len(selected) < total:
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(selected) < total:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    # Interleave strata before assigning splits so each split receives the same
    # deterministic category/type/complexity ordering rather than contiguous IDs.
    assignments = []
    targets = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    order = ["train", "val", "test"]
    cursor = 0
    for row in selected:
        while targets[order[cursor % 3]] <= 0:
            cursor += 1
        split = order[cursor % 3]
        targets[split] -= 1
        cursor += 1
        assignments.append({**row, "split": split})
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "pilot_manifest.tsv"
    fields = ["object_id", "tracks_path", "features_npz", "split"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in assignments])
    (output / "pilot_selection.json").write_text(
        json.dumps({
            "seed": 20260831,
            "counts": {name: sum(row["split"] == name for row in assignments) for name in order},
            "objects": [
                {key: row[key] for key in ("object_id", "split", "category", "joint_types", "complexity", "gt_part_count")}
                for row in assignments
            ],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _training_command(
    args: argparse.Namespace, manifest: Path, variant: str, seed: int, output: Path,
    *, stage2: bool = False, initial: Path | None = None,
) -> list[str]:
    settings = VARIANTS[variant]
    command = [
        args.python, "-m", "rgbd_urdf_mvp", "train-slot-relation-head",
        str(manifest), str(args.slot_model), "--output-dir", str(output),
        "--epochs", str(args.stage2_epochs if stage2 else args.epochs),
        "--object-batch-size", str(args.object_batch_size),
        "--seed", str(seed), "--axis-geometry-branch",
        "--axis-head-type", settings["axis_head_type"],
        "--axis-equivariance-loss-weight", (
            "1.0" if settings.get("paired_consistency", True) else "0.0"
        ),
        "--axis-line-equivariance-loss-weight", (
            "0.5" if settings.get("paired_consistency", True) else "0.0"
        ),
        "--edge-consistency-loss-weight", (
            "0.1" if settings.get("paired_consistency", True) else "0.0"
        ),
        "--type-consistency-loss-weight", (
            "0.1" if settings.get("paired_consistency", True) else "0.0"
        ),
        "--joint-type-loss-weight", str(settings.get("joint_type_loss_weight", 1.0)),
    ]
    if settings["augmentation"]:
        command += [
            "--rotation-augmentation", "--rotation-augmentation-probability", "1.0",
            "--rotation-augmentation-mode", "uniform_quaternion",
        ]
    if stage2:
        command += [
            "--unfreeze-slot-backbone", "--slot-unfreeze-scope", "decoder",
            "--slot-learning-rate-scale", "0.1", "--learning-rate", "1e-4",
        ]
    if initial is not None:
        command += ["--initial-relation-model", str(initial)]
    return command


def _run_job(command: list[str], output: Path, device: str, dry_run: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        return {"output": str(output), "status": "dry_run"}
    environment = dict(os.environ)
    if device.startswith("cuda:"):
        environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        command = [*command, "--device", "cuda"]
    else:
        command = [*command, "--device", device]
    with (output / "train.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return {"output": str(output), "status": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def run_stage1(args: argparse.Namespace, manifest: Path) -> list[dict[str, Any]]:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    jobs = []
    for variant in _selected_variants(args):
        for seed in seeds:
            output = args.output_dir / "stage1" / variant / f"seed_{seed}"
            device = devices[len(jobs) % len(devices)]
            jobs.append((_training_command(args, manifest, variant, seed, output), output, device))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {
            executor.submit(_run_job, command, output, device, args.dry_run): output
            for command, output, device in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    for result in results:
        output = Path(result["output"])
        audit_output = output / "equivariance_audit"
        audit_command = [
            args.python, "scripts/audit_relation_so3_equivariance.py",
            str(manifest), str(args.slot_model), str(output / "slot_relation_head.pt"),
            "--output-dir", str(audit_output), "--sampler-count", "1000",
            "--rotations-per-sample", "1", "--seed", "20260831",
            "--catalog", str(args.output_dir / "pilot_selection.json"),
        ]
        (output / "audit_command.json").write_text(
            json.dumps(audit_command, indent=2) + "\n", encoding="utf-8"
        )
        if result["status"] == "ok" and not args.dry_run:
            with (output / "audit.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    audit_command, stdout=log, stderr=subprocess.STDOUT
                )
            result["audit_status"] = "ok" if completed.returncode == 0 else "failed"
    return results


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    aggregate = {}
    for variant in _selected_variants(args):
        rows = []
        for summary_path in sorted((args.output_dir / "stage1" / variant).glob("seed_*/training_summary.json")):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if payload.get("test_metrics"):
                rows.append(payload["test_metrics"])
        if not rows:
            continue
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        aggregate[variant] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in keys if isinstance(rows[0][key], (int, float)) and math.isfinite(float(rows[0][key]))
        }
        aggregate[variant]["seed_count"] = len(rows)
        audit_rows = []
        for path in sorted((args.output_dir / "stage1" / variant).glob(
            "seed_*/equivariance_audit/so3_equivariance_audit.json"
        )):
            audit = json.loads(path.read_text(encoding="utf-8"))
            overall = audit.get("model_equivariance", {}).get("overall", {})
            if overall.get("median_deg") is not None:
                audit_rows.append(overall)
        if audit_rows:
            aggregate[variant]["equivariance_median_deg"] = sum(
                float(row["median_deg"]) for row in audit_rows
            ) / len(audit_rows)
            aggregate[variant]["equivariance_p90_deg"] = sum(
                float(row["p90_deg"]) for row in audit_rows
            ) / len(audit_rows)
    baseline = aggregate.get("direct_no_aug", {})
    recommendations = {}
    candidates_to_score = [
        variant for variant in _selected_variants(args)
        if variant not in {"direct_no_aug", "direct_haar"}
    ]
    for variant in candidates_to_score:
        row = aggregate.get(variant, {})
        recommendations[variant] = {
            "axis_better": all(
                row.get(key, math.inf) < baseline.get(key, math.inf)
                for key in ("axis_error_deg", "axis_error_median_deg", "axis_error_p90_deg")
            ),
            "catastrophic_halved": row.get("axis_error_gt_80_count", math.inf) <= 0.5 * baseline.get("axis_error_gt_80_count", 0),
            "type_gate": row.get("joint_type_accuracy", 0) >= baseline.get("joint_type_accuracy", 0) - 0.02,
            "edge_gate": row.get("edge_f1", 0) >= baseline.get("edge_f1", 0) - 0.02,
            "illegal_graph_gate": row.get("illegal_graph_rate", 1) <= baseline.get("illegal_graph_rate", 1),
            "candidate_coverage_gate": row.get("candidate_coverage", 0) >= 0.9,
            "equivariance_gate": (
                row.get("equivariance_median_deg", math.inf) < 5.0
                and row.get("equivariance_p90_deg", math.inf) < 10.0
            ),
        }
        recommendations[variant]["passes_metric_gates"] = all(recommendations[variant].values())
    result = {"aggregate": aggregate, "recommendations": recommendations}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_recommendation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    manifest = prepare_manifest(args)
    result: dict[str, Any] = {"manifest": str(manifest)}
    if args.phase in {"stage1", "all"}:
        result["stage1"] = run_stage1(args, manifest)
    if args.phase in {"summarize", "all"}:
        result["summary"] = summarize(args)
    if args.phase == "stage2":
        summary = summarize(args)
        candidates = [
            name for name, row in summary["recommendations"].items()
            if row["passes_metric_gates"]
        ]
        if not candidates:
            raise RuntimeError("No architecture passed Stage-1 metric gates; Stage 2 was not launched.")
        winner = min(
            candidates,
            key=lambda name: summary["aggregate"][name].get("axis_error_p90_deg", math.inf),
        )
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        jobs = []
        for index, seed in enumerate(seeds):
            initial = args.output_dir / "stage1" / winner / f"seed_{seed}" / "slot_relation_head.pt"
            output = args.output_dir / "stage2" / winner / f"seed_{seed}"
            jobs.append((
                _training_command(
                    args, manifest, winner, seed, output, stage2=True, initial=initial
                ),
                output,
                devices[index % len(devices)],
            ))
        with ThreadPoolExecutor(max_workers=min(len(jobs), max(1, args.max_parallel))) as executor:
            outputs = list(executor.map(
                lambda job: _run_job(job[0], job[1], job[2], args.dry_run), jobs
            ))
        result["stage2"] = {"winner": winner, "jobs": outputs}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
