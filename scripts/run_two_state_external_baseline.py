#!/usr/bin/env python3
"""Plan or run one released DTA/ArtGS baseline on a two-state package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_runners import (
    execute_run_plan,
    plan_artgs_run,
    plan_ditto_run,
    plan_dta_run,
    plan_gaussianart_run,
    plan_paris_run,
    write_run_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method", choices=("dta", "artgs", "gaussianart", "paris", "ditto")
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--gt-part-count", type=int, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--config-dir",
        default="config/release",
        help="DTA config directory relative to the released repository.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute official code. Without this flag only a dry-run plan is written.",
    )
    return parser.parse_args()


def register_artgs_slot_count(
    repo: Path,
    *,
    dataset: str,
    subset: str,
    object_id: str,
    gt_part_count: int,
) -> None:
    """Register the oracle slot count required by the released ArtGS code."""
    path = repo / "arguments" / "num_slots.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault(dataset, {}).setdefault(subset, {})[object_id] = gt_part_count
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    common = {
        "object_id": args.object_id,
        "gt_part_count": args.gt_part_count,
        "python": args.python,
    }
    if args.method == "dta":
        plan = plan_dta_run(
            args.repo,
            args.package_dir,
            args.output_dir,
            config_dir=args.config_dir,
            **common,
        )
    elif args.method == "artgs":
        plan = plan_artgs_run(args.repo, args.package_dir, args.output_dir, **common)
        register_artgs_slot_count(
            args.repo,
            dataset="external_suite",
            subset="aligned",
            object_id=args.object_id,
            gt_part_count=args.gt_part_count,
        )
    elif args.method == "gaussianart":
        plan = plan_gaussianart_run(
            args.repo, args.package_dir, args.output_dir, gpu=args.gpu, **common
        )
    elif args.method == "paris":
        plan = plan_paris_run(args.repo, args.package_dir, args.output_dir, **common)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for Ditto")
        plan = plan_ditto_run(
            args.repo,
            args.package_dir,
            args.output_dir,
            checkpoint=args.checkpoint,
            **common,
        )
    write_run_plan(plan, args.output_dir / "run_plan.json")
    result = execute_run_plan(
        plan,
        repo=args.repo,
        metrics_path=args.output_dir / "metrics.json",
        dry_run=not args.run,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
