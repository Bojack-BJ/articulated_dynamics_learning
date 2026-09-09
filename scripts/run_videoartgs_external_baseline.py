#!/usr/bin/env python3
"""Plan or execute the released VideoArtGS optimization pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_runners import (
    execute_run_plan,
    plan_videoartgs_run,
    write_run_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--joint-inventory-source",
        choices=("vlm", "manual", "gt_oracle"),
        required=True,
    )
    parser.add_argument("--python", default="python")
    parser.add_argument("--dataset", default="external_suite")
    parser.add_argument("--subset", default="aligned")
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Resume at canonical=0, deformation=1, or final optimization=2.",
    )
    parser.add_argument(
        "--skip-reconstruction-export",
        action="store_true",
        help="Skip official render.py reconstruction export (debug runs only).",
    )
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = plan_videoartgs_run(
        args.repo,
        args.source_root,
        args.output_dir,
        object_id=args.object_id,
        joint_inventory_source=args.joint_inventory_source,
        python=args.python,
        dataset=args.dataset,
        subset=args.subset,
        start_stage=args.start_stage,
        include_reconstruction_export=not args.skip_reconstruction_export,
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
