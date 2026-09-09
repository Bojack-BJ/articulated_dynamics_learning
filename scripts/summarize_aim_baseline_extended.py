#!/usr/bin/env python3
"""Build the paper-facing extended AiM baseline report from a run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_baseline_report import write_extended_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--additional-manifest",
        action="append",
        type=Path,
        default=[],
        help="Append protocol-ablation runs from another manifest.",
    )
    parser.add_argument("--failure-details", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/aim_baseline_extended_v1"),
    )
    args = parser.parse_args()
    report = write_extended_report(
        args.manifest,
        args.output_root,
        additional_manifests=args.additional_manifest,
        failure_details_path=args.failure_details,
    )
    print(json.dumps({
        "summary": str((args.output_root / "summary.md").resolve()),
        "aggregate_metrics": str(
            (args.output_root / "raw" / "aggregate_metrics.json").resolve()
        ),
        "protocol_pair_count": len(report["protocol_pair_aggregates"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
