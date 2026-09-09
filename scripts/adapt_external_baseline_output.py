#!/usr/bin/env python3
"""Convert one released baseline output into suite-native metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_outputs import (
    adapt_gaussianart_output,
    adapt_paris_output,
    adapt_videoartgs_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("paris", "gaussianart", "videoartgs"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    args = parser.parse_args()
    if args.method == "paris":
        payload = adapt_paris_output(args.output_dir)
    elif args.method == "gaussianart":
        payload = adapt_gaussianart_output(args.output_dir)
    else:
        payload = adapt_videoartgs_output(args.output_dir)
    metrics_path = args.metrics_path or args.output_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
