#!/usr/bin/env python3
"""Merge adapted native metrics into external-suite per-object records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--method", action="append", required=True)
    args = parser.parse_args()

    merged = []
    for method in args.method:
        for native_path in sorted(
            (args.suite_root / "per_object").glob(
                f"*/{method}/native_metrics.json"
            )
        ):
            metrics_path = native_path.with_name("metrics.json")
            metadata = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {}
            )
            native = json.loads(native_path.read_text(encoding="utf-8"))
            for key in (
                "status",
                "oracle",
                "segmentation",
                "kinematics",
                "geometry",
                "metric_support",
                "artifacts",
            ):
                if key in native:
                    metadata[key] = native[key]
            metadata.pop("stage", None)
            metadata.pop("reason", None)
            metadata["failure"] = None
            metrics_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            merged.append(str(metrics_path))
    print(json.dumps({"merged_n": len(merged), "paths": merged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
