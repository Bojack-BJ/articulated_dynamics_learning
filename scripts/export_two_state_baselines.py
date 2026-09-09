#!/usr/bin/env python3
"""Export one fixed-start/fixed-end episode for DTA and ArtGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.two_state_export import export_two_state_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--category")
    parser.add_argument("--gt-part-count", type=int)
    parser.add_argument("--required-views-per-state", type=int, default=100)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_two_state_package(
        args.episode,
        args.output_dir,
        object_id=args.object_id,
        category=args.category,
        gt_part_count=args.gt_part_count,
        required_views_per_state=args.required_views_per_state,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps({"two_state_package": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
