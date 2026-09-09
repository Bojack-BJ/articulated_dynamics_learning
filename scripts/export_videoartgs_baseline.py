#!/usr/bin/env python3
"""Export a recorded interaction and 3D tracks for VideoArtGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.videoartgs_export import (
    export_videoartgs_native_package,
    export_videoartgs_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("tracks", type=Path, nargs="?")
    parser.add_argument("joint_infos", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joint-inventory-source",
        choices=("vlm", "manual", "gt_oracle"),
        required=True,
    )
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument(
        "--static-cameras",
        type=Path,
        help="Enable the native 100-view canonical-scan plus interaction export.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.static_cameras:
        manifest = export_videoartgs_native_package(
            args.static_cameras,
            args.episode,
            args.joint_infos,
            args.output_dir,
            joint_inventory_source=args.joint_inventory_source,
            overwrite=args.overwrite,
        )
    else:
        if args.tracks is None:
            parser.error("tracks is required unless --static-cameras is used")
        manifest = export_videoartgs_package(
            args.episode,
            args.tracks,
            args.joint_infos,
            args.output_dir,
            joint_inventory_source=args.joint_inventory_source,
            view_index=args.view_index,
            overwrite=args.overwrite,
        )
    print(json.dumps({"videoartgs_package": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
