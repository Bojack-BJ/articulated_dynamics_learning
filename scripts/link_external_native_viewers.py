#!/usr/bin/env python3
"""Attach development-machine AiM/ReArt viewer paths to suite metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--remote-project-root", required=True)
    parser.add_argument("--axis-counts", type=Path, default=None)
    args = parser.parse_args()
    axis_counts = (
        json.loads(args.axis_counts.read_text(encoding="utf-8"))
        if args.axis_counts
        else {}
    )

    count = 0
    for object_dir in sorted((args.suite_root / "per_object").glob("partnet_*")):
        viewer = (
            Path(args.remote_project_root)
            / "outputs/external_baseline_suite_v1/per_object"
            / object_dir.name
            / "native_comparison/viewer.html"
        )
        for method in ("aim_aligned", "reart"):
            metrics_path = object_dir / method / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.setdefault("artifacts", {})
            metrics["artifacts"]["viewer_html"] = str(viewer)
            metrics["artifacts"]["viewer_location"] = "development_machine"
            metrics.setdefault("kinematics", {})
            count_key = "aim" if method == "aim_aligned" else "reart"
            axis_count = int(
                axis_counts.get(object_dir.name, {}).get(count_key, 0)
            )
            metrics["kinematics"]["viewer_axis_count"] = axis_count
            if method == "aim_aligned":
                metrics["kinematics"]["axis_visualization"] = (
                    "native_motion_json"
                    if axis_count
                    else "native_motion_json_no_dynamic_axis"
                )
            else:
                metrics["kinematics"]["axis_visualization"] = (
                    "converted_from_native_part_poses_not_native_reart_output"
                    if axis_count
                    else "conversion_unavailable_no_observable_relative_motion"
                )
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            count += 1
    print(json.dumps({"updated": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
