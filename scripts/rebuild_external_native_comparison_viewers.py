#!/usr/bin/env python3
"""Rebuild aligned AiM/ReArt viewers from retained native artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.perception.baseline_viewer import (
    BaselineComparisonViewerBuilder,
    BaselineViewerConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--suite-root", type=Path, default=Path("outputs/external_baseline_suite_v1"))
    parser.add_argument("--timeline-steps", type=int, default=31)
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    suite = (root / args.suite_root).resolve()
    results = []
    for object_dir in sorted((suite / "per_object").glob("partnet_*")):
        object_id = object_dir.name
        reference = object_dir / "evaluation_reference" / "two_state_start.ply"
        aim_run = root / "outputs/external_baselines_v1/aim_style_aligned_runs" / object_id
        reart = (
            root
            / "outputs/external_baselines_v1/reart_runs_aligned_4x512"
            / object_id
            / object_id
            / "result.pkl"
        )
        recording = root / "outputs/aim_style_protocol_v1_recordings" / object_id
        output = object_dir / "native_comparison" / "viewer.html"
        if not reference.is_file():
            results.append({"object_id": object_id, "status": "missing_reference"})
            continue
        try:
            BaselineComparisonViewerBuilder().build(
                BaselineViewerConfig(
                    output_html=output,
                    reference_ply=reference,
                    aim_run=aim_run if (aim_run / "seg_init_dual/segmented_point.ply").is_file() else None,
                    aim_metrics=object_dir / "aim_aligned" / "metrics.json",
                    reart_prediction=reart if reart.is_file() else None,
                    reart_metrics=object_dir / "reart" / "metrics.json",
                    gt_frames_dir=recording / "pointcloud_4d_partseg" / "frames",
                    gt_mesh_episode=recording / "episode.json",
                    timeline_steps=max(2, args.timeline_steps),
                    axis_remap="x,y,z",
                )
            )
            methods = [
                path.stem
                for path in sorted((output.parent / "viewer_data").glob("*.json"))
                if path.stem not in {"metrics", "gt_mesh"}
            ]
            results.append(
                {"object_id": object_id, "status": "success", "methods": methods}
            )
        except Exception as error:
            results.append(
                {
                    "object_id": object_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summary = {
        "attempted": len(results),
        "success": sum(row["status"] == "success" for row in results),
        "results": results,
    }
    (suite / "native_comparison_rebuild.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
