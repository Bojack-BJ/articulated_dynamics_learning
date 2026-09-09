#!/usr/bin/env python3
"""Validate a generated GT/Ours/AiM/ReArt viewer gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("gt", "hybrid", "aim", "reart")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("--min-timesteps", type=int, default=30)
    parser.add_argument("--min-gt-points", type=int, default=500)
    parser.add_argument("--expected-objects", type=int, default=4)
    args = parser.parse_args()

    root = args.viewer_root.expanduser().resolve()
    object_dirs = sorted(path.parent for path in root.glob("*/viewer.html"))
    if len(object_dirs) != args.expected_objects:
        raise ValueError(
            f"Expected {args.expected_objects} object viewers, found {len(object_dirs)}"
        )

    report = []
    for object_dir in object_dirs:
        methods = {}
        for method in METHODS:
            path = object_dir / "viewer_data" / f"{method}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {method} payload: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            trajectory = payload.get("trajectory", [])
            if not trajectory or len(trajectory[0]) < args.min_timesteps:
                raise ValueError(
                    f"{object_dir.name}/{method} has fewer than "
                    f"{args.min_timesteps} timesteps"
                )
            methods[method] = {
                "point_count": len(payload.get("points", [])),
                "timesteps": len(trajectory[0]),
            }

        gt = json.loads(
            (object_dir / "viewer_data" / "gt.json").read_text(encoding="utf-8")
        )
        geometry_frames = gt.get("geometry_frames", [])
        if len(geometry_frames) < args.min_timesteps:
            raise ValueError(f"{object_dir.name}/gt lacks dense temporal geometry")
        minimum_gt_points = min(len(frame.get("points", [])) for frame in geometry_frames)
        if minimum_gt_points < args.min_gt_points:
            raise ValueError(
                f"{object_dir.name}/gt has only {minimum_gt_points} points in a frame"
            )

        html = (object_dir / "viewer.html").read_text(encoding="utf-8")
        for marker in (
            'aspectmode:"cube"',
            "Point size",
            "Point opacity",
            "Show GT mesh replay",
            "Show GT point-cloud overlap",
        ):
            if marker not in html:
                raise ValueError(f"{object_dir.name}/viewer.html lacks {marker!r}")

        bounds = gt.get("bounds", {})
        spans = [
            float(bounds["max"][axis]) - float(bounds["min"][axis])
            for axis in range(3)
        ]
        if max(spans) - min(spans) > 1e-6:
            raise ValueError(f"{object_dir.name} has unequal axis spans: {spans}")
        report.append(
            {
                "object": object_dir.name,
                "methods": methods,
                "gt_geometry_frames": len(geometry_frames),
                "minimum_gt_points": minimum_gt_points,
                "axis_spans": spans,
            }
        )

    output = root / "validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
