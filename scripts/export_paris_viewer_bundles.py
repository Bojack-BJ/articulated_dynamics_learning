#!/usr/bin/env python3
"""Export compact PARIS part/axis bundles from retained native checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.perception.baseline_viewer import _read_ply


COLORS = np.asarray([[91, 192, 235], [255, 126, 95]], dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--points-per-part", type=int, default=8_000)
    args = parser.parse_args()
    output = []
    for object_dir in sorted((args.suite_root / "per_object").glob("partnet_*")):
        method_dir = object_dir / "paris"
        motions = sorted(method_dir.glob("raw/*/save/it*_test_motion.json"))
        if not motions:
            continue
        save = motions[-1].parent
        static_paths = sorted(save.glob("it*_static_*.ply"))
        dynamic_paths = sorted(save.glob("it*_dynamic_*.ply"))
        if not static_paths or not dynamic_paths:
            continue
        adapter = method_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        points = []
        colors = []
        for part_id, path in enumerate((static_paths[-1], dynamic_paths[-1])):
            values, _, _ = _read_ply(path)
            if len(values) > args.points_per_part:
                indices = np.linspace(
                    0, len(values) - 1, args.points_per_part
                ).round().astype(int)
                values = values[indices]
            points.append(values)
            colors.append(np.repeat(COLORS[part_id][None], len(values), axis=0))
        ply = adapter / "labeled_parts.ply"
        _write_ply(ply, np.concatenate(points), np.concatenate(colors))
        motion = json.loads(motions[-1].read_text(encoding="utf-8"))
        joint_type = (
            "revolute"
            if str(motion.get("type", "")).lower() in {"rotate", "revolute", "r"}
            else "prismatic"
        )
        joint = {
            "type": joint_type,
            "axis_position": motion.get(
                "R_axis_o", motion.get("axis_position", [0.0, 0.0, 0.0])
            ),
            "axis_direction": (
                motion.get("R_axis_d", motion.get("axis_direction"))
                if joint_type == "revolute"
                else motion.get("t_axis_d", motion.get("axis_direction"))
            ),
        }
        prediction = {
            "schema": "paris-viewer-bundle-v1",
            "object_id": object_dir.name,
            "labeled_point_cloud": str(ply),
            "joints": [joint],
            "source_motion_json": str(motions[-1]),
            "source_part_plys": [str(static_paths[-1]), str(dynamic_paths[-1])],
        }
        prediction_path = adapter / "predictions.json"
        prediction_path.write_text(
            json.dumps(prediction, indent=2) + "\n", encoding="utf-8"
        )
        output.append({"object_id": object_dir.name, "adapter": str(adapter)})
    print(json.dumps({"exported": len(output), "objects": output}, indent=2))
    return 0


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
        f"{int(color[0])} {int(color[1])} {int(color[2])}"
        for point, color in zip(points, colors)
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    raise SystemExit(main())
