#!/usr/bin/env python3
"""Export compact labeled point/axis bundles from VideoArtGS native meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


PALETTE = (
    (78, 205, 196),
    (255, 107, 107),
    (190, 174, 255),
    (255, 196, 61),
    (57, 211, 167),
    (255, 132, 51),
    (235, 88, 139),
    (88, 166, 255),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--points-per-object", type=int, default=12_000)
    args = parser.parse_args()

    exported = []
    for metrics_path in sorted(
        (args.suite_root / "per_object").glob("partnet_*/videoartgs/metrics.json")
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "success_native_metrics":
            continue
        artifacts = metrics.get("artifacts", {})
        meshes = [Path(path) for path in artifacts.get("part_meshes", [])]
        joint_path = Path(str(artifacts.get("joint_info_json", "")))
        if not meshes or not joint_path.exists() or not all(path.exists() for path in meshes):
            continue
        points = []
        colors = []
        per_part = max(100, args.points_per_object // len(meshes))
        for index, path in enumerate(meshes):
            mesh = trimesh.load(path, force="mesh", process=False)
            xyz, _ = trimesh.sample.sample_surface(mesh, per_part)
            points.append(xyz)
            colors.append(np.tile(PALETTE[index % len(PALETTE)], (len(xyz), 1)))
        xyz = np.concatenate(points, axis=0)
        rgb = np.concatenate(colors, axis=0)
        joints_raw = json.loads(joint_path.read_text(encoding="utf-8"))
        joints = []
        for joint in joints_raw:
            if int(joint.get("parent", -1)) < 0:
                continue
            axis = (joint.get("jointData") or {}).get("axis") or {}
            joints.append(
                {
                    "type": (
                        "revolute"
                        if joint.get("joint") == "hinge"
                        else "prismatic"
                    ),
                    "axis_position": axis.get("origin", joint.get("center", [0, 0, 0])),
                    "axis_direction": axis.get("direction", [1, 0, 0]),
                    "parent": joint.get("parent"),
                    "child": joint.get("id"),
                    "name": joint.get("name"),
                }
            )
        output = metrics_path.parent / "adapter"
        output.mkdir(exist_ok=True)
        ply_path = output / "start_labeled_parts.ply"
        _write_ply(ply_path, xyz, rgb)
        prediction_path = output / "predictions.json"
        prediction_path.write_text(
            json.dumps(
                {
                    "schema": "videoartgs-viewer-adapter-v1",
                    "method": "VideoArtGS",
                    "labeled_point_cloud": str(ply_path),
                    "joints": joints,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        exported.append(str(metrics_path.parents[1].name))
    print(json.dumps({"exported": len(exported), "objects": exported}, indent=2))
    return 0


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for point, color in zip(points, colors, strict=True):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
