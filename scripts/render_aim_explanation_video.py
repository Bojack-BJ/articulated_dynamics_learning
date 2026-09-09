#!/usr/bin/env python3
"""Render a three-panel GT-mesh/AiM motion explanation video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


VIVID = (
    "#00e5ff",
    "#ff1744",
    "#76ff03",
    "#ffea00",
    "#d500f9",
    "#ff9100",
    "#00e676",
    "#ff4081",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-aim-points", type=int, default=3000)
    parser.add_argument("--max-faces-per-geometry", type=int, default=4000)
    args = parser.parse_args()

    viewer_dir = args.viewer_dir.expanduser().resolve()
    aim = _load(viewer_dir / "viewer_data" / "aim.json")
    mesh = _load(viewer_dir / "viewer_data" / "gt_mesh.json")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else viewer_dir / "aim_explanation.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(aim["trajectory"], dtype=float)
    frame_count = trajectory.shape[1]
    selected = _sample_indices(len(trajectory), args.max_aim_points)
    predicted = np.asarray(aim["pred_part_id"], dtype=int)[selected]
    dynamic = np.asarray(
        aim.get("dynamic_flag", [1] * len(trajectory)), dtype=int
    )[selected]
    bounds = aim["bounds"]

    figure = plt.figure(figsize=(15, 5), facecolor="#071018")
    axes = [
        figure.add_subplot(1, 3, index + 1, projection="3d")
        for index in range(3)
    ]
    writer = FFMpegWriter(
        fps=max(1, args.fps),
        metadata={"title": "AiM motion decomposition explanation"},
        bitrate=5000,
    )
    with writer.saving(figure, str(output), dpi=140):
        for frame_index in range(frame_count):
            for axis in axes:
                axis.clear()
                _style_axis(axis, bounds)
            _draw_mesh(
                axes[0], mesh, frame_index, args.max_faces_per_geometry, opacity=0.75
            )
            axes[0].set_title("GT articulated mesh", color="white")

            points = trajectory[selected, frame_index]
            colors = np.where(dynamic > 0, "#ff3d71", "#45d7ff")
            axes[1].scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=5,
                alpha=0.9,
                depthshade=False,
            )
            axes[1].set_title("AiM Gaussian motion\ncyan=static, pink=dynamic", color="white")

            _draw_mesh(
                axes[2], mesh, frame_index, args.max_faces_per_geometry, opacity=0.16
            )
            part_colors = [
                "#64748b" if value < 0 else VIVID[value % len(VIVID)]
                for value in predicted
            ]
            axes[2].scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=part_colors,
                s=6,
                alpha=0.95,
                depthshade=False,
            )
            axes[2].set_title(
                "AiM final motion components\non GT mesh", color="white"
            )
            figure.suptitle(
                f"AiM decomposition, normalized time {frame_index + 1}/{frame_count}",
                color="white",
                fontsize=14,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.92))
            writer.grab_frame(facecolor=figure.get_facecolor())
    plt.close(figure)
    print(output)
    return 0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum).round().astype(int)


def _style_axis(axis, bounds: dict) -> None:
    axis.set_facecolor("#071018")
    axis.set_xlim(bounds["min"][0], bounds["max"][0])
    axis.set_ylim(bounds["min"][1], bounds["max"][1])
    axis.set_zlim(bounds["min"][2], bounds["max"][2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=20, azim=-55)
    axis.tick_params(colors="#8ca1b3", labelsize=7)
    axis.xaxis.label.set_color("#8ca1b3")
    axis.yaxis.label.set_color("#8ca1b3")
    axis.zaxis.label.set_color("#8ca1b3")


def _draw_mesh(axis, payload: dict, frame_index: int, max_faces: int, opacity: float) -> None:
    transforms = {
        int(item["payload_id"]): item
        for item in payload.get("frames", {}).get(str(frame_index), [])
    }
    for geometry in payload.get("geometries", []):
        transform = transforms.get(int(geometry["payload_id"]))
        if transform is None:
            continue
        vertices = np.asarray(geometry["vertices"], dtype=float)
        rotation = np.asarray(transform["rotation"], dtype=float)
        position = np.asarray(transform["position"], dtype=float)
        vertices = vertices @ rotation.T + position
        faces = np.asarray(geometry["faces"], dtype=int)
        if len(faces) > max_faces:
            indices = np.linspace(0, len(faces) - 1, max_faces).round().astype(int)
            faces = faces[indices]
        collection = Poly3DCollection(
            vertices[faces],
            facecolor=geometry["color"],
            edgecolor="none",
            alpha=opacity,
        )
        axis.add_collection3d(collection)


if __name__ == "__main__":
    raise SystemExit(main())
