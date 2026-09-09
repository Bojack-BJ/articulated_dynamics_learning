#!/usr/bin/env python3
"""Adapt released DTA part meshes and motion hypotheses for common evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


_PALETTE = (
    (78, 205, 196),
    (255, 107, 107),
    (196, 181, 253),
    (250, 204, 21),
    (251, 146, 60),
    (244, 114, 182),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def latest_result_step(run_dir: Path) -> Path:
    steps = sorted((run_dir / "results").glob("step_*"), reverse=True)
    for step in steps:
        if any(step.glob("init_part_[0-9]*_clustered.obj")):
            return step
    raise FileNotFoundError(
        f"No complete DTA result steps with clustered part meshes under {run_dir}"
    )


def sample_part_meshes(
    mesh_paths: list[Path], *, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Area-proportionally sample native DTA part meshes."""
    if sample_count < len(mesh_paths):
        raise ValueError("sample_count must be at least the number of parts")
    meshes = [trimesh.load_mesh(path, process=False) for path in mesh_paths]
    areas = np.asarray([max(float(mesh.area), 0.0) for mesh in meshes])
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("DTA part meshes have no finite positive surface area")
    counts = np.maximum(1, np.rint(sample_count * areas / areas.sum()).astype(int))
    counts[np.argmax(counts)] += sample_count - int(counts.sum())
    np.random.seed(seed)
    points = []
    labels = []
    for part_id, (mesh, count) in enumerate(zip(meshes, counts)):
        sampled, _ = trimesh.sample.sample_surface(mesh, int(count))
        points.append(sampled)
        labels.append(np.full(int(count), part_id, dtype=np.int64))
    return np.concatenate(points), np.concatenate(labels)


def write_labeled_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    colors = np.asarray([_PALETTE[int(label) % len(_PALETTE)] for label in labels])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    step_dir = (
        run_dir / "results" / f"step_{args.step:07d}"
        if args.step is not None
        else latest_result_step(run_dir)
    )
    part_meshes = sorted(step_dir.glob("init_part_[0-9]*_clustered.obj"))
    if not part_meshes:
        raise FileNotFoundError(f"No clustered DTA part meshes in {step_dir}")
    points, labels = sample_part_meshes(
        part_meshes, sample_count=args.sample_count, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labeled_ply = args.output_dir / "start_labeled_parts.ply"
    write_labeled_ply(labeled_ply, points, labels)

    hypotheses = {}
    for joint_type in ("prismatic", "revolute"):
        path = step_dir / f"init_{joint_type}_motion.json"
        if path.exists():
            hypotheses[joint_type] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema": "dta-prediction-adapter-v1",
        "method": "DTA",
        "step": int(step_dir.name.removeprefix("step_")),
        "predicted_part_count": len(part_meshes),
        "part_meshes": [str(path.resolve()) for path in part_meshes],
        "labeled_point_cloud": str(labeled_ply.resolve()),
        "joint_type_selection": "unsupported_by_released_non-GT_path",
        "joint_hypotheses": hypotheses,
    }
    output_json = args.output_dir / "predictions.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dta_predictions": str(output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
