#!/usr/bin/env python3
"""Select DTA revolute/prismatic hypotheses by no-GT target-state replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from rgbd_urdf_mvp.benchmarks.dta_hypothesis_selection import (
    select_joint_hypothesis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--target-point-cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--moving-part-indices",
        help="Comma-separated mesh indices in official DTA hypothesis order; defaults to 0..J-1.",
    )
    parser.add_argument("--sample-count", type=int, default=20_000)
    parser.add_argument("--trim-quantile", type=float, default=0.9)
    parser.add_argument("--ambiguity-relative-margin", type=float, default=0.08)
    parser.add_argument("--minimum-motion-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_points(path: Path) -> np.ndarray:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.dump(concatenate=True)
    if hasattr(geometry, "vertices"):
        return np.asarray(geometry.vertices, dtype=np.float64)
    raise ValueError(f"Unsupported target geometry: {path}")


def main() -> int:
    args = parse_args()
    prediction_path = args.predictions.expanduser().resolve()
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    hypotheses = payload["joint_hypotheses"]
    joint_count = min(len(hypotheses["prismatic"]), len(hypotheses["revolute"]))
    indices = (
        [int(value) for value in args.moving_part_indices.split(",")]
        if args.moving_part_indices
        else list(range(joint_count))
    )
    if len(indices) != joint_count:
        raise ValueError("moving-part-indices must contain one index per hypothesis")

    target = _load_points(args.target_point_cloud.expanduser().resolve())
    rng = np.random.default_rng(args.seed)
    selections = []
    for joint_index, mesh_index in enumerate(indices):
        mesh = trimesh.load_mesh(payload["part_meshes"][mesh_index], process=False)
        sample_count = min(args.sample_count, max(len(mesh.vertices), 1))
        if len(mesh.faces):
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            source, _ = trimesh.sample.sample_surface(mesh, sample_count)
        else:
            source = np.asarray(mesh.vertices)[
                rng.choice(len(mesh.vertices), sample_count, replace=False)
            ]
        result = select_joint_hypothesis(
            source,
            target,
            prismatic=hypotheses["prismatic"][joint_index],
            revolute=hypotheses["revolute"][joint_index],
            trim_quantile=args.trim_quantile,
            ambiguity_relative_margin=args.ambiguity_relative_margin,
            minimum_motion_fraction=args.minimum_motion_fraction,
        )
        result.update({"joint_index": joint_index, "moving_part_index": mesh_index})
        selections.append(result)

    output_payload = {
        "schema": "dta-no-gt-replay-selection-v1",
        "method": "DTA + no-GT replay model selection",
        "official_dta_output_modified": False,
        "source_predictions": str(prediction_path),
        "target_point_cloud": str(args.target_point_cloud.expanduser().resolve()),
        "selections": selections,
    }
    output = args.output or prediction_path.with_name("replay_model_selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "joints": joint_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
