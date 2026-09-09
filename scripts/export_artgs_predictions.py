#!/usr/bin/env python3
"""Export ArtGS slot labels and joint parameters without requiring native GT files."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np


_PALETTE = (
    (78, 205, 196),
    (255, 107, 107),
    (196, 181, 253),
    (250, 204, 21),
    (251, 146, 60),
    (244, 114, 182),
    (74, 222, 128),
    (96, 165, 250),
)


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_labeled_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    """Write an evaluator-compatible ASCII PLY with deterministic slot colors."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if points.shape != (len(labels), 3):
        raise ValueError("points must have shape (N, 3) and match labels")
    colors = np.asarray([_PALETTE[int(label) % len(_PALETTE)] for label in labels])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def main() -> int:
    # These imports intentionally resolve against the released ArtGS checkout.
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from pytorch_lightning import seed_everything
    from scene import Scene
    from scene.deform_model import DeformModel
    from utils.general_utils import safe_state

    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.source_path = f"data/{args.dataset}/{args.subset}/{args.scene_name}"

    safe_state(args.quiet)
    seed_everything(args.seed)
    dataset = model.extract(args)
    with torch.no_grad():
        deform = DeformModel(dataset)
        if not deform.load_weights(dataset.model_path, iteration=args.iteration):
            raise RuntimeError(f"Failed to load ArtGS weights from {dataset.model_path}")
        deform.update(args.iteration)
        gaussians = GaussianModel(dataset.sh_degree)
        Scene(dataset, gaussians, load_iteration=args.iteration)
        state_values = deform.step(gaussians, is_training=False)
        base_points = gaussians.get_xyz.detach()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        states: list[dict[str, Any]] = []
        for state_index, values in enumerate(state_values):
            labels = values["mask"].detach().cpu().numpy().astype(np.int64)
            points = (base_points + values["d_xyz"]).detach().cpu().numpy()
            state_name = "start" if state_index == 0 else "end"
            ply_path = args.output_dir / f"{state_name}_labeled_gaussians.ply"
            write_labeled_ply(ply_path, points, labels)
            states.append(
                {
                    "name": state_name,
                    "point_count": int(len(points)),
                    "predicted_part_count": int(len(np.unique(labels))),
                    "labels": sorted(int(label) for label in np.unique(labels)),
                    "ply": str(ply_path.resolve()),
                }
            )

        joint_types = list(deform.deform.joint_types[1:])
        joint_parameters = deform.deform.get_joint_param(joint_types)
        payload = {
            "schema": "artgs-prediction-adapter-v1",
            "method": "ArtGS",
            "scene_name": args.scene_name,
            "iteration": args.iteration,
            "oracle": {
                "gt_part_count_used_as_num_slots": int(dataset.num_slots),
            },
            "states": states,
            "joint_types": joint_types,
            "joints": _json_value(joint_parameters),
        }
        output_json = args.output_dir / "predictions.json"
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"artgs_predictions": str(output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
