#!/usr/bin/env python3
"""Run the released Ditto model on an exported two-state point cloud pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment", default="Ditto_s2m.yaml")
    return parser.parse_args()


def _normalize(tensor, dim: int):
    return tensor / ((tensor**2).sum(dim, keepdim=True).sqrt() + 1.0e-5)


def _load_model(repo: Path, checkpoint: Path, experiment: str, device):
    import hydra
    import torch
    from hydra.experimental import compose, initialize_config_dir

    config_dir = str((repo / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
    model = hydra.utils.instantiate(config.model)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def _estimate_joint(model, mobile_points, latent):
    import torch

    from src.utils.joint_estimation import aggregate_dense_prediction_r

    with torch.no_grad():
        type_logits, revolute, prismatic = model.model.decode_joints(
            mobile_points, latent
        )
    prismatic_probability = float(type_logits.sigmoid().mean().item())
    points = mobile_points[0].detach().cpu().numpy()
    if prismatic_probability < 0.5:
        axes = _normalize(revolute[:, :, :3], -1)[0].detach().cpu().numpy()
        configurations = revolute[:, :, 3][0].detach().cpu().numpy()
        point_to_line = (
            _normalize(revolute[:, :, 4:7], -1)[0].detach().cpu().numpy()
        )
        distances = revolute[:, :, 7][0].detach().cpu().numpy()
        pivots = points + point_to_line * distances[:, None]
        axis, pivot, configuration = aggregate_dense_prediction_r(
            axes, pivots, configurations, method="mean"
        )
        joint_type = "revolute"
    else:
        axes = _normalize(prismatic[:, :, :3], -1)[0].detach().cpu().numpy()
        axis = axes.mean(axis=0)
        axis /= np.linalg.norm(axis) + 1.0e-12
        configuration = float(
            prismatic[:, :, 3][0].detach().cpu().numpy().mean()
        )
        pivot = points.mean(axis=0)
        joint_type = "prismatic"
    return {
        "joint_type": joint_type,
        "joint_type_prismatic_probability": prismatic_probability,
        "axis_normalized": np.asarray(axis, dtype=float).tolist(),
        "pivot_normalized": np.asarray(pivot, dtype=float).tolist(),
        "configuration": float(configuration),
    }


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Ditto checkpoint not found: {checkpoint}. The official Box "
            "checkpoint links currently return HTTP 404; do not run with "
            "randomly initialized weights."
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Ditto input not found: {input_path}")

    sys.path.insert(0, str(repo))
    import torch

    from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D

    device = torch.device(args.device)
    model = _load_model(repo, checkpoint, args.experiment, device)
    generator = Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type="pointcloud",
        refinement_step=0,
        padding=0.1,
        resolution0=32,
    )

    arrays = np.load(input_path)
    sample = {
        "pc_start": torch.from_numpy(arrays["pc_start"])
        .unsqueeze(0)
        .float()
        .to(device),
        "pc_end": torch.from_numpy(arrays["pc_end"])
        .unsqueeze(0)
        .float()
        .to(device),
    }
    with torch.no_grad():
        meshes, mobile_points, latent, stats = generator.generate_mesh(sample)
    if len(meshes) != 2:
        raise RuntimeError(f"Ditto returned {len(meshes)} meshes; expected 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    meshes[0].export(output_dir / "static_normalized.ply")
    meshes[1].export(output_dir / "mobile_normalized.ply")
    joint = _estimate_joint(model, mobile_points, latent)
    center = np.asarray(arrays["normalization_center"], dtype=float)
    scale = float(arrays["normalization_scale"])
    joint["pivot_world"] = (
        np.asarray(joint["pivot_normalized"]) * scale + center
    ).tolist()
    joint["normalization_center"] = center.tolist()
    joint["normalization_scale"] = scale

    result = {
        "schema": "ditto-native-prediction-v1",
        "checkpoint": str(checkpoint),
        "input": str(input_path),
        "static_mesh": "static_normalized.ply",
        "mobile_mesh": "mobile_normalized.ply",
        "joint": joint,
        "generator_stats": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in stats.items()
            if isinstance(value, (str, int, float)) or hasattr(value, "item")
        },
    }
    (output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
