#!/usr/bin/env python3
"""Evaluate one GaussianArt checkpoint with the suite-common joint metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rgbd_urdf_mvp.benchmarks.gaussianart_evaluation import (
    JointTransform,
    evaluate_joint_transforms,
    transform_to_joint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("gt_transforms", type=Path)
    parser.add_argument("--bbox-diagonal", type=float, required=True)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    iteration = args.checkpoint or _selected_iteration(args.model_dir / "results.txt")
    checkpoint_path = args.model_dir / "ckpts" / f"ours_{iteration}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = checkpoint["articulation_params"]
    rotations = _quaternions_to_matrices(parameters["art_R"].detach().cpu().numpy())
    translations = parameters["art_T"].detach().cpu().numpy()

    # GaussianArt stores the static/base slot last, matching its released
    # evaluator. It has no associated parent joint.
    predicted = [
        transform_to_joint(rotation, translation)
        for rotation, translation in zip(rotations[:-1], translations[:-1])
    ]
    ground_truth = _load_ground_truth(args.gt_transforms)
    evaluation = evaluate_joint_transforms(
        predicted,
        ground_truth,
        bbox_diagonal=args.bbox_diagonal,
    )
    payload = {
        "schema": "gaussianart-common-kinematics-v1",
        "selected_checkpoint": iteration,
        "checkpoint": str(checkpoint_path),
        "gt_transforms": str(args.gt_transforms),
        "bbox_diagonal": args.bbox_diagonal,
        "oracle": {
            "used_for_inference": True,
            "inputs": [
                "part_count",
                "part_semantic_initialization",
                "gt_motion_metadata",
            ],
        },
        "evaluation": evaluation,
    }
    output = args.output or args.model_dir / "common_kinematics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _selected_iteration(results: Path) -> int:
    match = re.search(r"^The best:\s*(\d+)$", results.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise ValueError(f"Cannot find selected checkpoint in {results}")
    return int(match.group(1))


def _quaternions_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    # GaussianArt uses scalar-first (w, x, y, z); SciPy uses scalar-last.
    return Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()


def _load_ground_truth(path: Path) -> list[JointTransform]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transforms = payload["trans_info"]
    if isinstance(transforms, dict):
        transforms = [transforms]
    coordinate_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ).T
    joints: list[JointTransform] = []
    for transform in transforms:
        native_type = str(transform["type"])
        axis = coordinate_rotation @ np.asarray(transform["axis"]["d"], dtype=float)
        pivot = coordinate_rotation @ np.asarray(transform["axis"]["o"], dtype=float)
        lower = float(transform[native_type]["l"])
        upper = float(transform[native_type]["r"])
        travel = upper - lower
        if native_type == "rotate":
            rotation = Rotation.from_rotvec(axis * np.deg2rad(travel)).as_matrix()
            translation = (np.eye(3) - rotation) @ pivot
            joint = transform_to_joint(rotation, translation)
        else:
            rotation = np.eye(3)
            translation = axis * travel
            joint = transform_to_joint(rotation, translation)
        joints.append(
            JointTransform(
                joint_type=joint.joint_type,
                axis=joint.axis,
                pivot=pivot if joint.joint_type == "revolute" else joint.pivot,
                rotation=joint.rotation,
                translation=joint.translation,
                motion=abs(joint.motion),
            )
        )
    return joints


if __name__ == "__main__":
    raise SystemExit(main())
