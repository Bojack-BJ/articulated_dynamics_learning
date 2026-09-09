"""Adapters for released baseline outputs with explicit metric support."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def adapt_paris_output(output_dir: Path) -> dict[str, Any]:
    """Convert one completed PARIS optimization into suite-native metrics."""
    saves = sorted(output_dir.glob("raw/*/save"))
    if not saves:
        raise FileNotFoundError(f"No PARIS save directory under {output_dir}")
    save = saves[-1]
    motions = sorted(save.glob("it*_test_motion.json"))
    validations = sorted(save.glob("it*_val_metrics.json"))
    if not motions or not validations:
        raise FileNotFoundError(f"Incomplete PARIS native output under {save}")

    motion = json.loads(motions[-1].read_text(encoding="utf-8"))
    validation = json.loads(validations[-1].read_text(encoding="utf-8"))
    predicted_type = {
        "rotate": "revolute",
        "translate": "prismatic",
    }.get(motion.get("type"), motion.get("type"))
    gt_type = {
        "rotate": "revolute",
        "translate": "prismatic",
    }.get(motion.get("gt_type"), motion.get("gt_type"))
    motion_metrics = validation.get("motion", {})

    return {
        "status": "success_native_metrics",
        "segmentation": {},
        "kinematics": {
            "predicted_joint_type": predicted_type,
            "gt_joint_type": gt_type,
            "joint_type_correct": predicted_type == gt_type,
            "axis_angle_error_deg": motion_metrics.get("ang_err"),
            "axis_position_error": motion_metrics.get("pos_err"),
            "geometry_distance": motion_metrics.get("geo_dist"),
            "predicted_motion": motion,
        },
        "geometry": {
            "novel_view_psnr": validation.get("nvs", {}).get("psnr"),
            "novel_view_ssim": validation.get("nvs", {}).get("ssim"),
        },
        "metric_support": {
            "segmentation": "unsupported_without_native_part_mesh_export",
            "kinematics": "native",
            "geometry": "native_novel_view_metrics",
        },
        "artifacts": {
            "motion_json": str(motions[-1]),
            "validation_json": str(validations[-1]),
        },
    }


def adapt_gaussianart_output(output_dir: Path) -> dict[str, Any]:
    """Parse GaussianArt's official axis evaluation summary."""
    results = output_dir / "results.txt"
    if not results.exists():
        raise FileNotFoundError(results)
    text = results.read_text(encoding="utf-8")

    def number(label: str) -> float:
        match = re.search(rf"^{re.escape(label)}:\s*([-+0-9.eE]+)$", text, re.M)
        if match is None:
            raise ValueError(f"Missing {label!r} in {results}")
        return float(match.group(1))

    checkpoint = re.search(r"^The best:\s*(\d+)$", text, re.M)
    part_count = re.search(r"^Parts num:\s*(\d+)$", text, re.M)
    if checkpoint is None or part_count is None:
        raise ValueError(f"Incomplete GaussianArt summary in {results}")
    axis_angle = number("Angle mean")
    axis_position_x10 = number("Distance mean")
    motion_error = number("Theta diff mean")
    return {
        "status": "success_native_metrics",
        "oracle": {
            "used_for_inference": True,
            "inputs": ["part_count", "part_semantic_initialization", "gt_motion_metadata"],
        },
        "segmentation": {
            "predicted_part_count": int(part_count.group(1)),
        },
        "kinematics": {
            # These are the released evaluator's aggregate metrics. The
            # position error is multiplied by 10 inside eval_axis.py and is
            # therefore not interchangeable with our bbox-normalized line
            # distance. The evaluator does not report type correctness.
            "axis_angle_error_deg": axis_angle,
            "axis_angle_error_deg_native_all": axis_angle,
            "axis_position_error": axis_position_x10,
            "axis_position_error_native_x10": axis_position_x10,
            "motion_error": motion_error,
            "motion_error_native": motion_error,
        },
        "geometry": {},
        "metric_support": {
            "segmentation": "part_count_only",
            "kinematics": (
                "native_oracle_aggregate; no type-correct conditioning; "
                "position distance is evaluator-native x10"
            ),
            "geometry": "unsupported_without_native_mesh_adapter",
        },
        "artifacts": {
            "results_txt": str(results),
            "selected_checkpoint": int(checkpoint.group(1)),
        },
    }


def adapt_videoartgs_output(output_dir: Path) -> dict[str, Any]:
    """Convert VideoArtGS' official reconstruction export into native metrics."""
    candidates = sorted(
        output_dir.glob("**/final/train/ours_*/joint_info.json"),
        key=lambda path: _iteration_from_parent(path.parent.name),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No VideoArtGS final/train/ours_*/joint_info.json under {output_dir}"
        )
    joint_info_path = candidates[-1]
    reconstruction_dir = joint_info_path.parent
    joints = json.loads(joint_info_path.read_text(encoding="utf-8"))
    if not isinstance(joints, list):
        raise ValueError(f"Expected a joint list in {joint_info_path}")
    movable = [
        joint
        for joint in joints
        if _videoartgs_joint_type(joint) not in {"", "f", "fixed", "static", "heavy"}
    ]
    meshes_dir = reconstruction_dir / "meshes"
    part_meshes = sorted(meshes_dir.glob("part_*.ply"))
    whole_mesh = meshes_dir / "whole_mesh.ply"
    joint_values = reconstruction_dir / "joint_value.npy"
    return {
        "status": "success_native_export",
        "oracle": {
            "used_for_inference": True,
            "inputs": ["joint_count", "joint_types", "parent_topology"],
        },
        "segmentation": {
            "predicted_part_count": len(joints),
        },
        "kinematics": {
            "predicted_joint_count": len(movable),
            "predicted_joints": movable,
        },
        "geometry": {
            "predicted_part_mesh_count": len(part_meshes),
        },
        "metric_support": {
            "segmentation": (
                "native part assignment exported; common point metrics require "
                "GT-domain transfer"
            ),
            "kinematics": (
                "native prediction exported; common errors require frame-aligned "
                "GT evaluation"
            ),
            "geometry": (
                "native TSDF meshes exported; common voxel/Chamfer evaluation "
                "remains required"
            ),
        },
        "artifacts": {
            "joint_info_json": str(joint_info_path),
            "joint_value_npy": str(joint_values) if joint_values.exists() else None,
            "part_meshes": [str(path) for path in part_meshes],
            "whole_mesh": str(whole_mesh) if whole_mesh.exists() else None,
            "selected_iteration": _iteration_from_parent(reconstruction_dir.name),
        },
    }


def _iteration_from_parent(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


def _videoartgs_joint_type(joint: dict[str, Any]) -> str:
    """Read both released and legacy VideoArtGS joint schema variants."""
    return str(
        joint.get(
            "joint_type",
            joint.get("type", joint.get("joint", "")),
        )
    ).lower()
