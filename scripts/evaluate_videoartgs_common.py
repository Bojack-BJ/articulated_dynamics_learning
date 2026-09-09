#!/usr/bin/env python3
"""Evaluate completed VideoArtGS exports on the suite's common GT domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_labeled_points_on_reference,
    read_ascii_labeled_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--surface-samples-per-part", type=int, default=100_000)
    return parser.parse_args()


def _joint_type(value: str) -> str:
    return {
        "r": "revolute",
        "hinge": "revolute",
        "revolute": "revolute",
        "p": "prismatic",
        "s": "static",
        "slider": "prismatic",
        "prismatic": "prismatic",
    }.get(str(value).lower(), str(value).lower())


def _unit(vector: Any) -> np.ndarray | None:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    return result / norm if result.shape == (3,) and norm > 1e-9 else None


def axis_angle_deg(first: Any, second: Any) -> float | None:
    a, b = _unit(first), _unit(second)
    if a is None or b is None:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def axis_line_distance(
    first_origin: Any,
    first_direction: Any,
    second_origin: Any,
    second_direction: Any,
) -> float | None:
    a, b = _unit(first_direction), _unit(second_direction)
    if a is None or b is None:
        return None
    offset = np.asarray(second_origin, dtype=np.float64) - np.asarray(
        first_origin, dtype=np.float64
    )
    normal = np.cross(a, b)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-8:
        return abs(float(offset @ (normal / normal_norm)))
    return float(np.linalg.norm(offset - a * float(offset @ a)))


def _predicted_joint(row: dict[str, Any]) -> dict[str, Any]:
    axis = row.get("jointData", {}).get("axis", {})
    return {
        "type": _joint_type(row.get("joint", row.get("joint_type", ""))),
        "axis": axis.get("direction"),
        "origin": axis.get("origin"),
        "raw": row,
    }


def _gt_joint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": _joint_type(row.get("joint_type", "")),
        "axis": row.get("direction"),
        "origin": row.get("origin"),
        "raw": row,
    }


def evaluate_joints(
    predicted_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    *,
    bbox_diagonal: float,
) -> dict[str, Any]:
    predicted = [_predicted_joint(row) for row in predicted_rows]
    gt = [
        _gt_joint(row)
        for row in gt_rows
        if _joint_type(row.get("joint_type", "")) != "static"
    ]
    if not predicted or not gt:
        return {
            "predicted_joint_count": len(predicted),
            "gt_joint_count": len(gt),
            "matched_joint_count": 0,
            "joint_type_accuracy": None,
            "axis_angle_deg_type_correct": None,
            "revolute_axis_line_bbox": None,
            "matches": [],
        }
    cost = np.empty((len(predicted), len(gt)), dtype=np.float64)
    for pred_index, pred in enumerate(predicted):
        for gt_index, target in enumerate(gt):
            angle = axis_angle_deg(pred["axis"], target["axis"])
            cost[pred_index, gt_index] = (
                (angle if angle is not None else 90.0) / 90.0
                + (0.0 if pred["type"] == target["type"] else 2.0)
            )
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matches = []
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        pred, target = predicted[int(pred_index)], gt[int(gt_index)]
        type_correct = pred["type"] == target["type"]
        angle = axis_angle_deg(pred["axis"], target["axis"]) if type_correct else None
        line = (
            axis_line_distance(
                pred["origin"], pred["axis"], target["origin"], target["axis"]
            )
            if type_correct
            and target["type"] == "revolute"
            and pred["origin"] is not None
            and target["origin"] is not None
            else None
        )
        matches.append(
            {
                "predicted_index": int(pred_index),
                "gt_index": int(gt_index),
                "predicted_type": pred["type"],
                "gt_type": target["type"],
                "type_correct": type_correct,
                "axis_angle_deg": angle,
                "axis_line_distance_m": line,
                "axis_line_distance_bbox": (
                    line / bbox_diagonal if line is not None and bbox_diagonal > 0 else None
                ),
            }
        )
    correct = [row for row in matches if row["type_correct"]]
    angles = [row["axis_angle_deg"] for row in correct if row["axis_angle_deg"] is not None]
    lines = [
        row["axis_line_distance_bbox"]
        for row in correct
        if row["axis_line_distance_bbox"] is not None
    ]
    return {
        "predicted_joint_count": len(predicted),
        "gt_joint_count": len(gt),
        "matched_joint_count": len(matches),
        "joint_type_accuracy": float(np.mean([row["type_correct"] for row in matches])),
        "axis_angle_deg_type_correct": float(np.mean(angles)) if angles else None,
        "revolute_axis_line_bbox": float(np.mean(lines)) if lines else None,
        "matches": matches,
    }


def sample_part_meshes(
    paths: list[Path], samples_per_part: int
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks, label_chunks = [], []
    for label, path in enumerate(paths):
        mesh = trimesh.load(path, process=False, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            continue
        count = min(samples_per_part, max(10_000, len(mesh.faces) * 2))
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=label)
        point_chunks.append(np.asarray(points, dtype=np.float64))
        label_chunks.append(np.full(len(points), label, dtype=np.int64))
    if not point_chunks:
        raise ValueError("No nonempty VideoArtGS part meshes")
    return np.concatenate(point_chunks), np.concatenate(label_chunks)


def symmetric_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return float((np.mean(first_to_second) + np.mean(second_to_first)) / 2.0)


def evaluate_object(
    object_id: str,
    category: str,
    suite_root: Path,
    native_root: Path,
    samples_per_part: int,
) -> dict[str, Any]:
    method_root = suite_root / "per_object" / object_id / "videoartgs"
    native_metrics = json.loads(
        (method_root / "native_metrics.json").read_text(encoding="utf-8")
    )
    scene_root = native_root / "videoartgs" / "realscan" / object_id
    reference = suite_root / "per_object" / object_id / "evaluation_reference" / "two_state_start.ply"
    part_paths = [Path(path) for path in native_metrics["artifacts"]["part_meshes"]]
    points, labels = sample_part_meshes(part_paths, samples_per_part)
    evaluation = evaluate_labeled_points_on_reference(points, labels, reference)
    evaluation_path = method_root / "segmentation_evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    reference_points, _ = read_ascii_labeled_ply(reference, label_mode="part_id")
    primary = evaluation["primary"]
    covered = primary["covered_only_metrics"]
    aware = primary["coverage_aware_metrics"]
    bbox_diagonal = float(evaluation["reference_bbox_diagonal_m"])
    gt_inventory = json.loads((scene_root / "joint_infos.json").read_text(encoding="utf-8"))
    joints = evaluate_joints(
        native_metrics["kinematics"]["predicted_joints"],
        gt_inventory,
        bbox_diagonal=bbox_diagonal,
    )
    return {
        **native_metrics,
        "schema": "external-baseline-result-v1",
        "object_id": object_id,
        "category": category,
        "status": "success_native_metrics",
        "segmentation": {
            "point_iou": aware["one_to_one_mean_iou"],
            "ari": covered["adjusted_rand_index"] if covered else None,
            "ri": covered["rand_index"] if covered else None,
            "predicted_part_count": evaluation["predicted_part_count"],
            "gt_part_count": evaluation["gt_part_count"],
            "undersegmented": evaluation["predicted_part_count"] < evaluation["gt_part_count"],
            "unmatched_gt_part_count": aware["unmatched_gt_part_count"],
            "largest_cluster_ratio": covered["largest_cluster_ratio"] if covered else None,
            "geometry_coverage": primary["geometry_coverage"],
            "metric_domain": "two_state_start_points3d_semantics",
        },
        "kinematics": joints,
        "geometry": {
            **native_metrics.get("geometry", {}),
            "chamfer": symmetric_chamfer(points, reference_points),
        },
        "metric_support": {
            "segmentation": "converted_common_evaluator",
            "kinematics": "converted_against_oracle_inventory_used_by_method",
            "geometry": "common_reference_symmetric_chamfer",
        },
        "artifacts": {
            **native_metrics["artifacts"],
            "segmentation_evaluation": str(evaluation_path),
            "reference": str(reference),
        },
    }


def main() -> int:
    args = parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    native_root = args.native_root.expanduser().resolve()
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        manifest = {row["object_id"]: row for row in csv.DictReader(stream)}
    summary = []
    for native_path in sorted(
        (suite_root / "per_object").glob("*/videoartgs/native_metrics.json")
    ):
        object_id = native_path.parent.parent.name
        try:
            metrics = evaluate_object(
                object_id,
                manifest[object_id]["category"],
                suite_root,
                native_root,
                args.surface_samples_per_part,
            )
        except Exception as error:
            summary.append(
                {
                    "object_id": object_id,
                    "status": "failed_common_evaluation",
                    "exception": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps(summary[-1]), flush=True)
            continue
        metrics_path = native_path.with_name("metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        summary.append({"object_id": object_id, "status": metrics["status"]})
        print(json.dumps(summary[-1]), flush=True)
    print(json.dumps({"results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
