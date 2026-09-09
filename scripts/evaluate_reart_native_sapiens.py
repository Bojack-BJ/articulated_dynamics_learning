#!/usr/bin/env python3
"""Evaluate official ReArt Sapiens artifacts and converted SE(3) joint axes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, rand_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    joints = []
    for result_path in sorted(args.projection_root.glob("sapien_*/result.pkl")):
        row, joint_rows = evaluate_result(result_path)
        rows.append(row)
        joints.extend(joint_rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "reart_native_per_sequence.csv", rows)
    _write_csv(output / "reart_native_converted_joints.csv", joints)
    summary = summarize(rows, joints)
    (output / "reart_native_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def evaluate_result(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    with result_path.open("rb") as stream:
        result = pickle.load(stream)
    checkpoint = torch.load(result_path.with_name("model.pth.tar"), map_location="cpu")
    pred = np.asarray(result["pred_cano_part"], dtype=int)
    gt = np.asarray(result["gt_cano_part"], dtype=int)
    pred_labels = sorted(np.unique(pred).tolist())
    gt_labels = sorted(np.unique(gt).tolist())
    overlap = np.asarray(
        [[np.sum((pred == p) & (gt == g)) for g in gt_labels] for p in pred_labels],
        dtype=float,
    )
    union = np.asarray(
        [[np.sum((pred == p) | (gt == g)) for g in gt_labels] for p in pred_labels],
        dtype=float,
    )
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    row_ids, col_ids = linear_sum_assignment(-iou)
    mapping = {pred_labels[r]: gt_labels[c] for r, c in zip(row_ids, col_ids)}
    matched_iou = [float(iou[r, c]) for r, c in zip(row_ids, col_ids)]
    sequence_id = int(result_path.parent.name.split("_")[-1])
    official = _parse_result_txt(result_path.with_name("result.txt"))
    row = {
        "sequence_index": sequence_id,
        "frame_count": int(np.asarray(result["complete_pc_list"]).shape[0]),
        "point_count": int(len(pred)),
        "predicted_part_count": len(pred_labels),
        "gt_part_count": len(gt_labels),
        "point_iou": float(np.mean(matched_iou)) if matched_iou else None,
        "ari": float(adjusted_rand_score(gt, pred)),
        "rand_index": float(rand_score(gt, pred)),
        "official_per_scan_ri": official.get("per_scan_seg_ri"),
        "official_multi_scan_ri": official.get("multi_scan_seg_ri"),
        "official_recon_error": official.get("recon_err"),
        "official_flow_epe": official.get("flow_epe"),
    }
    return row, _converted_joint_rows(sequence_id, result, checkpoint, mapping)


def _converted_joint_rows(
    sequence_id: int,
    result: dict[str, Any],
    checkpoint: dict[str, Any],
    mapping: dict[int, int],
) -> list[dict[str, Any]]:
    state = checkpoint["state_dict"]
    axes = np.asarray(state["axis_list"].detach().cpu(), dtype=float)
    edge_index = checkpoint["edge_index"]
    joint_types = checkpoint["joint_type_list"]
    gt_poses = np.asarray(result["gt_pose_list"], dtype=float)
    rows = []
    for edge_name, edge_id in edge_index.items():
        child_pred, parent_pred = (int(value) for value in edge_name.split("_"))
        child_gt = mapping.get(child_pred)
        parent_gt = mapping.get(parent_pred)
        gt_type = None
        gt_axis = None
        if child_gt is not None and parent_gt is not None:
            gt_type, gt_axis = _fit_relative_joint(
                gt_poses[:, parent_gt], gt_poses[:, child_gt]
            )
        pred_type = str(joint_types[edge_id])
        pred_axis = _normalize(axes[edge_id])
        type_correct = gt_type is not None and pred_type == gt_type
        axis_error = (
            _axis_error_deg(pred_axis, gt_axis)
            if type_correct and gt_axis is not None
            else None
        )
        rows.append(
            {
                "sequence_index": sequence_id,
                "predicted_edge": edge_name,
                "predicted_parent_part": parent_pred,
                "predicted_child_part": child_pred,
                "mapped_gt_parent_part": parent_gt,
                "mapped_gt_child_part": child_gt,
                "predicted_joint_type": pred_type,
                "converted_gt_joint_type": gt_type,
                "type_correct": type_correct,
                "predicted_axis_x": float(pred_axis[0]),
                "predicted_axis_y": float(pred_axis[1]),
                "predicted_axis_z": float(pred_axis[2]),
                "converted_axis_error_deg": axis_error,
                "metric_scope": "mapped-predicted-edge conditional; GT type/axis fitted from relative GT SE(3)",
            }
        )
    return rows


def _fit_relative_joint(
    parent_poses: np.ndarray, child_poses: np.ndarray
) -> tuple[str | None, np.ndarray | None]:
    transforms = [np.linalg.inv(parent) @ child for parent, child in zip(parent_poses, child_poses)]
    rotations = []
    translations = []
    for transform in transforms:
        axis, angle = _rotation_axis_angle(transform[:3, :3])
        if angle > math.radians(1.0):
            rotations.append((axis, angle))
        translations.append(transform[:3, 3])
    if rotations and max(angle for _, angle in rotations) > math.radians(3.0):
        reference = rotations[0][0]
        aligned = [axis if axis @ reference >= 0 else -axis for axis, _ in rotations]
        weights = np.asarray([angle for _, angle in rotations])
        return "revolute", _normalize(np.average(aligned, axis=0, weights=weights))
    centered = np.asarray(translations) - np.asarray(translations)[0]
    if np.linalg.norm(centered, axis=1).max(initial=0.0) > 1e-4:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return "prismatic", _normalize(vt[0])
    return None, None


def _rotation_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-8:
        return np.asarray([1.0, 0.0, 0.0]), 0.0
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return _normalize(axis), angle


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _axis_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(_normalize(first) @ _normalize(second)), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def _parse_result_txt(path: Path) -> dict[str, float]:
    values = {}
    aliases = {
        "per_scan_seg_ri": "per_scan_seg_ri",
        "multi_scan_seg_ri": "multi_scan_seg_ri",
        "recon_err": "recon_err",
        "flow_epe": "flow_epe",
    }
    for key, value in re.findall(r"([a-z_]+):\s*([-+0-9.eE]+)", path.read_text()):
        if key in aliases:
            values[aliases[key]] = float(value)
    return values


def summarize(rows: list[dict[str, Any]], joints: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in joints if row["converted_axis_error_deg"] is not None]
    return {
        "protocol": "official ReArt Sapiens; 4 frames x 512 points",
        "sequence_count": len(rows),
        "point_iou_mean": _mean(rows, "point_iou"),
        "ari_mean": _mean(rows, "ari"),
        "rand_index_mean": _mean(rows, "rand_index"),
        "official_multi_scan_ri_mean": _mean(rows, "official_multi_scan_ri"),
        "predicted_part_count_mean": _mean(rows, "predicted_part_count"),
        "gt_part_count_mean": _mean(rows, "gt_part_count"),
        "converted_joint_count": len(joints),
        "converted_type_accuracy": (
            float(np.mean([row["type_correct"] for row in joints])) if joints else None
        ),
        "converted_axis_error_mean_deg": _mean(evaluable, "converted_axis_error_deg"),
        "converted_axis_error_median_deg": _median(evaluable, "converted_axis_error_deg"),
        "axis_metric_scope": (
            "Conditional on ReArt predicted edges mapped by canonical segmentation; "
            "GT type/axis are fitted from relative GT SE(3), not an official ReArt metric."
        ),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
