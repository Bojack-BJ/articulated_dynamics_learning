#!/usr/bin/env python3
"""Evaluate real-scene part and joint predictions against manual annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from rgbd_urdf_mvp.perception.cotracker_features import load_cotracker_feature_map
from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments
from rgbd_urdf_mvp.perception.motion_part_slots import _sample_from_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_tracks", type=Path)
    parser.add_argument("prediction_joints", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument(
        "--features-npz",
        type=Path,
        help="CoTracker features used at inference; required to align embedded slot assignments.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def unit(vector: Any) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 3 or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 else None


def axis_error_deg(left: Any, right: Any) -> float | None:
    a, b = unit(left), unit(right)
    if a is None or b is None:
        return None
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def line_distance(point_a: Any, axis_a: Any, point_b: Any, axis_b: Any) -> float | None:
    p1, p2 = np.asarray(point_a, dtype=np.float64), np.asarray(point_b, dtype=np.float64)
    a, b = unit(axis_a), unit(axis_b)
    if p1.shape != (3,) or p2.shape != (3,) or a is None or b is None:
        return None
    cross = np.cross(a, b)
    norm = float(np.linalg.norm(cross))
    delta = p2 - p1
    if norm > 1e-7:
        return abs(float(np.dot(delta, cross / norm)))
    return float(np.linalg.norm(delta - np.dot(delta, a) * a))


def prediction_joints(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    joints = artifact.get("joints")
    if isinstance(joints, list) and joints:
        return [row for row in joints if isinstance(row, dict)]
    rows = []
    for edge in artifact.get("selected_edges", []):
        if not isinstance(edge, dict):
            continue
        rows.append(
            {
                "name": edge.get("name", f"slot_{edge.get('child_slot_id', -1)}_joint"),
                "joint_type": edge.get("joint_type"),
                "parent_part_id": edge.get("parent_slot_id"),
                "child_part_id": edge.get("child_slot_id"),
                "axis": edge.get("axis_world"),
                "pivot": edge.get("axis_line_point_world"),
            }
        )
    return rows


def track_labels(annotation: dict[str, Any]) -> dict[int, int]:
    raw = annotation.get("track_labels", annotation.get("parts", {}).get("track_labels", {}))
    if isinstance(raw, dict):
        return {int(key): int(value) for key, value in raw.items()}
    labels: dict[int, int] = {}
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and "track_id" in row and "part_id" in row:
                labels[int(row["track_id"])] = int(row["part_id"])
    return labels


def bbox_diagonal(tracks: list[dict[str, Any]]) -> float:
    points = []
    for track in tracks:
        for sample in track.get("samples", []):
            if sample.get("visible", False) and sample.get("xyz_world") is not None:
                points.append(sample["xyz_world"])
    if not points:
        return 1.0
    xyz = np.asarray(points, dtype=np.float64)
    return max(1e-9, float(np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0))))


def main() -> None:
    args = parse_args()
    tracks_artifact = load(args.prediction_tracks)
    joint_artifact = load(args.prediction_joints)
    annotation = load(args.annotation)
    tracks = [row for row in tracks_artifact.get("tracks", []) if isinstance(row, dict)]
    embedded_assignments = joint_artifact.get("track_slot_assignments")
    assignment_by_track_id: dict[int, int] = {}
    if embedded_assignments is not None:
        embedded_track_ids = joint_artifact.get("track_ids")
        if embedded_track_ids is not None:
            if len(embedded_track_ids) != len(embedded_assignments):
                raise ValueError("Embedded track_ids and assignments have different lengths")
            assignment_by_track_id = {
                int(track_id): int(slot)
                for track_id, slot in zip(embedded_track_ids, embedded_assignments)
            }
        elif args.features_npz is None:
            raise ValueError(
                "Prediction contains track_slot_assignments; pass --features-npz "
                "to reproduce inference-time track filtering and ordering"
            )
        else:
            sample = _sample_from_artifact(
                tracks_artifact,
                load_cotracker_feature_map(args.features_npz),
                object_id=str(tracks_artifact.get("object_instance_id", args.prediction_tracks.stem)),
                require_labels=False,
                canonicalize_geometry=False,
            )
            sample_tracks = sample["tracks"]
            if len(sample_tracks) != len(embedded_assignments):
                raise ValueError(
                    "Inference assignment count does not match feature-aligned tracks: "
                    f"{len(embedded_assignments)} != {len(sample_tracks)}"
                )
            assignment_by_track_id = {
                int(track["track_id"]): int(slot)
                for track, slot in zip(sample_tracks, embedded_assignments)
            }
    manual_labels = track_labels(annotation)

    pred, gt, used_ids = [], [], []
    for track in tracks:
        track_id = int(track.get("track_id", -1))
        gt_id = manual_labels.get(track_id, track.get("original_part_id"))
        if gt_id is None:
            continue
        pred_id = assignment_by_track_id.get(
            track_id,
            int(track.get("part_id", track.get("pred_cluster", -1))),
        )
        if pred_id < 0:
            continue
        pred.append(pred_id)
        gt.append(int(gt_id))
        used_ids.append(track_id)
    if not pred:
        raise ValueError("No tracks overlap the manual part annotation")

    pred_ids, gt_ids = sorted(set(pred)), sorted(set(gt))
    overlap = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    for p, g in zip(pred, gt):
        overlap[pred_index[p], gt_index[g]] += 1
    rows, cols = linear_sum_assignment(-overlap)
    pred_to_gt = {pred_ids[int(row)]: gt_ids[int(col)] for row, col in zip(rows, cols) if overlap[row, col] > 0}

    segmentation = evaluate_slot_assignments(np.asarray(pred), np.asarray(gt))
    segmentation.update(
        {
            "evaluated_track_count": len(pred),
            "annotated_track_coverage": len(pred) / max(1, len(tracks)),
            "predicted_part_count": len(pred_ids),
            "gt_part_count": len(gt_ids),
            "pred_to_gt": {str(key): value for key, value in pred_to_gt.items()},
            "overlap_matrix": overlap.tolist(),
        }
    )

    gt_joints = [row for row in annotation.get("joints", []) if isinstance(row, dict)]
    predicted_joint_rows = prediction_joints(joint_artifact)

    # Relation inference may use a virtual root/base slot with no assigned tracks.
    # Map it only when both the predicted and annotated graphs have a unique root.
    gt_parents = {int(row.get("parent_part_id", -1)) for row in gt_joints}
    gt_children = {int(row.get("child_part_id", -1)) for row in gt_joints}
    gt_roots = gt_parents - gt_children
    predicted_parents = {int(row.get("parent_part_id", -1)) for row in predicted_joint_rows}
    predicted_children = {int(row.get("child_part_id", -1)) for row in predicted_joint_rows}
    unmapped_predicted_roots = (predicted_parents - predicted_children) - set(pred_to_gt)
    implicit_root_mapping: dict[int, int] = {}
    if len(gt_roots) == 1 and len(unmapped_predicted_roots) == 1:
        implicit_root_mapping[next(iter(unmapped_predicted_roots))] = next(iter(gt_roots))
    joint_part_mapping = {**pred_to_gt, **implicit_root_mapping}
    segmentation["implicit_root_mapping"] = {
        str(key): value for key, value in implicit_root_mapping.items()
    }
    gt_by_edge = {
        (int(row.get("parent_part_id", -1)), int(row.get("child_part_id", -1))): row
        for row in gt_joints
    }
    diagonal = bbox_diagonal(tracks)
    per_joint = []
    for joint in predicted_joint_rows:
        parent = joint_part_mapping.get(int(joint.get("parent_part_id", -1)))
        child = joint_part_mapping.get(int(joint.get("child_part_id", -1)))
        gt_joint = gt_by_edge.get((parent, child)) if parent is not None and child is not None else None
        row: dict[str, Any] = {
            "name": joint.get("name"),
            "pred_parent_part_id": joint.get("parent_part_id"),
            "pred_child_part_id": joint.get("child_part_id"),
            "mapped_parent_gt": parent,
            "mapped_child_gt": child,
            "matched": gt_joint is not None,
            "predicted_joint_type": joint.get("joint_type"),
        }
        if gt_joint is not None:
            type_correct = str(joint.get("joint_type")) == str(gt_joint.get("joint_type"))
            row.update(
                {
                    "gt_name": gt_joint.get("name"),
                    "gt_joint_type": gt_joint.get("joint_type"),
                    "joint_type_correct": type_correct,
                    "axis_angle_error_deg": axis_error_deg(joint.get("axis"), gt_joint.get("axis")) if type_correct else None,
                }
            )
            if type_correct and str(gt_joint.get("joint_type")) == "revolute":
                distance = line_distance(joint.get("pivot"), joint.get("axis"), gt_joint.get("pivot"), gt_joint.get("axis"))
                row["axis_line_distance_m"] = distance
                row["axis_line_distance_bbox_normalized"] = distance / diagonal if distance is not None else None
        per_joint.append(row)

    matched = [row for row in per_joint if row.get("matched")]
    type_correct = [row for row in matched if row.get("joint_type_correct")]
    angles = [float(row["axis_angle_error_deg"]) for row in type_correct if row.get("axis_angle_error_deg") is not None]
    line_errors = [
        float(row["axis_line_distance_bbox_normalized"])
        for row in type_correct
        if row.get("axis_line_distance_bbox_normalized") is not None
    ]
    kinematics = {
        "gt_joint_count": len(gt_joints),
        "predicted_joint_count": len(per_joint),
        "directed_joint_coverage": len(matched) / max(1, len(gt_joints)),
        "joint_type_accuracy": len(type_correct) / max(1, len(matched)),
        "type_correct_axis_error_deg_mean": float(np.mean(angles)) if angles else None,
        "type_correct_axis_error_deg_median": float(np.median(angles)) if angles else None,
        "revolute_axis_line_bbox_normalized_mean": float(np.mean(line_errors)) if line_errors else None,
        "per_joint": per_joint,
    }
    output = {
        "evaluation_type": "manual_real_scene_part_and_joint_evaluation",
        "prediction_tracks": str(args.prediction_tracks.resolve()),
        "prediction_joints": str(args.prediction_joints.resolve()),
        "annotation": str(args.annotation.resolve()),
        "annotation_source": annotation.get("annotation_source", "manual"),
        "segmentation": segmentation,
        "kinematics": kinematics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json.resolve()), "segmentation": segmentation, "kinematics": kinematics}, indent=2))


if __name__ == "__main__":
    main()
