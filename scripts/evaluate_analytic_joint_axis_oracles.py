#!/usr/bin/env python3
"""Evaluate analytic joint axes and decompose relation-pipeline error sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.analytic_joint_axis import (
    AnalyticAxisConfig,
    axis_angle_error_deg,
    axis_line_distance,
    estimate_analytic_joint_axis,
    select_analytic_joint_model,
)
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    JOINT_TYPES,
    _build_relation_model,
    _load_relation_samples,
    _load_slot_model,
    _relation_targets,
    _slot_motion_summary,
    _trajectory_tensors,
)
from rgbd_urdf_mvp.kinematics.so3_augmentation import geometry_registry_report
from rgbd_urdf_mvp.perception.motion_part_slots import _require_torch, _resolve_device
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


SETTING_NAMES = {
    "oracle_gt_parts_analytic": "GT parts + GT edges + GT type + analytic axis",
    "oracle_gt_pair_neural": "GT parts + GT edges + GT type + neural axis",
    "oracle_pred_parts_analytic": "Predicted parts + GT edges + GT type + analytic axis",
    "detected_pred_parts_analytic": "Predicted parts + predicted edges/type + analytic axis",
    "full_neural": "Full neural pipeline",
    "detected_pred_parts_full_analytic": (
        "Predicted parts + predicted edges + analytic type/axis"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("relation_model", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--min-rotation-deg", type=float, default=2.0)
    parser.add_argument("--min-total-rotation-deg", type=float, default=8.0)
    parser.add_argument("--min-total-translation-normalized", type=float, default=0.01)
    parser.add_argument("--catastrophic-threshold-deg", type=float, default=80.0)
    parser.add_argument("--correct-threshold-deg", type=float, default=10.0)
    parser.add_argument(
        "--oracle-geometry-slots", action="store_true",
        help="Feed GT part memberships, mapped to matched slot IDs, to relation geometry.",
    )
    return parser.parse_args()


def load_categories(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    return {
        str(row["object_id"]): str(row.get("category", row.get("source_category", "unknown"))).lower()
        for row in rows
    }


def _canonical_points(sample: dict[str, Any]) -> np.ndarray:
    center = np.asarray(sample["canonical_center_m"], dtype=float)
    scale = max(float(sample["canonical_scale_m"]), 1e-8)
    return (np.asarray(sample["points"], dtype=float) - center) / scale


def _estimate_for_masks(
    sample: dict[str, Any], parent_mask: np.ndarray, child_mask: np.ndarray,
    joint_type: str, config: AnalyticAxisConfig,
) -> dict[str, Any]:
    points = _canonical_points(sample)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    if int(parent_mask.sum()) < config.min_points_per_frame or int(child_mask.sum()) < config.min_points_per_frame:
        return {
            "joint_type": joint_type, "valid": False, "axis": None, "line_point": None,
            "valid_frame_count": 0, "total_motion": 0.0, "residual": math.inf,
            "confidence": 0.0, "track_count_parent": int(parent_mask.sum()),
            "track_count_child": int(child_mask.sum()), "reason": "insufficient_tracks",
        }
    return estimate_analytic_joint_axis(
        points[parent_mask], visibility[parent_mask], points[child_mask], visibility[child_mask],
        joint_type, config,
    ).to_dict()


def _slot_track_statistics(
    sample: dict[str, Any], probabilities: np.ndarray, slot: int, max_tracks: int
) -> dict[str, Any]:
    points = _canonical_points(sample)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    center = np.asarray(sample["canonical_center_m"], dtype=float)
    scale = max(float(sample["canonical_scale_m"]), 1e-8)
    references = (np.asarray(sample["references"], dtype=float) - center) / scale
    valid = visibility.any(axis=1)
    hard_assignment = probabilities.argmax(axis=1)
    hard = valid & (hard_assignment == int(slot))
    ranking = np.where(valid, probabilities[:, int(slot)], -np.inf)
    selected_count = min(max(1, int(max_tracks)), int(valid.sum()))
    selected = (
        np.argsort(-ranking, kind="stable")[:selected_count]
        if selected_count > 0
        else np.asarray([], dtype=int)
    )
    selected = selected[np.isfinite(ranking[selected])]
    raw_weights = probabilities[selected, int(slot)] if len(selected) else np.zeros(0)
    weight_sum = float(raw_weights.sum())
    weights = raw_weights / weight_sum if weight_sum > 1e-8 else np.zeros_like(raw_weights)
    effective = (
        float(1.0 / max(float(np.sum(weights * weights)), 1e-12))
        if weight_sum > 1e-8
        else 0.0
    )
    entropy = 0.0
    if len(weights) > 1 and weight_sum > 1e-8:
        entropy = float(
            -np.sum(weights * np.log(np.maximum(weights, 1e-12))) / math.log(len(weights))
        )
    positions = references[selected]
    if len(positions):
        center_unweighted = positions.mean(axis=0)
        spatial_rms = float(np.sqrt(np.mean(np.sum((positions - center_unweighted) ** 2, axis=1))))
        center_weighted = np.sum(positions * weights[:, None], axis=0)
        weighted_spatial_rms = float(
            np.sqrt(np.sum(weights * np.sum((positions - center_weighted) ** 2, axis=1)))
        )
        pairwise = positions[:, None, :] - positions[None, :, :]
        max_pair_distance = float(np.sqrt(np.max(np.sum(pairwise * pairwise, axis=-1))))
        singular = np.linalg.svd(positions - center_unweighted, compute_uv=False)
        spatial_condition = float(
            singular[0] / max(float(singular[-1]), 1e-8)
        ) if len(singular) else math.inf
    else:
        spatial_rms = weighted_spatial_rms = max_pair_distance = 0.0
        spatial_condition = math.inf
    visibility_lengths = visibility[selected].sum(axis=1).astype(float)
    displacement = []
    for track_index in selected:
        frames = np.flatnonzero(visibility[track_index])
        displacement.append(
            float(np.linalg.norm(points[track_index, frames[-1]] - points[track_index, frames[0]]))
            if len(frames) >= 2 else 0.0
        )
    displacement_array = np.asarray(displacement, dtype=float)
    return {
        "all_track_count": int(len(points)),
        "visible_track_count": int(valid.sum()),
        "hard_assigned_track_count": int(hard.sum()),
        "selected_track_count": int(len(selected)),
        "effective_track_count": effective,
        "pooling_entropy": entropy,
        "spatial_rms": spatial_rms,
        "weighted_spatial_rms": weighted_spatial_rms,
        "max_pair_distance": max_pair_distance,
        "spatial_condition_number": spatial_condition if math.isfinite(spatial_condition) else None,
        "mean_visibility_frames": (
            float(np.sum(weights * visibility_lengths)) if len(weights) else 0.0
        ),
        "mean_total_displacement": (
            float(np.sum(weights * displacement_array)) if len(weights) else 0.0
        ),
    }


def _joint_row(
    *, setting: str, sample: dict[str, Any], relation: dict[str, Any], category: str,
    axis: list[float] | None, line_point: list[float] | None, estimated_type: str,
    edge_detected: bool, estimate: dict[str, Any] | None, track_count: int,
    require_estimate_valid: bool = True,
) -> dict[str, Any]:
    gt_type = str(relation["joint_type"])
    type_correct = estimated_type == gt_type
    valid = axis is not None and (
        not require_estimate_valid or estimate is None or bool(estimate.get("valid"))
    )
    error = axis_angle_error_deg(axis, relation["axis"]) if valid and type_correct else None
    line_error = None
    if valid and type_correct and gt_type == "revolute" and line_point is not None:
        line_error = axis_line_distance(line_point, axis, relation["pivot"], relation["axis"])
    def finite_or_none(value: Any) -> float | None:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    return {
        "setting": setting,
        "setting_description": SETTING_NAMES[setting],
        "object_id": str(sample["object_id"]),
        "category": category,
        "joint_id": str(relation["joint_name"]),
        "joint_type": gt_type,
        "estimated_joint_type": estimated_type,
        "edge_detected": bool(edge_detected),
        "type_correct": bool(type_correct),
        "valid": bool(valid),
        "gt_axis": [float(value) for value in relation["axis"]],
        "estimated_axis": [float(value) for value in axis] if axis is not None else None,
        "gt_line_point": [float(value) for value in relation["pivot"]],
        "estimated_line_point": [float(value) for value in line_point] if line_point is not None else None,
        "axis_error_deg": error,
        "axis_line_error_bbox_normalized": line_error,
        "track_count": int(track_count),
        "valid_frame_count": int((estimate or {}).get("valid_frame_count", 0)),
        "motion_magnitude": finite_or_none((estimate or {}).get("total_motion", math.nan)),
        "analytic_residual": finite_or_none((estimate or {}).get("residual", math.nan)),
        "analytic_confidence": finite_or_none((estimate or {}).get("confidence", math.nan)),
        "analytic_reason": (estimate or {}).get("reason"),
    }


def evaluate_samples(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str, categories: dict[str, str],
    config: AnalyticAxisConfig, edge_threshold: float,
    oracle_geometry_slots: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slot_model.eval()
    relation_model.eval()
    with torch.no_grad():
        for sample in samples:
            features = torch.from_numpy(sample["features"]).to(device)
            logits, _, slots = slot_model((features - mean) / std, return_slots=True)
            probabilities = torch.softmax(logits, dim=-1)
            target = _relation_targets(sample, logits, int(slots.shape[0]), torch, device)
            geometry_probabilities = probabilities
            if oracle_geometry_slots:
                labels_tensor = torch.as_tensor(
                    sample["labels"], dtype=torch.long, device=device
                )
                geometry_probabilities = probabilities.clone()
                label_to_slot: dict[int, int] = {}
                for relation in target["relations"]:
                    label_to_slot[int(relation["parent_label"])] = int(relation["parent_slot"])
                    label_to_slot[int(relation["child_label"])] = int(relation["child_slot"])
                for label, slot in label_to_slot.items():
                    selected = labels_tensor == label
                    geometry_probabilities[selected] = 0.0
                    geometry_probabilities[selected, slot] = 1.0
            trajectory_tokens, trajectory_visibility = _trajectory_tensors(
                sample, torch, device,
                sample_count=int(getattr(relation_model, "trajectory_samples", 32)),
            )
            prediction = relation_model(
                slots,
                _slot_motion_summary(
                    features, geometry_probabilities, int(sample["embedding_dim"]), torch
                ),
                trajectory_tokens=trajectory_tokens,
                trajectory_visibility=trajectory_visibility,
                slot_probabilities=geometry_probabilities,
            )
            predicted_assignment = geometry_probabilities.argmax(dim=-1).cpu().numpy()
            probabilities_np = geometry_probabilities.cpu().numpy()
            labels = np.asarray(sample["labels"], dtype=int)
            category = categories.get(str(sample["object_id"]), "unknown")
            for relation in target["relations"]:
                parent_slot = int(relation["parent_slot"])
                child_slot = int(relation["child_slot"])
                max_tracks = int(
                    getattr(relation_model, "geometry_max_tracks", len(probabilities_np))
                )
                parent_track_stats = _slot_track_statistics(
                    sample, probabilities_np, parent_slot, max_tracks
                )
                child_track_stats = _slot_track_statistics(
                    sample, probabilities_np, child_slot, max_tracks
                )
                gt_type = str(relation["joint_type"])
                predicted_type = JOINT_TYPES[
                    int(prediction["type_logits"][parent_slot, child_slot].argmax().cpu())
                ]
                edge_probability = float(torch.sigmoid(
                    prediction["edge_logits"][parent_slot, child_slot]
                ).cpu())
                edge_detected = edge_probability >= edge_threshold
                gt_parent = labels == int(relation["parent_label"])
                gt_child = labels == int(relation["child_label"])
                pred_parent = predicted_assignment == parent_slot
                pred_child = predicted_assignment == child_slot
                analytic_gt = _estimate_for_masks(sample, gt_parent, gt_child, gt_type, config)
                analytic_pred_gt_type = _estimate_for_masks(
                    sample, pred_parent, pred_child, gt_type, config
                )
                analytic_pred_pred_type = _estimate_for_masks(
                    sample, pred_parent, pred_child, predicted_type, config
                )
                points = _canonical_points(sample)
                visibility = np.asarray(sample["visibility"], dtype=bool)
                analytic_selection = select_analytic_joint_model(
                    points[pred_parent], visibility[pred_parent],
                    points[pred_child], visibility[pred_child], config,
                )
                neural_axis = [
                    float(value) for value in prediction["axes"][parent_slot, child_slot].cpu()
                ]
                neural_pivot = [
                    float(value) for value in prediction["pivots"][parent_slot, child_slot].cpu()
                ]
                rows.extend([
                    _joint_row(
                        setting="oracle_gt_parts_analytic", sample=sample, relation=relation,
                        category=category, axis=analytic_gt.get("axis"),
                        line_point=analytic_gt.get("line_point"), estimated_type=gt_type,
                        edge_detected=True, estimate=analytic_gt,
                        track_count=int(gt_parent.sum() + gt_child.sum()),
                    ),
                    _joint_row(
                        setting="oracle_gt_pair_neural", sample=sample, relation=relation,
                        category=category, axis=neural_axis, line_point=neural_pivot,
                        estimated_type=gt_type, edge_detected=True, estimate=analytic_gt,
                        track_count=int(gt_parent.sum() + gt_child.sum()),
                        require_estimate_valid=False,
                    ),
                    _joint_row(
                        setting="oracle_pred_parts_analytic", sample=sample, relation=relation,
                        category=category, axis=analytic_pred_gt_type.get("axis"),
                        line_point=analytic_pred_gt_type.get("line_point"), estimated_type=gt_type,
                        edge_detected=True, estimate=analytic_pred_gt_type,
                        track_count=int(pred_parent.sum() + pred_child.sum()),
                    ),
                    _joint_row(
                        setting="detected_pred_parts_analytic", sample=sample, relation=relation,
                        category=category, axis=analytic_pred_pred_type.get("axis"),
                        line_point=analytic_pred_pred_type.get("line_point"),
                        estimated_type=predicted_type, edge_detected=edge_detected,
                        estimate=analytic_pred_pred_type,
                        track_count=int(pred_parent.sum() + pred_child.sum()),
                    ),
                    _joint_row(
                        setting="full_neural", sample=sample, relation=relation,
                        category=category, axis=neural_axis, line_point=neural_pivot,
                        estimated_type=predicted_type, edge_detected=edge_detected,
                        estimate=analytic_pred_gt_type,
                        track_count=int(pred_parent.sum() + pred_child.sum()),
                        require_estimate_valid=False,
                    ),
                    _joint_row(
                        setting="detected_pred_parts_full_analytic",
                        sample=sample, relation=relation, category=category,
                        axis=(
                            analytic_selection.selected_estimate.axis
                            if analytic_selection.selected_estimate is not None else None
                        ),
                        line_point=(
                            analytic_selection.selected_estimate.line_point
                            if analytic_selection.selected_estimate is not None else None
                        ),
                        estimated_type=str(analytic_selection.selected_type),
                        edge_detected=edge_detected,
                        estimate=(
                            analytic_selection.selected_estimate.to_dict()
                            if analytic_selection.selected_estimate is not None else None
                        ),
                        track_count=int(pred_parent.sum() + pred_child.sum()),
                    ),
                ])
                for row in rows[-6:]:
                    row["edge_probability"] = edge_probability
                    row["parent_slot"] = parent_slot
                    row["child_slot"] = child_slot
                    for prefix, statistics in (
                        ("parent", parent_track_stats),
                        ("child", child_track_stats),
                    ):
                        for name, value in statistics.items():
                            row[f"{prefix}_{name}"] = value
                    row["gt_relative_motion_magnitude"] = (
                        float(analytic_gt["total_motion"])
                        if analytic_gt.get("valid")
                        and math.isfinite(float(analytic_gt.get("total_motion", math.nan)))
                        else None
                    )
                rows[-1]["analytic_model_selection"] = analytic_selection.to_dict()
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["valid"] and row["type_correct"] and row["edge_detected"]
        and row["axis_error_deg"] is not None
    ]
    errors = [float(row["axis_error_deg"]) for row in eligible]
    lines = [
        float(row["axis_line_error_bbox_normalized"])
        for row in eligible if row["axis_line_error_bbox_normalized"] is not None
    ]
    return {
        "joint_count": len(rows),
        "edge_detected_count": sum(bool(row["edge_detected"]) for row in rows),
        "type_correct_count": sum(bool(row["type_correct"]) for row in rows),
        "axis_evaluated_count": len(errors),
        "axis_error_mean_deg": float(np.mean(errors)) if errors else None,
        "axis_error_median_deg": _percentile(errors, 50),
        "axis_error_p75_deg": _percentile(errors, 75),
        "axis_error_p90_deg": _percentile(errors, 90),
        "axis_error_above_10_count": sum(value > 10.0 for value in errors),
        "axis_error_above_30_count": sum(value > 30.0 for value in errors),
        "axis_error_above_60_count": sum(value > 60.0 for value in errors),
        "axis_error_above_80_count": sum(value > 80.0 for value in errors),
        "axis_line_error_mean_bbox_normalized": float(np.mean(lines)) if lines else None,
        "axis_line_error_median_bbox_normalized": _percentile(lines, 50),
        "axis_line_error_count": len(lines),
    }


def _bin(value: float, boundaries: list[float], labels: list[str]) -> str:
    if not math.isfinite(value):
        return "unavailable"
    for boundary, label in zip(boundaries, labels):
        if value < boundary:
            return label
    return labels[-1]


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dimensions = {
        "by_category": lambda row: str(row["category"]),
        "by_joint_type": lambda row: str(row["joint_type"]),
        "by_motion_magnitude": lambda row: _bin(
            float(row["motion_magnitude"]) if row["motion_magnitude"] is not None else math.nan,
            [0.03, 0.1, 0.3, math.inf],
            ["very_low", "low", "medium", "high"],
        ),
        "by_track_count": lambda row: _bin(
            float(row["track_count"]), [40, 100, 250, math.inf],
            ["small", "medium", "large", "very_large"],
        ),
        "by_confidence": lambda row: _bin(
            float(row["analytic_confidence"]) if row["analytic_confidence"] is not None else math.nan,
            [0.25, 0.5, 0.75, math.inf],
            ["very_low", "low", "medium", "high"],
        ),
    }
    for name, key_fn in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        result[name] = {key: metric_summary(value) for key, value in sorted(groups.items())}
    return result


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settings = {}
    for setting in SETTING_NAMES:
        selected = [row for row in rows if row["setting"] == setting]
        settings[setting] = {
            "description": SETTING_NAMES[setting],
            "overall": metric_summary(selected),
            **grouped_summaries(selected),
        }
    return {
        "settings": settings,
        "paired_comparison": paired_comparison(rows),
        "end_to_end": end_to_end_metrics(rows),
        "track_selection_diagnostics": track_selection_diagnostics(rows),
        "coordinate_frame_audit": coordinate_frame_audit(),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "median": _percentile(finite, 50),
        "p10": _percentile(finite, 10),
        "p90": _percentile(finite, 90),
    }


def _pearson(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    pairs = [
        (float(row[field]), float(row["axis_error_deg"]))
        for row in rows
        if row.get(field) is not None and row.get("axis_error_deg") is not None
        and math.isfinite(float(row[field])) and math.isfinite(float(row["axis_error_deg"]))
    ]
    if len(pairs) < 2:
        return {"count": len(pairs), "pearson_r": None}
    values = np.asarray(pairs, dtype=float)
    if float(values[:, 0].std()) < 1e-12 or float(values[:, 1].std()) < 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return {"count": len(pairs), "pearson_r": correlation}


def track_selection_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Relate slot track support and geometry coverage to neural axis failures."""
    neural = [
        row for row in rows
        if row["setting"] == "full_neural" and row.get("axis_error_deg") is not None
    ]
    fields = (
        "child_visible_track_count",
        "child_hard_assigned_track_count",
        "child_selected_track_count",
        "child_effective_track_count",
        "child_pooling_entropy",
        "child_spatial_rms",
        "child_weighted_spatial_rms",
        "child_max_pair_distance",
        "child_spatial_condition_number",
        "child_mean_visibility_frames",
        "child_mean_total_displacement",
        "gt_relative_motion_magnitude",
        "analytic_residual",
        "analytic_confidence",
    )
    catastrophic = [row for row in neural if float(row["axis_error_deg"]) > 80.0]
    regular = [row for row in neural if float(row["axis_error_deg"]) <= 80.0]
    return {
        "scope": (
            "full_neural rows with a type-correct, evaluable axis; GT is used only "
            "for diagnostic error and relative-motion stratification"
        ),
        "selection_stages": {
            "visible": "tracks visible in at least one sampled frame",
            "hard_assigned": "visible tracks whose highest slot probability is this slot",
            "selected": "deterministic top-k tracks by soft slot probability",
            "effective": "inverse squared normalized soft-weight sum among selected tracks",
        },
        "joint_count": len(neural),
        "catastrophic_axis_error_above_80_count": len(catastrophic),
        "distributions": {
            field: _distribution([row[field] for row in neural if row.get(field) is not None])
            for field in fields
        },
        "axis_error_correlations": {
            field: _pearson(neural, field) for field in fields
        },
        "catastrophic_vs_other": {
            field: {
                "catastrophic": _distribution(
                    [row[field] for row in catastrophic if row.get(field) is not None]
                ),
                "other": _distribution(
                    [row[field] for row in regular if row.get(field) is not None]
                ),
            }
            for field in fields
        },
    }


def coordinate_frame_audit() -> dict[str, Any]:
    return {
        "frame_chain": [
            {
                "field": "RGB and CoTracker image features",
                "frame": "camera/image frame",
                "current_so3_rule": "unchanged",
            },
            {
                "field": "camera extrinsics",
                "frame": "world_from_camera during RGB-D lifting",
                "current_so3_rule": "not consumed after 3D tracks are lifted",
            },
            {
                "field": "raw lifted 3D tracks",
                "frame": "recording world frame",
                "current_so3_rule": (
                    "Q @ x for relation trajectories; explicit slot geometry is also "
                    "rotated only under slot_and_relation_geometry scope"
                ),
            },
            {
                "field": "relation geometry and trajectory tokens",
                "frame": (
                    "centered/scaled object-canonical coordinates without rotational "
                    "canonicalization; axes remain aligned with the recording world basis"
                ),
                "current_so3_rule": "Q @ x or Q @ v",
            },
            {
                "field": "GT joint axis and pivot",
                "frame": "same centered/scaled orientation basis as relation geometry",
                "current_so3_rule": "axis'=Q@axis; pivot'=Q@pivot",
            },
        ],
        "consistency_conclusion": (
            "The current augmentation is internally consistent as a post-lifting change "
            "of 3D coordinate basis: geometry'=Q geometry, axis'=Q axis, pivot'=Q pivot. "
            "Camera extrinsics should not be transformed because they are not model inputs "
            "at this stage."
        ),
        "physical_scene_rotation_caveat": (
            "It is not equivalent to physically rotating the object relative to the camera. "
            "A physical augmentation would require rerendering/retracking or consistently "
            "changing object-camera pose and recomputing visual features."
        ),
        "geometry_registry": geometry_registry_report(),
    }


def _paired_method_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["axis_error_deg"]) for row in rows]
    return {
        "count": len(errors),
        "mean_deg": float(np.mean(errors)) if errors else None,
        "median_deg": _percentile(errors, 50),
        "p75_deg": _percentile(errors, 75),
        "p90_deg": _percentile(errors, 90),
        "above_80_count": sum(value > 80.0 for value in errors),
    }


def paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare neural and analytic axes on exactly the same GT joints."""
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["object_id"], row["joint_id"])][row["setting"]] = row
    pairs = []
    for settings in by_key.values():
        neural = settings.get("oracle_gt_pair_neural")
        analytic = settings.get("oracle_gt_parts_analytic")
        if neural is None or analytic is None or not analytic["valid"]:
            continue
        if neural["axis_error_deg"] is None or analytic["axis_error_deg"] is None:
            continue
        pairs.append({"neural": neural, "analytic": analytic})

    subsets = {
        "intersection_valid": pairs,
        "analytic_confidence_above_0_5": [
            pair for pair in pairs
            if (pair["analytic"]["analytic_confidence"] or 0.0) > 0.5
        ],
        "analytic_confidence_above_0_8": [
            pair for pair in pairs
            if (pair["analytic"]["analytic_confidence"] or 0.0) > 0.8
        ],
        "analytic_residual_below_0_05": [
            pair for pair in pairs
            if pair["analytic"]["analytic_residual"] is not None
            and pair["analytic"]["analytic_residual"] < 0.05
        ],
        "neural_analytic_disagreement_above_60": [
            pair for pair in pairs
            if axis_angle_error_deg(
                pair["neural"]["estimated_axis"], pair["analytic"]["estimated_axis"]
            ) > 60.0
        ],
    }
    output = {}
    for name, subset in subsets.items():
        output[name] = {
            "joint_count": len(subset),
            "neural": _paired_method_metrics([pair["neural"] for pair in subset]),
            "analytic": _paired_method_metrics([pair["analytic"] for pair in subset]),
            "analytic_better_count": sum(
                float(pair["analytic"]["axis_error_deg"])
                < float(pair["neural"]["axis_error_deg"])
                for pair in subset
            ),
        }
    return output


def end_to_end_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Penalize missing edges and wrong types instead of conditioning them away."""
    result = {}
    for setting in (
        "detected_pred_parts_analytic",
        "detected_pred_parts_full_analytic",
        "full_neural",
    ):
        selected = [row for row in rows if row["setting"] == setting]
        penalized = [
            float(row["axis_error_deg"])
            if row["edge_detected"] and row["type_correct"] and row["axis_error_deg"] is not None
            else 90.0
            for row in selected
        ]
        result[setting] = {
            "joint_count": len(selected),
            "edge_recall": sum(bool(row["edge_detected"]) for row in selected) / max(1, len(selected)),
            "type_accuracy": sum(bool(row["type_correct"]) for row in selected) / max(1, len(selected)),
            "failure_penalized_axis_error_mean_deg": float(np.mean(penalized)) if penalized else None,
            "joint_success_at_10_deg": sum(value < 10.0 for value in penalized) / max(1, len(penalized)),
            "joint_success_at_20_deg": sum(value < 20.0 for value in penalized) / max(1, len(penalized)),
        }
    return result


def catastrophic_audit(
    rows: list[dict[str, Any]], catastrophic_threshold: float, correct_threshold: float
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["object_id"], row["joint_id"])][row["setting"]] = row
    audit_rows = []
    contingency = {
        "neural_wrong_analytic_correct": 0,
        "neural_wrong_analytic_wrong": 0,
        "neural_correct_analytic_wrong": 0,
        "both_correct": 0,
        "unavailable": 0,
    }
    for settings in by_key.values():
        neural = settings.get("oracle_gt_pair_neural")
        analytic = settings.get("oracle_gt_parts_analytic")
        full = settings.get("full_neural")
        if neural is None or analytic is None:
            continue
        neural_error = neural.get("axis_error_deg")
        analytic_error = analytic.get("axis_error_deg")
        if neural_error is None:
            contingency["unavailable"] += 1
            continue
        if analytic_error is None:
            contingency["unavailable"] += 1
            if float(neural_error) > catastrophic_threshold:
                audit_rows.append({
                    "object_id": neural["object_id"], "category": neural["category"],
                    "joint_id": neural["joint_id"], "joint_type": neural["joint_type"],
                    "gt_axis": neural["gt_axis"], "neural_axis": neural["estimated_axis"],
                    "analytic_axis": analytic["estimated_axis"],
                    "neural_error_deg": neural_error, "analytic_error_deg": None,
                    "edge_detected": bool((full or {}).get("edge_detected", False)),
                    "type_correct": bool((full or {}).get("type_correct", False)),
                    "track_count": analytic["track_count"],
                    "valid_frames": analytic["valid_frame_count"],
                    "motion_magnitude": analytic["motion_magnitude"],
                    "analytic_confidence": analytic["analytic_confidence"],
                    "analytic_residual": analytic["analytic_residual"],
                    "analytic_reason": analytic["analytic_reason"],
                })
            continue
        neural_correct = float(neural_error) <= correct_threshold
        analytic_correct = float(analytic_error) <= correct_threshold
        if neural_correct and analytic_correct:
            contingency["both_correct"] += 1
        elif neural_correct:
            contingency["neural_correct_analytic_wrong"] += 1
        elif analytic_correct:
            contingency["neural_wrong_analytic_correct"] += 1
        else:
            contingency["neural_wrong_analytic_wrong"] += 1
        if float(neural_error) <= catastrophic_threshold:
            continue
        audit_rows.append({
            "object_id": neural["object_id"], "category": neural["category"],
            "joint_id": neural["joint_id"], "joint_type": neural["joint_type"],
            "gt_axis": neural["gt_axis"], "neural_axis": neural["estimated_axis"],
            "analytic_axis": analytic["estimated_axis"],
            "neural_error_deg": neural_error, "analytic_error_deg": analytic_error,
            "edge_detected": bool((full or {}).get("edge_detected", False)),
            "type_correct": bool((full or {}).get("type_correct", False)),
            "track_count": analytic["track_count"],
            "valid_frames": analytic["valid_frame_count"],
            "motion_magnitude": analytic["motion_magnitude"],
            "analytic_confidence": analytic["analytic_confidence"],
            "analytic_residual": analytic["analytic_residual"],
            "analytic_reason": analytic["analytic_reason"],
        })
    return {"thresholds": {
        "catastrophic_deg": catastrophic_threshold, "correct_deg": correct_threshold,
    }, "contingency": contingency, "joints": audit_rows}


def _csv_value(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in keys} for row in rows)


def main() -> int:
    args = parse_args()
    torch = _require_torch()
    device = _resolve_device(torch, args.device)
    slot_checkpoint = torch.load(args.slot_model.expanduser().resolve(), map_location="cpu", weights_only=True)
    relation_checkpoint = torch.load(
        args.relation_model.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    slot_model = _load_slot_model(slot_checkpoint, torch, device)
    if relation_checkpoint.get("slot_state_dict") is not None:
        slot_model.load_state_dict(relation_checkpoint["slot_state_dict"])
    relation_model = _build_relation_model(
        torch, slot_dim=int(relation_checkpoint["slot_dim"]),
        hidden_dim=int(relation_checkpoint["hidden_dim"]),
        motion_summary_dim=int(relation_checkpoint.get("motion_summary_dim", 0)),
        axis_geometry_branch=bool(relation_checkpoint.get("axis_geometry_branch", False)),
        axis_head_type=str(relation_checkpoint.get("axis_head_type", "direct")),
        vector_pivot_parameterization=str(
            relation_checkpoint.get("vector_pivot_parameterization", "legacy_center_delta")
        ),
        geometry_encoder_type=str(
            relation_checkpoint.get("geometry_encoder_type", "track_gru_average")
        ),
        trajectory_hidden_dim=int(relation_checkpoint.get("trajectory_hidden_dim", 128)),
        geometry_max_tracks=int(relation_checkpoint.get("geometry_max_tracks", 64)),
        geometry_attention_heads=int(
            relation_checkpoint.get("geometry_attention_heads", 4)
        ),
        geometry_transformer_layers=int(
            relation_checkpoint.get("geometry_transformer_layers", 1)
        ),
        joint_type_head_type=str(
            relation_checkpoint.get("joint_type_head_type", "pair_context")
        ),
        edge_head_type=str(relation_checkpoint.get("edge_head_type", "pair_context")),
    ).to(device)
    relation_model.load_state_dict(relation_checkpoint["state_dict"])
    relation_model.trajectory_samples = int(
        relation_checkpoint.get("trajectory_samples", 32)
    )
    samples = _load_relation_samples(
        load_pairwise_manifest(args.manifest.expanduser().resolve()), args.split, slot_checkpoint
    )
    config = AnalyticAxisConfig(
        min_rotation_rad=math.radians(args.min_rotation_deg),
        min_total_rotation_rad=math.radians(args.min_total_rotation_deg),
        min_total_translation=args.min_total_translation_normalized,
    )
    rows = evaluate_samples(
        samples, slot_model, relation_model, slot_checkpoint["feature_mean"].to(device),
        slot_checkpoint["feature_std"].to(device), torch, device, load_categories(args.catalog),
        config, args.edge_threshold,
        oracle_geometry_slots=bool(args.oracle_geometry_slots),
    )
    report = build_report(rows)
    report["split"] = args.split
    report["config"] = vars(args) | {"analytic": config.__dict__ if hasattr(config, "__dict__") else {
        field: getattr(config, field) for field in config.__slots__
    }}
    report["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["config"].items()}
    audit = catastrophic_audit(rows, args.catastrophic_threshold_deg, args.correct_threshold_deg)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analytic_axis_oracle_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "analytic_axis_per_joint.json").write_text(
        json.dumps({"joints": rows}, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "catastrophic_axis_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(output / "analytic_axis_per_joint.csv", rows)
    write_csv(output / "catastrophic_axis_audit.csv", audit["joints"])
    print(json.dumps({
        "output_dir": str(output), "object_count": len(samples),
        "joint_count": len(rows) // len(SETTING_NAMES),
        "catastrophic_joint_count": len(audit["joints"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
