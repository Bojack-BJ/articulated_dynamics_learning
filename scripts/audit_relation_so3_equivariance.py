#!/usr/bin/env python3
"""Audit SO(3) sampling, geometry consistency, and Relation Head equivariance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    JOINT_TYPES,
    _augment_relation_sample,
    _apply_slot_geometry_representation,
    _apply_slot_input_mode,
    _build_relation_model,
    _load_relation_samples,
    _load_slot_model,
    _relation_targets,
    _slot_motion_summary,
    _trajectory_tensors,
)
from rgbd_urdf_mvp.kinematics.so3_augmentation import (
    geometry_registry_report,
    nearest_canonical_axis_angle_deg,
    nearest_canonical_axis_id,
    sample_uniform_so3,
    sample_uniform_so3_batch,
)
from rgbd_urdf_mvp.perception.motion_part_slots import _require_torch, _resolve_device
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("relation_model", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--sampler-count", type=int, default=100_000)
    parser.add_argument("--rotations-per-sample", type=int, default=1)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--catastrophic-threshold-deg", type=float, default=80.0)
    parser.add_argument(
        "--slot-input-mode", choices=("full", "geometry_only", "embedding_only"), default="full",
        help="Apply the same slot-input ablation used during training.",
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


def _axis_error(axis_a: Any, axis_b: Any) -> float:
    a = np.asarray(axis_a, dtype=float)
    b = np.asarray(axis_b, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(float(np.clip(abs(a @ b), 0.0, 1.0))))


def _child_motion(sample: dict[str, Any], child_label: int) -> float:
    points = np.asarray(sample["points"], dtype=float)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    mask = np.asarray(sample["labels"]) == int(child_label)
    motions = []
    for trajectory, visible in zip(points[mask], visibility[mask]):
        frames = np.flatnonzero(visible)
        if len(frames) >= 2:
            motions.append(float(np.linalg.norm(trajectory[frames[-1]] - trajectory[frames[0]])))
    scale = max(float(sample["canonical_scale_m"]), 1e-8)
    return float(np.median(motions) / scale) if motions else 0.0


def _motion_bin(value: float) -> str:
    if value < 0.03:
        return "very_low"
    if value < 0.1:
        return "low"
    if value < 0.3:
        return "medium"
    return "high"


def _observability_bin(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    if value < 0.02:
        return "very_low"
    if value < 0.1:
        return "low"
    if value < 0.3:
        return "medium"
    return "high"


def _predict(sample: dict[str, Any], slot_model: Any, relation_model: Any, mean: Any, std: Any, torch: Any, device: str) -> tuple[Any, Any, Any]:
    relation_features = torch.from_numpy(sample["features"]).to(device)
    slot_features = torch.from_numpy(sample.get("slot_features", sample["features"])).to(device)
    logits, _, slots = slot_model((slot_features - mean) / std, return_slots=True)
    probabilities = torch.softmax(logits, dim=-1)
    trajectory_tokens, trajectory_visibility = _trajectory_tensors(
        sample, torch, device,
        sample_count=int(getattr(relation_model, "trajectory_samples", 32)),
    )
    prediction = relation_model(
        slots,
        _slot_motion_summary(
            relation_features, probabilities, int(sample["embedding_dim"]), torch
        ),
        trajectory_tokens=trajectory_tokens,
        trajectory_visibility=trajectory_visibility,
        slot_probabilities=probabilities,
    )
    target = _relation_targets(sample, logits, int(slots.shape[0]), torch, device)
    return logits, prediction, target


def _direction_row(source: str, axis: Any, **metadata: Any) -> dict[str, Any]:
    vector = np.asarray(axis, dtype=float)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    nearest = int(nearest_canonical_axis_id(vector[None])[0])
    angle = float(nearest_canonical_axis_angle_deg(vector[None])[0])
    return {
        "source": source,
        "axis_x": float(vector[0]), "axis_y": float(vector[1]), "axis_z": float(vector[2]),
        "nearest_axis": "XYZ"[nearest], "nearest_canonical_angle_deg": angle,
        **metadata,
    }


def evaluate(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any, mean: Any, std: Any,
    torch: Any, device: str, categories: dict[str, str], args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    direction_rows: list[dict[str, Any]] = []
    equivariance_rows: list[dict[str, Any]] = []
    slot_model.eval()
    relation_model.eval()
    with torch.no_grad():
        for sample in samples:
            _, original_prediction, original_target = _predict(
                sample, slot_model, relation_model, mean, std, torch, device
            )
            category = categories.get(str(sample["object_id"]), "unknown")
            for relation in original_target["relations"]:
                parent, child = int(relation["parent_slot"]), int(relation["child_slot"])
                gt_axis = np.asarray(relation["axis"], dtype=float)
                predicted_axis = original_prediction["axes"][parent, child].cpu().numpy()
                predicted_type = JOINT_TYPES[
                    int(original_prediction["type_logits"][parent, child].argmax().cpu())
                ]
                neural_error = _axis_error(predicted_axis, gt_axis)
                motion = _child_motion(sample, int(relation["child_label"]))
                observability_tensor = original_prediction.get("candidate_observability")
                observability = (
                    float(observability_tensor[parent, child].cpu())
                    if observability_tensor is not None else None
                )
                metadata = {
                    "object_id": str(sample["object_id"]), "category": category,
                    "joint_id": str(relation["joint_name"]), "joint_type": str(relation["joint_type"]),
                    "motion_magnitude": motion, "motion_bin": _motion_bin(motion),
                    "observability": observability,
                    "observability_bin": _observability_bin(observability),
                    "augmented": False,
                }
                direction_rows.append(_direction_row("raw_gt", gt_axis, **metadata))
                direction_rows.append(_direction_row(
                    "prediction", predicted_axis,
                    predicted_type=predicted_type,
                    type_correct=predicted_type == str(relation["joint_type"]),
                    edge_detected=bool(
                        original_prediction["edge_logits"][parent, child].cpu() > 0
                    ),
                    axis_error_deg=neural_error,
                    **metadata,
                ))
                if neural_error > args.catastrophic_threshold_deg:
                    direction_rows.append(_direction_row("catastrophic_prediction", predicted_axis, **metadata))
            for rotation_index in range(max(1, int(args.rotations_per_sample))):
                rotation = sample_uniform_so3(rng)
                for scope, rotate_slot_geometry in (
                    ("relation_geometry_only", False),
                    ("slot_and_relation_geometry", True),
                ):
                    augmented = _augment_relation_sample(
                        sample, rng=rng, np=np, rotation=rotation,
                        sampling_mode="uniform_quaternion",
                        rotate_slot_geometry=rotate_slot_geometry,
                    )
                    _, rotated_prediction, rotated_target = _predict(
                        augmented, slot_model, relation_model, mean, std, torch, device
                    )
                    rotated_by_name = {
                        str(row["joint_name"]): row for row in rotated_target["relations"]
                    }
                    for relation in original_target["relations"]:
                        rotated_relation = rotated_by_name[str(relation["joint_name"])]
                        parent, child = int(relation["parent_slot"]), int(relation["child_slot"])
                        rotated_parent = int(rotated_relation["parent_slot"])
                        rotated_child = int(rotated_relation["child_slot"])
                        original_axis = original_prediction["axes"][parent, child].cpu().numpy()
                        rotated_axis = rotated_prediction["axes"][rotated_parent, rotated_child].cpu().numpy()
                        expected_axis = rotation @ original_axis
                        error = _axis_error(expected_axis, rotated_axis)
                        gt_axis_rotated = np.asarray(rotated_relation["axis"], dtype=float)
                        motion = _child_motion(sample, int(relation["child_label"]))
                        observability_tensor = rotated_prediction.get(
                            "candidate_observability"
                        )
                        observability = (
                            float(
                                observability_tensor[rotated_parent, rotated_child].cpu()
                            )
                            if observability_tensor is not None else None
                        )
                        metadata = {
                            "object_id": str(sample["object_id"]), "category": category,
                            "joint_id": str(relation["joint_name"]), "joint_type": str(relation["joint_type"]),
                            "motion_magnitude": motion, "motion_bin": _motion_bin(motion),
                            "observability": observability,
                            "observability_bin": _observability_bin(observability),
                            "augmented": True, "rotation_index": rotation_index,
                            "augmentation_scope": scope,
                        }
                        if scope == "relation_geometry_only":
                            direction_rows.append(_direction_row("augmented_gt", gt_axis_rotated, **metadata))
                        direction_rows.append(_direction_row(f"augmented_prediction:{scope}", rotated_axis, **metadata))
                        equivariance_rows.append({
                            **metadata,
                            "equivariance_error_deg": error,
                            "original_axis": original_axis.tolist(),
                            "expected_rotated_axis": expected_axis.tolist(),
                            "predicted_rotated_axis": rotated_axis.tolist(),
                        })
    return direction_rows, equivariance_rows


def _dependency_variant(
    sample: dict[str, Any], mode: str, feature_mean: Any, rng: np.random.Generator,
) -> dict[str, Any]:
    result = dict(sample)
    features = np.asarray(sample["features"], dtype=np.float32).copy()
    embedding_dim = int(sample["embedding_dim"])
    if mode == "full":
        pass
    elif mode == "embedding_mean_mask":
        features[:, :embedding_dim] = feature_mean[:embedding_dim]
    elif mode == "geometry_mean_mask":
        features[:, embedding_dim:] = feature_mean[embedding_dim:]
    elif mode == "embedding_permute":
        features[:, :embedding_dim] = features[rng.permutation(len(features)), :embedding_dim]
    elif mode == "geometry_permute":
        features[:, embedding_dim:] = features[rng.permutation(len(features)), embedding_dim:]
    else:
        raise ValueError(f"Unknown dependency mode: {mode}")
    result["features"] = features
    result["slot_features"] = features
    return result


def dependency_sensitivity(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any, mean: Any, std: Any,
    torch: Any, device: str, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modes = (
        "full", "embedding_mean_mask", "embedding_permute",
        "geometry_mean_mask", "geometry_permute",
    )
    values: dict[str, dict[str, list[float]]] = {
        mode: defaultdict(list) for mode in modes
    }
    rows: list[dict[str, Any]] = []
    mean_np = mean.detach().cpu().numpy()
    slot_model.eval()
    relation_model.eval()
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            predictions: dict[str, Any] = {}
            targets: dict[str, Any] = {}
            for mode_index, mode in enumerate(modes):
                rng = np.random.default_rng(seed + 1009 * sample_index + 7919 * mode_index)
                variant = _dependency_variant(sample, mode, mean_np, rng)
                _, prediction, target = _predict(
                    variant, slot_model, relation_model, mean, std, torch, device
                )
                predictions[mode] = prediction
                targets[mode] = target

            full_prediction = predictions["full"]
            full_target = targets["full"]
            for mode in modes:
                prediction = predictions[mode]
                target = targets[mode]
                positive = target["positive"]
                predicted_edges = prediction["edge_logits"] > 0
                tp = float((predicted_edges & positive).sum().cpu())
                fp = float((predicted_edges & ~positive).sum().cpu())
                fn = float((~predicted_edges & positive).sum().cpu())
                values[mode]["edge_tp"].append(tp)
                values[mode]["edge_fp"].append(fp)
                values[mode]["edge_fn"].append(fn)
                for relation in target["relations"]:
                    parent, child = int(relation["parent_slot"]), int(relation["child_slot"])
                    predicted_type = JOINT_TYPES[
                        int(prediction["type_logits"][parent, child].argmax().cpu())
                    ]
                    type_correct = predicted_type == str(relation["joint_type"])
                    axis = prediction["axes"][parent, child].cpu().numpy()
                    gt_axis = np.asarray(relation["axis"], dtype=float)
                    axis_error = _axis_error(axis, gt_axis)
                    full_relation = next(
                        row for row in full_target["relations"]
                        if str(row["joint_name"]) == str(relation["joint_name"])
                    )
                    full_axis = full_prediction["axes"][
                        int(full_relation["parent_slot"]), int(full_relation["child_slot"])
                    ].cpu().numpy()
                    drift = _axis_error(axis, full_axis)
                    values[mode]["type_correct"].append(float(type_correct))
                    values[mode]["axis_drift_deg"].append(drift)
                    if type_correct:
                        values[mode]["axis_error_deg"].append(axis_error)
                    rows.append({
                        "object_id": str(sample["object_id"]),
                        "mode": mode,
                        "joint_id": str(relation["joint_name"]),
                        "joint_type": str(relation["joint_type"]),
                        "predicted_type": predicted_type,
                        "type_correct": type_correct,
                        "axis_error_deg": axis_error if type_correct else None,
                        "axis_drift_from_full_deg": drift,
                    })

    report: dict[str, Any] = {}
    for mode in modes:
        bucket = values[mode]
        tp, fp, fn = map(sum, (bucket["edge_tp"], bucket["edge_fp"], bucket["edge_fn"]))
        axis_errors = bucket["axis_error_deg"]
        drifts = bucket["axis_drift_deg"]
        report[mode] = {
            "edge_f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12),
            "joint_type_accuracy": float(np.mean(bucket["type_correct"])),
            "type_correct_axis_count": len(axis_errors),
            "axis_error_mean_deg": float(np.mean(axis_errors)) if axis_errors else None,
            "axis_error_median_deg": float(np.median(axis_errors)) if axis_errors else None,
            "axis_error_p90_deg": float(np.percentile(axis_errors, 90)) if axis_errors else None,
            "axis_drift_from_full_mean_deg": float(np.mean(drifts)),
            "axis_drift_from_full_median_deg": float(np.median(drifts)),
        }
    return report, rows


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    axes = np.asarray([[row["axis_x"], row["axis_y"], row["axis_z"]] for row in rows])
    nearest = nearest_canonical_axis_id(axes)
    angles = nearest_canonical_axis_angle_deg(axes)
    component_bins = np.linspace(-1.0, 1.0, 21)
    angle_bins = np.asarray([0, 5, 10, 15, 20, 30, 40, 55], dtype=float)
    return {
        "count": len(rows),
        "nearest_axis_count": {axis: int(np.sum(nearest == index)) for index, axis in enumerate("XYZ")},
        "nearest_axis_fraction": {axis: float(np.mean(nearest == index)) for index, axis in enumerate("XYZ")},
        "nearest_angle_mean_deg": float(np.mean(angles)),
        "nearest_angle_median_deg": float(np.median(angles)),
        "nearest_angle_histogram": {
            "bin_edges_deg": angle_bins.tolist(),
            "counts": np.histogram(angles, bins=angle_bins)[0].tolist(),
        },
        "component_histograms": {
            axis: {"bin_edges": component_bins.tolist(), "counts": np.histogram(axes[:, index], bins=component_bins)[0].tolist()}
            for index, axis in enumerate("XYZ")
        },
    }


def distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = {"overall": {}, "by_joint_type": {}, "by_category": {}, "by_motion_bin": {}}
    for source in sorted({row["source"] for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        report["overall"][source] = _distribution_summary(selected)
        for dimension, key in (("by_joint_type", "joint_type"), ("by_category", "category"), ("by_motion_bin", "motion_bin")):
            report[dimension][source] = {
                value: _distribution_summary([row for row in selected if row[key] == value])
                for value in sorted({row[key] for row in selected})
            }
    return report


def equivariance_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["equivariance_error_deg"]) for row in selected]
        return {
            "count": len(values),
            "mean_deg": float(np.mean(values)) if values else None,
            "median_deg": float(np.median(values)) if values else None,
            "p75_deg": float(np.percentile(values, 75)) if values else None,
            "p90_deg": float(np.percentile(values, 90)) if values else None,
            "above_10_count": sum(value > 10 for value in values),
            "above_30_count": sum(value > 30 for value in values),
            "above_60_count": sum(value > 60 for value in values),
        }
    return {
        "overall": summarize(rows),
        "by_augmentation_scope": {
            value: summarize([row for row in rows if row["augmentation_scope"] == value])
            for value in sorted({row["augmentation_scope"] for row in rows})
        },
        "by_joint_type": {
            value: summarize([row for row in rows if row["joint_type"] == value])
            for value in sorted({row["joint_type"] for row in rows})
        },
        "by_category": {
            value: summarize([row for row in rows if row["category"] == value])
            for value in sorted({row["category"] for row in rows})
        },
    }


def prediction_performance_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [row for row in rows if row["source"] == "prediction"]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        type_correct = [bool(row["type_correct"]) for row in selected]
        type_correct_errors = [
            float(row["axis_error_deg"])
            for row in selected if bool(row["type_correct"])
        ]
        return {
            "count": len(selected),
            "joint_type_accuracy": (
                float(np.mean(type_correct)) if type_correct else None
            ),
            "edge_recall": (
                float(np.mean([bool(row["edge_detected"]) for row in selected]))
                if selected else None
            ),
            "type_correct_axis_count": len(type_correct_errors),
            "axis_mean_deg": (
                float(np.mean(type_correct_errors)) if type_correct_errors else None
            ),
            "axis_median_deg": (
                float(np.median(type_correct_errors)) if type_correct_errors else None
            ),
            "axis_p75_deg": (
                float(np.percentile(type_correct_errors, 75))
                if type_correct_errors else None
            ),
            "axis_p90_deg": (
                float(np.percentile(type_correct_errors, 90))
                if type_correct_errors else None
            ),
            "axis_above_10_count": sum(value > 10 for value in type_correct_errors),
            "axis_above_30_count": sum(value > 30 for value in type_correct_errors),
            "axis_above_60_count": sum(value > 60 for value in type_correct_errors),
            "axis_above_80_count": sum(value > 80 for value in type_correct_errors),
        }

    report = {"overall": summarize(predictions)}
    for name, key in (
        ("by_joint_type", "joint_type"),
        ("by_category", "category"),
        ("by_motion_bin", "motion_bin"),
        ("by_observability_bin", "observability_bin"),
    ):
        report[name] = {
            value: summarize([row for row in predictions if row[key] == value])
            for value in sorted({row[key] for row in predictions})
        }
    return report


def sampler_report(count: int, seed: int) -> dict[str, Any]:
    rotations = sample_uniform_so3_batch(count, seed=seed)
    axes = rotations @ np.asarray([1.0, 0.0, 0.0])
    summary = _distribution_summary([
        _direction_row("sampler", axis) for axis in axes
    ])
    orthogonality = np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2))
    determinants = np.linalg.det(rotations)
    return {
        **summary,
        "orthogonality_error_max": float(np.max(orthogonality)),
        "determinant_mean": float(np.mean(determinants)),
        "balanced_nearest_axis_max_deviation": float(max(
            abs(value - 1.0 / 3.0) for value in summary["nearest_axis_fraction"].values()
        )),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key)
            for key in keys
        } for row in rows)


def plot_directions(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    sources = [source for source in (
        "raw_gt", "augmented_gt", "prediction",
        "augmented_prediction:relation_geometry_only",
        "augmented_prediction:slot_and_relation_geometry",
        "catastrophic_prediction",
    ) if any(row["source"] == source for row in rows)]
    figure, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), squeeze=False)
    for plot, source in zip(axes[0], sources):
        selected = [row for row in rows if row["source"] == source][:5000]
        vectors = np.asarray([[row["axis_x"], row["axis_y"], row["axis_z"]] for row in selected])
        longitude = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        latitude = np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))
        plot.scatter(longitude, latitude, s=4, alpha=0.35)
        plot.set(title=source, xlabel="longitude (deg)", ylabel="latitude (deg)", xlim=(-180, 180), ylim=(-90, 90))
        plot.grid(alpha=0.2)
    figure.tight_layout()
    path = output / "axis_spherical_distribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    torch = _require_torch()
    device = _resolve_device(torch, args.device)
    slot_checkpoint = torch.load(args.slot_model.expanduser().resolve(), map_location="cpu", weights_only=True)
    relation_checkpoint = torch.load(args.relation_model.expanduser().resolve(), map_location="cpu", weights_only=True)
    slot_model = _load_slot_model(slot_checkpoint, torch, device)
    if relation_checkpoint.get("slot_state_dict") is not None:
        slot_model.load_state_dict(relation_checkpoint["slot_state_dict"])
    relation_model = _build_relation_model(
        torch, slot_dim=int(relation_checkpoint["slot_dim"]), hidden_dim=int(relation_checkpoint["hidden_dim"]),
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
    relation_model.trajectory_samples = int(relation_checkpoint.get("trajectory_samples", 32))
    relation_model.load_state_dict(relation_checkpoint["state_dict"])
    samples = _load_relation_samples(
        load_pairwise_manifest(args.manifest.expanduser().resolve()), "test", slot_checkpoint
    )
    samples = _apply_slot_input_mode(
        samples, args.slot_input_mode, slot_checkpoint["feature_mean"].cpu().numpy()
    )
    samples = _apply_slot_geometry_representation(
        samples,
        str(relation_checkpoint.get("slot_geometry_representation", "raw")),
        np,
    )
    if args.max_objects is not None:
        samples = samples[: max(1, int(args.max_objects))]
    slot_mean = relation_checkpoint.get(
        "slot_feature_mean", slot_checkpoint["feature_mean"]
    ).to(device)
    slot_std = relation_checkpoint.get(
        "slot_feature_std", slot_checkpoint["feature_std"]
    ).to(device)
    direction_rows, equivariance_rows = evaluate(
        samples, slot_model, relation_model, slot_mean, slot_std,
        torch, device, load_categories(args.catalog), args,
    )
    dependency_report, dependency_rows = dependency_sensitivity(
        samples, slot_model, relation_model, slot_mean, slot_std,
        torch, device, args.seed,
    )
    registry = geometry_registry_report()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "sampler": sampler_report(max(100_000, int(args.sampler_count)), args.seed),
        "geometry_registry": registry,
        "unresolved_geometry_fields": [row for row in registry if row["status"] != "supported"],
        "direction_distribution": distribution_report(direction_rows),
        "prediction_performance": prediction_performance_report(direction_rows),
        "model_equivariance": equivariance_report(equivariance_rows),
        "dependency_sensitivity": dependency_report,
        "object_count": len(samples),
        "notes": [
            "The slot encoder receives its original cached input during relation-level rotation augmentation.",
            "Opaque cached track embeddings are not recomputed and remain a possible canonical-frame leakage path.",
        ],
    }
    (output / "so3_equivariance_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "axis_direction_samples.csv", direction_rows)
    write_csv(output / "axis_equivariance_per_joint.csv", equivariance_rows)
    write_csv(output / "dependency_sensitivity_per_joint.csv", dependency_rows)
    plots = plot_directions(output, direction_rows)
    print(json.dumps({
        "output_dir": str(output), "object_count": len(samples),
        "equivariance_joint_count": len(equivariance_rows), "plots": plots,
        "unresolved_geometry_field_count": len(report["unresolved_geometry_fields"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
