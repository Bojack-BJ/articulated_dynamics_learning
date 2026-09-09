"""Adapters for evaluating official ReArt optimization outputs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .aim_pointcloud_iou import evaluate_labeled_points_on_reference


def load_reart_result(path: Path) -> dict[str, Any]:
    """Load a trusted result.pkl written by the official ReArt pipeline."""
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    required = {"cano_pc", "pred_cano_part", "pred_pose_list"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"ReArt result is missing required keys {missing}: {path}")
    return payload


def evaluate_reart_segmentation(
    result_pkl: Path,
    reference_ply: Path,
    *,
    reference_part_ids: Iterable[int] | None = None,
    reference_part_id_map: dict[int, int] | None = None,
    distance_ratios: Iterable[float] = (0.01, 0.02, 0.05),
    primary_distance_ratio: float = 1.0,
) -> dict[str, Any]:
    """Evaluate canonical ReArt labels on the common observed-point GT domain."""
    result_path = result_pkl.expanduser().resolve()
    payload = load_reart_result(result_path)
    points = np.asarray(payload["cano_pc"], dtype=np.float64)
    labels = np.asarray(payload["pred_cano_part"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected ReArt cano_pc with shape [N, 3], got {points.shape}")
    if labels.shape != (len(points),):
        raise ValueError(
            f"Expected one ReArt label per canonical point, got points={points.shape}, labels={labels.shape}"
        )

    evaluation = evaluate_labeled_points_on_reference(
        points,
        labels,
        reference_ply,
        distance_ratios=distance_ratios,
        primary_distance_ratio=primary_distance_ratio,
        reference_part_ids=reference_part_ids,
        reference_part_id_map=reference_part_id_map,
    )
    unique_parts = sorted(int(value) for value in np.unique(labels))
    connections = [
        [int(endpoint) for endpoint in edge]
        for edge in payload.get("joint_connection", [])
    ]
    evaluation.update(
        {
            "prediction_source": "reart-result-pkl",
            "result_pkl": str(result_path),
            "canonical_frame_index": int(payload.get("cano_idx", 0)),
            "predicted_part_ids": unique_parts,
            "predicted_part_count": len(unique_parts),
            "native_outputs": {
                "part_segmentation": True,
                "per_part_se3_trajectory": True,
                "undirected_part_connections": bool(connections),
                "joint_type": False,
                "joint_axis": False,
                "joint_pivot": False,
            },
            "converted_outputs": {
                "joint_connection": connections,
                "joint_parameters_from_pose_trajectory": "not_implemented",
            },
            "metric_support": {
                "point_iou": "converted_common_evaluator",
                "ari": "converted_common_evaluator",
                "predicted_gt_part_count": "converted_common_evaluator",
                "undirected_edge_metrics": "convertible_when_gt_mapping_is_available",
                "directed_edge_f1": "unsupported_native",
                "joint_type_axis_pivot": "unsupported_in_current_adapter",
            },
        }
    )
    return evaluation
