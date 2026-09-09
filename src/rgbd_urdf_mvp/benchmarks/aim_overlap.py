"""Definitions and aggregation helpers for the four-object AiM overlap benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class AimObjectSpec:
    object_id: str
    source_id: str
    category: str
    total_part_count: int
    revolute_count: int
    prismatic_count: int

    @property
    def movable_joint_count(self) -> int:
        return self.revolute_count + self.prismatic_count


AIM_OBJECTS = (
    AimObjectSpec("partnet_47024", "47024", "Storage", 3, 1, 1),
    AimObjectSpec("partnet_11304", "11304", "Fridge", 3, 2, 0),
    AimObjectSpec("partnet_47648", "47648", "Storage", 7, 4, 2),
    AimObjectSpec("partnet_31249", "31249", "Table", 5, 2, 2),
)
AIM_OBJECT_IDS = tuple(spec.object_id for spec in AIM_OBJECTS)

AIM_REFERENCE = {
    "partnet_47024": {
        "mesh_part_ious": [94.20, 94.95, 79.75],
        "mean_mesh_part_iou": 89.63,
        "revolute_axis_error_deg": 0.56,
    },
    "partnet_11304": {
        "mesh_part_ious": [95.42, 89.95, 91.42],
        "mean_mesh_part_iou": 92.26,
        "reported_revolute_axis_errors_deg": [0.68, 1.67],
    },
    "partnet_47648": {
        "dynamic_part_mean_mesh_iou": 81.87,
        "average_revolute_axis_error_deg": 0.58,
    },
    "partnet_31249": {
        "dynamic_part_mean_mesh_iou": 53.49,
        "average_revolute_axis_error_deg": 1.19,
    },
}

METRIC_DISCLAIMER = (
    "AiM segmentation values are 3D IoU on voxelized reconstructed meshes, whereas this "
    "benchmark evaluates segmentation on visible tracked 3D points. These are not identical "
    "metrics, so the side-by-side table is descriptive and must not be used for direct ranking."
)


def select_explicit_objects(catalog_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select all benchmark IDs without applying any category allowlist."""
    rows = {str(row["object_id"]): dict(row) for row in catalog_rows}
    missing = [object_id for object_id in AIM_OBJECT_IDS if object_id not in rows]
    if missing:
        raise ValueError(f"AiM benchmark objects missing from catalog: {missing}")
    return [rows[object_id] for object_id in AIM_OBJECT_IDS]


def validate_object_metadata(spec: AimObjectSpec, metadata: dict[str, Any]) -> None:
    actual_parts = int(metadata["total_part_count"])
    joints = list(metadata["movable_joints"])
    counts = {
        kind: sum(str(joint["joint_type"]) == kind for joint in joints)
        for kind in ("revolute", "prismatic")
    }
    errors = []
    if actual_parts != spec.total_part_count:
        errors.append(f"parts expected={spec.total_part_count} actual={actual_parts}")
    if len(joints) != spec.movable_joint_count:
        errors.append(f"joints expected={spec.movable_joint_count} actual={len(joints)}")
    if counts["revolute"] != spec.revolute_count:
        errors.append(f"revolute expected={spec.revolute_count} actual={counts['revolute']}")
    if counts["prismatic"] != spec.prismatic_count:
        errors.append(f"prismatic expected={spec.prismatic_count} actual={counts['prismatic']}")
    if errors:
        raise ValueError(f"{spec.object_id} metadata mismatch: " + "; ".join(errors))


def sequence_plan(movable_joints: Iterable[dict[str, Any]], simultaneous: bool = False) -> list[dict[str, Any]]:
    joints = [dict(joint) for joint in movable_joints]
    if not joints:
        raise ValueError("Cannot plan an articulation benchmark without movable joints")
    plans = [
        {"name": f"isolated_{joint['name']}", "mode": "isolated", "joint_names": [joint["name"]]}
        for joint in joints
    ]
    plans.append({"name": "combined_sequential", "mode": "staggered", "joint_names": [j["name"] for j in joints]})
    if simultaneous:
        plans.append({"name": "combined_simultaneous", "mode": "simultaneous", "joint_names": [j["name"] for j in joints]})
    return plans


def joint_success(rows: Iterable[dict[str, Any]], threshold_deg: float) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    successes = sum(
        bool(row.get("edge_detected"))
        and bool(row.get("type_correct"))
        and row.get("axis_error_deg") is not None
        and float(row["axis_error_deg"]) < threshold_deg
        for row in rows
    )
    return successes / len(rows)


def axis_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(row["axis_error_deg"])
        for row in rows
        if row.get("axis_error_deg") is not None and math.isfinite(float(row["axis_error_deg"]))
    )
    percentile = lambda q: _percentile(values, q)
    return {
        "count": len(values),
        "mean_deg": sum(values) / len(values) if values else None,
        "median_deg": percentile(50),
        "p75_deg": percentile(75),
        "p90_deg": percentile(90),
        **{f"above_{threshold}_count": sum(value > threshold for value in values) for threshold in (10, 30, 60, 80)},
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def benchmark_metadata() -> dict[str, Any]:
    return {
        "objects": [asdict(spec) | {"movable_joint_count": spec.movable_joint_count} for spec in AIM_OBJECTS],
        "aim_reference": AIM_REFERENCE,
        "metric_disclaimer": METRIC_DISCLAIMER,
    }
