"""Object selection and metric contracts for external baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


CORE_OBJECT_IDS = (
    "partnet_102018",  # oven, 2 parts
    "partnet_10849",  # refrigerator, 2 parts
    "partnet_46556",  # drawer, 2 parts
    "partnet_9388",  # door, 2 parts
    "partnet_10638",  # refrigerator, 3 parts
    "partnet_7263",  # microwave, 3 parts
    "partnet_20745",  # drawer, 4 parts
    "partnet_101604",  # door, 4 parts
    "partnet_45332",  # door, 5 parts
    "partnet_103069",  # coffee machine, 6 parts
    "partnet_7179",  # oven, 6 parts
    "partnet_47466",  # drawer, 11 parts
)

CORE16_OBJECT_IDS = (
    # Two-part / one-joint scope.
    "partnet_102018",
    "partnet_10849",
    "partnet_10944",
    "partnet_46556",
    "partnet_9388",
    # Three-to-four-part objects, balanced across categories.
    "partnet_10638",
    "partnet_7263",
    "partnet_101604",
    "partnet_102055",
    "partnet_20745",
    # Complex objects. Native failures remain intentionally represented.
    "partnet_45332",
    "partnet_102149",
    "partnet_45261",
    "partnet_7179",
    "partnet_103064",
    "partnet_47466",
)


@dataclass(frozen=True)
class MethodContract:
    protocol: str
    scope: str
    oracle_inputs: tuple[str, ...]
    predicted_metrics: tuple[str, ...]
    conditional_metrics: tuple[str, ...] = ()
    notes: str = ""


METHOD_CONTRACTS = {
    "ours_hybrid": MethodContract(
        protocol="continuous_rgbd_interaction",
        scope="all",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "edge_f1",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
    ),
    "aim": MethodContract(
        protocol="paper_like_or_aim_style_rgb_video",
        scope="all",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
        notes="Directed topology is not a native AiM output.",
    ),
    "reart": MethodContract(
        protocol="native_sapiens_4d_point_cloud",
        scope="all",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
        notes="Only report topology if the released projection exposes it.",
    ),
    "dta": MethodContract(
        protocol="two_state_multiview_rgbd",
        scope="all",
        oracle_inputs=("gt_part_count",),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
    ),
    "artgs": MethodContract(
        protocol="two_state_multiview_rgbd",
        scope="all",
        oracle_inputs=("gt_part_count",),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
    ),
    "videoartgs": MethodContract(
        protocol="continuous_monocular_rgb_video",
        scope="all",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
        conditional_metrics=("edge_f1",),
        notes="Record whether joint inventory is VLM, manual, or GT-oracle.",
    ),
    "gaussianart": MethodContract(
        protocol="two_state_multiview_rgbd",
        scope="all",
        oracle_inputs=(
            "gt_part_count",
            "part_semantic_initialization",
            "gt_motion_metadata",
        ),
        predicted_metrics=(
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
        conditional_metrics=("point_iou", "ari", "ri"),
        notes="Segmentation is oracle-initialized and must not enter the non-oracle table.",
    ),
    "paris": MethodContract(
        protocol="two_state_multiview_rgb",
        scope="two_part_one_joint",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
            "revolute_motion_error",
            "prismatic_motion_error",
        ),
    ),
    "ditto": MethodContract(
        protocol="two_state_fused_rgbd_point_cloud",
        scope="two_part_one_joint",
        oracle_inputs=(),
        predicted_metrics=(
            "point_iou",
            "ari",
            "ri",
            "predicted_part_count",
            "joint_type_accuracy",
            "axis_angle_type_correct",
            "revolute_axis_line",
        ),
    ),
}


def complexity_bucket(gt_part_count: int) -> str:
    if gt_part_count == 2:
        return "2"
    if gt_part_count <= 4:
        return "3-4"
    return ">=5"


def method_applicable(method: str, gt_part_count: int, gt_joint_count: int | None) -> bool:
    contract = METHOD_CONTRACTS[method]
    if contract.scope == "all":
        return True
    # Restricted baselines must not be dispatched from an inferred topology.
    return gt_part_count == 2 and gt_joint_count == 1


def build_aligned_rows(source_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand object metadata into deterministic method-specific run rows."""
    normalized = []
    for source in source_rows:
        gt_parts = int(source["gt_part_count"])
        raw_joint_count = source.get("gt_joint_count")
        gt_joints = (
            int(raw_joint_count)
            if raw_joint_count not in (None, "")
            else None
        )
        joint_count_source = str(source.get("gt_joint_count_source", "")).strip()
        if not joint_count_source:
            joint_count_source = (
                "manifest_gt" if gt_joints is not None else "unknown"
            )
        object_id = str(source["object_id"])
        suite_tiers = ["full20"]
        if object_id in CORE_OBJECT_IDS:
            suite_tiers.append("core12")
        if gt_parts == 2 and gt_joints == 1:
            suite_tiers.append("two_part")

        for method, contract in METHOD_CONTRACTS.items():
            applicable = method_applicable(method, gt_parts, gt_joints)
            normalized.append(
                {
                    "object_id": object_id,
                    "category": source["category"],
                    "gt_part_count": gt_parts,
                    "gt_joint_count": "" if gt_joints is None else gt_joints,
                    "gt_joint_count_source": joint_count_source,
                    "complexity_bucket": complexity_bucket(gt_parts),
                    "suite_tiers": ",".join(suite_tiers),
                    "method": method,
                    "applicable": applicable,
                    "protocol": contract.protocol,
                    "scope": contract.scope,
                    "oracle_inputs": ",".join(contract.oracle_inputs),
                    "predicted_metrics": ",".join(contract.predicted_metrics),
                    "conditional_metrics": ",".join(contract.conditional_metrics),
                    "notes": contract.notes,
                }
            )
    return sorted(
        normalized,
        key=lambda row: (
            row["gt_part_count"],
            row["category"],
            row["object_id"],
            row["method"],
        ),
    )
