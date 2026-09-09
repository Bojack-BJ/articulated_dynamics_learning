#!/usr/bin/env python3
"""Initialize and summarize the articulated-reconstruction baseline suite."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    "ours_hybrid",
    "aim_aligned",
    "reart",
    "dta",
    "artgs",
    "videoartgs",
    "gaussianart",
    "paris",
    "ditto",
)
SUCCESS_STATUSES = {
    "success",
    "success_native_metrics",
    "completed_with_exporter_failure",
}
METHOD_PROTOCOLS = {
    "ours_hybrid": "continuous_interaction",
    "aim_aligned": "aim_style_cross_protocol_generalization",
    "reart": "native_4d_point_cloud",
    "dta": "two_state_multiview_rgbd",
    "artgs": "two_state_multiview_rgbd_released_path",
    "videoartgs": "continuous_monocular_interaction",
    "gaussianart": "two_state_multiview_oracle_parts",
    "paris": "two_state_multiview_rgb_two_part_only",
    "ditto": "two_state_fused_point_cloud_one_joint_only",
}
ORACLES = {
    "dta": ["gt_part_count"],
    "artgs": ["gt_part_count"],
    "videoartgs": ["joint_count", "joint_types", "parent_topology"],
    "gaussianart": [
        "part_count",
        "part_semantic_initialization",
        "gt_motion_metadata",
    ],
    "paris": ["two_part_method_scope"],
    "ditto": ["one_joint_method_scope"],
}
METHOD_GROUPS = {
    "ours_hybrid": "non_oracle_part_discovery",
    "aim_aligned": "non_oracle_part_discovery",
    "reart": "non_oracle_part_discovery",
    "dta": "exact_count_oracle",
    "artgs": "exact_count_oracle",
    "videoartgs": "inventory_or_semantic_oracle",
    "gaussianart": "inventory_or_semantic_oracle",
    "paris": "restricted_two_part_scope",
    "ditto": "restricted_two_part_scope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned_per_object_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.aligned_per_object_csv.open(encoding="utf-8")))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_object").mkdir(exist_ok=True)

    manifest = []
    for source in rows:
        gt_count = int(source["gt_part_count"])
        object_id = source["object_id"]
        manifest.append(
            {
                "object_id": object_id,
                "category": source["category"],
                "gt_part_count": gt_count,
                "complexity_bucket": _bucket(gt_count),
                "continuous_interaction_available": True,
                "two_state_export_required": True,
                "reart_native_available": source["reart_status"] == "success",
                "paris_applicable": gt_count == 2,
                "ditto_applicable": gt_count == 2,
                **{
                    f"{prefix}_observed_gt_parts": _integer(
                        source.get(f"{prefix}_observed_gt_parts", "")
                    )
                    for prefix in ("ours", "aim", "reart")
                },
            }
        )
        object_dir = output / "per_object" / object_id
        object_dir.mkdir(exist_ok=True)
        for method in METHODS:
            result_path = object_dir / method / "metrics.json"
            if result_path.exists():
                continue
            payload = _pending_result(source, method)
            result_path.parent.mkdir(exist_ok=True)
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

    _write_csv(output / "manifest.csv", manifest)
    results = _load_results(output, manifest)
    _write_csv(output / "segmentation_summary.csv", _segmentation_summary(results))
    _write_csv(output / "kinematics_summary.csv", _kinematics_summary(results))
    _write_csv(output / "geometry_summary.csv", _geometry_summary(results))
    _write_csv(output / "complexity_summary.csv", _complexity_summary(results))
    _write_csv(output / "per_object_metrics.csv", _per_object_rows(results))
    _write_csv(
        output / "grouped_segmentation_summary.csv",
        _grouped_segmentation_summary(results),
    )
    _write_csv(
        output / "common_success_summary.csv",
        _common_success_summary(results),
    )
    _write_csv(
        output / "non_oracle_main_table.csv",
        _paper_method_summary(
            results,
            ("ours_hybrid", "aim_aligned", "reart"),
        ),
    )
    _write_csv(
        output / "oracle_assisted_table.csv",
        _paper_method_summary(
            results,
            ("dta", "artgs", "videoartgs", "gaussianart"),
        ),
    )
    _write_csv(
        output / "restricted_two_part_table.csv",
        _paper_method_summary(results, ("paris", "ditto")),
    )
    _write_csv(output / "gt_domain_issues.csv", _gt_domain_issues(results))
    failures = [
        result
        for result in results
        if result["status"] in {"failure", "failed"}
    ]
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary(output / "summary.md", manifest, results)
    _write_protocol_audit(output / "protocol_audit.md")
    return 0


def _pending_result(source: dict[str, str], method: str) -> dict[str, Any]:
    status = "pending"
    segmentation: dict[str, Any] = {}
    if method in {"ours_hybrid", "aim_aligned", "reart"}:
        prefix = {
            "ours_hybrid": "ours",
            "aim_aligned": "aim",
            "reart": "reart",
        }[method]
        status = source[f"{prefix}_status"]
        if status == "success":
            segmentation = {
                "point_iou": _number(source[f"{prefix}_iou"]),
                "ari": _number(source[f"{prefix}_ari"]),
                "ri": _number(source[f"{prefix}_ri"]),
                "predicted_part_count": _integer(source[f"{prefix}_pred_parts"]),
                "gt_part_count": int(source["gt_part_count"]),
                "undersegmented": _boolean(source[f"{prefix}_undersegmented"]),
                "largest_cluster_ratio": _number(
                    source[f"{prefix}_largest_cluster_ratio"]
                ),
            }
    applicable = not (
        method in {"paris", "ditto"} and int(source["gt_part_count"]) != 2
    )
    if not applicable:
        status = "not_applicable"
    return {
        "schema": "external-baseline-result-v1",
        "object_id": source["object_id"],
        "category": source["category"],
        "method": method,
        "protocol": METHOD_PROTOCOLS[method],
        "status": status,
        "applicable": applicable,
        "oracle_requirements": ORACLES.get(method, []),
        "segmentation": segmentation,
        "kinematics": {},
        "geometry": {},
        "metric_support": {
            "segmentation": "converted_common_evaluator",
            "kinematics": "unsupported_until_native_output_adapter",
            "geometry": "unsupported_until_mesh_adapter",
        },
        "failure": None,
    }


def _load_results(
    output: Path, manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for row in manifest:
        for method in METHODS:
            path = output / "per_object" / row["object_id"] / method / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result.setdefault("object_id", row["object_id"])
            result.setdefault("category", row["category"])
            result.setdefault("method", method)
            result.setdefault("protocol", METHOD_PROTOCOLS[method])
            result.setdefault("applicable", not (
                method in {"paris", "ditto"} and row["gt_part_count"] != 2
            ))
            result.setdefault("oracle_requirements", ORACLES.get(method, []))
            result.setdefault("segmentation", {})
            result.setdefault("kinematics", {})
            result.setdefault("geometry", {})
            result["_manifest_gt_part_count"] = int(row["gt_part_count"])
            prefix = {
                "ours_hybrid": "ours",
                "aim_aligned": "aim",
                "reart": "reart",
            }.get(method)
            if prefix is not None:
                observed = row.get(f"{prefix}_observed_gt_parts")
                has_union_domain = (
                    result["segmentation"].get("evaluation_domain")
                    == "time_aligned_multiframe_union"
                )
                if observed is not None and not has_union_domain:
                    result["segmentation"]["observed_gt_part_count"] = observed
                    result["segmentation"]["gt_domain_valid"] = (
                        int(observed) == int(row["gt_part_count"])
                    )
            results.append(result)
    return results


def _segmentation_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "protocol": METHOD_PROTOCOLS[method],
            "requested_n": sum(
                result["method"] == method and result["applicable"] for result in results
            ),
            "success_n": len(selected),
            "metric_n": _metric_count(selected, "segmentation", "point_iou"),
            "point_iou": _mean_valid_gt(selected, "point_iou"),
            "ari": _mean_valid_gt(selected, "ari"),
            "ri": _mean_valid_gt(selected, "ri"),
            "undersegmentation_rate": _mean(
                _valid_gt_results(selected), "segmentation", "undersegmented"
            ),
        }
        for method in METHODS
        for selected in [[
            result
            for result in results
            if result["method"] == method and result["status"] in SUCCESS_STATUSES
        ]]
    ]


def _kinematics_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "edge_f1",
        "joint_type_accuracy",
        "axis_angle_deg_type_correct",
        "axis_angle_error_deg",
        "axis_angle_error_deg_native_all",
        "revolute_axis_line_bbox",
        "axis_position_error",
        "axis_position_error_native_x10",
        "revolute_motion_error_rad",
        "prismatic_motion_error_m",
        "motion_error_native",
    )
    return [
        {
            "method": method,
            "success_n": len(selected),
            **{field: _mean(selected, "kinematics", field) for field in fields},
        }
        for method in METHODS
        for selected in [[
            result
            for result in results
            if result["method"] == method and result["status"] in SUCCESS_STATUSES
        ]]
    ]


def _geometry_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "full_object_voxel_iou",
        "part_voxel_iou",
        "chamfer",
        "novel_view_psnr",
        "novel_view_ssim",
    )
    return [
        {
            "method": method,
            "success_n": len(selected),
            **{field: _mean(selected, "geometry", field) for field in fields},
        }
        for method in METHODS
        for selected in [[
            result
            for result in results
            if result["method"] == method and result["status"] in SUCCESS_STATUSES
        ]]
    ]


def _complexity_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_gt_counts = {
        result["object_id"]: result["_manifest_gt_part_count"] for result in results
    }
    rows = []
    for bucket in ("2", "3-4", ">=5"):
        for method in METHODS:
            selected = [
                result
                for result in results
                if result["method"] == method
                and result["status"] in SUCCESS_STATUSES
                and object_gt_counts.get(result["object_id"]) is not None
                and _bucket(int(object_gt_counts[result["object_id"]])) == bucket
            ]
            rows.append(
                {
                    "complexity_bucket": bucket,
                    "method": method,
                    "success_n": len(selected),
                    "metric_n": _metric_count(
                        selected, "segmentation", "point_iou"
                    ),
                    "point_iou": _mean_valid_gt(selected, "point_iou"),
                    "ari": _mean_valid_gt(selected, "ari"),
                    "ri": _mean_valid_gt(selected, "ri"),
                    "undersegmentation_rate": _mean(
                        _valid_gt_results(selected),
                        "segmentation",
                        "undersegmented",
                    ),
                }
            )
    return rows


def _per_object_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        segmentation = result.get("segmentation", {})
        kinematics = result.get("kinematics", {})
        geometry = result.get("geometry", {})
        rows.append(
            {
                "object_id": result["object_id"],
                "category": result["category"],
                "method": result["method"],
                "protocol": result["protocol"],
                "status": result["status"],
                "oracle_requirements": ",".join(result["oracle_requirements"]),
                "benchmark_group": METHOD_GROUPS[result["method"]],
                "point_iou": segmentation.get("point_iou"),
                "ari": segmentation.get("ari"),
                "ri": segmentation.get("ri"),
                "predicted_part_count": segmentation.get("predicted_part_count"),
                "gt_part_count": result["_manifest_gt_part_count"],
                "observed_gt_part_count": segmentation.get(
                    "observed_gt_part_count",
                    segmentation.get("gt_part_count"),
                ),
                "gt_domain_valid": segmentation.get("gt_domain_valid", True),
                "edge_f1": kinematics.get("edge_f1"),
                "joint_type_accuracy": kinematics.get("joint_type_accuracy"),
                "axis_angle_deg_type_correct": kinematics.get(
                    "axis_angle_deg_type_correct"
                ),
                "axis_angle_error_deg": kinematics.get("axis_angle_error_deg"),
                "axis_angle_error_deg_native_all": kinematics.get(
                    "axis_angle_error_deg_native_all"
                ),
                "revolute_axis_line_bbox": kinematics.get(
                    "revolute_axis_line_bbox"
                ),
                "axis_position_error_native_x10": kinematics.get(
                    "axis_position_error_native_x10"
                ),
                "axis_position_error": kinematics.get("axis_position_error"),
                "motion_error_native": kinematics.get("motion_error_native"),
                "full_object_voxel_iou": geometry.get("full_object_voxel_iou"),
                "part_voxel_iou": geometry.get("part_voxel_iou"),
                "chamfer": geometry.get("chamfer"),
                "novel_view_psnr": geometry.get("novel_view_psnr"),
                "novel_view_ssim": geometry.get("novel_view_ssim"),
            }
        )
    return rows


def _grouped_segmentation_summary(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for group in dict.fromkeys(METHOD_GROUPS.values()):
        for method in METHODS:
            if METHOD_GROUPS[method] != group:
                continue
            selected = [
                result
                for result in results
                if result["method"] == method
                and result["status"] in SUCCESS_STATUSES
            ]
            rows.append(
                {
                    "benchmark_group": group,
                    "method": method,
                    "requested_n": sum(
                        result["method"] == method and result["applicable"]
                        for result in results
                    ),
                    "success_n": len(selected),
                    "metric_n": _metric_count(
                        selected, "segmentation", "point_iou"
                    ),
                    "point_iou": _mean_valid_gt(selected, "point_iou"),
                    "ari": _mean_valid_gt(selected, "ari"),
                    "ri": _mean_valid_gt(selected, "ri"),
                }
            )
    return rows


def _common_success_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups = {
        "non_oracle_part_discovery": (
            "ours_hybrid",
            "aim_aligned",
            "reart",
        ),
        "non_oracle_plus_exact_count_oracle": (
            "ours_hybrid",
            "aim_aligned",
            "reart",
            "dta",
            "artgs",
        ),
    }
    by_method_object = {
        (result["method"], result["object_id"]): result for result in results
    }
    all_objects = {result["object_id"] for result in results}
    for comparison, methods in groups.items():
        common = sorted(
            object_id
            for object_id in all_objects
            if all(
                (method, object_id) in by_method_object
                and by_method_object[(method, object_id)]["status"]
                in SUCCESS_STATUSES
                for method in methods
            )
        )
        for method in methods:
            selected = [by_method_object[(method, object_id)] for object_id in common]
            rows.append(
                {
                    "comparison": comparison,
                    "method": method,
                    "common_success_n": len(common),
                    "metric_n": _metric_count(
                        selected, "segmentation", "point_iou"
                    ),
                    "object_ids": ",".join(common),
                    "point_iou": _mean_valid_gt(selected, "point_iou"),
                    "ari": _mean_valid_gt(selected, "ari"),
                    "ri": _mean_valid_gt(selected, "ri"),
                }
            )
    return rows


def _paper_method_summary(
    results: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        selected = [
            result
            for result in results
            if result["method"] == method
            and result["status"] in SUCCESS_STATUSES
        ]
        rows.append(
            {
                "method": method,
                "protocol": METHOD_PROTOCOLS[method],
                "oracle_or_scope": ",".join(ORACLES.get(method, [])),
                "requested_n": sum(
                    result["method"] == method and result["applicable"]
                    for result in results
                ),
                "success_n": len(selected),
                "metric_n": _metric_count(
                    selected, "segmentation", "point_iou"
                ),
                "point_iou": _mean_valid_gt(selected, "point_iou"),
                "ari": _mean_valid_gt(selected, "ari"),
                "ri": _mean_valid_gt(selected, "ri"),
                "edge_f1": _mean(selected, "kinematics", "edge_f1"),
                "joint_type_accuracy": _mean(
                    selected, "kinematics", "joint_type_accuracy"
                ),
                "axis_angle_deg_type_correct": _first_mean(
                    selected,
                    "kinematics",
                    (
                        "axis_angle_deg_type_correct",
                        "axis_angle_error_deg",
                    ),
                ),
                "revolute_axis_line_bbox": _mean(
                    selected, "kinematics", "revolute_axis_line_bbox"
                ),
                "native_axis_error_all": _mean(
                    selected, "kinematics", "axis_angle_error_deg_native_all"
                ),
                "native_axis_position_error": _first_mean(
                    selected,
                    "kinematics",
                    (
                        "axis_position_error_native_x10",
                        "axis_position_error",
                    ),
                ),
            }
        )
    return rows


def _gt_domain_issues(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        segmentation = result.get("segmentation", {})
        if segmentation.get("gt_domain_valid", True):
            continue
        rows.append(
            {
                "object_id": result["object_id"],
                "category": result["category"],
                "method": result["method"],
                "manifest_gt_part_count": result["_manifest_gt_part_count"],
                "observed_gt_part_count": segmentation.get(
                    "observed_gt_part_count"
                ),
                "missing_gt_part_count": (
                    result["_manifest_gt_part_count"]
                    - int(segmentation["observed_gt_part_count"])
                ),
                "required_action": "rerun_multiframe_union_gt_evaluation",
            }
        )
    return rows


def _mean(results: list[dict[str, Any]], section: str, field: str) -> float | None:
    values = [
        result.get(section, {}).get(field)
        for result in results
        if isinstance(result.get(section, {}).get(field), (int, float))
    ]
    return statistics.fmean(values) if values else None


def _valid_gt_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if result.get("segmentation", {}).get("gt_domain_valid", True)
    ]


def _mean_valid_gt(results: list[dict[str, Any]], field: str) -> float | None:
    return _mean(_valid_gt_results(results), "segmentation", field)


def _metric_count(
    results: list[dict[str, Any]], section: str, field: str
) -> int:
    selected = (
        _valid_gt_results(results) if section == "segmentation" else results
    )
    return sum(
        isinstance(result.get(section, {}).get(field), (int, float))
        for result in selected
    )


def _write_summary(
    path: Path, manifest: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    success = {
        method: sum(
            result["method"] == method and result["status"] in SUCCESS_STATUSES
            for result in results
        )
        for method in METHODS
    }
    applicable = {
        method: sum(
            result["method"] == method and result["applicable"]
            for result in results
        )
        for method in METHODS
    }
    lines = [
        "# Articulated Reconstruction External Baseline Suite V1",
        "",
        "## Comparison contract",
        "",
        "- Object identity follows the existing aligned PartNet manifest.",
        "- Metrics use the shared evaluator; each method retains its declared native input protocol.",
        "- `aim_aligned` is a cross-protocol generalization result. Exact-paper AiM reproductions are reported separately and are not averaged into this manifest.",
        "- Unsupported outputs remain empty and are not inferred from another method.",
        "- GT part count is an explicit oracle for DTA and ArtGS runs.",
        "- PARIS and Ditto are evaluated only on their native two-part/one-joint scope.",
        "",
        "## Current status",
        "",
        "| Method | Protocol | Success | Oracle / scope |",
        "|---|---|---:|---|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {METHOD_PROTOCOLS[method]} | "
            f"{success[method]}/{applicable[method]} "
            f"| {', '.join(ORACLES.get(method, [])) or 'none declared'} |"
        )
    native_rows = []
    for method in METHODS:
        selected = [
            result
            for result in results
            if result["method"] == method and result["status"] in SUCCESS_STATUSES
        ]
        native_rows.append(
            {
                "method": method,
                "axis": _first_mean(
                    selected,
                    "kinematics",
                    (
                        "axis_angle_deg_type_correct",
                        "axis_angle_error_deg",
                        "axis_angle_error_deg_native_all",
                    ),
                ),
                "axis_position": _first_mean(
                    selected,
                    "kinematics",
                    (
                        "revolute_axis_line_bbox",
                        "axis_position_error",
                        "axis_position_error_native_x10",
                    ),
                ),
                "motion": _first_mean(
                    selected,
                    "kinematics",
                    (
                        "revolute_motion_error_rad",
                        "prismatic_motion_error_m",
                        "motion_error_native",
                    ),
                ),
                "psnr": _mean(selected, "geometry", "novel_view_psnr"),
            }
        )
    lines.extend(
        [
            "",
            "## Common segmentation metrics",
            "",
            "Metrics with an incomplete observed GT domain are excluded. `Metric N` "
            "can therefore be lower than successful runs until multi-frame union "
            "evaluation is completed.",
            "",
            "| Method | Success | Metric N | Point IoU | ARI | RI |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row['method']} | {row['success_n']}/{row['requested_n']} | "
                f"{row['metric_n']} | {_format_metric(row['point_iou'])} | "
                f"{_format_metric(row['ari'])} | {_format_metric(row['ri'])} |"
                for row in _segmentation_summary(results)
            ],
            "",
            "## Native kinematics and geometry",
            "",
            "These columns retain each method's native units and are not interchangeable "
            "unless the corresponding adapter reports a shared metric.",
            "",
            "| Method | Axis error | Axis-position / line error | Motion error | Novel-view PSNR |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row['method']} | {_format_metric(row['axis'])} | "
                f"{_format_metric(row['axis_position'])} | "
                f"{_format_metric(row['motion'])} | "
                f"{_format_metric(row['psnr'])} |"
                for row in native_rows
            ],
            "",
            "## Complexity buckets",
            "",
            "The manifest records `2`, `3-4`, and `>=5` buckets. "
            "Only methods with common segmentation outputs contribute segmentation metrics.",
            "",
            "## Failure analysis",
            "",
            "Failures are recorded in per-object result JSON and aggregated into `failures.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_mean(
    results: list[dict[str, Any]],
    section: str,
    fields: tuple[str, ...],
) -> float | None:
    for field in fields:
        value = _mean(results, section, field)
        if value is not None:
            return value
    return None


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _write_protocol_audit(path: Path) -> None:
    lines = [
        "# External Baseline Protocol Audit",
        "",
        "| Method | Native/suite input | Oracle or scope requirement | Unsupported outputs are left empty |",
        "|---|---|---|---|",
        "| Ours Hybrid | Continuous RGB-D interaction | None | No |",
        "| AiM aligned | AiM-style static scan + monocular interaction | Cross-protocol generalization; paper-like exact cases are separate | Directed topology is not native |",
        "| ReArt | Native/adapted 4D point clouds | No exact-K oracle | Directed topology is not native |",
        "| DTA | Two-state calibrated RGB-D | Exact `--num_parts` oracle | Any output absent from official artifacts |",
        "| ArtGS | Two-state multi-view RGBA/depth | Per-scene `num_slots` oracle | Any output absent from official artifacts |",
        "| VideoArtGS | Native monocular interaction | Oracle joint count, types, and parent topology | Common point segmentation, type-correct axis, axis-line, and Chamfer are converted when native mesh/joint export succeeds |",
        "| GaussianArt | Two-state multi-view RGB-D | Oracle part count, semantic initialization, and GT motion metadata | Common segmentation and geometry remain unsupported by the released aggregate output |",
        "| PARIS | Two-state multi-view RGB | Two-part method scope | General multi-part topology |",
        "| Ditto | Start/end fused RGB-D point clouds | One-joint method scope | General multi-part topology |",
        "",
        "DTA and ArtGS share the same 100-view start/end acquisition package, but each receives its own released file layout. GT part masks are not exported. Exact GT part count is stored separately and must be labeled as oracle input.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bucket(count: int) -> str:
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return ">=5"


def _number(value: str) -> float | None:
    return None if value == "" else float(value)


def _integer(value: str) -> int | None:
    return None if value == "" else int(float(value))


def _boolean(value: str) -> bool | None:
    if value == "":
        return None
    return value.lower() == "true"


if __name__ == "__main__":
    raise SystemExit(main())
