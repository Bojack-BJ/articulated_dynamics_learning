#!/usr/bin/env python3
"""Select and summarize the controlled feature/backend ablation for Ours."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


FEATURES = ("cotracker_only", "tapip_only", "hybrid")
BACKENDS = {
    "full_neural": "full_neural",
    "neural_type_analytic_axis": "detected_pred_parts_analytic",
    "full_analytic": "detected_pred_parts_full_analytic",
}
SLOT_DIRS = {
    "cotracker_only": "cotracker_slots",
    "tapip_only": "tapip_slots",
    "hybrid": "hybrid_slots",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--slot-training-root", type=Path, required=True)
    parser.add_argument("--segmentation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def feature_validation_scores(root: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for feature, directory in SLOT_DIRS.items():
        summary = load_json(root / directory / "training_summary.json")
        if summary.get("validation_split") != "val":
            raise ValueError(f"{feature} was not selected on the validation split")
        scores[feature] = float(summary["best_val_assignment_loss"])
    return scores


def load_kinematic_reports(root: Path, split: str) -> dict[str, dict[str, Any]]:
    reports = {}
    for feature in FEATURES:
        path = root / feature / split / "analytic_axis_oracle_report.json"
        report = load_json(path)
        if report.get("split") != split:
            raise ValueError(f"Expected {split} report at {path}")
        reports[feature] = report
    return reports


def choose_backend(validation_report: dict[str, Any]) -> str:
    """Select without test access: success@20, penalized error, then type accuracy."""
    metrics = validation_report["end_to_end"]
    return max(
        BACKENDS,
        key=lambda backend: (
            float(metrics[BACKENDS[backend]]["joint_success_at_20_deg"]),
            -float(metrics[BACKENDS[backend]]["failure_penalized_axis_error_mean_deg"]),
            float(metrics[BACKENDS[backend]]["type_accuracy"]),
        ),
    )


def segmentation_per_object(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    return {
        (str(row["method"]), str(row["object_id"])): row
        for row in payload["per_object"]
        if str(row.get("split", "test")) == "test"
    }


def selected_joint_rows(report_root: Path, feature: str, backend: str) -> list[dict[str, Any]]:
    payload = load_json(report_root / feature / "test" / "analytic_axis_per_joint.json")
    setting = BACKENDS[backend]
    return [row for row in payload["joints"] if row["setting"] == setting]


def penalized_axis_error(row: dict[str, Any]) -> float:
    if not row.get("edge_detected") or not row.get("type_correct") or not row.get("valid"):
        return 90.0
    value = row.get("axis_error_deg")
    return 90.0 if value is None else float(value)


def difficulty_bucket(gt_part_count: int) -> str:
    if gt_part_count <= 2:
        return "2"
    if gt_part_count <= 4:
        return "3-4"
    return ">=5"


def per_object_rows(
    report_root: Path,
    configurations: list[tuple[str, str]],
    selected_feature: str,
    selected_backend: str,
    segmentation_rows: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    object_rows: list[dict[str, Any]] = []
    joint_output: list[dict[str, Any]] = []
    joints_by_configuration = {
        (feature, backend): selected_joint_rows(report_root, feature, backend)
        for feature, backend in configurations
    }
    availability: dict[str, list[str]] = {}
    common_objects: set[str] | None = None
    for feature, backend in configurations:
        joint_objects = {
            str(row["object_id"]) for row in joints_by_configuration[(feature, backend)]
        }
        segmentation_objects = {
            object_id for method, object_id in segmentation_rows if method == feature
        }
        available = joint_objects & segmentation_objects
        availability[f"{feature} + {backend}"] = sorted(available)
        common_objects = available if common_objects is None else common_objects & available
    common_objects = common_objects or set()

    for feature, backend in configurations:
        joints = joints_by_configuration[(feature, backend)]
        by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for joint in joints:
            by_object[str(joint["object_id"])].append(joint)
            joint_output.append(
                {
                    "configuration": f"{feature} + {backend}",
                    "feature": feature,
                    "kinematic_backend": backend,
                    "ours_main": feature == selected_feature and backend == selected_backend,
                    "aligned_common_object": str(joint["object_id"]) in common_objects,
                    **joint,
                    "failure_penalized_axis_error_deg": penalized_axis_error(joint),
                }
            )
        for object_id, object_joints in sorted(by_object.items()):
            if object_id not in common_objects:
                continue
            segmentation = segmentation_rows[(feature, object_id)]
            conditional = [
                row
                for row in object_joints
                if row.get("edge_detected")
                and row.get("type_correct")
                and row.get("valid")
                and row.get("axis_error_deg") is not None
            ]
            line_rows = [
                row for row in conditional if row.get("axis_line_error_bbox_normalized") is not None
            ]
            penalized = [penalized_axis_error(row) for row in object_joints]
            gt_parts = int(segmentation["gt_part_count"])
            object_rows.append(
                {
                    "configuration": f"{feature} + {backend}",
                    "feature": feature,
                    "kinematic_backend": backend,
                    "ours_main": feature == selected_feature and backend == selected_backend,
                    "object_id": object_id,
                    "category": segmentation["category"],
                    "difficulty_bucket": difficulty_bucket(gt_parts),
                    "gt_part_count": gt_parts,
                    "predicted_part_count": int(segmentation["predicted_part_count"]),
                    "point_iou": float(segmentation["one_to_one_mean_iou"]),
                    "ari": float(segmentation["adjusted_rand_index"]),
                    "rand_index": float(segmentation["rand_index"]),
                    "exact_part_count": bool(segmentation["part_count_exact"]),
                    "joint_count": len(object_joints),
                    "edge_recall": mean(float(row["edge_detected"]) for row in object_joints),
                    "joint_type_accuracy": mean(
                        float(row["type_correct"]) for row in object_joints
                    ),
                    "type_correct_axis_error_mean_deg": (
                        mean(float(row["axis_error_deg"]) for row in conditional)
                        if conditional
                        else None
                    ),
                    "type_correct_axis_error_median_deg": (
                        median(float(row["axis_error_deg"]) for row in conditional)
                        if conditional
                        else None
                    ),
                    "revolute_axis_line_error_bbox": (
                        mean(float(row["axis_line_error_bbox_normalized"]) for row in line_rows)
                        if line_rows
                        else None
                    ),
                    "failure_penalized_axis_error_mean_deg": mean(penalized),
                    "joint_success_at_10_deg": mean(float(value <= 10.0) for value in penalized),
                    "joint_success_at_20_deg": mean(float(value <= 20.0) for value in penalized),
                }
            )
    all_objects = sorted(set().union(*(set(values) for values in availability.values())))
    alignment = {
        "common_object_count": len(common_objects),
        "common_objects": sorted(common_objects),
        "available_objects_by_configuration": availability,
        "dropped_objects_by_configuration": {
            configuration: sorted(set(all_objects) - set(objects))
            for configuration, objects in availability.items()
        },
    }
    return object_rows, joint_output, alignment


def aligned_result_row(
    feature: str,
    backend: str,
    selected_feature: str,
    selected_backend: str,
    object_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    configuration = f"{feature} + {backend}"
    objects = [row for row in object_rows if row["configuration"] == configuration]
    joints = [
        row
        for row in joint_rows
        if row["configuration"] == configuration and row["aligned_common_object"]
    ]
    conditional = [
        row
        for row in joints
        if row.get("edge_detected")
        and row.get("type_correct")
        and row.get("valid")
        and row.get("axis_error_deg") is not None
    ]
    line_rows = [
        row for row in conditional if row.get("axis_line_error_bbox_normalized") is not None
    ]
    penalized = [float(row["failure_penalized_axis_error_deg"]) for row in joints]
    return {
        "configuration": configuration,
        "feature": feature,
        "kinematic_backend": backend,
        "ours_main": feature == selected_feature and backend == selected_backend,
        "test_object_count": len(objects),
        "test_joint_count": len(joints),
        "point_iou": mean(float(row["point_iou"]) for row in objects),
        "ari": mean(float(row["ari"]) for row in objects),
        "rand_index": mean(float(row["rand_index"]) for row in objects),
        "exact_part_count_rate": mean(float(row["exact_part_count"]) for row in objects),
        "edge_recall": mean(float(row["edge_detected"]) for row in joints),
        "joint_type_accuracy": mean(float(row["type_correct"]) for row in joints),
        "type_correct_axis_error_mean_deg": mean(
            float(row["axis_error_deg"]) for row in conditional
        ),
        "type_correct_axis_error_median_deg": median(
            float(row["axis_error_deg"]) for row in conditional
        ),
        "revolute_axis_line_error_bbox": mean(
            float(row["axis_line_error_bbox_normalized"]) for row in line_rows
        ),
        "failure_penalized_axis_error_mean_deg": mean(penalized),
        "joint_success_at_10_deg": mean(float(value <= 10.0) for value in penalized),
        "joint_success_at_20_deg": mean(float(value <= 20.0) for value in penalized),
    }


def grouped_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["configuration"]), str(row[group_key]))].append(row)
    metrics = (
        "point_iou",
        "ari",
        "rand_index",
        "edge_recall",
        "joint_type_accuracy",
        "failure_penalized_axis_error_mean_deg",
        "joint_success_at_20_deg",
    )
    output = []
    for (configuration, group), values in sorted(groups.items()):
        output.append(
            {
                "configuration": configuration,
                group_key: group,
                "object_count": len(values),
                **{metric: mean(float(row[metric]) for row in values) for metric in metrics},
                "exact_part_count_rate": mean(float(row["exact_part_count"]) for row in values),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output_dir = (args.output_dir or root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_scores = feature_validation_scores(args.slot_training_root.expanduser().resolve())
    selected_feature = min(feature_scores, key=feature_scores.get)
    validation = load_kinematic_reports(root, "val")
    selected_backend = choose_backend(validation[selected_feature])
    segmentation_objects = segmentation_per_object(args.segmentation_report)

    configurations = [(feature, selected_backend) for feature in FEATURES]
    configurations.extend((selected_feature, backend) for backend in BACKENDS)
    configurations = list(dict.fromkeys(configurations))
    objects, joints, alignment = per_object_rows(
        root,
        configurations,
        selected_feature,
        selected_backend,
        segmentation_objects,
    )
    rows = [
        aligned_result_row(
            feature,
            backend,
            selected_feature,
            selected_backend,
            objects,
            joints,
        )
        for feature, backend in configurations
    ]
    main_row = next(row for row in rows if row["ours_main"])

    selection = {
        "selection_policy": {
            "feature": "minimum best validation assignment loss",
            "backend": [
                "maximum validation end-to-end joint_success_at_20_deg",
                "minimum validation failure_penalized_axis_error_mean_deg",
                "maximum validation joint_type_accuracy",
            ],
            "test_metrics_used_for_selection": False,
        },
        "feature_validation_scores": feature_scores,
        "selected_feature": selected_feature,
        "backend_validation_end_to_end": validation[selected_feature]["end_to_end"],
        "selected_backend": selected_backend,
        "test_object_alignment": alignment,
        "ours_main": main_row,
    }
    (output_dir / "validation_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "test_object_alignment.json").write_text(
        json.dumps(alignment, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "ours_five_config_test.csv", rows)
    write_csv(output_dir / "ours_five_config_per_object.csv", objects)
    write_csv(output_dir / "ours_five_config_per_joint.csv", joints)
    write_csv(output_dir / "ours_five_config_by_category.csv", grouped_rows(objects, "category"))
    write_csv(
        output_dir / "ours_five_config_by_complexity.csv",
        grouped_rows(objects, "difficulty_bucket"),
    )
    write_csv(
        output_dir / "feature_ablation.csv",
        [row for row in rows if row["kinematic_backend"] == selected_backend],
    )
    write_csv(
        output_dir / "kinematics_backend_ablation.csv",
        [row for row in rows if row["feature"] == selected_feature],
    )

    lines = [
        "# Ours Controlled Ablation",
        "",
        "The feature and kinematic backend are selected on validation data only. "
        "The Cartesian 3x3 grid is not reported: the feature ablation fixes the "
        "selected backend, and the backend ablation fixes the selected feature.",
        "",
        f"- Selected feature: `{selected_feature}` (validation assignment loss "
        f"{feature_scores[selected_feature]:.6f})",
        f"- Selected backend: `{selected_backend}`",
        f"- Ours Main: `{main_row['configuration']}`",
        "",
        "| Configuration | Main | IoU | ARI | RI | Edge recall | Type acc. | "
        "Axis mean | Axis median | Axis-line | Penalized axis | Joint@20 |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {'yes' if row['ours_main'] else ''} | "
            f"{fmt(row['point_iou'])} | {fmt(row['ari'])} | {fmt(row['rand_index'])} | "
            f"{fmt(row['edge_recall'])} | {fmt(row['joint_type_accuracy'])} | "
            f"{fmt(row['type_correct_axis_error_mean_deg'])} | "
            f"{fmt(row['type_correct_axis_error_median_deg'])} | "
            f"{fmt(row['revolute_axis_line_error_bbox'])} | "
            f"{fmt(row['failure_penalized_axis_error_mean_deg'])} | "
            f"{fmt(row['joint_success_at_20_deg'])} |"
        )
    lines.extend(
        [
            "",
            "`neural_type_analytic_axis` keeps neural topology and joint type, then "
            "replaces only axis/line estimation with relative-SE(3) fitting. "
            "`full_analytic` keeps neural topology proposals but selects joint type "
            "and estimates axis/line analytically from a common point-replay error.",
            "",
            "`edge_recall` is reported instead of Edge F1 because the oracle "
            "decomposition evaluator scores recovery of GT pairs and does not count "
            "all unmatched predicted edges.",
            "",
            "Raw analysis tables are retained at per-object and per-joint resolution. "
            "Category and complexity summaries use macro averages over objects; the "
            "complexity buckets are GT part counts 2, 3-4, and >=5.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **selection}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
