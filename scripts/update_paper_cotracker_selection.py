#!/usr/bin/env python3
"""Build paper tables after validation selects the kinematic-ontology CoTracker model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FEATURES = ("cotracker_only", "tapip_only", "hybrid")
BUCKETS = ("2", "3-4", ">=5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ontology-root",
        type=Path,
        default=Path(
            "backups/dev-f-20260803/shared_results/partnet_core_v1_training/"
            "kinematic_ontology_v1"
        ),
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path(
            "outputs/external_baseline_suite_v1/aligned_v1/method_object_manifest.csv"
        ),
    )
    parser.add_argument(
        "--external-complexity",
        type=Path,
        default=Path(
            "outputs/external_baseline_suite_v1/kinematic_domain_revaluation_v1/"
            "complexity_summary.csv"
        ),
    )
    parser.add_argument(
        "--external-segmentation",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/segmentation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_experiments/cotracker_selection_v2"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bucket(part_count: int) -> str:
    if part_count <= 2:
        return "2"
    if part_count <= 4:
        return "3-4"
    return ">=5"


def feature_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    validation = aggregate["validation_selection"]
    methods = aggregate["common_test_domain"]["methods"]
    selected = validation["selected_feature_path"]
    rows = []
    for feature in FEATURES:
        metrics = methods[feature]
        rows.append(
            {
                "feature": feature,
                "selected": feature == selected,
                "validation_assignment_loss": validation["assignment_loss"][feature],
                "object_count": aggregate["common_test_domain"]["object_count"],
                **metrics,
            }
        )
    return rows


def aligned_ids(manifest_rows: list[dict[str, str]]) -> set[str]:
    return {
        row["object_id"]
        for row in manifest_rows
        if row["method"] == "ours_hybrid" and "full20" in row["suite_tiers"]
    }


def cotracker_per_object(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["method"] != "cotracker_only" or row["domain"] != "observable":
            continue
        output.append(
            {
                **row,
                "point_iou": float(row["point_iou"]),
                "ari": float(row["ari"]),
                "rand_index": float(row["rand_index"]),
                "predicted_part_count": int(row["predicted_part_count"]),
                "gt_part_count": int(row["gt_part_count"]),
                "part_count_exact": row["part_count_exact"].lower() == "true",
            }
        )
    return output


def aggregate_segmentation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "requested_n": len(rows),
        "success_n": len(rows),
        "metric_n": len(rows),
        "point_iou": mean(row["point_iou"] for row in rows),
        "ari": mean(row["ari"] for row in rows),
        "ri": mean(row["rand_index"] for row in rows),
        "exact_part_count_rate": mean(float(row["part_count_exact"]) for row in rows),
        "undersegmentation_rate": mean(
            float(row["predicted_part_count"] < row["gt_part_count"]) for row in rows
        ),
        "oversegmentation_rate": mean(
            float(row["predicted_part_count"] > row["gt_part_count"]) for row in rows
        ),
    }


def complexity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[bucket(row["gt_part_count"])].append(row)
    output = []
    for difficulty in BUCKETS:
        values = grouped[difficulty]
        metrics = aggregate_segmentation(values)
        output.append(
            {
                "complexity_bucket": difficulty,
                "method": "ours_cotracker",
                "success_n": metrics["success_n"],
                "metric_n": metrics["metric_n"],
                "point_iou": metrics["point_iou"],
                "ari": metrics["ari"],
                "ri": metrics["ri"],
                "undersegmentation_rate": metrics["undersegmentation_rate"],
            }
        )
    return output


def latex_fragment(
    features: list[dict[str, Any]],
    relation: dict[str, Any],
    aligned: dict[str, Any],
) -> str:
    feature_lines = []
    labels = {
        "cotracker_only": "CoTracker only",
        "tapip_only": "TAPIP3D only",
        "hybrid": "Hybrid",
    }
    for row in features:
        values = (
            labels[row["feature"]],
            row["point_iou"],
            row["ari"],
            row["rand_index"],
            row["exact_part_count_rate"],
            row["undersegmented_object_rate"],
            row["oversegmented_object_rate"],
        )
        line = "%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f" % values
        if row["selected"]:
            line = "\\textbf{" + line.replace(" & ", "} & \\textbf{") + "}"
        feature_lines.append(line + " \\\\")
    test = relation["test_metrics"]
    return "\n".join(
        [
            "% Generated by scripts/update_paper_cotracker_selection.py",
            "% Feature ablation on the fixed-connected-body collapsed ontology:",
            *feature_lines,
            "",
            "% Validation-selected main configuration: CoTracker only + full neural.",
            f"% Edge F1: {test['edge_f1']:.3f}",
            f"% Type accuracy: {test['joint_type_accuracy']:.3f}",
            f"% Axis mean / median: {test['axis_error_mean_deg']:.2f} / {test['axis_error_median_deg']:.2f} deg",
            f"% Axis-line: {test['axis_line_error_normalized']:.3f}",
            f"% Illegal graph rate: {test['illegal_graph_rate']:.3f}",
            "",
            "% Aligned-20 CoTracker segmentation row:",
            "Track2Art & Continuous multi-view RGB-D interaction & None "
            f"& {aligned['success_n']}/{aligned['requested_n']} & {aligned['metric_n']} "
            f"& {aligned['point_iou']:.3f} & {aligned['ari']:.3f} & {aligned['ri']:.3f} \\\\" ,
            "",
            "% Suggested model-selection text:",
            "% All feature-path choices were made on the validation split after applying",
            "% the same fixed-connected-body collapsed kinematic ontology used at test time.",
            "% CoTracker-only achieved the lowest validation assignment loss (0.108469),",
            "% compared with Hybrid (0.131409) and TAPIP3D-only (0.290504), and is therefore",
            "% used for the main model without test-set reselection.",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    root = args.ontology_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    per_object = cotracker_per_object(
        read_csv(root / "controlled_common_domain_v2/merged/per_object_metrics.csv")
    )
    ids = aligned_ids(read_csv(args.external_manifest.expanduser().resolve()))
    aligned_rows = [row for row in per_object if row["object_id"] in ids]
    if len(ids) != 20 or len(aligned_rows) != 20:
        raise ValueError(f"Expected aligned 20 rows, got ids={len(ids)} metrics={len(aligned_rows)}")

    features = feature_rows(aggregate)
    aligned = aggregate_segmentation(aligned_rows)
    relation = aggregate["selected_relation_head"]
    write_csv(output / "feature_ablation.csv", features)
    write_csv(output / "aligned20_cotracker_per_object.csv", aligned_rows)

    old_complexity = [
        row
        for row in read_csv(args.external_complexity.expanduser().resolve())
        if row["method"] != "ours_hybrid"
    ]
    new_complexity = complexity_rows(aligned_rows) + old_complexity
    write_csv(output / "external_complexity_summary.csv", new_complexity)

    old_segmentation = [
        row
        for row in read_csv(args.external_segmentation.expanduser().resolve())
        if row["method"] != "ours_hybrid"
    ]
    new_segmentation = [
        {
            "method": "ours_cotracker",
            "protocol": "continuous_interaction",
            **{key: aligned[key] for key in (
                "requested_n", "success_n", "metric_n", "point_iou", "ari", "ri",
                "undersegmentation_rate",
            )},
        },
        *old_segmentation,
    ]
    write_csv(output / "external_segmentation_summary.csv", new_segmentation)

    summary = {
        "schema": "paper-experiments-cotracker-selection-v2",
        "ontology": aggregate["common_test_domain"]["ontology"],
        "validation_selection": aggregate["validation_selection"],
        "controlled_feature_ablation": features,
        "selected_relation_head": relation,
        "aligned20_segmentation": aligned,
        "source_artifacts": {
            "ontology_root": str(root),
            "external_manifest": str(args.external_manifest),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "paper_experiment_replacements.tex").write_text(
        latex_fragment(features, relation, aligned), encoding="utf-8"
    )
    (output / "summary.md").write_text(
        "# CoTracker validation-selected experiment update\n\n"
        f"Validation selects **CoTracker only** ({aggregate['validation_selection']['assignment_loss']['cotracker_only']:.6f}) "
        f"over Hybrid ({aggregate['validation_selection']['assignment_loss']['hybrid']:.6f}) and "
        f"TAPIP3D ({aggregate['validation_selection']['assignment_loss']['tapip_only']:.6f}).\n\n"
        f"On the aligned 20-object subset, CoTracker obtains Point IoU {aligned['point_iou']:.3f}, "
        f"ARI {aligned['ari']:.3f}, and RI {aligned['ri']:.3f}. Its exact part-count rate is "
        f"{aligned['exact_part_count_rate']:.3f}, with under-/over-segmentation rates of "
        f"{aligned['undersegmentation_rate']:.3f}/{aligned['oversegmentation_rate']:.3f}.\n\n"
        "The aligned-20 result must replace the previous Hybrid row rather than be compared "
        "against it: feature selection and the fixed-body-collapsed ontology changed together.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
