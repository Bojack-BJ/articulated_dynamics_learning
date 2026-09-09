#!/usr/bin/env python3
"""Build provenance-aware AiM, ReArt, and Ours comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PILOT_IDS = {"partnet_103069", "partnet_20745", "partnet_10638"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-segmentation-csv", type=Path, required=True)
    parser.add_argument("--ours-relation-csv", type=Path, required=True)
    parser.add_argument("--ours-analytic-report", type=Path, required=True)
    parser.add_argument("--aim-style-metrics", type=Path, required=True)
    parser.add_argument("--aim-axis-summary", type=Path)
    parser.add_argument("--reart-native-summary", type=Path, required=True)
    parser.add_argument("--reart-run-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ours_rows = _read_csv(args.ours_segmentation_csv)
    aim_rows = json.loads(args.aim_style_metrics.read_text(encoding="utf-8"))
    aim_axis = (
        json.loads(args.aim_axis_summary.read_text(encoding="utf-8"))
        if args.aim_axis_summary
        else None
    )
    reart = json.loads(args.reart_native_summary.read_text(encoding="utf-8"))
    reart_run = (
        json.loads(args.reart_run_manifest.read_text(encoding="utf-8"))
        if args.reart_run_manifest
        else None
    )
    relation = _read_csv(args.ours_relation_csv)
    analytic = json.loads(args.ours_analytic_report.read_text(encoding="utf-8"))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot_rows = _pilot_segmentation_rows(ours_rows, aim_rows)
    native_rows = _native_segmentation_rows(ours_rows, aim_rows, reart)
    kinematic_rows = _kinematic_rows(relation, analytic, reart, aim_axis)
    _write_csv(output / "common_pilot_segmentation.csv", pilot_rows)
    _write_csv(output / "native_protocol_segmentation.csv", native_rows)
    _write_csv(output / "kinematics_axis_comparison.csv", kinematic_rows)
    _write_report(
        output / "summary.md",
        pilot_rows,
        native_rows,
        kinematic_rows,
        reart_run,
    )
    return 0


def _pilot_segmentation_rows(
    ours_rows: list[dict[str, str]], aim_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for method in ("cotracker_only", "tapip_only", "hybrid"):
        rows = [
            row for row in ours_rows
            if row["method"] == method and row["object_id"] in PILOT_IDS
        ]
        output.append(_segmentation_summary(f"Ours/{method}", rows, "PartNet pilot; learned slots"))
    output.append(
        {
            "method": "AiM",
            "protocol": "AiM-style approximation: 24-view static scan + monocular interaction",
            "object_count": len(aim_rows),
            "point_iou": _mean(aim_rows, "point_iou"),
            "ari": _mean(aim_rows, "ari"),
            "rand_index": None,
            "predicted_parts_total": sum(int(row["predicted_parts"]) for row in aim_rows),
            "gt_parts_total": sum(int(row["gt_parts"]) for row in aim_rows),
            "comparison_scope": "same three PartNet object ids and common observed-point evaluator",
        }
    )
    return output


def _native_segmentation_rows(
    ours_rows: list[dict[str, str]],
    aim_rows: list[dict[str, Any]],
    reart: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for method in ("cotracker_only", "tapip_only", "hybrid"):
        rows = [row for row in ours_rows if row["method"] == method]
        summary = _segmentation_summary(
            f"Ours/{method}", rows, "PartNet held-out test; shared recording protocol"
        )
        summary["comparison_scope"] = "native/protocol aggregate; not paired across datasets"
        output.append(summary)
    output.append(
        {
            "method": "AiM",
            "protocol": "AiM-style approximation; three PartNet pilot objects",
            "object_count": len(aim_rows),
            "point_iou": _mean(aim_rows, "point_iou"),
            "ari": _mean(aim_rows, "ari"),
            "rand_index": None,
            "predicted_parts_total": sum(int(row["predicted_parts"]) for row in aim_rows),
            "gt_parts_total": sum(int(row["gt_parts"]) for row in aim_rows),
            "comparison_scope": "native/protocol aggregate; not paired across datasets",
        }
    )
    output.append(
        {
            "method": "ReArt",
            "protocol": str(reart["protocol"]),
            "object_count": reart["sequence_count"],
            "point_iou": reart["point_iou_mean"],
            "ari": reart["ari_mean"],
            "rand_index": reart["official_multi_scan_ri_mean"],
            "predicted_parts_total": None,
            "gt_parts_total": None,
            "comparison_scope": "official Sapiens subset; not PartNet object-paired",
        }
    )
    return output


def _kinematic_rows(
    relation_rows: list[dict[str, str]],
    analytic: dict[str, Any],
    reart: dict[str, Any],
    aim_axis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    output = []
    for row in relation_rows:
        if row["phase"] != "decoder_finetune":
            continue
        output.append(
            {
                "method": f"Ours/{row['method']}/relation_head",
                "dataset_protocol": "PartNet held-out test",
                "edge_f1": float(row["edge_f1"]),
                "joint_type_accuracy": float(row["joint_type_accuracy"]),
                "axis_error_mean_deg": float(row["axis_error_deg_type_correct"]),
                "axis_error_median_deg": None,
                "axis_line_error": float(row["axis_line_error"]),
                "metric_scope": "native learned relation head; type-correct axes",
            }
        )
    analytic_overall = analytic["settings"]["detected_pred_parts_analytic"]["overall"]
    output.append(
        {
            "method": "Ours/cotracker_only/analytic_SE3",
            "dataset_protocol": "PartNet held-out test",
            "edge_f1": (
                analytic_overall["edge_detected_count"] / analytic_overall["joint_count"]
            ),
            "joint_type_accuracy": (
                analytic_overall["type_correct_count"] / analytic_overall["joint_count"]
            ),
            "axis_error_mean_deg": analytic_overall["axis_error_mean_deg"],
            "axis_error_median_deg": analytic_overall["axis_error_median_deg"],
            "axis_line_error": analytic_overall["axis_line_error_mean_bbox_normalized"],
            "metric_scope": "analytic SE(3), predicted parts/edges; evaluable joints only",
        }
    )
    output.append(
        {
            "method": "ReArt/converted_SE3",
            "dataset_protocol": str(reart["protocol"]),
            "edge_f1": None,
            "joint_type_accuracy": reart["converted_type_accuracy"],
            "axis_error_mean_deg": reart["converted_axis_error_mean_deg"],
            "axis_error_median_deg": reart["converted_axis_error_median_deg"],
            "axis_line_error": None,
            "metric_scope": reart["axis_metric_scope"],
        }
    )
    output.append(
        {
            "method": "AiM/native_screw",
            "dataset_protocol": "AiM-style approximation; three PartNet pilot objects",
            "edge_f1": None,
            "joint_type_accuracy": aim_axis.get("joint_type_accuracy") if aim_axis else None,
            "axis_error_mean_deg": aim_axis.get("axis_error_mean_deg") if aim_axis else None,
            "axis_error_median_deg": aim_axis.get("axis_error_median_deg") if aim_axis else None,
            "axis_line_error": (
                aim_axis.get("axis_line_error_mean_bbox_normalized") if aim_axis else None
            ),
            "metric_scope": (
                aim_axis.get("metric_scope")
                if aim_axis
                else "motion.json was not supplied to the comparison report"
            ),
        }
    )
    return output


def _segmentation_summary(
    method: str, rows: list[dict[str, str]], protocol: str
) -> dict[str, Any]:
    return {
        "method": method,
        "protocol": protocol,
        "object_count": len(rows),
        "point_iou": _mean(rows, "one_to_one_mean_iou"),
        "ari": _mean(rows, "adjusted_rand_index"),
        "rand_index": _mean(rows, "rand_index"),
        "predicted_parts_total": sum(int(row["predicted_part_count"]) for row in rows),
        "gt_parts_total": sum(int(row["gt_part_count"]) for row in rows),
        "comparison_scope": "same three PartNet object ids and common observed-point evaluator",
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.fmean(values) if values else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    pilot: list[dict[str, Any]],
    native: list[dict[str, Any]],
    kinematics: list[dict[str, Any]],
    reart_run: dict[str, Any] | None,
) -> None:
    lines = [
        "# AiM, ReArt, and Ours Comparison",
        "",
        "## Common PartNet Pilot Segmentation",
        "",
        "| Method | Objects | Point IoU | ARI | RI | Pred./GT parts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['method']} | {row['object_count']} | {_fmt(row['point_iou'])} | "
        f"{_fmt(row['ari'])} | {_fmt(row['rand_index'])} | "
        f"{row['predicted_parts_total']}/{row['gt_parts_total']} |"
        for row in pilot
    )
    lines.extend(
        [
            "",
            "ReArt native Sapiens is intentionally excluded from this paired table because the official",
            "dataset does not expose PartNet object IDs corresponding to the three AiM pilot objects.",
            "",
            "## Native-Protocol Aggregates",
            "",
            "| Method | Protocol | N | Point IoU | ARI | RI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['method']} | {row['protocol']} | {row['object_count']} | "
        f"{_fmt(row['point_iou'])} | {_fmt(row['ari'])} | {_fmt(row['rand_index'])} |"
        for row in native
    )
    lines.extend(
        [
            "",
            "These rows are protocol-specific aggregates, not an apples-to-apples ranking.",
            "",
            "## Kinematics and Axis",
            "",
            "| Method | Edge F1 | Type accuracy | Axis mean | Axis median | Axis-line | Scope |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['method']} | {_fmt(row['edge_f1'])} | "
        f"{_fmt(row['joint_type_accuracy'])} | {_fmt(row['axis_error_mean_deg'])} | "
        f"{_fmt(row['axis_error_median_deg'])} | {_fmt(row['axis_line_error'])} | "
        f"{row['metric_scope']} |"
        for row in kinematics
    )
    lines.extend(
        [
            "",
            "## Provenance Notes",
            "",
            "- AiM-style results were reused; no camera/reconstruction rerun was performed.",
            "- AiM type/axis are evaluated from its native motion.json screw output after point-IoU part matching.",
            "- ReArt RI is native. ReArt Point IoU/ARI are converted on its canonical 512-point GT domain.",
            "- ReArt axis/type rows are conditional diagnostics converted from predicted edges and relative GT SE(3).",
            "- Ours reports all three part-segmentation settings and both learned relation-head and analytic SE(3) axes.",
        ]
    )
    if reart_run:
        runs = reart_run["runs"]
        successful = [run for run in runs if run["status"] == "success"]
        failed = [run for run in runs if run["status"] != "success"]
        mean_runtime = statistics.fmean(run["runtime_s"] for run in successful)
        lines.extend(
            [
                "",
                "## ReArt Native Run Status",
                "",
                f"- Successful sequences: {len(successful)}/{len(runs)}.",
                f"- Total wall time with parallel workers: {reart_run['elapsed_s']:.1f} s.",
                f"- Mean successful sequence runtime: {mean_runtime:.1f} s.",
                f"- Failed indices: {', '.join(str(run['index']) for run in failed) or 'none'}.",
                "- Failures remain unpatched to preserve the official ReArt optimization behavior.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
