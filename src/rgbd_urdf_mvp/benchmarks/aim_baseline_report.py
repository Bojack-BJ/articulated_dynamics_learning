"""Aggregate protocol-aligned AiM baseline evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


PROTOCOL_LABELS = {
    "shared_3view": "Shared 3-view",
    "continuous_orbit": "Continuous orbit",
    "aim_style": "AiM-style approximation",
}


def complexity_bucket(gt_part_count: int) -> str:
    if gt_part_count == 2:
        return "2"
    if gt_part_count <= 4:
        return "3-4"
    return ">=5"


def load_run_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("runs", payload) if isinstance(payload, dict) else payload
        return [dict(row) for row in rows]
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _threshold_at_ratio(evaluation: dict[str, Any], ratio: float = 0.02) -> dict[str, Any]:
    for row in evaluation.get("thresholds", []):
        if math.isclose(float(row["distance_ratio_bbox"]), ratio, abs_tol=1e-9):
            return row
    primary = evaluation.get("primary", {})
    if math.isclose(float(primary.get("distance_ratio_bbox", -1.0)), ratio, abs_tol=1e-9):
        return primary
    raise ValueError(f"Evaluation has no threshold at {ratio:g} bbox")


def flatten_run(row: dict[str, Any], *, manifest_root: Path) -> dict[str, Any]:
    output = {
        "object_id": str(row["object_id"]),
        "category": str(row.get("category", "unknown")),
        "method": str(row["method"]).lower(),
        "protocol": str(row["protocol"]),
        "status": str(row.get("status", "success")).lower(),
        "evaluation_json": str(row.get("evaluation_json", "")),
        "failure_stage": str(row.get("failure_stage", "")),
        "exception": str(row.get("exception", "")),
        "moving_gaussian_count": _optional_int(row.get("moving_gaussian_count")),
        "predicted_component_count": _optional_int(row.get("predicted_component_count")),
        "nn_ransac_statistics": row.get("nn_ransac_statistics", ""),
    }
    if output["status"] != "success":
        # No final segmentation exists, so a component count cannot be measured reliably.
        output["predicted_component_count"] = None
        return output
    evaluation_path = Path(output["evaluation_json"]).expanduser()
    if not evaluation_path.is_absolute():
        evaluation_path = manifest_root / evaluation_path
    elif not evaluation_path.is_file():
        portable_path = manifest_root / "raw" / output["object_id"] / f"{output['method']}.json"
        if portable_path.is_file():
            evaluation_path = portable_path
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    segmentation = evaluation["primary"]["covered_only_metrics"]
    coverage_row = _threshold_at_ratio(evaluation)
    gt_count = int(evaluation["gt_part_count"])
    predicted_count = int(evaluation["predicted_part_count"])
    output.update({
        "gt_part_count": gt_count,
        "predicted_part_count": predicted_count,
        "part_count_error": predicted_count - gt_count,
        "part_count_exact": predicted_count == gt_count,
        "undersegmented_gt_part_count": int(segmentation["undersegmented_gt_part_count"]),
        "unmatched_gt_part_count": int(segmentation["unmatched_gt_part_count"]),
        "largest_cluster_ratio": float(segmentation["largest_cluster_ratio"]),
        "point_iou": float(evaluation["primary"]["coverage_aware_metrics"]["one_to_one_mean_iou"]),
        "ari": float(segmentation["adjusted_rand_index"]),
        "geometry_coverage": float(coverage_row["geometry_coverage"]),
        "complexity_bucket": complexity_bucket(gt_count),
    })
    return output


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success"]
    attempted = list(rows)
    if not attempted:
        return {}
    return {
        "attempted": len(attempted),
        "successful": len(successful),
        "success_rate": len(successful) / len(attempted),
        "point_iou_mean": _mean(successful, "point_iou"),
        "point_iou_median": _median(successful, "point_iou"),
        "ari_mean": _mean(successful, "ari"),
        "ari_median": _median(successful, "ari"),
        "geometry_coverage_mean": _mean(successful, "geometry_coverage"),
        "exact_part_count_rate": _rate(successful, "part_count_exact"),
        "mean_part_count_error": _mean(successful, "part_count_error"),
        "undersegmentation_rate": (
            mean(float(row["undersegmented_gt_part_count"] > 0) for row in successful)
            if successful else None
        ),
        "mean_undersegmented_gt_parts": _mean(successful, "undersegmented_gt_part_count"),
        "mean_unmatched_gt_parts": _mean(successful, "unmatched_gt_part_count"),
        "mean_largest_cluster_ratio": _mean(successful, "largest_cluster_ratio"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return median(values) if values else None


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean(float(bool(row[key])) for row in rows) if rows else None


def grouped_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for keys in (("method", "protocol"), ("method", "protocol", "complexity_bucket")):
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if "complexity_bucket" in keys and row.get("complexity_bucket") is None:
                continue
            grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
        result["/".join(keys)] = {
            "/".join(group): aggregate(group_rows) for group, group_rows in sorted(grouped.items())
        }
    return result


def protocol_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aim = {
        (row["object_id"], row["protocol"]): row
        for row in rows
        if row["method"] == "aim" and row["status"] == "success"
    }
    comparisons = (
        ("shared_3view", "continuous_orbit"),
        ("continuous_orbit", "aim_style"),
        ("shared_3view", "aim_style"),
    )
    output = []
    for object_id in sorted({key[0] for key in aim}):
        for before, after in comparisons:
            if (object_id, before) not in aim or (object_id, after) not in aim:
                continue
            left, right = aim[(object_id, before)], aim[(object_id, after)]
            output.append({
                "object_id": object_id,
                "category": left["category"],
                "from_protocol": before,
                "to_protocol": after,
                "delta_point_iou": right["point_iou"] - left["point_iou"],
                "delta_ari": right["ari"] - left["ari"],
                "delta_coverage": right["geometry_coverage"] - left["geometry_coverage"],
                "delta_predicted_part_count": (
                    right["predicted_part_count"] - left["predicted_part_count"]
                ),
            })
    return output


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful_aim = [
        row for row in rows if row["method"] == "aim" and row["status"] == "success"
    ]
    output = []
    for protocol in sorted({row["protocol"] for row in successful_aim}):
        selected = [row for row in successful_aim if row["protocol"] == protocol]
        for metric in ("point_iou", "ari", "part_count_error"):
            x = [float(row["geometry_coverage"]) for row in selected]
            y = [float(row[metric]) for row in selected]
            pearson, pearson_p = _correlation(pearsonr, x, y)
            spearman, spearman_p = _correlation(spearmanr, x, y)
            output.append({
                "protocol": protocol,
                "target_metric": metric,
                "sample_count": len(selected),
                "pearson": pearson,
                "pearson_p": pearson_p,
                "spearman": spearman,
                "spearman_p": spearman_p,
            })
    return output


def _correlation(function: Any, x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    result = function(x, y)
    return float(result.statistic), float(result.pvalue)


def paired_aggregates(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["from_protocol"], row["to_protocol"])].append(row)
    output = []
    for (before, after), selected in sorted(grouped.items()):
        row: dict[str, Any] = {
            "from_protocol": before,
            "to_protocol": after,
            "object_count": len(selected),
        }
        for metric in (
            "delta_point_iou", "delta_ari", "delta_coverage", "delta_predicted_part_count"
        ):
            row[f"{metric}_mean"] = _mean(selected, metric)
            row[f"{metric}_median"] = _median(selected, metric)
        output.append(row)
    return output


def write_extended_report(
    manifest_path: Path,
    output_root: Path,
    *,
    additional_manifests: Iterable[Path] = (),
    failure_details_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for current_manifest in (manifest_path, *additional_manifests):
        current_manifest = current_manifest.expanduser().resolve()
        rows.extend(
            flatten_run(row, manifest_root=current_manifest.parent)
            for row in load_run_manifest(current_manifest)
        )
    if failure_details_path is not None:
        details = json.loads(failure_details_path.expanduser().resolve().read_text(encoding="utf-8"))
        for row in rows:
            if row["status"] != "success" and row["object_id"] in details:
                row.update(details[row["object_id"]])
    pairs = protocol_pairs(rows)
    correlations = correlation_rows(rows)
    grouped = grouped_aggregates(rows)
    pair_summary = paired_aggregates(pairs)
    failures = _failure_report(rows)
    high_coverage_failures = sorted(
        (
            row for row in rows
            if row["method"] == "aim"
            and row["status"] == "success"
            and row["geometry_coverage"] >= 0.75
            and row["ari"] <= 0.2
        ),
        key=lambda row: (-row["geometry_coverage"], row["ari"]),
    )

    _write_csv(output_root / "per_object_metrics.csv", rows)
    _write_csv(output_root / "protocol_pairs.csv", pairs)
    scatter = [
        {
            key: row.get(key)
            for key in (
                "object_id", "category", "protocol", "geometry_coverage", "point_iou", "ari",
                "part_count_error", "predicted_part_count", "gt_part_count",
            )
        }
        for row in rows if row["method"] == "aim" and row["status"] == "success"
    ]
    _write_csv(output_root / "coverage_segmentation_scatter.csv", scatter)
    _write_csv(output_root / "coverage_segmentation_correlation.csv", correlations)
    (output_root / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    raw = {
        "aggregates": grouped,
        "protocol_pair_aggregates": pair_summary,
        "coverage_segmentation_correlations": correlations,
        "high_coverage_low_ari_cases": high_coverage_failures,
    }
    raw_root = output_root / "raw"
    raw_root.mkdir(exist_ok=True)
    (raw_root / "aggregate_metrics.json").write_text(
        json.dumps(raw, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(
        _render_markdown(rows, grouped, pair_summary, correlations, failures, high_coverage_failures),
        encoding="utf-8",
    )
    return raw


def _failure_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aim = [row for row in rows if row["method"] == "aim"]
    failures = [row for row in aim if row["status"] != "success"]
    return {
        "attempted": len(aim),
        "successful": len(aim) - len(failures),
        "failed": len(failures),
        "success_rate": (len(aim) - len(failures)) / len(aim) if aim else None,
        "failure_rate": len(failures) / len(aim) if aim else None,
        "failure_reason_histogram": dict(Counter(row["failure_stage"] or "unknown" for row in failures)),
        "failures": failures,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_markdown(
    rows: list[dict[str, Any]],
    grouped: dict[str, Any],
    pair_summary: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    failures: dict[str, Any],
    high_coverage_failures: list[dict[str, Any]],
) -> str:
    method_protocol = grouped["method/protocol"]
    lines = [
        "# Extended AiM Baseline Evaluation",
        "",
        "## Main Equal-Input Comparison",
        "",
        "Shared 3-view is the primary equal-input-budget comparison.",
        "",
        "| Method | Protocol | N | Success | Point IoU | ARI | Coverage | Exact count | Underseg. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, protocol in (
        ("hybrid", "shared_3view"),
        ("aim", "shared_3view"),
        ("aim", "continuous_orbit"),
        ("aim", "aim_style"),
    ):
        aggregate_row = method_protocol.get(f"{method}/{protocol}")
        if not aggregate_row:
            continue
        lines.append(
            f"| {method.title()} | {PROTOCOL_LABELS[protocol]} | "
            f"{aggregate_row['attempted']} | {_fmt(aggregate_row['success_rate'])} | "
            f"{_fmt(aggregate_row['point_iou_mean'])} | {_fmt(aggregate_row['ari_mean'])} | "
            f"{_fmt(aggregate_row['geometry_coverage_mean'])} | "
            f"{_fmt(aggregate_row['exact_part_count_rate'])} | "
            f"{_fmt(aggregate_row['undersegmentation_rate'])} |"
        )
    lines.extend([
        "",
        "Continuous orbit is a moving-camera robustness ablation. AiM-style approximation is a "
        "project-defined acquisition ablation, not an exact reproduction of the official camera "
        "path. Neither replaces the shared-input main comparison.",
        "",
        "## Under-Segmentation By Complexity",
        "",
        "| Method / protocol | GT parts | N | IoU | ARI | Count error | Unmatched GT | Largest cluster |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, aggregate_row in grouped["method/protocol/complexity_bucket"].items():
        method, protocol, bucket = key.split("/")
        lines.append(
            f"| {method.title()} / {PROTOCOL_LABELS.get(protocol, protocol)} | {bucket} | "
            f"{aggregate_row['attempted']} | {_fmt(aggregate_row['point_iou_mean'])} | "
            f"{_fmt(aggregate_row['ari_mean'])} | {_fmt(aggregate_row['mean_part_count_error'])} | "
            f"{_fmt(aggregate_row['mean_unmatched_gt_parts'])} | "
            f"{_fmt(aggregate_row['mean_largest_cluster_ratio'])} |"
        )
    lines.extend([
        "",
        "## AiM Protocol Ablation",
        "",
        "| Transition | Paired N | Delta IoU | Delta ARI | Delta coverage | Delta predicted parts |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in pair_summary:
        lines.append(
            f"| {PROTOCOL_LABELS[row['from_protocol']]} -> {PROTOCOL_LABELS[row['to_protocol']]} | "
            f"{row['object_count']} | {_fmt(row['delta_point_iou_mean'])} | "
            f"{_fmt(row['delta_ari_mean'])} | {_fmt(row['delta_coverage_mean'])} | "
            f"{_fmt(row['delta_predicted_part_count_mean'])} |"
        )
    lines.extend([
        "",
        "## Coverage Versus Segmentation",
        "",
        "| Protocol | Target | N | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ])
    for row in correlations:
        lines.append(
            f"| {PROTOCOL_LABELS.get(row['protocol'], row['protocol'])} | "
            f"{row['target_metric']} | {row['sample_count']} | {_fmt(row['pearson'])} | "
            f"{_fmt(row['spearman'])} |"
        )
    lines.extend(["", "### High-Coverage, Low-ARI Cases", ""])
    if high_coverage_failures:
        lines.extend([
            "| Object | Category | Protocol | Coverage | ARI | Pred. / GT parts |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in high_coverage_failures[:10]:
            lines.append(
                f"| {row['object_id']} | {row['category']} | "
                f"{PROTOCOL_LABELS.get(row['protocol'], row['protocol'])} | "
                f"{_fmt(row['geometry_coverage'])} | {_fmt(row['ari'])} | "
                f"{row['predicted_part_count']} / {row['gt_part_count']} |"
            )
    else:
        lines.append("No successful AiM case met coverage >= 0.75 and ARI <= 0.2.")
    lines.extend([
        "",
        "## Failure Analysis",
        "",
        f"- AiM success rate: {_fmt(failures['success_rate'])} "
        f"({failures['successful']}/{failures['attempted']}).",
        f"- Failure-stage histogram: `{json.dumps(failures['failure_reason_histogram'], sort_keys=True)}`.",
        "",
        "## Interpretation",
        "",
        "This report separates geometry coverage from motion-part segmentation. Higher coverage supports "
        "the hypothesis only when it fails to produce consistent IoU, ARI, and part-count gains across "
        "paired protocols; the correlation and high-coverage failure tables provide that evidence.",
        "",
        "## Limitations",
        "",
        "- Point IoU is observed-reference-point IoU and is not numerically comparable to AiM mesh/voxel IoU.",
        "- Hybrid orbit results, when present, measure distribution-shift robustness rather than main method capability.",
        "- GT labels are used only for evaluation and never for AiM reconstruction or segmentation.",
        "",
    ])
    return "\n".join(lines)
