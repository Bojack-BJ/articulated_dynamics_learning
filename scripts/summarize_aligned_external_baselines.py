#!/usr/bin/env python3
"""Build an object-aligned, metric-aligned external baseline report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = ("Ours Hybrid", "ReArt", "AiM")
PROTOCOLS = {
    "Ours Hybrid": "PartNet RGB-D interaction recording",
    "ReArt": "PartNet object-aligned ReArt 4-frame point-cloud protocol",
    "AiM": "AiM-style 24-view static scan + 120-frame monocular interaction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_tsv", type=Path)
    parser.add_argument("--ours-csv", type=Path, required=True)
    parser.add_argument("--aim-style-root", type=Path, required=True)
    parser.add_argument("--aim-style-summary", type=Path)
    parser.add_argument("--reart-csv", type=Path, required=True)
    parser.add_argument("--reart-raw-root", type=Path)
    parser.add_argument("--reart-failures", type=Path)
    parser.add_argument("--reart-points-per-frame", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = _read_table(args.objects_tsv, delimiter="\t")
    ours = _load_ours(args.ours_csv)
    aim = _load_aim_style(args.aim_style_root, args.aim_style_summary)
    reart = _load_reart(args.reart_csv, args.reart_raw_root)
    reart_failures = _load_failures(args.reart_failures)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest, per_object, failures = build_report_rows(
        objects, ours, aim, reart, reart_failures
    )
    common_ids = {
        row["object_id"]
        for row in per_object
        if all(row[f"{prefix}_status"] == "success" for prefix in ("ours", "reart", "aim"))
    }
    protocols = dict(PROTOCOLS)
    protocols["ReArt"] = (
        f"PartNet object-aligned ReArt 4-frame x "
        f"{args.reart_points_per_frame}-point protocol"
    )
    requested_summary = summarize_methods(
        per_object, requested_count=len(manifest), protocols=protocols
    )
    common_summary = summarize_methods(
        [row for row in per_object if row["object_id"] in common_ids],
        requested_count=len(common_ids),
        protocols=protocols,
    )
    complexity = summarize_complexity(per_object, protocols=protocols)

    _write_csv(output / "aligned_manifest.csv", manifest)
    _write_csv(output / "aligned_per_object.csv", per_object)
    _write_csv(output / "requested_subset_summary.csv", requested_summary)
    _write_csv(output / "common_success_summary.csv", common_summary)
    _write_csv(output / "complexity_summary.csv", complexity)
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(
        output / "summary.md",
        manifest,
        requested_summary,
        common_summary,
        complexity,
        common_ids,
        failures,
    )
    return 0


def build_report_rows(
    objects: list[dict[str, str]],
    ours: dict[str, dict[str, Any]],
    aim: dict[str, dict[str, Any]],
    reart: dict[str, dict[str, Any]],
    reart_failures: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = []
    per_object = []
    failures = []
    for source in objects:
        object_id = source["object_id"]
        ours_row = ours.get(object_id)
        aim_row = aim.get(object_id)
        reart_row = reart.get(object_id)
        reart_failure = reart_failures.get(object_id)
        category = source["category"]
        gt_count = _first_int(
            source.get("gt_part_count"),
            _field(ours_row, "gt_part_count"),
            _field(aim_row, "gt_part_count"),
            _field(reart_row, "gt_part_count"),
        )
        manifest.append(
            {
                "object_id": object_id,
                "category": category,
                "gt_part_count": gt_count,
                "available_ours": ours_row is not None,
                "available_aim_style": aim_row is not None,
                "available_reart_native": reart_row is not None or reart_failure is not None,
                "reart_protocol_scope": (
                    "partnet_object_aligned_native_format"
                    if reart_row is not None or reart_failure is not None
                    else "missing"
                ),
            }
        )
        row = {
            "object_id": object_id,
            "category": category,
            "gt_part_count": gt_count,
        }
        _add_method_fields(row, "ours", ours_row, expected_gt_count=gt_count)
        _add_method_fields(
            row,
            "reart",
            reart_row,
            explicit_failure=reart_failure,
            expected_gt_count=gt_count,
        )
        _add_method_fields(row, "aim", aim_row, expected_gt_count=gt_count)
        row["aim_coverage"] = _field(aim_row, "geometry_coverage")
        per_object.append(row)
        for method, result, failure in (
            ("Ours Hybrid", ours_row, None),
            ("ReArt", reart_row, reart_failure),
            ("AiM", aim_row, None),
        ):
            method_prefix = {
                "Ours Hybrid": "ours",
                "ReArt": "reart",
                "AiM": "aim",
            }[method]
            method_status = row[f"{method_prefix}_status"]
            if method_status != "success":
                failures.append(
                    {
                        "object_id": object_id,
                        "category": category,
                        "method": method,
                        "status": method_status,
                        "failure": failure,
                        "expected_gt_part_count": gt_count,
                        "observed_gt_part_count": _field(result, "gt_part_count"),
                    }
                )
    per_object.sort(key=lambda row: (_sort_int(row["gt_part_count"]), row["object_id"]))
    return manifest, per_object, failures


def summarize_methods(
    rows: list[dict[str, Any]],
    requested_count: int,
    protocols: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    protocols = protocols or PROTOCOLS
    output = []
    for method, prefix in (("Ours Hybrid", "ours"), ("ReArt", "reart"), ("AiM", "aim")):
        selected = [row for row in rows if row[f"{prefix}_status"] == "success"]
        output.append(
            {
                "method": method,
                "protocol": protocols[method],
                "requested_n": requested_count,
                "success_n": len(selected),
                "point_iou": _mean(selected, f"{prefix}_iou"),
                "ari": _mean(selected, f"{prefix}_ari"),
                "ri": _mean(selected, f"{prefix}_ri"),
                "mean_part_count_error": _mean(selected, f"{prefix}_part_count_error"),
                "undersegmentation_rate": _mean(selected, f"{prefix}_undersegmented"),
                "mean_unmatched_gt_parts": _mean(
                    selected, f"{prefix}_unmatched_gt_parts"
                ),
                "mean_largest_cluster_ratio": _mean(
                    selected, f"{prefix}_largest_cluster_ratio"
                ),
            }
        )
    return output


def summarize_complexity(
    rows: list[dict[str, Any]], protocols: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    output = []
    for bucket in ("2", "3-4", ">=5"):
        bucket_rows = [
            row for row in rows if _complexity_bucket(row["gt_part_count"]) == bucket
        ]
        for summary in summarize_methods(
            bucket_rows, requested_count=len(bucket_rows), protocols=protocols
        ):
            output.append({"complexity_bucket": bucket, **summary})
    return output


def _load_ours(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_table(path)
    return {
        row["object_id"]: _normalize_metrics(row)
        for row in rows
        if row.get("method", "").lower() == "hybrid"
    }


def _load_reart(
    path: Path, raw_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    output = {
        row["object_id"]: _normalize_metrics(row)
        for row in _read_table(path)
        if row.get("status") == "success"
    }
    if raw_root is not None:
        for object_id, metrics in output.items():
            raw_path = raw_root / object_id / "reart.json"
            if not raw_path.is_file():
                continue
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            primary = payload.get("primary", {}).get("covered_only_metrics")
            if primary:
                raw_metrics = _normalize_metrics(primary)
                if raw_metrics["ri"] is not None:
                    metrics["ri"] = raw_metrics["ri"]
    return output


def _load_aim_style(
    root: Path, summary_path: Path | None = None
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    paths = list(root.glob("*_aim_style.json"))
    paths.extend(root.glob("*/aim.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("primary", {}).get("covered_only_metrics")
        if metrics is None:
            metrics = payload.get("primary")
        if metrics is None:
            continue
        object_id = (
            path.parent.name
            if path.name == "aim.json"
            else path.name.removesuffix("_aim_style.json")
        )
        if not object_id.startswith("partnet_"):
            object_id = f"partnet_{object_id}"
        normalized = _normalize_metrics(metrics)
        normalized["geometry_coverage"] = _first_number(
            _coverage_at_ratio(payload, 0.02),
            payload.get("primary", {}).get("geometry_coverage"),
            payload.get("coverage"),
        )
        output[object_id] = normalized
    if summary_path is not None and summary_path.is_file():
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            object_id = str(row["object_id"])
            if object_id in output:
                output[object_id]["geometry_coverage"] = _first_number(
                    row.get("coverage"), row.get("geometry_coverage")
                )
    return output


def _coverage_at_ratio(payload: dict[str, Any], ratio: float) -> float | None:
    for row in payload.get("thresholds", []):
        if math.isclose(float(row.get("distance_ratio_bbox", -1.0)), ratio):
            return _number(row.get("geometry_coverage"))
    return None


def _normalize_metrics(row: dict[str, Any]) -> dict[str, Any]:
    overlap = row.get("overlap_matrix")
    ri = _number(row.get("rand_index"))
    if ri is None and overlap:
        ri = rand_index_from_contingency(overlap)
    gt_count = _first_int(row.get("gt_part_count"), row.get("gt_parts"))
    pred_count = _first_int(row.get("predicted_part_count"), row.get("predicted_parts"))
    unmatched = _first_int(row.get("unmatched_gt_part_count"))
    under_count = _first_int(row.get("undersegmented_gt_part_count"))
    return {
        "gt_part_count": gt_count,
        "predicted_part_count": pred_count,
        "point_iou": _first_number(row.get("one_to_one_mean_iou"), row.get("point_iou")),
        "ari": _first_number(row.get("adjusted_rand_index"), row.get("ari")),
        "ri": ri,
        "unmatched_gt_part_count": unmatched,
        "undersegmented": (
            bool(under_count > 0)
            if under_count is not None
            else bool(pred_count is not None and gt_count is not None and pred_count < gt_count)
        ),
        "largest_cluster_ratio": _number(row.get("largest_cluster_ratio")),
        "geometry_coverage": _first_number(
            row.get("geometry_coverage"), row.get("coverage")
        ),
    }


def rand_index_from_contingency(matrix: Iterable[Iterable[int]]) -> float | None:
    rows = [[int(value) for value in row] for row in matrix]
    total = sum(sum(row) for row in rows)
    if total < 2:
        return None
    same_both = sum(_choose2(value) for row in rows for value in row)
    same_gt = sum(_choose2(sum(rows[r][c] for r in range(len(rows)))) for c in range(len(rows[0])))
    same_pred = sum(_choose2(sum(row)) for row in rows)
    total_pairs = _choose2(total)
    different_both = total_pairs - same_gt - same_pred + same_both
    return (same_both + different_both) / total_pairs


def _add_method_fields(
    target: dict[str, Any],
    prefix: str,
    metrics: dict[str, Any] | None,
    explicit_failure: dict[str, Any] | None = None,
    expected_gt_count: int | None = None,
) -> None:
    status = "success" if metrics is not None else ("failed" if explicit_failure else "missing")
    target[f"{prefix}_status"] = status
    target[f"{prefix}_gt_domain_valid"] = metrics is not None
    target[f"{prefix}_observed_gt_parts"] = (
        metrics["gt_part_count"] if metrics is not None else None
    )
    for name in (
        "iou",
        "ari",
        "ri",
        "pred_parts",
        "part_count_error",
        "undersegmented",
        "unmatched_gt_parts",
        "largest_cluster_ratio",
    ):
        target[f"{prefix}_{name}"] = None
    if metrics is None:
        return
    gt_count = expected_gt_count or metrics["gt_part_count"]
    pred_count = metrics["predicted_part_count"]
    target[f"{prefix}_iou"] = metrics["point_iou"]
    target[f"{prefix}_ari"] = metrics["ari"]
    target[f"{prefix}_ri"] = metrics["ri"]
    target[f"{prefix}_pred_parts"] = pred_count
    target[f"{prefix}_part_count_error"] = (
        pred_count - gt_count
        if pred_count is not None and gt_count is not None
        else None
    )
    target[f"{prefix}_undersegmented"] = (
        pred_count < gt_count
        if pred_count is not None and gt_count is not None
        else metrics["undersegmented"]
    )
    observed_unmatched = metrics["unmatched_gt_part_count"]
    count_unmatched = (
        max(0, gt_count - pred_count)
        if pred_count is not None and gt_count is not None
        else None
    )
    unmatched_values = [
        value for value in (observed_unmatched, count_unmatched) if value is not None
    ]
    target[f"{prefix}_unmatched_gt_parts"] = (
        max(unmatched_values) if unmatched_values else None
    )
    target[f"{prefix}_largest_cluster_ratio"] = metrics["largest_cluster_ratio"]


def _load_failures(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    return {
        row["object_id"]: row
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def _write_markdown(
    path: Path,
    manifest: list[dict[str, Any]],
    requested: list[dict[str, Any]],
    common: list[dict[str, Any]],
    complexity: list[dict[str, Any]],
    common_ids: set[str],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        "# Object-Aligned External Baseline Comparison",
        "",
        "Object identity and evaluation metrics are aligned across methods, while each external",
        "baseline retains its native/favorable input protocol. This is an object-aligned,",
        "metric-aligned, protocol-specific comparison, not an identical-input comparison.",
        "All segmentation metrics use semantic `original_part_id` labels and the articulated",
        "start-state (source frame 0) convention; point samples remain protocol-specific.",
        "",
        "## Comparison Scope",
        "",
        "This table is a **cross-protocol generalization benchmark**, not a reproduction of the",
        "published AiM or ReArt benchmark. The aligned subset contains substantially more",
        "high-part-count PartNet objects, while the published methods use their own selected",
        "object distributions, motion schedules, temporal lengths, and primary metrics.",
        "Absolute values must not be compared directly with published voxel IoU or native",
        "Sapiens RI tables.",
        "",
        "## Protocols",
        "",
        "- **Ours Hybrid:** PartNet RGB-D interaction recording.",
        "- **AiM:** AiM-style 24-view static start scan plus 120-frame/15 Hz monocular interaction.",
        "- **ReArt:** four point-cloud states using the ReArt Sapiens input format on the aligned",
        "  PartNet object. These are not the anonymous official Sapiens sequence indices.",
        "- The existing shared-3view Ours/AiM experiment remains the equal-input-budget comparison.",
        "",
        "## Requested Aligned Subset",
        "",
        _summary_table(requested),
        "",
        "## Common-Success Subset",
        "",
        f"N = {len(common_ids)}: {', '.join(sorted(common_ids)) or 'none'}",
        "",
        _summary_table(common),
        "",
        "## Complexity",
        "",
        "| GT parts | Method | Success | IoU | ARI | RI | Part count error | Underseg. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complexity:
        lines.append(
            f"| {row['complexity_bucket']} | {row['method']} | "
            f"{row['success_n']}/{row['requested_n']} | {_fmt(row['point_iou'])} | "
            f"{_fmt(row['ari'])} | {_fmt(row['ri'])} | "
            f"{_fmt(row['mean_part_count_error'])} | "
            f"{_fmt(row['undersegmentation_rate'])} |"
        )
    missing_counts = defaultdict(int)
    for row in failures:
        missing_counts[(row["method"], row["status"])] += 1
    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            *[
                f"- {method} / {status}: {count}"
                for (method, status), count in sorted(missing_counts.items())
            ],
            "",
            "Failures and missing runs remain in `aligned_manifest.csv` and `failures.json`; they",
            "are never silently removed from the requested-subset denominator.",
            "",
            "## Limitations",
            "",
            "- Official ReArt Sapiens outputs are indexed by anonymous sequence index in the released",
            "  artifacts. Without a verified PartNet-ID mapping they cannot be used for object alignment.",
            "- The aligned ReArt rows therefore use the same PartNet identities rendered into ReArt's",
            "  four-state input format. They must not be labeled as official Sapiens object instances.",
            "- Method-specific protocols differ in observation density and temporal sampling.",
            "- Source-frame alignment does not make the input observations identical: it only makes",
            "  the semantic segmentation evaluation domain consistent.",
            "- AiM uses 120 interaction frames here rather than the paper's object-dependent",
            "  200/500-frame schedules and explicitly prescribed large joint excursions.",
            "- ReArt uses aligned PartNet objects in the native Sapiens input format, not the",
            "  official Sapiens test distribution used to train/evaluate its correspondence model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Protocol | Requested | Success | Point IoU | ARI | RI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['method']} | {row['protocol']} | {row['requested_n']} | "
        f"{row['success_n']} | {_fmt(row['point_iou'])} | {_fmt(row['ari'])} | "
        f"{_fmt(row['ri'])} |"
        for row in rows
    )
    return "\n".join(lines)


def _read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _field(row: dict[str, Any] | None, key: str) -> Any:
    return row.get(key) if row else None


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value not in (None, ""):
            return int(value)
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.fmean(values) if values else None


def _choose2(value: int) -> int:
    return value * (value - 1) // 2


def _complexity_bucket(value: Any) -> str:
    count = int(value)
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return ">=5"


def _sort_int(value: Any) -> int:
    return int(value) if value not in (None, "") else math.inf


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
