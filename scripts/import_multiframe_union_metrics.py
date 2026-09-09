#!/usr/bin/env python3
"""Import multi-frame-union segmentation metrics into the baseline suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHOD_DIR = {
    "aim": "aim_aligned",
    "reart": "reart",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=tuple(METHOD_DIR))
    parser.add_argument("union_root", type=Path)
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    union_root = args.union_root.expanduser().resolve()
    suite_root = args.suite_root.expanduser().resolve()
    expected_gt_parts = _load_expected_gt_parts(suite_root / "manifest.csv")
    rows: list[dict[str, Any]] = []
    pattern = f"*/{args.method}_multiframe_union.json"
    for union_path in sorted(union_root.glob(pattern)):
        object_id = union_path.parent.name
        metrics_path = (
            suite_root
            / "per_object"
            / object_id
            / METHOD_DIR[args.method]
            / "metrics.json"
        )
        if not metrics_path.is_file():
            rows.append(
                {
                    "object_id": object_id,
                    "status": "missing_suite_metrics",
                    "metrics_path": str(metrics_path),
                }
            )
            continue
        payload = json.loads(union_path.read_text(encoding="utf-8"))
        target = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_gt_count = expected_gt_parts.get(object_id)
        observed_gt_count = int(payload["gt_part_count"])
        gt_domain_valid = (
            expected_gt_count is not None
            and observed_gt_count == expected_gt_count
        )
        if not gt_domain_valid:
            segmentation = target.setdefault("segmentation", {})
            segmentation.update(
                {
                    "observed_gt_part_count": observed_gt_count,
                    "gt_domain_valid": False,
                    "evaluation_domain": "time_aligned_multiframe_union_incomplete",
                    "evaluation_frame_count": payload["frame_count"],
                    "multiframe_union_artifact": str(union_path),
                }
            )
            target.setdefault("provenance", {})["segmentation_evaluation"] = {
                "artifact": str(union_path),
                "metric": payload["metric"],
                "domain_error": (
                    f"Expected {expected_gt_count} semantic GT parts, "
                    f"observed {observed_gt_count}"
                ),
            }
            if not args.dry_run:
                metrics_path.write_text(
                    json.dumps(target, indent=2) + "\n", encoding="utf-8"
                )
            rows.append(
                {
                    "object_id": object_id,
                    "status": "gt_domain_mismatch",
                    "expected_gt_part_count": expected_gt_count,
                    "observed_gt_part_count": observed_gt_count,
                    "metrics_path": str(metrics_path),
                }
            )
            continue
        target["status"] = "success"
        target["failure"] = None
        primary = payload["primary"]
        covered = primary["covered_only_metrics"]
        if covered is None:
            rows.append(
                {
                    "object_id": object_id,
                    "status": "empty_covered_domain",
                    "metrics_path": str(metrics_path),
                }
            )
            continue
        segmentation = target.setdefault("segmentation", {})
        segmentation.update(
            {
                "point_iou": covered["one_to_one_mean_iou"],
                "ari": covered["adjusted_rand_index"],
                "ri": covered["rand_index"],
                "predicted_part_count": payload["predicted_part_count"],
                "gt_part_count": payload["gt_part_count"],
                "observed_gt_part_count": payload["gt_part_count"],
                "undersegmented": (
                    payload["predicted_part_count"] < payload["gt_part_count"]
                ),
                "largest_cluster_ratio": covered["largest_cluster_ratio"],
                "geometry_coverage": primary["geometry_coverage"],
                "gt_domain_valid": gt_domain_valid,
                "evaluation_domain": "time_aligned_multiframe_union",
                "evaluation_frame_count": payload["frame_count"],
                "evaluation_source_frame_indices": payload.get(
                    "source_frame_indices", []
                ),
                "multiframe_union_artifact": str(union_path),
            }
        )
        support = target.setdefault("metric_support", {})
        support["segmentation"] = "converted_common_multiframe_union_evaluator"
        target.setdefault("provenance", {})["segmentation_evaluation"] = {
            "artifact": str(union_path),
            "metric": payload["metric"],
            "primary_distance_ratio_bbox": payload[
                "primary_distance_ratio_bbox"
            ],
        }
        if not args.dry_run:
            metrics_path.write_text(
                json.dumps(target, indent=2) + "\n", encoding="utf-8"
            )
        rows.append(
            {
                "object_id": object_id,
                "status": "would_update" if args.dry_run else "updated",
                "point_iou": segmentation["point_iou"],
                "ari": segmentation["ari"],
                "ri": segmentation["ri"],
                "gt_part_count": segmentation["gt_part_count"],
                "expected_gt_part_count": expected_gt_count,
                "predicted_part_count": segmentation["predicted_part_count"],
                "frame_count": segmentation["evaluation_frame_count"],
                "metrics_path": str(metrics_path),
            }
        )

    report_path = union_root / f"{args.method}_suite_import.csv"
    _write_csv(report_path, rows)
    print(json.dumps({"rows": len(rows), "report": str(report_path)}, indent=2))
    return 0


def _load_expected_gt_parts(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Suite manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["object_id"]: int(row["gt_part_count"])
            for row in csv.DictReader(handle)
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
