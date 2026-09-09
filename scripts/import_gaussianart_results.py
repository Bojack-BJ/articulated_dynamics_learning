#!/usr/bin/env python3
"""Import GaussianArt's per-object native results into the baseline suite."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_outputs import (
    adapt_gaussianart_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument(
        "--native-csv",
        type=Path,
        help="Import an already-synchronized native metric CSV instead of results.txt files.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    imported = []
    missing = []
    metric_rows = []
    csv_metrics = _load_native_csv(args.native_csv)
    for row in rows:
        object_id = row["object_id"]
        source = args.native_root / object_id / "results.txt"
        csv_row = csv_metrics.get(object_id)
        if not source.exists() and csv_row is None:
            missing.append(object_id)
            continue
        output = args.suite_root / "per_object" / object_id / "gaussianart"
        native = output / "native"
        native.mkdir(parents=True, exist_ok=True)
        if source.exists():
            destination = native / "results.txt"
            shutil.copy2(source, destination)
            payload = adapt_gaussianart_output(native)
        else:
            payload = _payload_from_csv(csv_row)
        payload.update(
            {
                "schema": "external-baseline-result-v1",
                "method": "gaussianart",
                "object_id": object_id,
                "category": row["category"],
                "protocol": "two_state_multiview_oracle_parts",
                "applicable": True,
                "oracle_requirements": [
                    "part_count",
                    "part_semantic_initialization",
                    "gt_motion_metadata",
                ],
            }
        )
        payload["segmentation"]["gt_part_count"] = int(row["gt_part_count"])
        payload["segmentation"]["part_count_is_oracle"] = True
        (output / "metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        imported.append(object_id)
        metric_rows.append(
            {
                "object_id": object_id,
                "category": row["category"],
                "gt_part_count": int(row["gt_part_count"]),
                "selected_checkpoint": payload["artifacts"]["selected_checkpoint"],
                "axis_angle_error_deg_native_all": payload["kinematics"][
                    "axis_angle_error_deg_native_all"
                ],
                "axis_position_error_native_x10": payload["kinematics"][
                    "axis_position_error_native_x10"
                ],
                "motion_error_native": payload["kinematics"]["motion_error_native"],
            }
        )

    _write_csv(args.suite_root / "gaussianart_native_per_object.csv", metric_rows)
    aggregate = {
        field: _distribution(metric_rows, field)
        for field in (
            "axis_angle_error_deg_native_all",
            "axis_position_error_native_x10",
            "motion_error_native",
        )
    }
    summary = {
        "requested_n": len(rows),
        "imported_n": len(imported),
        "missing_n": len(missing),
        "imported_objects": imported,
        "missing_objects": missing,
        "aggregate": aggregate,
        "comparison_note": (
            "GaussianArt uses oracle part count, semantic initialization, and GT "
            "motion metadata. Native axis metrics are not the common evaluator's "
            "type-correct/bbox-normalized metrics."
        ),
    }
    (args.suite_root / "gaussianart_import_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not missing else 1


def _load_native_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {
        row["object_id"]: row
        for row in csv.DictReader(path.open(encoding="utf-8"))
    }


def _payload_from_csv(row: dict[str, str]) -> dict[str, object]:
    return {
        "status": "success_native_metrics",
        "oracle": {
            "used_for_inference": True,
            "inputs": [
                "part_count",
                "part_semantic_initialization",
                "gt_motion_metadata",
            ],
        },
        "segmentation": {
            "predicted_part_count": int(row["gt_part_count"]),
        },
        "kinematics": {
            "axis_angle_error_deg": float(
                row["axis_angle_error_deg_native_all"]
            ),
            "axis_angle_error_deg_native_all": float(
                row["axis_angle_error_deg_native_all"]
            ),
            "axis_position_error": float(
                row["axis_position_error_native_x10"]
            ),
            "axis_position_error_native_x10": float(
                row["axis_position_error_native_x10"]
            ),
            "motion_error": float(row["motion_error_native"]),
            "motion_error_native": float(row["motion_error_native"]),
        },
        "geometry": {},
        "metric_support": {
            "segmentation": "oracle_part_count_only",
            "kinematics": (
                "native_oracle_aggregate; no type-correct conditioning; "
                "position distance is evaluator-native x10"
            ),
            "geometry": "unsupported_without_native_mesh_adapter",
        },
        "artifacts": {
            "selected_checkpoint": int(row["selected_checkpoint"]),
            "source": "synchronized_native_metric_csv",
        },
    }


def _distribution(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
