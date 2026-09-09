#!/usr/bin/env python3
"""Build aligned object tiers and per-method protocol/metric contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.benchmarks.baseline_alignment import (
    CORE16_OBJECT_IDS,
    CORE_OBJECT_IDS,
    METHOD_CONTRACTS,
    build_aligned_rows,
)
from rgbd_urdf_mvp.benchmarks.partnet_urdf import (
    load_partnet_joint_metadata,
    resolve_partnet_urdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--partnet-assets-root",
        type=Path,
        help="Optional root containing <object-id>/mobility.urdf assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = list(csv.DictReader(args.object_manifest.open(encoding="utf-8")))
    source = _attach_urdf_joint_metadata(source, args.partnet_assets_root)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = build_aligned_rows(source)
    _write_csv(output / "method_object_manifest.csv", rows)
    _write_csv(
        output / "core12_objects.csv",
        _object_rows(source, set(CORE_OBJECT_IDS)),
    )
    _write_csv(
        output / "core16_objects.csv",
        _object_rows(source, set(CORE16_OBJECT_IDS)),
    )
    _write_csv(output / "full20_objects.csv", _object_rows(source, None))
    _write_csv(
        output / "two_part_objects.csv",
        [
            row
            for row in _object_rows(source, None)
            if int(row["gt_part_count"]) == 2
            and row["gt_joint_count"] not in (None, "")
            and int(row["gt_joint_count"]) == 1
        ],
    )
    contracts = {
        method: {
            "protocol": contract.protocol,
            "scope": contract.scope,
            "oracle_inputs": list(contract.oracle_inputs),
            "predicted_metrics": list(contract.predicted_metrics),
            "conditional_metrics": list(contract.conditional_metrics),
            "notes": contract.notes,
        }
        for method, contract in METHOD_CONTRACTS.items()
    }
    (output / "method_contracts.json").write_text(
        json.dumps(contracts, indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(output / "README.md", source)
    return 0


def _object_rows(
    source: list[dict[str, str]], selected_ids: set[str] | None
) -> list[dict[str, Any]]:
    rows = []
    for row in source:
        if selected_ids is not None and row["object_id"] not in selected_ids:
            continue
        gt_parts = int(row["gt_part_count"])
        raw_joint_count = row.get("gt_joint_count")
        gt_joints = (
            int(raw_joint_count) if raw_joint_count not in (None, "") else None
        )
        rows.append(
            {
                "object_id": row["object_id"],
                "category": row["category"],
                "gt_part_count": gt_parts,
                "gt_joint_count": "" if gt_joints is None else gt_joints,
                "gt_joint_count_source": (
                    row.get("gt_joint_count_source")
                    or ("manifest_gt" if gt_joints is not None else "unknown")
                ),
                "complexity_bucket": (
                    "2" if gt_parts == 2 else "3-4" if gt_parts <= 4 else ">=5"
                ),
            }
        )
    return sorted(rows, key=lambda item: (item["gt_part_count"], item["category"]))


def _attach_urdf_joint_metadata(
    source: list[dict[str, str]], assets_root: Path | None
) -> list[dict[str, str]]:
    if assets_root is None:
        return source
    enriched: list[dict[str, str]] = []
    for original in source:
        row = dict(original)
        urdf_path = resolve_partnet_urdf(assets_root, row["object_id"])
        if urdf_path is not None:
            metadata = load_partnet_joint_metadata(urdf_path)
            row["gt_joint_count"] = str(metadata.joint_count)
            row["gt_joint_count_source"] = "partnet_mobility_urdf"
            row["gt_joint_types"] = ",".join(metadata.joint_types)
            row["gt_joint_names"] = ",".join(metadata.joint_names)
            row["gt_urdf_path"] = str(urdf_path.resolve())
        enriched.append(row)
    return enriched


def _write_readme(path: Path, source: list[dict[str, str]]) -> None:
    categories = sorted({row["category"] for row in source if row["object_id"] in CORE_OBJECT_IDS})
    lines = [
        "# Aligned External Baseline Objects",
        "",
        "This directory separates object alignment from method-specific acquisition.",
        "Object identity and metrics are aligned; each method keeps its declared native",
        "or favorable protocol.",
        "",
        "## Tiers",
        "",
        f"- `core12`: 12 objects, four per complexity bucket; categories: {', '.join(categories)}.",
        "- `core16`: frozen stratified paper subset with 5/5/6 objects in the 2/3-4/>=5 buckets.",
        "- `full20`: the complete existing aligned PartNet pool.",
        "- `two_part`: one-joint objects used for PARIS and Ditto.",
        "- Restricted methods are dispatched only when the joint count is verified from the manifest or PartNet URDF.",
        "",
        "The core tier is the first execution target. Full-20 is run only after each",
        "adapter passes its core-tier validation. PARIS and Ditto are never extrapolated",
        "to multi-part objects.",
        "",
        "## Metric policy",
        "",
        "- Segmentation uses Point IoU, ARI, RI, predicted/GT part count, and under-segmentation.",
        "- Axis error is undirected and is reported only for type-correct matched joints.",
        "- Revolute axis-line distance is normalized by object bbox diagonal.",
        "- Unsupported outputs stay absent.",
        "- Oracle-conditioned metrics are reported separately from non-oracle results.",
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


if __name__ == "__main__":
    raise SystemExit(main())
