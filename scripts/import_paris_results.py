#!/usr/bin/env python3
"""Import completed PARIS native outputs without fabricating segmentation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_outputs import adapt_paris_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        row["object_id"]: row
        for row in csv.DictReader(args.manifest.open(encoding="utf-8"))
        if int(row["gt_part_count"]) == 2
    }
    imported = []
    missing = []
    for object_id, row in manifest.items():
        output = args.suite_root / "per_object" / object_id / "paris"
        native_metrics = output / "native_metrics.json"
        try:
            if native_metrics.exists():
                payload = json.loads(native_metrics.read_text(encoding="utf-8"))
            else:
                payload = adapt_paris_output(output)
        except (FileNotFoundError, ValueError):
            missing.append(object_id)
            continue
        payload.update(
            {
                "schema": "external-baseline-result-v1",
                "method": "paris",
                "object_id": object_id,
                "category": row["category"],
                "protocol": "two_state_multiview_rgb_two_part_only",
                "applicable": True,
                "oracle_requirements": ["two_part_method_scope"],
            }
        )
        payload.setdefault("segmentation", {})
        payload["segmentation"]["gt_part_count"] = int(row["gt_part_count"])
        (output / "metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        imported.append(object_id)

    summary = {
        "requested_n": len(manifest),
        "imported_n": len(imported),
        "missing_n": len(missing),
        "imported_objects": sorted(imported),
        "missing_objects": sorted(missing),
        "note": (
            "PARIS is restricted to its native one-static/one-moving scope. "
            "Native motion and novel-view metrics are imported; common point "
            "segmentation remains unsupported until component geometry is exported."
        ),
    }
    (args.suite_root / "paris_import_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
