#!/usr/bin/env python3
"""Build a learning manifest from successful PartNet tracking artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks-name", default="part_tracks.json")
    parser.add_argument("--features-name", default="cotracker_features.npz")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.expanduser().resolve().read_text(encoding="utf-8"))
    objects = catalog.get("objects", []) if isinstance(catalog, dict) else catalog
    recording_root = args.recording_root.expanduser().resolve()
    rows = []
    missing = []
    for item in objects:
        object_id = str(item["object_id"])
        artifact_dir = recording_root / object_id / "pointcloud_4d_partseg"
        tracks = artifact_dir / args.tracks_name
        features = artifact_dir / args.features_name
        if not tracks.is_file() or not features.is_file():
            missing.append(object_id)
            continue
        rows.append(
            {
                "object_id": object_id,
                "tracks_path": str(tracks),
                "features_npz": str(features),
                "split": str(item["split"]),
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_id", "tracks_path", "features_npz", "split"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")
    }
    summary = {
        "catalog_objects": len(objects),
        "usable_objects": len(rows),
        "missing_objects": len(missing),
        "split_counts": split_counts,
        "missing_object_ids": missing,
        "manifest": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
