#!/usr/bin/env python3
"""Merge manually labeled Arti4D relations into a PartNet training manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partnet_manifest", type=Path)
    parser.add_argument("arti4d_output_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--train", action="append", default=[])
    parser.add_argument("--val", action="append", default=[])
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-partnet", action="store_true")
    args = parser.parse_args()

    with args.partnet_manifest.open(newline="") as stream:
        base_rows = list(csv.DictReader(stream, delimiter="\t"))
    fieldnames = list(base_rows[0])
    if "relation_gt_path" not in fieldnames:
        fieldnames.append("relation_gt_path")

    rows = [] if args.no_partnet else [
        {**row, "relation_gt_path": row.get("relation_gt_path", "")} for row in base_rows
    ]
    for split, names, repeats in (
        ("train", args.train, max(1, args.repeat)),
        ("val", args.val, 1),
        ("test", args.test, 1),
    ):
        for name in names:
            root = args.arti4d_output_root / name
            tracks = root / "track_quality/motion_part_tracks_with_quality.json"
            features = root / "tracking/cotracker_features.npz"
            gt = root / "relation_gt.json"
            missing = [path for path in (tracks, features, gt) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing Arti4D relation assets for {name}: {missing}")
            for replica in range(repeats):
                rows.append({
                    "object_id": f"arti4d_{name}_r{replica:02d}",
                    "tracks_path": str(tracks.resolve()),
                    "features_npz": str(features.resolve()),
                    "split": split,
                    "relation_gt_path": str(gt.resolve()),
                })

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
