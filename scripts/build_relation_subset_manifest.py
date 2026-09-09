#!/usr/bin/env python3
"""Build an evaluation-only relation manifest for an explicit object subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("object_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--force-split", default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.object_manifest.open(newline="", encoding="utf-8-sig") as stream:
        object_rows = list(csv.DictReader(stream))
    object_ids = [row["object_id"].strip() for row in object_rows]
    requested = set(object_ids)

    with args.source_manifest.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    source_by_id = {row["object_id"]: row for row in source_rows}
    missing = sorted(requested - source_by_id.keys())
    if missing:
        raise SystemExit(f"Missing objects in source manifest: {missing}")

    rows = []
    for object_id in object_ids:
        row = dict(source_by_id[object_id])
        row["original_split"] = row["split"]
        row["split"] = args.force_split
        rows.append(row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(source_rows[0]) + ["original_split"]
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) != len(object_ids) or {row["object_id"] for row in rows} != requested:
        raise RuntimeError("Subset manifest validation failed")
    print(f"wrote {len(rows)} objects to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
