#!/usr/bin/env python3
"""Filter learning manifests to their common objects and verify identical splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = {row["object_id"]: row for row in reader}
        return list(reader.fieldnames), rows


def main() -> int:
    args = parse_args()
    loaded = [read_rows(path) for path in args.manifests]
    common = sorted(set.intersection(*(set(rows) for _, rows in loaded)))
    for object_id in common:
        splits = {rows[object_id]["split"] for _, rows in loaded}
        if len(splits) != 1:
            raise ValueError(f"Split mismatch for {object_id}: {sorted(splits)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, (fieldnames, rows) in zip(args.manifests, loaded, strict=True):
        destination = output_dir / f"{source.stem}_common.tsv"
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows[object_id] for object_id in common)

    counts = {
        split: sum(loaded[0][1][object_id]["split"] == split for object_id in common)
        for split in ("train", "val", "test")
    }
    print(f"Aligned {len(common)} objects: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
