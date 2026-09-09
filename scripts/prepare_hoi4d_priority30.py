#!/usr/bin/env python3
"""Prepare selected HOI4D archives and annotations without expanding the full release."""

from __future__ import annotations

import argparse
import csv
import tarfile
import zipfile
from pathlib import Path


def sequence_path(sequence: str) -> str:
    return sequence.strip("/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("annotation_zip", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.output_root.mkdir(parents=True, exist_ok=True)

    extracted_raw = 0
    for row in rows:
        destination = args.output_root / "raw" / row["archive"].removesuffix(".tar.gz")
        marker = destination / ".prepared"
        if not marker.is_file():
            archive = args.archive_root / row["archive"]
            with tarfile.open(archive, "r:gz") as stream:
                stream.extractall(args.output_root / "raw", filter="data")
            marker.touch()
        extracted_raw += 1

    prefixes = tuple(
        f"HOI4D_annotations/{sequence_path(row['sequence'])}/" for row in rows
    )
    annotation_root = args.output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    extracted_annotations = 0
    with zipfile.ZipFile(args.annotation_zip) as stream:
        for member in stream.infolist():
            if member.filename.startswith(prefixes):
                stream.extract(member, annotation_root)
                extracted_annotations += 1

    print(
        f"sequences={len(rows)} raw={extracted_raw} "
        f"annotation_files={extracted_annotations} output={args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
