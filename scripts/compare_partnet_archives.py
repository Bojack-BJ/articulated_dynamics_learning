#!/usr/bin/env python3
"""Compare SAPIEN PartNet-Mobility and GAPartNet ZIP inventories."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OBJECT_PATH = re.compile(r"^[^/]+/(\d+)/(.*)$")
CORE_FILES = ("meta.json", "mobility.urdf", "semantics.txt")


def _inventory(archive: Path) -> tuple[zipfile.ZipFile, dict[str, dict[str, zipfile.ZipInfo]]]:
    handle = zipfile.ZipFile(archive)
    objects: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    for info in handle.infolist():
        match = OBJECT_PATH.match(info.filename)
        if match is None or not match.group(2) or info.is_dir():
            continue
        objects[match.group(1)][match.group(2)] = info
    return handle, dict(objects)


def _categories(
    handle: zipfile.ZipFile,
    objects: dict[str, dict[str, zipfile.ZipInfo]],
) -> tuple[dict[str, str], Counter[str]]:
    by_object: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for object_id, entries in objects.items():
        meta = entries.get("meta.json")
        if meta is None:
            continue
        payload = json.loads(handle.read(meta))
        category = str(payload.get("model_cat", "unknown"))
        by_object[object_id] = category
        counts[category] += 1
    return by_object, counts


def compare(sapien_archive: Path, gapartnet_archive: Path) -> dict[str, Any]:
    sapien_zip, sapien = _inventory(sapien_archive)
    gap_zip, gap = _inventory(gapartnet_archive)
    try:
        sapien_ids = set(sapien)
        gap_ids = set(gap)
        common_ids = sapien_ids & gap_ids
        sapien_categories, sapien_category_counts = _categories(sapien_zip, sapien)
        gap_categories, gap_category_counts = _categories(gap_zip, gap)

        core_stats: dict[str, dict[str, int]] = {}
        for relative_path in CORE_FILES:
            present_both = exact_matches = 0
            mismatch_ids: list[str] = []
            missing_from_sapien: list[str] = []
            missing_from_gapartnet: list[str] = []
            for object_id in common_ids:
                left = sapien[object_id].get(relative_path)
                right = gap[object_id].get(relative_path)
                if left is None:
                    missing_from_sapien.append(object_id)
                if right is None:
                    missing_from_gapartnet.append(object_id)
                if left is None or right is None:
                    continue
                present_both += 1
                exact = left.CRC == right.CRC and left.file_size == right.file_size
                exact_matches += int(exact)
                if not exact:
                    mismatch_ids.append(object_id)
            core_stats[relative_path] = {
                "present_in_both": present_both,
                "byte_exact_matches": exact_matches,
                "mismatch_ids": sorted(mismatch_ids, key=int),
                "missing_from_sapien_ids": sorted(missing_from_sapien, key=int),
                "missing_from_gapartnet_ids": sorted(missing_from_gapartnet, key=int),
            }

        shared_paths = exact_shared_files = 0
        per_object_exact_core = 0
        for object_id in common_ids:
            common_paths = set(sapien[object_id]) & set(gap[object_id])
            shared_paths += len(common_paths)
            exact_shared_files += sum(
                sapien[object_id][path].CRC == gap[object_id][path].CRC
                and sapien[object_id][path].file_size == gap[object_id][path].file_size
                for path in common_paths
            )
            per_object_exact_core += int(
                all(
                    path in sapien[object_id]
                    and path in gap[object_id]
                    and sapien[object_id][path].CRC == gap[object_id][path].CRC
                    and sapien[object_id][path].file_size == gap[object_id][path].file_size
                    for path in CORE_FILES
                )
            )

        category_agreement = sum(
            sapien_categories.get(object_id) == gap_categories.get(object_id)
            for object_id in common_ids
            if object_id in sapien_categories and object_id in gap_categories
        )
        category_comparable = sum(
            object_id in sapien_categories and object_id in gap_categories for object_id in common_ids
        )

        return {
            "sapien_archive": str(sapien_archive.resolve()),
            "gapartnet_archive": str(gapartnet_archive.resolve()),
            "sapien_object_count": len(sapien_ids),
            "gapartnet_object_count": len(gap_ids),
            "common_object_count": len(common_ids),
            "sapien_only_count": len(sapien_ids - gap_ids),
            "gapartnet_only_count": len(gap_ids - sapien_ids),
            "gapartnet_overlap_fraction": len(common_ids) / max(len(gap_ids), 1),
            "sapien_overlap_fraction": len(common_ids) / max(len(sapien_ids), 1),
            "category_comparable_count": category_comparable,
            "category_agreement_count": category_agreement,
            "core_file_comparison": core_stats,
            "objects_with_all_core_files_byte_exact": per_object_exact_core,
            "shared_relative_file_count": shared_paths,
            "byte_exact_shared_file_count": exact_shared_files,
            "sapien_category_counts": dict(sorted(sapien_category_counts.items())),
            "gapartnet_category_counts": dict(sorted(gap_category_counts.items())),
            "sapien_only_ids_sample": sorted(sapien_ids - gap_ids, key=int)[:25],
            "gapartnet_only_ids_sample": sorted(gap_ids - sapien_ids, key=int)[:25],
        }
    finally:
        sapien_zip.close()
        gap_zip.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sapien_archive", type=Path)
    parser.add_argument("gapartnet_archive", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = compare(args.sapien_archive, args.gapartnet_archive)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
