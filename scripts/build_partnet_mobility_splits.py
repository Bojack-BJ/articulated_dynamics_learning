#!/usr/bin/env python3
"""Build category and articulation-stratified PartNet-Mobility splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CORE_CATEGORIES = (
    "CoffeeMachine",
    "Dishwasher",
    "Door",
    "Laptop",
    "Microwave",
    "Oven",
    "Refrigerator",
    "Safe",
    "StorageFurniture",
    "Suitcase",
    "Table",
    "Toilet",
    "TrashCan",
    "WashingMachine",
)


@dataclass(frozen=True)
class AssetRecord:
    object_id: str
    category: str
    movable_joint_count: int
    revolute_joint_count: int
    prismatic_joint_count: int
    continuous_joint_count: int
    articulation_profile: str
    split: str = ""


def _profile(revolute: int, prismatic: int, continuous: int) -> str:
    rotary = revolute + continuous
    total = rotary + prismatic
    if total == 0:
        return "fixed"
    if total == 1 and rotary == 1:
        return "single_revolute"
    if total == 1 and prismatic == 1:
        return "single_prismatic"
    if prismatic == 0:
        return "multi_revolute"
    if rotary == 0:
        return "multi_prismatic"
    return "mixed"


def load_catalog(archive: Path) -> list[AssetRecord]:
    records: list[AssetRecord] = []
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        object_ids = sorted(
            {
                fields[1]
                for name in names
                if len((fields := name.split("/"))) >= 3 and fields[0] == "dataset" and fields[1].isdigit()
            },
            key=int,
        )
        for object_id in object_ids:
            meta_name = f"dataset/{object_id}/meta.json"
            urdf_name = f"dataset/{object_id}/mobility.urdf"
            if meta_name not in names or urdf_name not in names:
                continue
            category = str(json.loads(handle.read(meta_name)).get("model_cat", "unknown"))
            root = ET.fromstring(handle.read(urdf_name))
            types = [str(node.attrib.get("type", "fixed")).lower() for node in root.findall("joint")]
            revolute = sum(kind == "revolute" for kind in types)
            prismatic = sum(kind == "prismatic" for kind in types)
            continuous = sum(kind == "continuous" for kind in types)
            records.append(
                AssetRecord(
                    object_id=object_id,
                    category=category,
                    movable_joint_count=revolute + prismatic + continuous,
                    revolute_joint_count=revolute,
                    prismatic_joint_count=prismatic,
                    continuous_joint_count=continuous,
                    articulation_profile=_profile(revolute, prismatic, continuous),
                )
            )
    return records


def _seed_for(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_splits(
    records: list[AssetRecord],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> list[AssetRecord]:
    groups: dict[tuple[str, str], list[AssetRecord]] = defaultdict(list)
    for record in records:
        groups[(record.category, record.articulation_profile)].append(record)

    output: list[AssetRecord] = []
    for key, rows in sorted(groups.items()):
        rng = random.Random(_seed_for(seed, ":".join(key)))
        rows = sorted(rows, key=lambda item: int(item.object_id))
        rng.shuffle(rows)
        count = len(rows)
        val_count = int(round(count * val_fraction)) if count >= 5 else 0
        test_count = int(round(count * (1.0 - train_fraction - val_fraction))) if count >= 5 else 0
        if count >= 10:
            val_count = max(1, val_count)
            test_count = max(1, test_count)
        if val_count + test_count >= count:
            test_count = max(0, count - val_count - 1)
        train_count = count - val_count - test_count
        splits = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        output.extend(AssetRecord(**{**asdict(row), "split": split}) for row, split in zip(rows, splits))
    return sorted(output, key=lambda item: (item.category, int(item.object_id)))


def summarize(records: list[AssetRecord]) -> dict[str, Any]:
    return {
        "object_count": len(records),
        "category_count": len({row.category for row in records}),
        "split_counts": dict(sorted(Counter(row.split for row in records).items())),
        "category_counts": dict(sorted(Counter(row.category for row in records).items())),
        "articulation_profile_counts": dict(
            sorted(Counter(row.articulation_profile for row in records).items())
        ),
        "category_profile_counts": {
            category: dict(sorted(Counter(row.articulation_profile for row in records if row.category == category).items()))
            for category in sorted({row.category for row in records})
        },
    }


def write_outputs(records: list[AssetRecord], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(AssetRecord.__dataclass_fields__)
    with (output_dir / "catalog.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    for split in ("train", "val", "test"):
        with (output_dir / f"{split}.txt").open("w", encoding="utf-8") as stream:
            stream.writelines(f"{row.object_id}\n" for row in records if row.split == split)
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["core", "all"], default="core")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.train_fraction <= 0 or args.val_fraction < 0 or args.train_fraction + args.val_fraction >= 1:
        parser.error("fractions must leave positive train and test partitions")

    all_records = load_catalog(args.archive)
    categories = set(args.categories or (CORE_CATEGORIES if args.preset == "core" else []))
    selected = [row for row in all_records if not categories or row.category in categories]
    assigned = assign_splits(
        selected,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    metadata = {
        "source_archive": str(args.archive.resolve()),
        "preset": args.preset,
        "selected_categories": sorted(categories) if categories else "all",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        **summarize(assigned),
    }
    write_outputs(assigned, args.output_dir, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
