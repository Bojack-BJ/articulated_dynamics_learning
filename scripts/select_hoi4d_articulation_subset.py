#!/usr/bin/env python3
"""Select articulation-focused HOI4D sequences from the official release list."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "C3": "Laptop",
    "C4": "Storage Furniture",
    "C6": "Safe",
    "C9": "Scissors",
    "C11": "Pliers",
    "C14": "Trash Can",
    "C17": "Lamp",
    "C18": "Stapler",
}

# Favor large, clearly observable revolute/prismatic motions for the first pass.
PROFILES = {
    "core": {("C3", "T2"), ("C4", "T1"), ("C4", "T2"), ("C6", "T1"), ("C14", "T1")},
    "extended": {
        ("C3", "T2"), ("C4", "T1"), ("C4", "T2"), ("C6", "T1"),
        ("C14", "T1"), ("C17", "T2"), ("C9", "T2"), ("C11", "T4"),
        ("C18", "T2"),
    },
}


def select_sequences(release_path: Path, profile: str) -> list[dict[str, str]]:
    selected = []
    for sequence in release_path.read_text(encoding="utf-8").splitlines():
        sequence = sequence.strip()
        if not sequence:
            continue
        fields = sequence.split("/")
        if len(fields) != 7:
            raise ValueError(f"Unexpected HOI4D sequence key: {sequence}")
        camera, human, category, instance, scene, layout, task = fields
        if (category, task) not in PROFILES[profile]:
            continue
        selected.append({
            "sequence": sequence,
            "camera": camera,
            "human": human,
            "category_id": category,
            "category": CATEGORIES[category],
            "instance": instance,
            "scene": scene,
            "layout": layout,
            "task": task,
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path, help="Official HOI4D-Instructions/release.txt")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = select_sequences(args.release, args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"hoi4d_{args.profile}_sequences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sequence"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "profile": args.profile,
        "sequence_count": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "task_counts": dict(sorted(Counter(f'{row["category_id"]}/{row["task"]}' for row in rows).items())),
        "manifest": str(csv_path.resolve()),
    }
    (args.output_dir / f"hoi4d_{args.profile}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
