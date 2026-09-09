#!/usr/bin/env python3
"""Audit per-sequence HOI4D mask IDs and object-pose labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--categories", default="C3,C4,C6,C14")
    args = parser.parse_args()
    categories = set(args.categories.split(","))
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] in categories]

    report = []
    annotation_root = args.prepared_root / "annotations" / "HOI4D_annotations"
    for row in rows:
        sequence = annotation_root / row["sequence"]
        mask_counts: Counter[int] = Counter()
        for mask_path in sorted((sequence / "2Dseg/mask").glob("*.png"))[::30]:
            values, counts = np.unique(np.asarray(Image.open(mask_path)), return_counts=True)
            mask_counts.update({int(value): int(count) for value, count in zip(values, counts)})
        pose_labels: dict[int, Counter[str]] = {}
        effective_frames = 0
        for pose_path in sorted((sequence / "objpose").glob("*.json")):
            payload = json.loads(pose_path.read_text())
            if not payload.get("isEffective"):
                continue
            effective_frames += 1
            for item in payload.get("dataList", []):
                part_id = int(item["id"])
                pose_labels.setdefault(part_id, Counter())[str(item.get("label", ""))] += 1
        report.append({
            **row,
            "mask_ids": dict(sorted(mask_counts.items())),
            "pose_labels": {
                str(part_id): counts.most_common() for part_id, counts in sorted(pose_labels.items())
            },
            "effective_pose_frames": effective_frames,
        })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"sequences": report}, indent=2) + "\n")
    print(f"audited={len(report)} output={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
