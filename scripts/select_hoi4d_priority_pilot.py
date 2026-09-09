#!/usr/bin/env python3
"""Select a small, instance-disjoint HOI4D articulation training pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def score(seed: int, sequence: str) -> str:
    return hashlib.sha256(f"{seed}:{sequence}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--train-per-motion", type=int, default=2)
    parser.add_argument("--val-per-motion", type=int, default=1)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "usable_pose_gt":
                groups[row["motion"]].append(row)

    selected = []
    for motion, rows in sorted(groups.items()):
        rows.sort(key=lambda row: score(args.seed, row["sequence"]))
        needed = args.train_per_motion + args.val_per_motion
        chosen = []
        used_instances = set()
        for row in rows:
            identity = (row["category"], row["instance"])
            if identity in used_instances:
                continue
            chosen.append(row)
            used_instances.add(identity)
            if len(chosen) == needed:
                break
        if len(chosen) < needed:
            for row in rows:
                if row not in chosen:
                    chosen.append(row)
                if len(chosen) == needed:
                    break
        for index, row in enumerate(chosen):
            sequence = row["sequence"]
            archive = sequence.replace("/", "_") + ".tar.gz"
            selected.append({
                "split": "train" if index < args.train_per_motion else "val",
                "sequence": sequence,
                "archive": archive,
                "category": row["category"],
                "instance": row["instance"],
                "motion": motion,
                "effective_pose_ratio": row["effective_pose_ratio"],
                "mask_frames": row["mask_frames"],
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"selected={len(selected)} train={sum(r['split'] == 'train' for r in selected)} "
          f"val={sum(r['split'] == 'val' for r in selected)} output={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
