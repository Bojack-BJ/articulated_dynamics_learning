#!/usr/bin/env python3
"""Build object-balanced real-only or fixed-ratio PartNet/real manifests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partnet_manifest", type=Path)
    parser.add_argument("arti4d_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--train", action="append", default=[])
    parser.add_argument("--val", action="append", default=[])
    parser.add_argument("--real-repeats", type=int, default=20)
    parser.add_argument("--partnet-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.train or not args.val:
        raise ValueError("At least one --train and --val real object are required")
    if not 0.0 <= args.partnet_fraction < 1.0:
        raise ValueError("--partnet-fraction must be in [0, 1)")

    base_rows = read_rows(args.partnet_manifest)
    fieldnames = list(base_rows[0])
    if "relation_gt_path" not in fieldnames:
        fieldnames.append("relation_gt_path")

    real_rows: list[dict[str, str]] = []
    for split, names, repeats in (
        ("train", args.train, max(1, args.real_repeats)),
        ("val", args.val, 1),
    ):
        for name in names:
            root = args.arti4d_root / name
            values = {
                "tracks_path": root / "track_quality/motion_part_tracks_with_quality.json",
                "features_npz": root / "tracking/cotracker_features.npz",
                "relation_gt_path": root / "relation_gt.json",
            }
            missing = [str(path) for path in values.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing assets for {name}: {missing}")
            for replica in range(repeats):
                real_rows.append({
                    "object_id": f"arti4d_{name}_r{replica:03d}",
                    "tracks_path": str(values["tracks_path"].resolve()),
                    "features_npz": str(values["features_npz"].resolve()),
                    "split": split,
                    "relation_gt_path": str(values["relation_gt_path"].resolve()),
                })

    real_train_count = sum(row["split"] == "train" for row in real_rows)
    partnet_count = round(
        real_train_count * args.partnet_fraction / max(1e-9, 1.0 - args.partnet_fraction)
    )
    candidates = [row for row in base_rows if row.get("split") == "train"]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if partnet_count > len(candidates):
        raise ValueError(f"Requested {partnet_count} PartNet rows, only {len(candidates)} available")
    selected = [
        {**row, "relation_gt_path": row.get("relation_gt_path", "")}
        for row in candidates[:partnet_count]
    ]

    rows = selected + real_rows
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} rows: partnet_train={len(selected)}, "
        f"real_train={real_train_count}, real_val={len(args.val)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
