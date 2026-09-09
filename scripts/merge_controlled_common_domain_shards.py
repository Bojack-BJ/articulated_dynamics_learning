#!/usr/bin/env python3
"""Merge completed controlled common-domain evaluation shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_root = args.shard_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(
        path for path in shard_root.glob("shard_[0-9][0-9]") if (path / "results").is_dir()
    )
    rows: list[dict[str, Any]] = []
    controlled_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    for shard_dir in shard_dirs:
        results = shard_dir / "results"
        with (results / "per_object_metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = fieldnames or list(reader.fieldnames or [])
            rows.extend(dict(row) for row in reader)
        controlled_rows.extend(
            json.loads((results / "benchmark_metrics.json").read_text(encoding="utf-8"))["per_object"]
        )
        failures.extend(json.loads((results / "failures.json").read_text(encoding="utf-8")))

    if not fieldnames:
        raise ValueError(f"No completed shard metrics found under {shard_root}")
    rows.sort(key=lambda row: (row["object_id"], row["method"], row["domain"]))
    controlled_rows.sort(key=lambda row: (row["object_id"], row["method"]))
    with (output_dir / "per_object_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(
            {
                "schema": "controlled-common-kinematic-domain-v1",
                "domain": "observable",
                "per_object": controlled_rows,
                "summary": {
                    "shard_count": len(shard_dirs),
                    "object_count": len({row["object_id"] for row in controlled_rows}),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "shards": len(shard_dirs),
        "objects": len({row["object_id"] for row in controlled_rows}),
        "controlled_rows": len(controlled_rows),
        "all_domain_rows": len(rows),
        "failures": len(failures),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
