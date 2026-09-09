#!/usr/bin/env python3
"""Split an articulation batch manifest into deterministic round-robin shards."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    lines = [
        line for line in args.manifest.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for shard_index in range(args.shards):
        selected = lines[shard_index::args.shards]
        output = args.output_dir / f"shard_{shard_index:02d}.tsv"
        output.write_text(
            "# category\tmodel_path\tobject_id\tjoint_name\n" + "\n".join(selected) + "\n",
            encoding="utf-8",
        )
        counts.append(len(selected))
    print({"objects": len(lines), "shards": args.shards, "counts": counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
