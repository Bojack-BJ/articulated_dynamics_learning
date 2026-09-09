#!/usr/bin/env python3
"""Audit raw-body versus fixed-connected part counts on common slot validation objects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rgbd_urdf_mvp.perception.part_segmentation import collapse_fixed_connected_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return {
            row["object_id"]: row
            for row in csv.DictReader(stream, delimiter="\t")
            if row.get("split") == "val"
        }


def main() -> int:
    args = parse_args()
    manifests = [manifest_rows(path) for path in args.manifests]
    common = sorted(set.intersection(*(set(rows) for rows in manifests)))
    rows = []
    for object_id in common:
        tracks_path = Path(manifests[0][object_id]["tracks_path"]).expanduser().resolve()
        episode_path = tracks_path.parent.parent / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        segmentation = episode.get("metadata", {}).get("part_segmentation", {})
        raw_count = len(segmentation.get("parts", []))
        collapsed = collapse_fixed_connected_parts(segmentation)
        collapsed_count = len(collapsed.get("parts", []))
        rows.append({
            "object_id": object_id,
            "episode_json": str(episode_path),
            "raw_part_count": raw_count,
            "kinematic_part_count": collapsed_count,
            "collapsed_part_count": raw_count - collapsed_count,
            "changed": raw_count != collapsed_count,
        })
    output = {
        "common_validation_object_count": len(common),
        "changed_object_count": sum(row["changed"] for row in rows),
        "raw_part_count_total": sum(row["raw_part_count"] for row in rows),
        "kinematic_part_count_total": sum(row["kinematic_part_count"] for row in rows),
        "objects": rows,
    }
    destination = args.output_json.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "objects"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
