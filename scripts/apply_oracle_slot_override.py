#!/usr/bin/env python3
"""Apply a track-to-slot oracle override to a slot visualization artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slot_artifact", type=Path)
    parser.add_argument("override_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.slot_artifact.read_text(encoding="utf-8"))
    override = json.loads(args.override_json.read_text(encoding="utf-8"))
    track_to_slot = {str(key): int(value) for key, value in override["track_to_slot"].items()}
    missing = []
    counts: dict[int, int] = {}
    for track in artifact.get("tracks", []):
        track_id = str(track.get("track_id"))
        if track_id not in track_to_slot:
            missing.append(track_id)
            continue
        slot_id = track_to_slot[track_id]
        track["part_id"] = slot_id
        track["part_name"] = f"oracle_slot_{slot_id}"
        track["slot_initial_id"] = slot_id
        track["slot_confidence"] = 1.0
        counts[slot_id] = counts.get(slot_id, 0) + 1
    artifact["part_track_counts"] = {str(key): value for key, value in sorted(counts.items())}
    artifact["motion_segmentation"] = {
        **artifact.get("motion_segmentation", {}),
        "source": "oracle-track-assignment-override",
        "override_json": str(args.override_json.resolve()),
        "missing_track_ids": missing,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(artifact.get('tracks', [])) - len(missing)} tracks to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
