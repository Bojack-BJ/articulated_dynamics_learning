#!/usr/bin/env python3
"""Align GT track parts to predicted slots and emit an oracle assignment override."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def track_part_id(track: dict) -> int:
    for key in ("part_id", "gt_part_id", "raw_part_id"):
        if key in track:
            return int(track[key])
    raise KeyError(f"Track {track.get('track_id')} has no GT part id")


def build_override(tracks: dict, prediction: dict) -> dict:
    predicted = dict(zip(prediction["track_ids"], prediction["track_slot_assignments"]))
    overlap: dict[int, Counter] = defaultdict(Counter)
    for track in tracks["tracks"]:
        track_id = int(track["track_id"])
        overlap[track_part_id(track)][int(predicted[track_id])] += 1

    part_to_slot: dict[int, int] = {}
    used: set[int] = set()
    for part_id in sorted(overlap, key=lambda value: -sum(overlap[value].values())):
        candidates = sorted(overlap[part_id].items(), key=lambda item: (-item[1], item[0]))
        slot_id = next((slot for slot, _ in candidates if slot not in used), None)
        if slot_id is None:
            raise ValueError("Not enough distinct predicted slots for GT parts")
        part_to_slot[part_id] = slot_id
        used.add(slot_id)
    return {
        "schema_version": 1,
        "source": "gt-part-to-predicted-slot-max-overlap",
        "part_to_slot": {str(key): value for key, value in part_to_slot.items()},
        "track_to_slot": {
            str(int(track["track_id"])): part_to_slot[track_part_id(track)]
            for track in tracks["tracks"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = build_override(
        json.loads(args.tracks.read_text()), json.loads(args.prediction.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "part_to_slot": payload["part_to_slot"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
