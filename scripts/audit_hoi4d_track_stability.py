#!/usr/bin/env python3
"""Audit temporal coverage and continuity of HOI4D moving-part tracks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def audit(track_path: Path, episode_path: Path) -> dict:
    artifact = json.loads(track_path.read_text())
    episode = json.loads(episode_path.read_text())
    moving_ids = {
        int(part["part_id"])
        for part in episode["metadata"]["part_segmentation"]["parts"]
        if part["role"] == "moving"
    }
    frame_count = int(artifact["frame_count"])
    counts = np.zeros(frame_count, dtype=int)
    moving_tracks = [track for track in artifact["tracks"] if int(track["part_id"]) in moving_ids]
    lifetimes = []
    longest_lifetimes = []
    for track in moving_tracks:
        visible = np.asarray([
            bool(sample.get("visible")) and sample.get("xyz_world") is not None
            for sample in track["samples"]
        ])
        counts[: len(visible)] += visible
        lifetimes.append(int(visible.sum()))
        longest_lifetimes.append(longest_run(visible))
    peak = int(counts.max(initial=0))
    coverage_threshold = max(12, int(np.ceil(0.15 * peak)))
    covered = counts >= coverage_threshold
    longest_covered = longest_run(covered)
    coverage_ratio = float(covered.mean()) if frame_count else 0.0
    stable_segment_ratio = longest_covered / max(frame_count, 1)
    median_lifetime_ratio = float(np.median(lifetimes) / frame_count) if lifetimes else 0.0
    median_contiguous_ratio = float(np.median(longest_lifetimes) / frame_count) if lifetimes else 0.0
    accepted = (
        peak >= 40
        and coverage_ratio >= 0.60
        and stable_segment_ratio >= 0.45
        and median_contiguous_ratio >= 0.20
    )
    return {
        "sequence": track_path.parent.name,
        "moving_part_ids": sorted(moving_ids),
        "moving_track_count": len(moving_tracks),
        "frame_count": frame_count,
        "visible_count_by_frame": counts.tolist(),
        "peak_visible_tracks": peak,
        "coverage_threshold": coverage_threshold,
        "coverage_ratio": coverage_ratio,
        "longest_stable_segment_frames": longest_covered,
        "longest_stable_segment_ratio": stable_segment_ratio,
        "median_track_lifetime_ratio": median_lifetime_ratio,
        "median_contiguous_lifetime_ratio": median_contiguous_ratio,
        "training_status": "accept" if accepted else "reject_or_trim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for track_path in sorted(args.viewer_root.glob("*/part_tracks.json")):
        episode_path = args.episode_root / track_path.parent.name / "episode.json"
        if episode_path.is_file():
            rows.append(audit(track_path, episode_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sequences": rows}, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    fields = [key for key in rows[0] if key != "visible_count_by_frame"] if rows else []
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps({
        "count": len(rows),
        "accept": sum(row["training_status"] == "accept" for row in rows),
        "reject_or_trim": sum(row["training_status"] != "accept" for row in rows),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
