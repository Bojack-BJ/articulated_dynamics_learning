#!/usr/bin/env python3
"""Export Track2Art world-space tracks for an ArtiPoint backend diagnostic.

This adapter deliberately discards Track2Art part assignments.  It exports only
world-space XYZ trajectories and visibility, allowing ArtiPoint to run its own
motion filtering, reliability filtering, DBSCAN, and screw-motion fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_track2art_tracks(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tracks") or []
    if not rows:
        raise ValueError(f"No tracks found in {path}")

    sampled = [int(value) for value in payload.get("sampled_frame_indices") or []]
    frame_count = int(payload.get("frame_count") or len(rows[0].get("samples") or []))
    if not sampled:
        sampled = list(range(frame_count))
    if len(sampled) != frame_count:
        raise ValueError(
            f"sampled_frame_indices has {len(sampled)} entries, expected {frame_count}"
        )

    tracks = np.zeros((frame_count, len(rows), 3), dtype=np.float64)
    visibility = np.zeros((frame_count, len(rows)), dtype=bool)
    source_to_frame = {source: frame for frame, source in enumerate(sampled)}

    for track_index, row in enumerate(rows):
        for sample in row.get("samples") or []:
            frame = sample.get("frame_index")
            source = sample.get("source_frame_index")
            if frame is None and source is not None:
                frame = source_to_frame.get(int(source))
            if frame is None or not 0 <= int(frame) < frame_count:
                continue
            xyz = sample.get("xyz_world")
            valid = bool(sample.get("visible", False)) and xyz is not None
            if not valid:
                continue
            value = np.asarray(xyz, dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                continue
            tracks[int(frame), track_index] = value
            visibility[int(frame), track_index] = True

    return tracks, visibility, sampled


def write_artipoint_tracks(
    tracks_json: Path,
    output_npz: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    tracks, visibility, sampled = load_track2art_tracks(tracks_json)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    track_segments = np.empty(1, dtype=object)
    visibility_segments = np.empty(1, dtype=object)
    track_segments[0] = tracks
    visibility_segments[0] = visibility
    np.savez(output_npz, tracks=track_segments, visibility=visibility_segments)
    visible_counts = visibility.sum(axis=0)
    manifest = {
        "protocol": "track2art_tracks_to_artipoint_backend_diagnostic",
        "source_tracks": str(tracks_json.resolve()),
        "output_tracks_world": str(output_npz.resolve()),
        "passes_track2art_part_assignments": False,
        "passes_gt_part_or_joint_information": False,
        "artipoint_stages_to_run": [
            "static_and_jerky_filter",
            "reliability_filter",
            "largest_dbscan_cluster",
            "track_smoothing",
            "screw_motion_fitting",
        ],
        "frame_count": int(tracks.shape[0]),
        "track_count": int(tracks.shape[1]),
        "sampled_source_frame_indices": sampled,
        "visible_samples": int(visibility.sum()),
        "median_visible_frames_per_track": float(np.median(visible_counts)),
        "max_visible_frames_per_track": int(visible_counts.max(initial=0)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks_json", type=Path)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_artipoint_tracks(
        args.tracks_json.expanduser().resolve(),
        args.output_npz.expanduser().resolve(),
        args.manifest.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
