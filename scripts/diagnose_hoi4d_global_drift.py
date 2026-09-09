#!/usr/bin/env python3
"""Diagnose camera/global drift in an HOI4D part-track episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def percentile_range(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return float(np.percentile(distances, 95))


def track_centroids(tracks: list[dict], part_id: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = [[] for _ in range(frame_count)]
    for track in tracks:
        if int(track["part_id"]) != part_id:
            continue
        for sample in track["samples"]:
            xyz = sample.get("xyz_world")
            if sample.get("visible") and isinstance(xyz, list) and len(xyz) == 3:
                values[int(sample["frame_index"])].append(xyz)
    centers = np.full((frame_count, 3), np.nan)
    counts = np.zeros(frame_count, dtype=int)
    for index, points in enumerate(values):
        if points:
            centers[index] = np.median(np.asarray(points), axis=0)
            counts[index] = len(points)
    return centers, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.tracks.read_text())
    episode = json.loads(args.episode.read_text())
    parts = episode["metadata"]["part_segmentation"]["parts"]
    base_ids = [int(part["part_id"]) for part in parts if part["role"] == "base"]
    moving_ids = [int(part["part_id"]) for part in parts if part["role"] == "moving"]
    frame_count = int(artifact["frame_count"])
    centers, counts = {}, {}
    for part_id in base_ids + moving_ids:
        centers[part_id], counts[part_id] = track_centroids(artifact["tracks"], part_id, frame_count)
    base = centers[base_ids[0]]
    base_valid = np.isfinite(base).all(axis=1) & (counts[base_ids[0]] >= 12)
    base_drift = percentile_range(base[base_valid])
    camera_positions = np.asarray([
        np.asarray(frame["camera_pose"])[:3, 3]
        for frame in episode["frames"]
    ])
    sampled = np.asarray(artifact["sampled_frame_indices"], dtype=int)
    camera_positions = camera_positions[np.clip(sampled, 0, len(camera_positions) - 1)]
    camera_path = float(np.linalg.norm(np.diff(camera_positions, axis=0), axis=1).sum())
    relative_ranges = {}
    common_motion_cosines = {}
    for part_id in moving_ids:
        moving = centers[part_id]
        valid = base_valid & np.isfinite(moving).all(axis=1) & (counts[part_id] >= 12)
        relative_ranges[str(part_id)] = percentile_range((moving - base)[valid])
        base_delta = np.diff(base, axis=0)
        moving_delta = np.diff(moving, axis=0)
        pair_valid = valid[:-1] & valid[1:]
        denominator = np.linalg.norm(base_delta[pair_valid], axis=1) * np.linalg.norm(moving_delta[pair_valid], axis=1)
        usable = denominator > 1e-8
        cosine = np.sum(base_delta[pair_valid] * moving_delta[pair_valid], axis=1)[usable] / denominator[usable]
        common_motion_cosines[str(part_id)] = float(np.median(cosine)) if len(cosine) else None
    ratio = base_drift / max(max(relative_ranges.values(), default=0.0), 1e-9)
    result = {
        "sequence": args.tracks.parent.name,
        "base_part_id": base_ids[0],
        "base_world_centroid_drift_p95_m": base_drift,
        "camera_path_length_m": camera_path,
        "moving_relative_motion_p95_m": relative_ranges,
        "base_to_relative_motion_ratio": ratio,
        "base_moving_step_cosine_median": common_motion_cosines,
        "global_drift_failure": bool(base_drift > 0.08 and ratio > 0.5),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
