#!/usr/bin/env python3
"""Audit masks and lifted tracks before using real recordings for training."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image


MASK_FIELDS = ("mask_path", "part_mask_path")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _percentile(values: Iterable[float], q: float) -> float | None:
    data = list(values)
    return float(np.percentile(data, q)) if data else None


def _mask_stats(episode_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload.get("frames", [])
    result: dict[str, Any] = {}
    for field in MASK_FIELDS:
        areas: list[float] = []
        missing = 0
        for frame in frames:
            value = frame.get(field)
            if not value:
                missing += 1
                continue
            path = _resolve(episode_path.parent, value)
            if not path.exists():
                missing += 1
                continue
            mask = np.asarray(Image.open(path))
            areas.append(float(np.mean(mask > 0)))
        prefix = field.removesuffix("_path")
        result[f"{prefix}_frame_ratio"] = len(areas) / max(1, len(frames))
        result[f"{prefix}_area_median"] = median(areas) if areas else None
        result[f"{prefix}_area_p90"] = _percentile(areas, 90)
        if len(areas) > 1:
            diffs = np.abs(np.diff(np.asarray(areas)))
            result[f"{prefix}_area_step_p90"] = float(np.percentile(diffs, 90))
        else:
            result[f"{prefix}_area_step_p90"] = None
        result[f"{prefix}_missing_frames"] = missing
    return result


def _longest_run(indices: list[int]) -> int:
    if not indices:
        return 0
    longest = current = 1
    for left, right in zip(indices, indices[1:]):
        current = current + 1 if right == left + 1 else 1
        longest = max(longest, current)
    return longest


def _track_stats(payload: dict[str, Any], *, jump_m: float) -> dict[str, Any]:
    tracks = payload.get("tracks", [])
    frame_count = int(payload.get("frame_count") or 0)
    lifetimes: list[int] = []
    runs: list[int] = []
    steps: list[float] = []
    jump_count = step_count = 0
    quality: list[float] = []
    timestep_quality: list[float] = []
    part_counts: Counter[str] = Counter()
    for track in tracks:
        part_counts[str(track.get("part_name", track.get("part_id", "unknown")))] += 1
        visible_samples = [s for s in track.get("samples", []) if s.get("visible")]
        indices = [int(s["frame_index"]) for s in visible_samples]
        lifetimes.append(len(indices))
        runs.append(_longest_run(indices))
        points: list[np.ndarray] = []
        point_indices: list[int] = []
        for sample in visible_samples:
            xyz = sample.get("xyz_world")
            if xyz is None or len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
                continue
            points.append(np.asarray(xyz, dtype=np.float64))
            point_indices.append(int(sample["frame_index"]))
            value = sample.get("timestep_quality_score")
            if value is not None:
                timestep_quality.append(float(value))
        for idx in range(1, len(points)):
            if point_indices[idx] != point_indices[idx - 1] + 1:
                continue
            step = float(np.linalg.norm(points[idx] - points[idx - 1]))
            steps.append(step)
            step_count += 1
            jump_count += int(step > jump_m)
        value = track.get("track_quality_score")
        if value is None and isinstance(track.get("track_quality"), dict):
            value = track["track_quality"].get("track_quality_score")
        if value is not None:
            quality.append(float(value))
    return {
        "track_count": len(tracks),
        "part_track_counts": dict(sorted(part_counts.items())),
        "min_part_track_count": min(part_counts.values()) if part_counts else 0,
        "track_lifetime_median": median(lifetimes) if lifetimes else None,
        "track_lifetime_p10": _percentile(lifetimes, 10),
        "track_lifetime_ratio_median": (median(lifetimes) / frame_count) if lifetimes and frame_count else None,
        "longest_contiguous_run_median": median(runs) if runs else None,
        "step_m_p90": _percentile(steps, 90),
        "step_m_p99": _percentile(steps, 99),
        "jump_ratio": jump_count / step_count if step_count else None,
        "track_quality_median": median(quality) if quality else None,
        "timestep_quality_median": median(timestep_quality) if timestep_quality else None,
    }


def _find_track_file(episode_path: Path, roots: list[Path]) -> Path | None:
    resolved = episode_path.resolve()
    candidates: list[Path] = []
    for root in roots:
        for path in root.rglob("*.json"):
            if "track" not in path.name:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source = payload.get("input_episode_path")
            if source and Path(source).expanduser().resolve() == resolved and payload.get("tracks"):
                candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="*", type=Path)
    parser.add_argument("--tracks", action="append", type=Path, default=[])
    parser.add_argument("--track-root", action="append", type=Path, default=[])
    parser.add_argument("--jump-m", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episode_paths: list[Path] = []
    for value in args.episodes:
        episode_paths.extend(
            [Path(path) for path in sorted(glob.glob(str(value)))]
            if any(c in str(value) for c in "*?[")
            else [value]
        )
    rows: list[dict[str, Any]] = []
    for episode_path in episode_paths:
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "dataset": payload.get("metadata", {}).get("source", "unknown"),
            "object_id": payload.get("object_instance_id", episode_path.parent.name),
            "episode_path": str(episode_path.resolve()),
            "frame_count": len(payload.get("frames", [])),
        }
        row.update(_mask_stats(episode_path, payload))
        track_path = _find_track_file(episode_path, args.track_root)
        row["track_path"] = str(track_path.resolve()) if track_path else None
        if track_path:
            row.update(_track_stats(json.loads(track_path.read_text(encoding="utf-8")), jump_m=args.jump_m))
        rows.append(row)

    for track_pattern in args.tracks:
        paths = (
            [Path(path) for path in sorted(glob.glob(str(track_pattern)))]
            if any(c in str(track_pattern) for c in "*?[")
            else [track_pattern]
        )
        for track_path in paths:
            payload = json.loads(track_path.read_text(encoding="utf-8"))
            rows.append({
                "dataset": "track_artifact_only",
                "object_id": track_path.parent.name,
                "episode_path": payload.get("input_episode_path"),
                "frame_count": payload.get("frame_count"),
                "track_path": str(track_path.resolve()),
                **_track_stats(payload, jump_m=args.jump_m),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    csv_path = args.output_dir / "audit.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"episodes": len(rows), "with_tracks": sum(bool(r.get("track_path")) for r in rows), "output": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
