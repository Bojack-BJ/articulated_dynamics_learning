#!/usr/bin/env python3
"""Export compact native artifacts for the paper qualitative comparison.

The script intentionally preserves native predicted labels and joint outputs. It
only removes temporal samples that are not needed by the fixed-view figure and
deterministically downsamples dense point sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OBJECTS = (
    "partnet_46556",
    "partnet_10638",
    "partnet_45261",
    "partnet_9388",
    "partnet_103118",
    "partnet_102149",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _subset(values: Any, indices: np.ndarray) -> list[Any] | None:
    if values is None:
        return None
    return [values[int(index)] for index in indices]


def compact_native_viewer(path: Path, point_limit: int) -> dict[str, Any]:
    data = _load(path)
    points = data.get("points", [])
    indices = _indices(len(points), point_limit)
    result: dict[str, Any] = {
        "source_path": str(path),
        "source_method": data.get("source_method"),
        "display_name": data.get("display_name"),
        "points": _subset(points, indices),
        "gt_part_id": _subset(data.get("gt_part_id"), indices),
        "pred_part_id": _subset(data.get("pred_part_id"), indices),
        "joint_axes": data.get("joint_axes", []),
        "overlap": data.get("overlap"),
        "metrics": data.get("metrics", {}),
        "bounds": data.get("bounds"),
        "available_fields": data.get("available_fields", []),
        "full_point_count": len(points),
        "exported_point_count": len(indices),
    }
    return result


def compact_track2art(root: Path, point_limit: int) -> dict[str, Any]:
    tracks_path = root / "motion_part_tracks_slots.json"
    joints_path = root / "joint_inference.json"
    poses_path = root / "part_poses.json"
    tracks_data = _load(tracks_path)
    tracks = tracks_data.get("tracks", [])
    if not tracks:
        raise ValueError(f"No tracks in {tracks_path}")

    # Use one common start-state sample per track. This is the native observed
    # track domain, not a reconstructed or GT-replaced geometry.
    points: list[list[float]] = []
    pred_ids: list[Any] = []
    gt_ids: list[Any] = []
    track_ids: list[Any] = []
    for track in tracks:
        sample = next(
            (
                item
                for item in track.get("samples", [])
                if item.get("visible") and item.get("xyz_world") is not None
            ),
            None,
        )
        xyz = sample.get("xyz_world") if sample else track.get("reference_xyz_world")
        if xyz is None:
            continue
        points.append(xyz)
        pred_ids.append(track.get("part_id"))
        gt_ids.append(track.get("original_part_id"))
        track_ids.append(track.get("track_id"))

    indices = _indices(len(points), point_limit)
    poses = _load(poses_path)
    joints = _load(joints_path)
    return {
        "source_path": str(root),
        "source_method": "track2art_hybrid_full_neural",
        "display_name": "Track2Art",
        "points": _subset(points, indices),
        "gt_part_id": _subset(gt_ids, indices),
        "pred_part_id": _subset(pred_ids, indices),
        "track_id": _subset(track_ids, indices),
        "joint_axes": joints.get("joints", []),
        "anchor_part_id": joints.get("anchor_part_id", poses.get("anchor_part_id")),
        "anchor_part_name": joints.get("anchor_part_name", poses.get("anchor_part_name")),
        "frame_count": tracks_data.get("frame_count"),
        "full_point_count": len(points),
        "exported_point_count": len(indices),
        "coordinate_contract": "xyz_world from native RGB-D lifted tracks",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--track2art-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--point-limit", type=int, default=8000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"objects": {}, "point_limit": args.point_limit}
    for object_id in args.objects:
        output = args.output_dir / object_id
        output.mkdir(parents=True, exist_ok=True)
        object_summary: dict[str, Any] = {}
        for method in ("gt", "aim", "reart"):
            source = args.native_root / object_id / "native_comparison" / "viewer_data" / f"{method}.json"
            if not source.exists():
                object_summary[method] = {"status": "missing", "path": str(source)}
                continue
            payload = compact_native_viewer(source, args.point_limit)
            target = output / f"{method}.json"
            target.write_text(json.dumps(payload, separators=(",", ":")))
            object_summary[method] = {"status": "success", "path": str(target)}

        track2art_source = args.track2art_root / object_id
        if track2art_source.exists():
            payload = compact_track2art(track2art_source, args.point_limit)
            target = output / "track2art.json"
            target.write_text(json.dumps(payload, separators=(",", ":")))
            object_summary["track2art"] = {"status": "success", "path": str(target)}
        else:
            object_summary["track2art"] = {"status": "missing", "path": str(track2art_source)}
        summary["objects"][object_id] = object_summary

    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
