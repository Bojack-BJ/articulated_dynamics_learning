#!/usr/bin/env python3
"""Convert an ArtiPoint prediction into Track2Art's interactive viewer format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _surface_distances(episode_path: Path, tracks: np.ndarray, *, pixel_stride: int) -> np.ndarray:
    from PIL import Image
    from scipy.spatial import cKDTree

    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    root = episode_path.parent
    intrinsics = episode["camera_intrinsics"]
    result = np.full(tracks.shape[:2], np.inf, dtype=np.float64)
    for frame_index in range(min(len(episode["frames"]), tracks.shape[0])):
        frame = episode["frames"][frame_index]
        depth = np.asarray(Image.open(root / frame["depth_path"]), dtype=np.float64) * 0.001
        mask = np.asarray(Image.open(root / frame["mask_path"])) > 0
        vv, uu = np.where(mask & (depth > 0.0))
        uu, vv = uu[::pixel_stride], vv[::pixel_stride]
        if not len(uu):
            continue
        z = depth[vv, uu]
        camera = np.stack(
            [
                (uu - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
                -(vv - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
                z,
            ],
            axis=1,
        )
        pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        result[frame_index] = cKDTree(world).query(tracks[frame_index], k=1)[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks_npz", type=Path)
    parser.add_argument("axis_info_json", type=Path)
    parser.add_argument("--output-tracks", type=Path, required=True)
    parser.add_argument("--output-joints", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--max-surface-distance-m", type=float, default=0.0)
    parser.add_argument("--surface-pixel-stride", type=int, default=4)
    args = parser.parse_args()

    loaded = np.load(args.tracks_npz, allow_pickle=True)
    tracks = np.asarray(loaded["tracks"][0], dtype=np.float64)
    visibility = np.asarray(loaded["visibility"][0], dtype=bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"Expected [T,N,3] tracks, got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(f"Visibility {visibility.shape} does not match tracks {tracks.shape}")
    surface_distance = None
    if args.episode is not None:
        surface_distance = _surface_distances(
            args.episode.expanduser().resolve(), tracks,
            pixel_stride=max(1, int(args.surface_pixel_stride)),
        )

    rows = []
    for track_index in range(tracks.shape[1]):
        samples = []
        for frame_index in range(tracks.shape[0]):
            visible = bool(visibility[frame_index, track_index])
            xyz = tracks[frame_index, track_index]
            distance = (
                float(surface_distance[frame_index, track_index])
                if surface_distance is not None else None
            )
            surface_consistent = (
                distance is None or args.max_surface_distance_m <= 0.0
                or distance <= args.max_surface_distance_m
            )
            valid = visible and surface_consistent and bool(np.all(np.isfinite(xyz)))
            samples.append(
                {
                    "frame_index": frame_index,
                    "source_frame_index": frame_index,
                    "timestamp_s": frame_index / args.fps,
                    "xyz_world": xyz.tolist() if valid else None,
                    "visible": valid,
                    "confidence": 1.0 if valid else 0.0,
                    "surface_distance_m": distance,
                    "surface_consistent": surface_consistent,
                }
            )
        first = next((sample["xyz_world"] for sample in samples if sample["visible"]), None)
        rows.append(
            {
                "track_id": track_index,
                "part_id": 1,
                "part_name": "artipoint_moving_part",
                "view_index": 0,
                "reference_xyz_world": first,
                "samples": samples,
            }
        )

    tracks_payload = {
        "input_episode_path": None,
        "estimator": "artipoint-official",
        "frame_count": int(tracks.shape[0]),
        "sampled_frame_indices": list(range(tracks.shape[0])),
        "effective_tracking_fps_hz": args.fps,
        "tracks": rows,
        "part_track_counts": {"1": {"name": "artipoint_moving_part", "count": len(rows)}},
        "diagnostic_surface_filter": {
            "enabled": surface_distance is not None and args.max_surface_distance_m > 0.0,
            "episode": str(args.episode) if args.episode is not None else None,
            "max_surface_distance_m": float(args.max_surface_distance_m),
        },
    }
    axis_info = json.loads(args.axis_info_json.read_text(encoding="utf-8"))
    joint_type = str(axis_info.get("joint_type", "unknown"))
    joint_payload = {
        "source": "artipoint-official",
        "joints": [
            {
                "name": "artipoint_joint",
                "joint_type": joint_type,
                "parent_part_id": 0,
                "child_part_id": 1,
                "axis": axis_info.get("axis", [0.0, 0.0, 1.0]),
                "pivot": axis_info.get("center", [0.0, 0.0, 0.0]),
            }
        ],
    }
    args.output_tracks.parent.mkdir(parents=True, exist_ok=True)
    args.output_joints.parent.mkdir(parents=True, exist_ok=True)
    args.output_tracks.write_text(json.dumps(tracks_payload, indent=2) + "\n", encoding="utf-8")
    args.output_joints.write_text(json.dumps(joint_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tracks": str(args.output_tracks), "joints": str(args.output_joints)}, indent=2))


if __name__ == "__main__":
    main()
