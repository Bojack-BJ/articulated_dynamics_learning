#!/usr/bin/env python3
"""Extract one camera from a multi-view Track2Art episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def resolve(root: Path, value: str) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", type=int, default=0)
    args = parser.parse_args()
    root = args.episode.resolve().parent
    episode = json.loads(args.episode.read_text())

    for frame in episode["frames"]:
        for singular, plural in (
            ("rgb_path", "rgb_paths_by_view"),
            ("depth_path", "depth_paths_by_view"),
            ("mask_path", "mask_paths_by_view"),
            ("part_mask_path", "part_mask_paths_by_view"),
        ):
            values = frame.get(plural) or []
            value = values[args.view] if len(values) > args.view else frame.get(singular)
            if value:
                frame[singular] = resolve(root, value)
            frame.pop(plural, None)
        poses = frame.get("camera_poses_by_view") or []
        if len(poses) > args.view:
            frame["camera_pose"] = poses[args.view]
        frame.pop("camera_poses_by_view", None)

    metadata = episode.setdefault("metadata", {})
    intrinsics = metadata.get("camera_intrinsics_by_view") or []
    if len(intrinsics) > args.view:
        episode["camera_intrinsics"] = intrinsics[args.view]
    image = Image.open(episode["frames"][0]["rgb_path"])
    episode["camera_intrinsics"]["width"] = image.width
    episode["camera_intrinsics"]["height"] = image.height
    metadata["extracted_view_index"] = args.view
    metadata.pop("camera_intrinsics_by_view", None)
    metadata.pop("camera_names", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(episode, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
