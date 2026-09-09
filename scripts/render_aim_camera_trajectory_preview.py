#!/usr/bin/env python3
"""Render labeled contact sheets for AiM camera-trajectory verification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interaction-samples", type=int, default=16)
    return parser.parse_args()


def _actual_azimuth_deg(pose: list[list[float]], lookat: list[float]) -> float:
    position = np.asarray(pose, dtype=np.float64)[:3, 3]
    relative = position - np.asarray(lookat, dtype=np.float64)
    return math.degrees(math.atan2(float(relative[1]), float(relative[0])))


def _unwrap_degrees(values: list[float]) -> list[float]:
    return np.degrees(np.unwrap(np.radians(values))).tolist()


def _active_joint(frame: dict, previous: dict | None) -> str:
    if previous is None:
        return "none"
    current = frame["action_log"]["joint_positions"]
    before = previous["action_log"]["joint_positions"]
    changes = {
        name: abs(float(value) - float(before.get(name, value)))
        for name, value in current.items()
    }
    name, amount = max(changes.items(), key=lambda item: item[1])
    return name if amount > 1e-6 else "none"


def _tile(image_path: Path, lines: list[str], width: int = 280) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    ratio = width / image.width
    image = image.resize((width, round(image.height * ratio)), Image.Resampling.LANCZOS)
    header = 56
    tile = Image.new("RGB", (image.width, image.height + header), (11, 17, 25))
    tile.paste(image, (0, header))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    for index, line in enumerate(lines):
        draw.text((8, 7 + 17 * index), line, fill=(235, 241, 248), font=font)
    return tile


def _sheet(tiles: list[Image.Image], columns: int) -> Image.Image:
    rows = math.ceil(len(tiles) / columns)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    output = Image.new("RGB", (columns * width, rows * height), (4, 8, 13))
    for index, tile in enumerate(tiles):
        output.paste(tile, ((index % columns) * width, (index // columns) * height))
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    episode_dir = args.episode.parent
    lookat = episode["metadata"]["lookat"]
    frames = episode["frames"]
    sample_indices = sorted(
        set(
            np.linspace(
                0,
                len(frames) - 1,
                min(args.interaction_samples, len(frames)),
                dtype=int,
            ).tolist()
            + [len(frames) - 1]
        )
    )
    actual_wrapped = [
        _actual_azimuth_deg(frame["camera_pose"], lookat) for frame in frames
    ]
    actual_unwrapped = _unwrap_degrees(actual_wrapped)
    interaction_tiles = []
    for index in sample_indices:
        frame = frames[index]
        interaction_tiles.append(
            _tile(
                episode_dir / frame["rgb_path"],
                [
                    f"frame {index}/{len(frames) - 1} | t={frame['timestamp_s']:.2f}s",
                    f"actual az={actual_wrapped[index]:.1f} deg | unwrap={actual_unwrapped[index]:.1f}",
                    f"active joint={_active_joint(frame, frames[index - 1] if index else None)}",
                ],
            )
        )
    _sheet(interaction_tiles, columns=4).save(
        args.output_dir / "interaction_trajectory_contact_sheet.jpg",
        quality=92,
    )

    protocol = episode.get("metadata", {}).get("aim_protocol", {})
    static_relative = protocol.get("static_scan_manifest")
    if static_relative:
        static_manifest_path = episode_dir / static_relative
        static = json.loads(static_manifest_path.read_text(encoding="utf-8"))
        static_tiles = []
        static_actual = []
        for view in static["views"]:
            static_actual.append(_actual_azimuth_deg(view["camera_pose"], lookat))
        static_unwrapped = _unwrap_degrees(static_actual)
        for index, view in enumerate(static["views"]):
            static_tiles.append(
                _tile(
                    episode_dir / view["rgb_path"],
                    [
                        f"view {index:02d} | recorder az={view['azimuth_deg']:.1f} deg",
                        f"actual az={static_actual[index]:.1f} deg | unwrap={static_unwrapped[index]:.1f}",
                        "fixed start joint state",
                    ],
                )
            )
        _sheet(static_tiles, columns=6).save(
            args.output_dir / "static_scan_azimuth_contact_sheet.jpg",
            quality=92,
        )

    payload = {
        "episode": str(args.episode),
        "trajectory": protocol.get("interaction_camera_trajectory"),
        "sample_indices": sample_indices,
        "actual_azimuth_wrapped_deg": [actual_wrapped[index] for index in sample_indices],
        "actual_azimuth_unwrapped_deg": [actual_unwrapped[index] for index in sample_indices],
        "total_sweep_deg": actual_unwrapped[-1] - actual_unwrapped[0],
        "includes_final_frame": sample_indices[-1] == len(frames) - 1,
    }
    (args.output_dir / "trajectory_preview.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
