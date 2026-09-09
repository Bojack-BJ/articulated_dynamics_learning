#!/usr/bin/env python3
"""Download one interaction window from a remote Arti4D scene archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--axis-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=30)
    parser.add_argument("--include-video", action="store_true")
    return parser.parse_args()


def archive_scene_prefix(names: list[str], scene: str) -> str:
    suffix = f"/{scene}/"
    matches = [name[: name.index(suffix) + len(suffix)] for name in names if suffix in name]
    if not matches:
        raise ValueError(f"Scene {scene!r} was not found in the archive")
    return matches[0]


def read_cues(zf: RemoteZip, prefix: str) -> list[dict[str, str]]:
    text = zf.read(prefix + "matched_cues.csv").decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def extract_member(zf: RemoteZip, member: str, output_root: Path, prefix: str) -> Path:
    relative = PurePosixPath(member).relative_to(PurePosixPath(prefix))
    destination = output_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(zf.read(member))
    return destination


def main() -> None:
    args = parse_args()
    headers = {"User-Agent": "Mozilla/5.0"}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with RemoteZip(args.url, headers=headers) as zf:
        names = zf.namelist()
        prefix = archive_scene_prefix(names, args.scene)
        cues = read_cues(zf, prefix)
        cue = next((row for row in cues if row["AXIS_NAME"] == args.axis_name), None)
        if cue is None:
            available = ", ".join(row["AXIS_NAME"] for row in cues)
            raise ValueError(f"Unknown axis {args.axis_name!r}; available: {available}")

        cue_start = int(cue["CUE_START"])
        cue_end = int(cue["CUE_END"])
        rgb = sorted(name for name in names if name.startswith(prefix + "rgb/") and name.endswith(".jpg"))
        depth = sorted(name for name in names if name.startswith(prefix + "depth/") and name.endswith(".png"))
        if len(rgb) != len(depth):
            raise ValueError(f"RGB/depth count mismatch: {len(rgb)} != {len(depth)}")

        first = max(0, cue_start - args.context_frames)
        last = min(len(rgb) - 1, cue_end + args.context_frames)
        selected = rgb[first : last + 1] + depth[first : last + 1]
        metadata_members = [
            prefix + "rgb/camera_info.txt",
            prefix + "depth/camera_info.txt",
            prefix + "azure_tf.json",
            prefix + f"odom/{args.scene}.csv",
            prefix + f"{args.scene}.json",
        ]
        if args.include_video:
            metadata_members.append(prefix + f"{args.scene}.mp4")

        for member in metadata_members + selected:
            extract_member(zf, member, args.output_dir, prefix)

    local_start = cue_start - first
    local_end = cue_end - first
    with (args.output_dir / "matched_cues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AXIS_NAME", "CUE_START", "CUE_END", "VERIFICATION"])
        writer.writeheader()
        writer.writerow(
            {
                "AXIS_NAME": args.axis_name,
                "CUE_START": local_start,
                "CUE_END": local_end,
                "VERIFICATION": cue.get("VERIFICATION", "VERIFIED"),
            }
        )

    manifest = {
        "source_url": args.url,
        "scene": args.scene,
        "axis_name": args.axis_name,
        "source_frame_range": [first, last],
        "source_cue_range": [cue_start, cue_end],
        "local_cue_range": [local_start, local_end],
        "frame_count": last - first + 1,
        "context_frames": args.context_frames,
        "protocol": "official_arti4d_interaction_window",
    }
    (args.output_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
