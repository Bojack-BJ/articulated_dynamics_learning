#!/usr/bin/env python3
"""Convert manually curated indexed masks into binary object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def finalize_episode(source: Path, output: Path, mask_dir: Path) -> dict[str, object]:
    episode = json.loads(source.read_text(encoding="utf-8"))
    mask_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(episode.get("frames", [])):
        raw_path = frame.get("part_mask_path") or frame.get("mask_path")
        if not raw_path:
            raise ValueError(f"Frame {frame_index} has no mask path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = source.parent / path
        labels = np.asarray(Image.open(path))
        binary = np.where(labels > 0, 255, 0).astype(np.uint8)
        binary_path = mask_dir / f"mask_{frame_index:06d}.png"
        Image.fromarray(binary, mode="L").save(binary_path)
        frame["mask_path"] = str(binary_path.resolve())

    metadata = episode.setdefault("metadata", {})
    metadata["object_mask_source"] = "artipoint_manual_curated"
    metadata["uses_gt_part_masks"] = False
    metadata["curated_mask_source_episode"] = str(source.resolve())
    metadata.pop("part_segmentation", None)
    metadata["mask_provider"] = {
        "provider": "manual-curation",
        "mask_kind": "binary-object",
        "foreground_value": 255,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    args = parser.parse_args()
    episode = finalize_episode(args.source.resolve(), args.output.resolve(), args.mask_dir.resolve())
    print(json.dumps({"episode": str(args.output.resolve()), "frames": len(episode["frames"])}, indent=2))


if __name__ == "__main__":
    main()
