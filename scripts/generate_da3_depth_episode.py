#!/usr/bin/env python3
"""Replace an episode's sensor depth with DA3 monocular metric depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    episode = json.loads(args.episode.read_text())
    frames = episode["frames"]
    camera = episode["camera_intrinsics"]
    first_image = Image.open(frames[0]["rgb_path"])
    width = int(camera.get("width", first_image.width))
    height = int(camera.get("height", first_image.height))
    camera["width"], camera["height"] = width, height
    focal = 0.5 * (float(camera["fx"]) + float(camera["fy"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output_dir / "depth_mm"
    depth_dir.mkdir(exist_ok=True)
    model = DepthAnything3.from_pretrained(args.model).to(args.device)

    summaries = []
    for start in range(0, len(frames), args.batch_size):
        batch = frames[start : start + args.batch_size]
        prediction = model.inference(
            [frame["rgb_path"] for frame in batch],
            process_res=args.process_res,
        )
        processed_width = prediction.depth.shape[-1]
        metric_factor = focal * (processed_width / width) / 300.0
        for offset, (frame, raw_depth) in enumerate(zip(batch, prediction.depth)):
            frame_index = start + offset
            depth_m = cv2.resize(
                raw_depth * metric_factor,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < 65.535)
            depth_mm = np.zeros((height, width), dtype=np.uint16)
            depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
            output_path = depth_dir / f"{frame_index:05d}.png"
            Image.fromarray(depth_mm, mode="I;16").save(output_path)
            frame["sensor_depth_path"] = frame["depth_path"]
            frame["depth_path"] = str(output_path.resolve())
            values = depth_m[valid]
            summaries.append({
                "frame_index": frame_index,
                "source_frame_index": frame.get("action_log", {}).get("source_frame_index"),
                "valid_ratio": float(valid.mean()),
                "depth_median_m": float(np.median(values)),
                "depth_p01_m": float(np.percentile(values, 1)),
                "depth_p99_m": float(np.percentile(values, 99)),
            })
        print(f"generated {min(start + len(batch), len(frames))}/{len(frames)}", flush=True)

    metadata = episode.setdefault("metadata", {})
    metadata["depth_convention"] = {"unit": "millimeter", "scale_to_m": 0.001}
    metadata["depth_source"] = {
        "provider": "Depth Anything 3",
        "model": args.model,
        "process_res": args.process_res,
        "metric_formula": "focal_processed_px * network_output / 300",
        "sensor_depth_preserved_as": "sensor_depth_path",
    }
    output_episode = args.output_dir / "episode_da3.json"
    output_episode.write_text(json.dumps(episode, indent=2) + "\n")
    (args.output_dir / "depth_summary.json").write_text(
        json.dumps({"frames": summaries}, indent=2) + "\n"
    )
    print(output_episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
