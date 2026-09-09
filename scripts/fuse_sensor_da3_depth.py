#!/usr/bin/env python3
"""Fuse sensor depth with DA3 while preserving sensor metric consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_depth_mm(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source, allow_pickle=False).astype(np.uint16, copy=False)
    return np.asarray(Image.open(source), dtype=np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("da3_episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--spike-threshold-m", type=float, default=0.08)
    parser.add_argument("--da3-surface-threshold-m", type=float, default=0.05)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--scale-median-window", type=int, default=9)
    parser.add_argument("--temporal-alpha", type=float, default=0.5)
    parser.add_argument("--flow-scale", type=float, default=0.25)
    parser.add_argument("--flow-fb-threshold-px", type=float, default=1.5)
    args = parser.parse_args()

    episode = json.loads(args.da3_episode.read_text())
    depth_dir = args.output_dir / "depth_mm"
    depth_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    scales = []
    for frame in episode["frames"]:
        sensor = load_depth_mm(frame["sensor_depth_path"]).astype(np.float32) / 1000.0
        da3 = np.asarray(Image.open(frame["depth_path"]), dtype=np.float32) / 1000.0
        object_mask = np.asarray(Image.open(frame["part_mask_path"])) > 0
        overlap = (sensor > 0) & (da3 > 0) & object_mask
        ratio = sensor[overlap] / np.maximum(da3[overlap], 1e-4)
        ratio = ratio[np.isfinite(ratio)]
        scales.append(float(np.median(ratio)) if ratio.size else 1.0)
    scales = np.clip(np.asarray(scales), args.scale_min, args.scale_max)
    radius = max(0, args.scale_median_window // 2)
    stable_scales = np.asarray([
        np.median(scales[max(0, i - radius) : min(len(scales), i + radius + 1)])
        for i in range(len(scales))
    ])

    previous_gray = None
    previous_stable_da3 = None
    previous_object_mask = None
    for index, frame in enumerate(episode["frames"]):
        sensor_mm = load_depth_mm(frame["sensor_depth_path"])
        da3_mm = np.asarray(Image.open(frame["depth_path"]), dtype=np.uint16)
        object_mask = np.asarray(Image.open(frame["part_mask_path"])) > 0
        sensor = sensor_mm.astype(np.float32) / 1000.0
        da3 = da3_mm.astype(np.float32) / 1000.0
        sensor_valid = sensor > 0
        da3_valid = da3 > 0

        scale = float(stable_scales[index])
        da3_aligned = da3 * scale

        rgb = cv2.imread(frame["rgb_path"], cv2.IMREAD_GRAYSCALE)
        flow_size = (
            max(32, int(round(rgb.shape[1] * args.flow_scale))),
            max(32, int(round(rgb.shape[0] * args.flow_scale))),
        )
        gray = cv2.resize(rgb, flow_size, interpolation=cv2.INTER_AREA)
        temporal_valid = np.zeros_like(object_mask)
        stable_da3 = da3_aligned
        if previous_gray is not None and previous_stable_da3 is not None and previous_object_mask is not None:
            forward = cv2.calcOpticalFlowFarneback(
                previous_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            backward = cv2.calcOpticalFlowFarneback(
                gray, previous_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            h, w = gray.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            px, py = gx + backward[..., 0], gy + backward[..., 1]
            sampled_forward = cv2.remap(
                forward, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            consistent_small = np.linalg.norm(backward + sampled_forward, axis=-1) <= args.flow_fb_threshold_px
            back_full = cv2.resize(backward, (sensor.shape[1], sensor.shape[0]), interpolation=cv2.INTER_LINEAR)
            back_full[..., 0] /= args.flow_scale
            back_full[..., 1] /= args.flow_scale
            fx, fy = np.meshgrid(
                np.arange(sensor.shape[1], dtype=np.float32),
                np.arange(sensor.shape[0], dtype=np.float32),
            )
            warped_previous = cv2.remap(
                previous_stable_da3,
                fx + back_full[..., 0],
                fy + back_full[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_previous_mask = cv2.remap(
                previous_object_mask.astype(np.uint8),
                fx + back_full[..., 0],
                fy + back_full[..., 1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(bool)
            consistent = cv2.resize(
                consistent_small.astype(np.uint8),
                (sensor.shape[1], sensor.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            temporal_valid = (
                consistent & object_mask & warped_previous_mask
                & (warped_previous > 0) & da3_valid
            )
            stable_da3 = da3_aligned.copy()
            stable_da3[temporal_valid] = (
                args.temporal_alpha * da3_aligned[temporal_valid]
                + (1.0 - args.temporal_alpha) * warped_previous[temporal_valid]
            )

        # Median filtering is used only to detect isolated sensor spikes. It is
        # never copied wholesale, so valid edges retain the sensor measurement.
        local = cv2.medianBlur(sensor_mm, 5).astype(np.float32) / 1000.0
        local_valid = local > 0
        sensor_spike = (
            sensor_valid
            & object_mask
            & local_valid
            & (np.abs(sensor - local) > args.spike_threshold_m)
            & da3_valid
            & (np.abs(stable_da3 - local) < args.da3_surface_threshold_m)
        )
        holes = ~sensor_valid & da3_valid & object_mask

        fused = sensor.copy()
        fused[holes | sensor_spike] = stable_da3[holes | sensor_spike]
        fused_mm = np.zeros_like(sensor_mm)
        valid = np.isfinite(fused) & (fused > 0) & (fused < 65.535)
        fused_mm[valid] = np.rint(fused[valid] * 1000.0).astype(np.uint16)
        output_path = depth_dir / f"{index:05d}.png"
        Image.fromarray(fused_mm, mode="I;16").save(output_path)
        frame["da3_depth_path"] = frame["depth_path"]
        frame["depth_path"] = str(output_path.resolve())

        summaries.append({
            "frame_index": index,
            "raw_da3_alignment_scale": float(scales[index]),
            "da3_alignment_scale": scale,
            "temporal_consistent_ratio": float((temporal_valid & object_mask).sum() / max(1, object_mask.sum())),
            "sensor_valid_ratio": float(sensor_valid.mean()),
            "filled_hole_ratio": float(holes.mean()),
            "replaced_spike_ratio": float(sensor_spike.mean()),
            "fused_valid_ratio": float(valid.mean()),
        })
        previous_gray = gray
        previous_stable_da3 = stable_da3
        previous_object_mask = object_mask

    metadata = episode.setdefault("metadata", {})
    metadata["depth_source"] = {
        "provider": "sensor+temporally-stabilized Depth Anything 3 fusion",
        "policy": "keep sensor; flow-stabilized DA3 fills object holes and verified spikes",
        "spike_threshold_m": args.spike_threshold_m,
        "da3_surface_threshold_m": args.da3_surface_threshold_m,
        "da3_scale_clip": [args.scale_min, args.scale_max],
        "scale_median_window": args.scale_median_window,
        "temporal_alpha": args.temporal_alpha,
        "flow_scale": args.flow_scale,
        "flow_fb_threshold_px": args.flow_fb_threshold_px,
    }
    output_episode = args.output_dir / "episode_sensor_da3_fused.json"
    output_episode.write_text(json.dumps(episode, indent=2) + "\n")
    (args.output_dir / "fusion_summary.json").write_text(
        json.dumps({"frames": summaries}, indent=2) + "\n"
    )
    print(output_episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
