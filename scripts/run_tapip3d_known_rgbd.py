#!/usr/bin/env python3
"""Run TAPIP3D on known RGB-D/camera inputs without importing MegaSAM."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapip-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-factor", type=float, default=2.0)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--support-grid-size", type=int, default=16)
    parser.add_argument("--vis-threshold", type=float, default=0.9)
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tapip_root = args.tapip_root.expanduser().resolve()
    sys.path.insert(0, str(tapip_root))

    from datasets.data_ops import _filter_one_depth
    from utils.inference_utils import inference, load_model, resize_depth_bilinear

    with np.load(args.input.expanduser().resolve()) as payload:
        video = np.asarray(payload["video"], dtype=np.uint8)
        depths = np.asarray(payload["depths"], dtype=np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32).copy()
        extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
        query_point = np.asarray(payload["query_point"], dtype=np.float32)
        track_ids = np.asarray(payload["track_ids"], dtype=np.int64)
        source_frame_indices = np.asarray(payload["source_frame_indices"], dtype=np.int64)

    if video.shape[:3] != depths.shape or query_point.shape != (len(track_ids), 4):
        raise ValueError("TAPIP3D input video/depth/query dimensions are inconsistent.")

    model = load_model(str(args.checkpoint.expanduser().resolve())).to(args.device)
    inference_res = (
        int(model.image_size[0] * np.sqrt(args.resolution_factor)),
        int(model.image_size[1] * np.sqrt(args.resolution_factor)),
    )
    model.set_image_size(inference_res)
    original_height, original_width = depths.shape[1:]
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / max(1, original_width - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / max(1, original_height - 1)

    with ThreadPoolExecutor(max_workers=max(1, args.num_threads)) as executor:
        video = np.stack(
            list(
                executor.map(
                    lambda frame: cv2.resize(
                        frame, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR
                    ),
                    video,
                )
            )
        )
        depths = np.stack(
            list(
                executor.map(
                    lambda depth: resize_depth_bilinear(
                        depth, (inference_res[1], inference_res[0])
                    ),
                    depths,
                )
            )
        )
        depths = np.stack(
            list(
                executor.map(
                    lambda values: _filter_one_depth(values[0], 0.08, 15, values[1]),
                    zip(depths, intrinsics),
                )
            )
        )

    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().div_(255.0).to(args.device)
    depth_tensor = torch.from_numpy(depths).float().to(args.device)
    intrinsic_tensor = torch.from_numpy(intrinsics).float().to(args.device)
    extrinsic_tensor = torch.from_numpy(extrinsics).float().to(args.device)
    query_tensor = torch.from_numpy(query_point).float().to(args.device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=video_tensor,
            depths=depth_tensor,
            intrinsics=intrinsic_tensor,
            extrinsics=extrinsic_tensor,
            query_point=query_tensor,
            num_iters=max(1, args.num_iters),
            grid_size=max(0, args.support_grid_size),
            vis_threshold=float(args.vis_threshold),
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coords=coords.detach().float().cpu().numpy(),
        visibs=visibs.detach().cpu().numpy(),
        query_points=query_point,
        track_ids=track_ids,
        source_frame_indices=source_frame_indices,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
