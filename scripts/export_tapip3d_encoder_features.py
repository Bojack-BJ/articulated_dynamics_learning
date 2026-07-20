#!/usr/bin/env python3
"""Export frozen TAPIP3D per-track UpdateFormer tokens on CUDA.

Run this script from a CUDA TAPIP3D environment, not from the macOS project
environment. It writes the same ``track_ids`` / ``embeddings`` NPZ contract
used by the existing feature-probe command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export frozen TAPIP3D UpdateFormer track features.")
    parser.add_argument("--tapip-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="NPZ from prepare-tapip3d-input")
    parser.add_argument("--result", type=Path, required=True, help="TAPIP3D .result.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--support-grid-size", type=int, default=16)
    parser.add_argument("--resolution-factor", type=int, default=2)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, str(args.tapip_root.expanduser().resolve()))
    from utils.inference_utils import inference, load_model  # type: ignore[import-not-found]
    from inference import prepare_inputs  # type: ignore[import-not-found]

    with np.load(args.input) as input_data, np.load(args.result) as result_data:
        track_ids = np.asarray(input_data["track_ids"], dtype=np.int64)
        visibs = np.asarray(result_data["visibs"], dtype=bool)
        result_coords = np.asarray(result_data["coords"], dtype=np.float32)
    if result_coords.shape[:2] != visibs.shape or result_coords.shape[1] != track_ids.shape[0]:
        raise ValueError("Result trajectories do not align with input track_ids.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TAPIP3D encoder feature export requires CUDA PyTorch.")
    model = load_model(str(args.checkpoint)).to(device)
    inference_res = tuple(int(value * np.sqrt(args.resolution_factor)) for value in model.image_size)
    model.set_image_size(inference_res)
    video, depths, intrinsics, extrinsics, query_points, _ = prepare_inputs(
        str(args.input), inference_res, int(args.support_grid_size), device=str(device)
    )

    captures: list[torch.Tensor] = []
    flow_head = getattr(model.point_updater, "flow_head", None)
    if flow_head is None:
        raise RuntimeError("TAPIP3D point updater has no flow_head to hook.")

    def capture_tokens(_module: object, inputs: tuple[torch.Tensor, ...]) -> None:
        # The flow head input is [B,N,T,C], after temporal and cross-track attention.
        captures.append(inputs[0][:, : len(track_ids)].detach().float().cpu())

    hook = flow_head.register_forward_pre_hook(capture_tokens)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference(
                model=model,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_points,
                num_iters=int(args.num_iters),
                grid_size=int(args.support_grid_size),
            )
    finally:
        hook.remove()
    temporal_tokens = _assemble_final_iteration_tokens(
        captures,
        frame_count=int(video.shape[0]),
        track_count=len(track_ids),
        num_iters=int(args.num_iters),
    )
    valid = torch.from_numpy(visibs.T).bool()  # [N,T]
    if valid.shape != temporal_tokens.shape[:2]:
        raise ValueError(f"Visibility/token mismatch: {tuple(valid.shape)} vs {tuple(temporal_tokens.shape)}")
    weights = valid.to(temporal_tokens.dtype)[..., None]
    mean_tokens = (temporal_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    centered = (temporal_tokens - mean_tokens[:, None]) * weights
    std_tokens = torch.sqrt(centered.square().sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))
    endpoints = temporal_tokens[:, -1] - temporal_tokens[:, 0]
    embeddings = functional.normalize(torch.cat([mean_tokens, std_tokens, endpoints], dim=-1), dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        track_ids=track_ids,
        embeddings=embeddings.numpy().astype(np.float32),
        temporal_tokens=temporal_tokens.numpy().astype(np.float16),
        valid_timestep_count=valid.sum(dim=1).numpy().astype(np.int64),
        feature_source=np.asarray(["tapip3d_updateformer_mean_std_delta"]),
    )
    print(args.output.resolve())
    return 0


def _assemble_final_iteration_tokens(captures, *, frame_count: int, track_count: int, num_iters: int):
    """Stitch the final refinement token from overlapping TAPIP windows."""
    import torch

    if num_iters <= 0 or len(captures) % num_iters:
        raise RuntimeError(f"Unexpected TAPIP token captures: {len(captures)} for {num_iters} iterations.")
    windows = captures[num_iters - 1 :: num_iters]
    if not windows:
        raise RuntimeError("TAPIP token hook captured no completed windows.")
    feature_dim = int(windows[0].shape[-1])
    output = torch.zeros(track_count, frame_count, feature_dim, dtype=torch.float32)
    counts = torch.zeros(track_count, frame_count, 1, dtype=torch.float32)
    stride = int(windows[0].shape[2]) // 2
    for window_index, tokens in enumerate(windows):
        values = tokens[0, :track_count]  # [N,W,C]
        start = window_index * stride
        end = min(frame_count, start + values.shape[1])
        output[:, start:end] += values[:, : end - start]
        counts[:, start:end] += 1.0
    if bool((counts == 0).any()):
        raise RuntimeError("TAPIP token windows did not cover every requested frame.")
    return output / counts


if __name__ == "__main__":
    raise SystemExit(main())
