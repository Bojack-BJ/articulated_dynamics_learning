#!/usr/bin/env python3
"""Run TAPIP3D on known RGB-D/camera inputs without importing MegaSAM."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapip-root", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument(
        "--jobs-manifest",
        type=Path,
        help="TSV with input, output, and optional features_output columns; the model is loaded once.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-factor", type=float, default=2.0)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--support-grid-size", type=int, default=16)
    parser.add_argument("--vis-threshold", type=float, default=0.9)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--uncompressed-output", action="store_true")
    parser.add_argument(
        "--wait-for-input-s", type=float, default=0.0,
        help="Wait for producer-created input files when consuming a live jobs manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tapip_root = args.tapip_root.expanduser().resolve()
    sys.path.insert(0, str(tapip_root))

    from datasets.data_ops import _filter_one_depth
    from utils.inference_utils import inference, load_model, resize_depth_bilinear

    jobs = _load_jobs(args)
    model = load_model(str(args.checkpoint.expanduser().resolve())).to(args.device)
    failures = []
    for job_index, job in enumerate(jobs, start=1):
        started = time.perf_counter()
        try:
            deadline = time.monotonic() + max(0.0, float(args.wait_for_input_s))
            while not job["input"].is_file() and time.monotonic() < deadline:
                time.sleep(0.25)
            if not job["input"].is_file():
                raise FileNotFoundError(job["input"])
            _run_job(args, job, model, inference, resize_depth_bilinear, _filter_one_depth, torch)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append({"input": str(job["input"]), "output": str(job["output"]), "error": repr(exc)})
            print(f"[{job_index}/{len(jobs)}] FAILED {job['input']}: {exc}", flush=True)
            continue
        finally:
            # TAPIP has large per-video workspaces. Release cached blocks between
            # heterogeneous jobs so a long-lived worker does not fragment VRAM.
            gc.collect()
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"[{job_index}/{len(jobs)}] {job['output']} elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    if args.failures_output is not None:
        failure_path = args.failures_output.expanduser().resolve()
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return 0


def _load_jobs(args: argparse.Namespace) -> list[dict[str, Path | None]]:
    if args.jobs_manifest is not None:
        with args.jobs_manifest.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if not rows or not all(row.get("input") and row.get("output") for row in rows):
            raise ValueError("TAPIP jobs manifest requires input and output columns.")
        return [
            {
                "input": Path(row["input"]).expanduser().resolve(),
                "output": Path(row["output"]).expanduser().resolve(),
                "features_output": (
                    Path(row["features_output"]).expanduser().resolve()
                    if row.get("features_output")
                    else None
                ),
            }
            for row in rows
        ]
    if args.input is None or args.output is None:
        raise ValueError("Provide --input and --output, or --jobs-manifest.")
    return [{
        "input": args.input.expanduser().resolve(),
        "output": args.output.expanduser().resolve(),
        "features_output": args.features_output.expanduser().resolve() if args.features_output else None,
    }]


def _run_job(args, job, model, inference, resize_depth_bilinear, filter_one_depth, torch_module) -> None:
    with np.load(job["input"]) as payload:
        video = np.asarray(payload["video"], dtype=np.uint8)
        depths = np.asarray(payload["depths"], dtype=np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32).copy()
        extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
        query_point = np.asarray(payload["query_point"], dtype=np.float32)
        track_ids = np.asarray(payload["track_ids"], dtype=np.int64)
        source_frame_indices = np.asarray(payload["source_frame_indices"], dtype=np.int64)

    if video.shape[:3] != depths.shape or query_point.shape != (len(track_ids), 4):
        raise ValueError("TAPIP3D input video/depth/query dimensions are inconsistent.")

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
                    lambda values: filter_one_depth(values[0], 0.08, 15, values[1]),
                    zip(depths, intrinsics),
                )
            )
        )

    video_tensor = torch_module.from_numpy(video).permute(0, 3, 1, 2).float().div_(255.0).to(args.device)
    depth_tensor = torch_module.from_numpy(depths).float().to(args.device)
    intrinsic_tensor = torch_module.from_numpy(intrinsics).float().to(args.device)
    extrinsic_tensor = torch_module.from_numpy(extrinsics).float().to(args.device)
    query_tensor = torch_module.from_numpy(query_point).float().to(args.device)

    captures = []
    hook = None
    if job["features_output"] is not None:
        flow_head = getattr(model.point_updater, "flow_head", None)
        if flow_head is None:
            raise RuntimeError("TAPIP3D point updater has no flow_head to hook.")

        def capture_tokens(_module, inputs):
            captures.append(inputs[0][:, : len(track_ids)].detach().float().cpu())

        hook = flow_head.register_forward_pre_hook(capture_tokens)
    try:
        with torch_module.inference_mode(), torch_module.autocast("cuda", dtype=torch_module.bfloat16):
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
    finally:
        if hook is not None:
            hook.remove()

    output = job["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez if args.uncompressed_output else np.savez_compressed
    save(
        output,
        coords=coords.detach().float().cpu().numpy(),
        visibs=visibs.detach().cpu().numpy(),
        query_points=query_point,
        track_ids=track_ids,
        source_frame_indices=source_frame_indices,
    )
    if job["features_output"] is not None:
        _save_features(
            captures, visibs, track_ids, frame_count=len(video), num_iters=max(1, args.num_iters),
            output=job["features_output"], torch_module=torch_module,
            compressed=not args.uncompressed_output,
        )


def _save_features(
    captures, visibs, track_ids, *, frame_count, num_iters, output, torch_module, compressed=True
) -> None:
    import torch.nn.functional as functional

    if not captures or len(captures) % num_iters:
        raise RuntimeError(f"Unexpected TAPIP token captures: {len(captures)} for {num_iters} iterations.")
    windows = captures[num_iters - 1 :: num_iters]
    feature_dim = int(windows[0].shape[-1])
    temporal = torch_module.zeros(len(track_ids), frame_count, feature_dim, dtype=torch_module.float32)
    counts = torch_module.zeros(len(track_ids), frame_count, 1, dtype=torch_module.float32)
    stride = int(windows[0].shape[2]) // 2
    for window_index, tokens in enumerate(windows):
        values = tokens[0, : len(track_ids)]
        start = window_index * stride
        if start >= frame_count:
            break
        end = min(frame_count, start + values.shape[1])
        temporal[:, start:end] += values[:, : end - start]
        counts[:, start:end] += 1.0
    if bool((counts == 0).any()):
        raise RuntimeError("TAPIP token windows did not cover every requested frame.")
    temporal = temporal / counts
    valid = visibs.detach().cpu().bool().transpose(0, 1)
    weights = valid.to(temporal.dtype)[..., None]
    mean = (temporal * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    centered = (temporal - mean[:, None]) * weights
    std = torch_module.sqrt(centered.square().sum(1) / weights.sum(1).clamp_min(1.0))
    embeddings = functional.normalize(torch_module.cat([mean, std, temporal[:, -1] - temporal[:, 0]], -1), dim=-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    save(
        output,
        track_ids=track_ids,
        embeddings=embeddings.numpy().astype(np.float32),
        temporal_tokens=temporal.numpy().astype(np.float16),
        valid_timestep_count=valid.sum(1).numpy().astype(np.int64),
        feature_source=np.asarray(["tapip3d_updateformer_mean_std_delta"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
