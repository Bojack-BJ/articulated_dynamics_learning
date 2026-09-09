#!/usr/bin/env python3
"""Render an AiM checkpoint with the official Gaussian rasterizer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aim_repo", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--include-test-cameras", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    aim_repo = args.aim_repo.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    sys.path.insert(0, str(aim_repo))
    for relative in (
        "submodules/diff-gaussian-rasterization1",
        "submodules/depth-diff-gaussian-rasterization",
        "lib/pointops",
    ):
        dependency = aim_repo / relative
        if dependency.exists():
            sys.path.insert(0, str(dependency))
    rasterizer_builds = sorted(
        (aim_repo / "submodules/diff-gaussian-rasterization1/build").glob("lib.*")
    )
    if rasterizer_builds:
        sys.path.insert(0, str(rasterizer_builds[-1]))
    simple_knn_builds = sorted(
        (aim_repo / "submodules/simple-knn/build").glob("lib.*")
    )
    if simple_knn_builds:
        sys.path.insert(0, str(simple_knn_builds[-1]))
    pointops_builds = sorted((aim_repo / "lib/pointops/build").glob("lib.*"))
    if pointops_builds:
        sys.path.insert(0, str(pointops_builds[-1]))

    import torch
    from gaussian_renderer import render_mix
    from scene import DeformModel, GaussianModel, Scene
    from utils.system_utils import searchForMaxIteration

    cfg = _load_cfg(model_dir / "cfg_args")
    cfg.model_path = str(model_dir)
    if args.source_path is not None:
        cfg.source_path = str(args.source_path.expanduser().resolve())
    cfg.data_device = "cuda"
    cfg.eval = True

    static_iteration = searchForMaxIteration(model_dir / "point_cloud_start")
    motion_iteration = searchForMaxIteration(model_dir / "point_cloud_motion")
    deform_iteration = searchForMaxIteration(model_dir / "deform")
    available_iteration = min(static_iteration, motion_iteration, deform_iteration)
    iteration = available_iteration if args.iteration < 0 else args.iteration
    if iteration > available_iteration:
        raise ValueError(
            f"Requested iteration {iteration}, but the latest common checkpoint is "
            f"{available_iteration}."
        )

    static_gaussians = GaussianModel(cfg.sh_degree)
    static_scene = Scene(
        cfg,
        gaussians=static_gaussians,
        load_iteration=iteration,
        state="start",
        shuffle=False,
    )
    moving_gaussians = GaussianModel(cfg.sh_degree)
    motion_scene = Scene(
        cfg,
        gaussians=moving_gaussians,
        load_iteration=iteration,
        state="motion",
        shuffle=False,
        init_point_size="small",
    )
    deform = DeformModel(is_blender=cfg.is_blender, is_6dof=cfg.is_6dof)
    deform.load_weights(str(model_dir), iteration=iteration)
    deform.deform.eval()

    cameras = list(motion_scene.getTrainCameras())
    if args.include_test_cameras:
        cameras.extend(motion_scene.getTestCameras())
    cameras.sort(key=lambda camera: (float(camera.fid.item()), camera.image_name))
    if args.max_frames > 0 and len(cameras) > args.max_frames:
        indices = np.linspace(0, len(cameras) - 1, args.max_frames).round().astype(int)
        cameras = [cameras[index] for index in indices]

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else model_dir / f"aim_3dgs_reconstruction_iter_{iteration}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_root = (
        output.with_suffix("").with_name(output.stem + "_frames")
        if args.keep_frames
        else Path(tempfile.mkdtemp(prefix="aim_3dgs_render_"))
    )
    frame_root.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | int | str]] = []
    moving_part_masks, articulated_part_ids = _load_moving_part_masks(
        Path(cfg.source_path)
    )

    with torch.no_grad():
        for frame_index, camera in enumerate(cameras):
            time = camera.fid.reshape(1, 1).expand(moving_gaussians.get_xyz.shape[0], 1)
            displacement, rotation = deform.step(
                moving_gaussians.get_xyz.detach(), time
            )
            prediction = torch.clamp(
                render_mix(
                    camera,
                    moving_gaussians,
                    static_gaussians,
                    pipe,
                    background,
                    displacement,
                    rotation,
                    random_bg_color=False,
                )["render"],
                0.0,
                1.0,
            )
            target = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
            prediction_np = _image_array(prediction)
            target_np = _image_array(target)
            error = np.abs(prediction_np.astype(np.float32) - target_np.astype(np.float32))
            mse = float(np.mean((error / 255.0) ** 2))
            psnr = -10.0 * math.log10(max(mse, 1e-12))
            mae = float(np.mean(error) / 255.0)
            mask = (
                camera.gt_alpha_mask.detach().cpu().numpy().squeeze() >= 0.5
                if camera.gt_alpha_mask is not None
                else np.ones(target_np.shape[:2], dtype=bool)
            )
            masked_error = error[mask] / 255.0
            masked_mse = float(np.mean(masked_error**2))
            masked_psnr = -10.0 * math.log10(max(masked_mse, 1e-12))
            masked_mae = float(np.mean(masked_error))
            frame_metrics: dict[str, float | int | str] = {
                "frame_index": frame_index,
                "image_name": str(camera.image_name),
                "normalized_time": float(camera.fid.item()),
                "psnr_db": psnr,
                "mae": mae,
                "object_psnr_db": masked_psnr,
                "object_mae": masked_mae,
                "object_pixel_ratio": float(mask.mean()),
            }
            moving_mask = moving_part_masks.get(str(camera.image_name))
            if moving_mask is not None:
                if moving_mask.shape != target_np.shape[:2]:
                    raise ValueError(
                        f"Moving-part mask shape {moving_mask.shape} does not match "
                        f"render shape {target_np.shape[:2]} for {camera.image_name}"
                    )
                moving_error = error[moving_mask] / 255.0
                if moving_error.size:
                    moving_mse = float(np.mean(moving_error**2))
                    frame_metrics.update(
                        {
                            "moving_part_psnr_db": -10.0
                            * math.log10(max(moving_mse, 1e-12)),
                            "moving_part_mae": float(np.mean(moving_error)),
                            "moving_part_pixel_ratio": float(moving_mask.mean()),
                        }
                    )
            metrics.append(frame_metrics)
            panel = _make_panel(
                target_np,
                prediction_np,
                error,
                frame_index=frame_index,
                frame_count=len(cameras),
                normalized_time=float(camera.fid.item()),
                psnr=psnr,
                mae=mae,
                object_psnr=masked_psnr,
                object_mae=masked_mae,
            )
            panel.save(frame_root / f"{frame_index:06d}.png")

    _encode_video(frame_root, output, fps=args.fps)
    summary = {
        "model_dir": str(model_dir),
        "source_path": str(cfg.source_path),
        "checkpoint_iteration": iteration,
        "camera_count": len(cameras),
        "included_test_cameras": bool(args.include_test_cameras),
        "fps": float(args.fps),
        "psnr_db_mean": float(np.mean([row["psnr_db"] for row in metrics])),
        "psnr_db_median": float(np.median([row["psnr_db"] for row in metrics])),
        "mae_mean": float(np.mean([row["mae"] for row in metrics])),
        "object_psnr_db_mean": float(
            np.mean([row["object_psnr_db"] for row in metrics])
        ),
        "object_psnr_db_median": float(
            np.median([row["object_psnr_db"] for row in metrics])
        ),
        "object_mae_mean": float(np.mean([row["object_mae"] for row in metrics])),
        "object_pixel_ratio_mean": float(
            np.mean([row["object_pixel_ratio"] for row in metrics])
        ),
        "moving_part_diagnostic": {
            "available": any("moving_part_psnr_db" in row for row in metrics),
            "used_by_aim": False,
            "articulated_part_ids": articulated_part_ids,
            **_optional_metric_summary(metrics, "moving_part_psnr_db", "moving_part_mae"),
        },
        "temporal_quartiles": _temporal_quartile_metrics(metrics),
        "frames": metrics,
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.keep_frames:
        shutil.rmtree(frame_root)
    print(json.dumps({"video": str(output), "metrics": str(metrics_path)}, indent=2))
    return 0


def _load_cfg(path: Path) -> Namespace:
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        raise ValueError(f"Unsupported AiM cfg_args format: {path}")
    return eval(text, {"Namespace": Namespace}, {})  # noqa: S307


def _image_array(tensor) -> np.ndarray:
    return (
        tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .clip(0.0, 1.0)
        .__mul__(255.0)
        .round()
        .astype(np.uint8)
    )


def _load_moving_part_masks(
    source_path: Path,
) -> tuple[dict[str, np.ndarray], list[int]]:
    manifest_path = source_path / "aim_export_manifest.json"
    if not manifest_path.is_file():
        return {}, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostic = ((manifest.get("diagnostics") or {}).get("part_masks") or {})
    articulated_part_ids = [int(value) for value in diagnostic.get("articulated_part_ids", [])]
    if not articulated_part_ids:
        return {}, []

    masks: dict[str, np.ndarray] = {}
    for split_name in ("train", "test"):
        transforms_path = source_path / "motion" / f"transforms_{split_name}.json"
        if not transforms_path.is_file():
            continue
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
        for frame in transforms.get("frames", []):
            relative = frame.get("diagnostic_part_mask_path")
            if not relative:
                continue
            image_name = Path(str(frame["file_path"])).name
            part_ids = np.asarray(
                Image.open(source_path / "motion" / str(relative)).convert("I")
            )
            masks[image_name] = np.isin(part_ids, articulated_part_ids)
    return masks, articulated_part_ids


def _optional_metric_summary(
    rows: list[dict[str, float | int | str]],
    psnr_key: str,
    mae_key: str,
) -> dict[str, float]:
    selected = [row for row in rows if psnr_key in row and mae_key in row]
    if not selected:
        return {}
    return {
        "frame_count": len(selected),
        "psnr_db_mean": float(np.mean([row[psnr_key] for row in selected])),
        "psnr_db_median": float(np.median([row[psnr_key] for row in selected])),
        "mae_mean": float(np.mean([row[mae_key] for row in selected])),
    }


def _make_panel(
    target: np.ndarray,
    prediction: np.ndarray,
    error: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    normalized_time: float,
    psnr: float,
    mae: float,
    object_psnr: float,
    object_mae: float,
) -> Image.Image:
    height, width = target.shape[:2]
    error_intensity = np.clip(error.mean(axis=2) * 4.0, 0.0, 255.0).astype(np.uint8)
    error_rgb = np.stack(
        [
            error_intensity,
            (error_intensity.astype(np.float32) * 0.25).astype(np.uint8),
            np.zeros_like(error_intensity),
        ],
        axis=2,
    )
    header = 58
    panel = Image.new("RGB", (width * 3, height + header), (7, 16, 24))
    panel.paste(Image.fromarray(target), (0, header))
    panel.paste(Image.fromarray(prediction), (width, header))
    panel.paste(Image.fromarray(error_rgb), (width * 2, header))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), "Input RGB", fill="white")
    draw.text((width + 12, 8), "AiM 3DGS render", fill="white")
    draw.text((width * 2 + 12, 8), "Absolute error (4x)", fill="white")
    draw.text(
        (12, 31),
        (
            f"frame {frame_index + 1}/{frame_count}  t={normalized_time:.4f}  "
            f"PSNR={psnr:.2f} dB  object={object_psnr:.2f} dB  "
            f"MAE={mae:.4f}  object={object_mae:.4f}"
        ),
        fill=(183, 215, 232),
    )
    return panel


def _temporal_quartile_metrics(
    metrics: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    """Expose temporal fit imbalance that a single video-wide mean can hide."""
    result: list[dict[str, float | int]] = []
    for quartile, indices in enumerate(np.array_split(np.arange(len(metrics)), 4), start=1):
        rows = [metrics[int(index)] for index in indices]
        row: dict[str, float | int] = {
            "quartile": quartile,
            "frame_start": int(indices[0]),
            "frame_end": int(indices[-1]),
            "psnr_db_mean": float(np.mean([item["psnr_db"] for item in rows])),
            "object_psnr_db_mean": float(
                np.mean([item["object_psnr_db"] for item in rows])
            ),
            "mae_mean": float(np.mean([item["mae"] for item in rows])),
            "object_mae_mean": float(
                np.mean([item["object_mae"] for item in rows])
            ),
        }
        optional = _optional_metric_summary(
            rows, "moving_part_psnr_db", "moving_part_mae"
        )
        if optional:
            row["moving_part_psnr_db_mean"] = optional["psnr_db_mean"]
            row["moving_part_mae_mean"] = optional["mae_mean"]
        result.append(row)
    return result


def _encode_video(frame_root: Path, output: Path, *, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_root / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
