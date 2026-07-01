#!/usr/bin/env python3
"""Seed and propagate cardboard-box part masks with SAM2.

The script creates a two-part indexed keyframe mask from prompt bboxes/points,
then runs the existing `propagate-episode-masks --backend sam2-video` path and
exports overlay previews for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PART_COLORS = {
    1: np.array([82, 220, 105], dtype=np.float32),
    2: np.array([178, 95, 238], dtype=np.float32),
}


@dataclass(frozen=True)
class PartPrompt:
    part_id: int
    bbox: tuple[float, float, float, float]
    pos: tuple[tuple[float, float], ...]
    neg: tuple[tuple[float, float], ...] = ()
    selection: str = "constrained"


PROMPTS: dict[str, dict[int, tuple[PartPrompt, ...]]] = {
    "cardboardbox01_o": {
        80: (
            PartPrompt(1, (250, 185, 426, 286), ((306, 232), (382, 246), (415, 220)), ((330, 145), (480, 320)), "constrained"),
            PartPrompt(2, (286, 68, 445, 205), ((360, 104), (425, 136), (426, 84)), ((330, 156), (355, 236)), "constrained"),
        ),
        0: (
            PartPrompt(1, (315, 232, 532, 340), ((365, 292), (498, 286)), ((420, 205),), "constrained"),
            PartPrompt(2, (325, 168, 532, 268), ((415, 218), (505, 228)), ((405, 300),), "constrained"),
        ),
    },
    "cardboardbox02_o": {
        80: (
            PartPrompt(1, (140, 142, 420, 256), ((215, 220), (352, 220), (405, 214)), ((245, 92), (520, 310)), "constrained"),
            PartPrompt(2, (180, 76, 420, 178), ((284, 118), (376, 120), (410, 132)), ((186, 92), (330, 220)), "constrained"),
        ),
        0: (
            PartPrompt(1, (315, 232, 532, 340), ((365, 292), (498, 286)), ((420, 205),), "constrained"),
            PartPrompt(2, (325, 168, 532, 268), ((415, 218), (505, 228)), ((405, 300),), "constrained"),
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "objects",
        nargs="*",
        default=["cardboardbox01_o", "cardboardbox02_o"],
        help="Recording ids under --recordings-root.",
    )
    parser.add_argument("--recordings-root", type=Path, default=Path("outputs/real_recordings"))
    parser.add_argument("--reference-frame", type=int, default=80)
    parser.add_argument("--mask-name", default="manual_partmask_sam2_keyframe")
    parser.add_argument("--sam2-root", type=Path, default=Path("sam2"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--sam2-part-mode", default="independent", choices=["independent", "joint"])
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--skip-propagation", action="store_true")
    args = parser.parse_args()

    for object_id in args.objects:
        build_for_object(object_id, args)
    return 0


def build_for_object(object_id: str, args: argparse.Namespace) -> None:
    if object_id not in PROMPTS:
        raise ValueError(f"No prompt template for {object_id}")
    prompts_by_frame = PROMPTS[object_id]
    if args.reference_frame not in prompts_by_frame:
        raise ValueError(f"No prompts for {object_id} frame {args.reference_frame}")

    root = args.recordings_root / object_id
    episode_path = root / "episode.json"
    seed_episode_path = root / f"episode.{args.mask_name}.seed.json"
    output_episode_path = root / f"episode.{args.mask_name}.json"
    output_dir = root / "assets" / "masks" / args.mask_name
    seed_mask_path = output_dir / "seed" / "view_0" / f"frame_{args.reference_frame:04d}_mask.png"

    episode = json.loads(episode_path.read_text())
    seed_mask = load_preferred_seed(root, object_id, args.reference_frame)
    if seed_mask is None:
        seed_mask = predict_seed_mask(root, object_id, args.reference_frame, prompts_by_frame[args.reference_frame], args)
    seed_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(seed_mask.astype(np.uint16), mode="I;16").save(seed_mask_path)

    frame = episode["frames"][args.reference_frame]
    rel_seed = seed_mask_path.relative_to(root).as_posix()
    frame["part_mask_paths_by_view"] = [rel_seed]
    frame["part_mask_path"] = rel_seed
    episode.setdefault("metadata", {})["part_segmentation"] = {
        "provider": "sam2-keyframe-prompts",
        "mask_kind": "part",
        "mask_encoding": "indexed-mask-u16",
        "part_ids": [1, 2],
        "parts": [
            {"part_id": 1, "name": "base", "role": "base"},
            {"part_id": 2, "name": "lid", "role": "moving"},
        ],
        "reference_frame": int(args.reference_frame),
    }
    seed_episode_path.write_text(json.dumps(episode, indent=2), encoding="utf-8")
    write_seed_debug(root, object_id, args.reference_frame, seed_mask, output_dir)

    if not args.skip_propagation:
        run_propagation(seed_episode_path, output_episode_path, output_dir, args)
    if output_episode_path.exists():
        write_overlay_preview(output_episode_path, output_dir, args.fps)
    print(f"{object_id}: seed={seed_mask_path} episode={output_episode_path}")


def load_preferred_seed(root: Path, object_id: str, frame_index: int) -> np.ndarray | None:
    paths: list[Path]
    if object_id == "cardboardbox01_o":
        paths = [
            root / "assets/masks/live_sam2_video_f0000_v0_1781098040/view_0" / f"frame_{frame_index:04d}_mask.png",
        ]
    elif object_id == "cardboardbox02_o":
        paths = [
            root
            / "assets/masks/manual_partmask_test_independent/seed/view_0"
            / f"frame_{frame_index:04d}_mask.png",
            root
            / "assets/masks/manual_partmask_test/live_sam2_video_f0080_v0_1781021853/view_0"
            / f"frame_{frame_index:04d}_mask.png",
        ]
    else:
        paths = []
    for path in paths:
        if path.exists():
            mask = np.asarray(Image.open(path), dtype=np.uint16)
            cleaned = np.zeros(mask.shape, dtype=np.uint16)
            cleaned[mask == 1] = 1
            cleaned[mask == 2] = 2
            if int((cleaned > 0).sum()) > 0:
                return cleaned
    return None


def predict_seed_mask(
    root: Path,
    object_id: str,
    frame_index: int,
    prompts: tuple[PartPrompt, ...],
    args: argparse.Namespace,
) -> np.ndarray:
    import torch

    sam2_root = args.sam2_root.expanduser().resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = resolve_device(torch, args.sam2_device)
    model = build_sam2(
        args.sam2_config,
        str(args.sam2_checkpoint.expanduser().resolve()),
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    image_path = root / "assets" / "view_0" / f"frame_{frame_index:04d}_rgb.png"
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    predictor.set_image(image)
    seed = np.zeros(image.shape[:2], dtype=np.uint16)
    debug: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt in prompts:
            point_coords, point_labels, box = prompt_to_sam_inputs(prompt)
            masks, ious, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
            areas = [int(mask.sum()) for mask in masks]
            index = select_mask(prompt, masks, ious, areas)
            mask = np.asarray(masks[index], dtype=bool)
            mask &= padded_bbox_mask(mask.shape, prompt.bbox, padding=0)
            mask = keep_components_touching_points(mask, prompt.pos)
            seed[(seed == 0) & mask] = int(prompt.part_id)
            seed[mask & (seed == int(prompt.part_id))] = int(prompt.part_id)
            debug.append(
                {
                    "object_id": object_id,
                    "frame_index": frame_index,
                    "part_id": int(prompt.part_id),
                    "bbox": prompt.bbox,
                    "pos": prompt.pos,
                    "neg": prompt.neg,
                    "selection": prompt.selection,
                    "choice": int(index),
                    "areas": areas,
                    "ious": [float(v) for v in np.asarray(ious).ravel()],
                    "pixels": int(mask.sum()),
                }
            )
    debug_path = root / "assets" / "masks" / args.mask_name / "seed" / "prompt_debug.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return seed


def resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested SAM2 MPS, but torch.backends.mps.is_available() is false.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested SAM2 CUDA, but torch.cuda.is_available() is false.")
    return requested


def prompt_to_sam_inputs(prompt: PartPrompt):
    points = [[float(x), float(y)] for x, y in prompt.pos] + [[float(x), float(y)] for x, y in prompt.neg]
    labels = [1] * len(prompt.pos) + [0] * len(prompt.neg)
    return (
        np.asarray(points, dtype=np.float32) if points else None,
        np.asarray(labels, dtype=np.int32) if labels else None,
        np.asarray(prompt.bbox, dtype=np.float32),
    )


def select_mask(prompt: PartPrompt, masks: np.ndarray, ious, areas: list[int]) -> int:
    selection = prompt.selection
    if selection == "largest":
        return int(np.argmax(np.asarray(areas)))
    if selection == "smallest":
        return int(np.argmin(np.asarray(areas)))
    if selection == "constrained":
        return select_constrained_mask(prompt, masks, ious, areas)
    return int(np.argmax(np.asarray(ious).ravel())) if len(areas) else 0


def select_constrained_mask(prompt: PartPrompt, masks: np.ndarray, ious, areas: list[int]) -> int:
    x1, y1, x2, y2 = [int(round(value)) for value in prompt.bbox]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    best_index = 0
    best_score = -1.0e9
    iou_values = np.asarray(ious).ravel() if len(areas) else np.zeros((0,), dtype=np.float32)
    for index, mask in enumerate(masks):
        mask_bool = np.asarray(mask, dtype=bool)
        area = max(1, int(mask_bool.sum()))
        bbox_pixels = int(mask_bool[y1:y2, x1:x2].sum())
        bbox_precision = bbox_pixels / area
        bbox_recall = bbox_pixels / bbox_area
        area_ratio = area / bbox_area
        pos_hits = sum(1 for x, y in prompt.pos if _mask_value(mask_bool, x, y))
        neg_hits = sum(1 for x, y in prompt.neg if _mask_value(mask_bool, x, y))
        iou_hint = float(iou_values[index]) if index < len(iou_values) else 0.0
        # Strongly penalize masks that flood outside the prompt box. This is the
        # failure mode that makes SAM choose the tabletop as the "base" part.
        area_penalty = max(0.0, area_ratio - 1.35) * 3.0
        score = (
            2.0 * pos_hits
            - 3.0 * neg_hits
            + 1.7 * bbox_precision
            + 0.8 * min(bbox_recall, 1.0)
            + 0.1 * iou_hint
            - area_penalty
        )
        if score > best_score:
            best_score = score
            best_index = index
    return int(best_index)


def _mask_value(mask: np.ndarray, x: float, y: float) -> bool:
    yy = min(max(0, int(round(y))), mask.shape[0] - 1)
    xx = min(max(0, int(round(x))), mask.shape[1] - 1)
    return bool(mask[yy, xx])


def padded_bbox_mask(shape: tuple[int, int], bbox: tuple[float, float, float, float], padding: int) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def keep_components_touching_points(mask: np.ndarray, points: tuple[tuple[float, float], ...]) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    seeds: list[tuple[int, int]] = []
    for x, y in points:
        yy = min(max(0, int(round(y))), height - 1)
        xx = min(max(0, int(round(x))), width - 1)
        if mask[yy, xx]:
            seeds.append((yy, xx))
            continue
        # SAM boundaries are sometimes a few pixels away from the clicked point.
        y0, y1 = max(0, yy - 5), min(height, yy + 6)
        x0, x1 = max(0, xx - 5), min(width, xx + 6)
        local = np.argwhere(mask[y0:y1, x0:x1])
        if len(local):
            ly, lx = local[0]
            seeds.append((int(y0 + ly), int(x0 + lx)))
    if not seeds:
        return mask
    output = np.zeros(mask.shape, dtype=bool)
    for seed in seeds:
        if visited[seed]:
            continue
        stack = [seed]
        component: list[tuple[int, int]] = []
        visited[seed] = True
        while stack:
            y, x = stack.pop()
            if not mask[y, x]:
                continue
            component.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if component:
            ys, xs = zip(*component)
            output[np.asarray(ys), np.asarray(xs)] = True
    return output


def run_propagation(seed_episode: Path, output_episode: Path, output_dir: Path, args: argparse.Namespace) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        "-m",
        "rgbd_urdf_mvp",
        "propagate-episode-masks",
        str(seed_episode),
        "--backend",
        "sam2-video",
        "--mask-kind",
        "part",
        "--reference-frame",
        str(args.reference_frame),
        "--view-indices",
        "0",
        "--sam2-root",
        str(args.sam2_root),
        "--sam2-config",
        args.sam2_config,
        "--sam2-checkpoint",
        str(args.sam2_checkpoint),
        "--sam2-device",
        args.sam2_device,
        "--sam2-part-mode",
        args.sam2_part_mode,
        "--output-episode",
        str(output_episode),
        "--output-dir",
        str(output_dir / "propagated"),
        "--force",
    ]
    subprocess.run(cmd, check=True, env=env)


def write_seed_debug(root: Path, object_id: str, frame_index: int, labels: np.ndarray, output_dir: Path) -> None:
    rgb = np.asarray(Image.open(root / "assets" / "view_0" / f"frame_{frame_index:04d}_rgb.png").convert("RGB"))
    overlay_mask(rgb, labels, f"{object_id} seed f{frame_index:04d}").save(output_dir / "seed_overlay.jpg", quality=92)


def write_overlay_preview(episode_path: Path, output_dir: Path, fps: float) -> None:
    episode = json.loads(episode_path.read_text())
    root = episode_path.parent
    preview_dir = output_dir / "_overlay_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(episode["frames"]):
        rgb_path = root / frame["rgb_paths_by_view"][0]
        mask_path = root / frame["part_mask_paths_by_view"][0]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        labels = np.asarray(Image.open(mask_path), dtype=np.uint16)
        overlay_mask(rgb, labels, f"{root.name} f{frame_index:04d}").save(preview_dir / f"frame_{frame_index:04d}.jpg", quality=90)
    run_ffmpeg(preview_dir, output_dir, fps)
    build_contact_sheet(preview_dir, output_dir / "overlay_contact_sheet.jpg")


def overlay_mask(rgb: np.ndarray, labels: np.ndarray, title: str) -> Image.Image:
    arr = rgb.astype(np.float32)
    for part_id, color in PART_COLORS.items():
        sel = labels == part_id
        arr[sel] = 0.58 * arr[sel] + 0.42 * color
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 230, 28), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return image


def run_ffmpeg(preview_dir: Path, output_dir: Path, fps: float) -> None:
    mp4 = output_dir / "overlay_preview.mp4"
    gif = output_dir / "overlay_preview.gif"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(preview_dir / "frame_%04d.jpg"),
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4), "-vf", "fps=6,scale=480:-1:flags=lanczos", str(gif)],
        check=True,
    )


def build_contact_sheet(preview_dir: Path, output_path: Path) -> None:
    paths = sorted(preview_dir.glob("frame_*.jpg"))
    if not paths:
        return
    selected = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, 12).round()]
    thumbs = [Image.open(path).convert("RGB").resize((240, 180)) for path in selected]
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 240, rows * 180), (18, 20, 24))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 240, (idx // cols) * 180))
    sheet.save(output_path, quality=92)


if __name__ == "__main__":
    raise SystemExit(main())
