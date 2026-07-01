#!/usr/bin/env python3
"""Build cleaned manual part masks for the two cardboard-box real recordings.

This is intentionally conservative: it reuses the best previous propagation
attempts as seeds, removes hand-colored pixels, drops tiny components, and
writes a fresh episode JSON that points at the cleaned indexed masks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PART_COLORS = {
    1: np.array([82, 220, 105], dtype=np.float32),
    2: np.array([178, 95, 238], dtype=np.float32),
}


@dataclass(frozen=True)
class RecordingSpec:
    object_id: str
    mode: str


SPECS = [
    RecordingSpec("cardboardbox01_o", "box01"),
    RecordingSpec("cardboardbox02_o", "box02"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("outputs/real_recordings"),
        help="Root containing cardboardbox01_o/cardboardbox02_o.",
    )
    parser.add_argument("--mask-name", default="manual_partmask_offline")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for spec in SPECS:
        process_recording(args.recordings_root / spec.object_id, spec, args.mask_name, args.fps, args.dry_run)
    return 0


def process_recording(root: Path, spec: RecordingSpec, mask_name: str, fps: float, dry_run: bool) -> None:
    episode_path = root / "episode.json"
    episode = json.loads(episode_path.read_text())
    output_dir = root / "assets" / "masks" / mask_name / "view_0"
    preview_dir = root / "assets" / "masks" / mask_name / "_preview_frames"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    per_frame_stats = []
    overlay_frames = []
    for frame_index, frame in enumerate(episode["frames"]):
        rgb_path = root / frame["rgb_paths_by_view"][0]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        labels = build_labels(root, spec, frame_index, rgb)
        labels = clean_labels(labels, rgb)
        labels = repair_with_previous(labels, per_frame_stats[-1]["labels"] if per_frame_stats else None)

        rel_mask = Path("assets") / "masks" / mask_name / "view_0" / f"frame_{frame_index:04d}_mask.png"
        if not dry_run:
            Image.fromarray(labels.astype(np.uint16), mode="I;16").save(root / rel_mask)
        frame["part_mask_paths_by_view"] = [str(rel_mask)]
        frame["part_mask_path"] = str(rel_mask)

        stats = {
            "frame_index": frame_index,
            "part_pixels": {str(part): int(np.count_nonzero(labels == part)) for part in PART_COLORS},
            "labels": labels,
        }
        per_frame_stats.append(stats)

        if not dry_run:
            overlay = overlay_mask(rgb, labels, f"{spec.object_id} f{frame_index:04d}")
            overlay_path = preview_dir / f"frame_{frame_index:04d}.jpg"
            overlay.save(overlay_path, quality=90)
            overlay_frames.append(overlay_path)

    for stats in per_frame_stats:
        stats.pop("labels", None)

    episode.setdefault("metadata", {})["part_segmentation"] = {
        "provider": "manual-cardboardbox-offline-cleanup",
        "mask_kind": "part",
        "mask_encoding": "indexed-mask-u16",
        "part_ids": [1, 2],
        "parts": [
            {"id": 1, "name": "base"},
            {"id": 2, "name": "lid"},
        ],
        "notes": "Cleaned from prior manual/SAM propagation with component filtering and hand-color suppression.",
    }
    episode.setdefault("metadata", {})["mask_provider"] = {
        "provider": "manual-cardboardbox-offline-cleanup",
        "mask_kind": "part",
    }
    output_episode = root / f"episode.{mask_name}.json"
    summary_path = root / "assets" / "masks" / mask_name / "mask_summary.json"
    if not dry_run:
        output_episode.write_text(json.dumps(episode, indent=2), encoding="utf-8")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"object_id": spec.object_id, "frames": per_frame_stats}, indent=2), encoding="utf-8")
        write_preview_artifacts(root, mask_name, preview_dir, fps)
    print(f"{spec.object_id}: wrote {output_episode}")


def build_labels(root: Path, spec: RecordingSpec, frame_index: int, rgb: np.ndarray) -> np.ndarray:
    if spec.mode == "box01":
        labels = load_mask(
            root / "assets/masks/live_sam2_video_f0000_v0_1781098040/view_0" / f"frame_{frame_index:04d}_mask.png",
            rgb.shape[:2],
        )
        if not np.any(labels):
            labels = load_mask(
                root / "assets/masks/part_depthmotion_handcut/view_0" / f"frame_{frame_index:04d}_mask.png",
                rgb.shape[:2],
            )
        return remap_to_two_parts(labels)

    joint = load_mask(
        root / "assets/masks/manual_partmask_test/live_sam2_video_f0080_v0_1781021853/view_0"
        / f"frame_{frame_index:04d}_mask.png",
        rgb.shape[:2],
    )
    independent = load_mask(
        root / "assets/masks/manual_partmask_test_independent/live_sam2_video_f0080_v0_1781022722/view_0"
        / f"frame_{frame_index:04d}_mask.png",
        rgb.shape[:2],
    )
    labels = np.zeros(rgb.shape[:2], dtype=np.uint16)
    labels[independent == 1] = 1
    labels[joint == 1] = 1
    labels[independent == 2] = 2
    labels[joint == 2] = 2
    return labels


def remap_to_two_parts(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.uint16)
    out[labels == 1] = 1
    out[labels == 2] = 2
    return out


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint16)
    arr = np.asarray(Image.open(path), dtype=np.uint16)
    if arr.shape != shape:
        arr = np.asarray(Image.fromarray(arr).resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.uint16)
    return arr


def clean_labels(labels: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(labels, dtype=np.uint16)
    skin = skin_mask(rgb)
    for part_id in [1, 2]:
        binary = labels == part_id
        binary &= ~skin
        binary = ndimage.binary_opening(binary, structure=np.ones((2, 2), dtype=bool))
        binary = keep_components(binary, max_components=3 if part_id == 1 else 2, min_area=80)
        binary = ndimage.binary_closing(binary, structure=np.ones((5, 5), dtype=bool))
        binary = ndimage.binary_fill_holes(binary)
        cleaned[binary] = part_id
    cleaned[(cleaned == 1) & (labels == 2)] = 2
    return cleaned


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    # Tuned for these recordings: catches arms/hands while sparing the off-white
    # cardboard and most of the dark brown lid interior.
    bright_skin = (r > 95) & (g > 45) & (b > 30) & ((r - g) > 10) & ((r - b) > 35) & (g > b)
    dark_skin = (r > 70) & (g > 35) & (b > 20) & ((r - g) > 12) & ((r - b) > 28) & (g > b) & (r < 150)
    return bright_skin | dark_skin


def keep_components(binary: np.ndarray, max_components: int, min_area: int) -> np.ndarray:
    labeled, count = ndimage.label(binary)
    if count == 0:
        return binary
    areas = np.bincount(labeled.ravel())
    keep = []
    for component_id in np.argsort(areas[1:])[::-1] + 1:
        if areas[component_id] < min_area:
            continue
        keep.append(component_id)
        if len(keep) >= max_components:
            break
    if not keep:
        return np.zeros_like(binary, dtype=bool)
    return np.isin(labeled, keep)


def repair_with_previous(labels: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return labels
    repaired = labels.copy()
    for part_id in [1, 2]:
        pixels = int(np.count_nonzero(repaired == part_id))
        prev_pixels = int(np.count_nonzero(previous == part_id))
        if pixels < 150 and prev_pixels > 800:
            repaired[previous == part_id] = part_id
    return repaired


def overlay_mask(rgb: np.ndarray, labels: np.ndarray, title: str) -> Image.Image:
    arr = rgb.astype(np.float32)
    for part_id, color in PART_COLORS.items():
        sel = labels == part_id
        arr[sel] = 0.58 * arr[sel] + 0.42 * color
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 220, 28), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return image


def write_preview_artifacts(root: Path, mask_name: str, preview_dir: Path, fps: float) -> None:
    mask_root = root / "assets" / "masks" / mask_name
    mp4_path = mask_root / "overlay_preview.mp4"
    gif_path = mask_root / "overlay_preview.gif"
    contact_path = mask_root / "overlay_contact_sheet.jpg"
    pattern = str(preview_dir / "frame_%04d.jpg")
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
            pattern,
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=6,scale=480:-1:flags=lanczos",
            str(gif_path),
        ],
        check=True,
    )
    build_contact_sheet(preview_dir.glob("frame_*.jpg"), contact_path)


def build_contact_sheet(paths: Iterable[Path], output_path: Path) -> None:
    selected = []
    all_paths = sorted(paths)
    if not all_paths:
        return
    for idx in np.linspace(0, len(all_paths) - 1, 12).round().astype(int):
        selected.append(all_paths[int(idx)])
    thumbs = [Image.open(path).convert("RGB").resize((240, 180)) for path in selected]
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 240, rows * 180), (18, 20, 24))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 240, (index // cols) * 180))
    sheet.save(output_path, quality=92)


if __name__ == "__main__":
    raise SystemExit(main())
