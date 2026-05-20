from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.serialization import load_episode, load_json, save_json
from ..sim.mujoco_recorder import _read_png_rgb
from .pointcloud_fusion import _load_depth_u16


DEFAULT_HUNYUAN_VIEW_ORDER = ("front", "left", "right", "back")


@dataclass(slots=True)
class GenerationImagePreparationConfig:
    episode_path: str | Path
    output_dir: str | Path | None = None
    frame_index: int = 0
    view_indices: Sequence[int] | None = None
    image_views: Sequence[str] | None = None
    mask_source: str = "auto"
    background: str = "transparent"
    padding_ratio: float = 0.15
    min_mask_pixels: int = 16


class GenerationImagePreparer:
    """Prepare object-centric images for feedforward 3D generation.

    The current implementation uses prior masks already present in simulator
    recordings. Future real segmentation modules should emit the same masked
    image manifest contract so Hunyuan3D/remote generation does not need to
    know where the masks came from.
    """

    def prepare(self, config: GenerationImagePreparationConfig) -> Path:
        episode_path = Path(config.episode_path).expanduser().resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)
        if not episode.frames:
            raise ValueError("Episode has no frames.")
        frame_index = min(max(0, int(config.frame_index)), len(episode.frames) - 1)
        frame = episode.frames[frame_index]

        rgb_paths = _resolve_view_rgb_paths(frame, episode_root)
        mask_paths, mask_kind = _resolve_mask_paths(frame, episode_root, str(config.mask_source))
        if not rgb_paths:
            raise ValueError("Selected frame has no RGB paths.")
        if not mask_paths:
            raise ValueError(
                "Selected frame has no usable object/part masks. Re-record with --segmentation-masks "
                "or --part-segmentation-masks, or add a real segmentation provider."
            )

        view_indices = (
            [int(index) for index in config.view_indices]
            if config.view_indices is not None
            else list(range(min(len(rgb_paths), len(mask_paths))))
        )
        image_views = _resolve_image_views(config.image_views, len(view_indices))
        output_dir = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir is not None
            else episode_root / "generation_images"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, Any]] = []
        for output_index, view_index in enumerate(view_indices):
            if view_index < 0 or view_index >= len(rgb_paths) or view_index >= len(mask_paths):
                raise ValueError(f"View index {view_index} is out of range for RGB/mask paths.")
            image_info = self._prepare_view(
                rgb_path=rgb_paths[view_index],
                mask_path=mask_paths[view_index],
                output_path=output_dir / f"{image_views[output_index]}.png",
                background=str(config.background),
                padding_ratio=float(config.padding_ratio),
                min_mask_pixels=max(1, int(config.min_mask_pixels)),
            )
            images.append(
                {
                    **image_info,
                    "view": image_views[output_index],
                    "view_index": int(view_index),
                    "source_rgb_path": str(rgb_paths[view_index]),
                    "source_mask_path": str(mask_paths[view_index]),
                    "mask_source": mask_kind,
                }
            )

        manifest_path = output_dir / "generation_images.json"
        save_json(
            {
                "source": "generation-image-preparation",
                "mask_provider": "recorded-prior-mask",
                "episode_path": str(episode_path),
                "object_instance_id": episode.object_instance_id,
                "category": episode.category,
                "frame_index": frame_index,
                "timestamp_s": float(frame.timestamp_s),
                "background": str(config.background),
                "padding_ratio": float(config.padding_ratio),
                "image_paths": [item["path"] for item in images],
                "image_views": [item["view"] for item in images],
                "images": images,
                "contract": {
                    "image_paths": "Clean object-centric PNGs for Hunyuan3D.",
                    "image_views": "Named Hunyuan3D views matching image_paths order.",
                    "future_segmentation": "Replace mask_provider while keeping this manifest schema.",
                },
            },
            manifest_path,
        )
        return manifest_path

    def _prepare_view(
        self,
        rgb_path: Path,
        mask_path: Path,
        output_path: Path,
        background: str,
        padding_ratio: float,
        min_mask_pixels: int,
    ) -> dict[str, Any]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Generation image preparation requires Pillow and NumPy.") from exc

        rgb = _load_rgb(rgb_path)
        mask_u16 = _load_depth_u16(mask_path)
        mask = np.asarray(mask_u16, dtype=np.uint16) > 0
        if int(mask.sum()) < min_mask_pixels:
            raise ValueError(f"Mask has too few foreground pixels: {mask_path}")
        height, width = mask.shape
        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise ValueError(f"RGB/mask shape mismatch: {rgb_path} {rgb.shape} vs {mask_path} {mask.shape}")

        left, top, right, bottom = _padded_square_bbox(mask, padding_ratio)
        rgb_crop = rgb[top:bottom, left:right]
        mask_crop = mask[top:bottom, left:right]

        if background == "transparent":
            alpha = (mask_crop.astype(np.uint8) * 255)[:, :, None]
            output = np.concatenate([rgb_crop, alpha], axis=2)
            image = Image.fromarray(output, mode="RGBA")
        elif background in {"white", "black"}:
            fill = 255 if background == "white" else 0
            output = np.full_like(rgb_crop, fill)
            output[mask_crop] = rgb_crop[mask_crop]
            image = Image.fromarray(output, mode="RGB")
        elif background == "original":
            image = Image.fromarray(rgb_crop, mode="RGB")
        else:
            raise ValueError("background must be one of: transparent, white, black, original")

        image.save(output_path)
        return {
            "path": str(output_path),
            "bbox_xyxy": [int(left), int(top), int(right), int(bottom)],
            "foreground_pixel_count": int(mask.sum()),
            "output_size": [int(image.width), int(image.height)],
        }


def load_generation_image_manifest(path: str | Path) -> tuple[list[Path], list[str]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = load_json(manifest_path)
    image_paths_raw = payload.get("image_paths")
    image_views_raw = payload.get("image_views")
    if not isinstance(image_paths_raw, list) or not image_paths_raw:
        raise ValueError(f"Generation image manifest has no image_paths: {manifest_path}")
    if not isinstance(image_views_raw, list) or len(image_views_raw) != len(image_paths_raw):
        raise ValueError(f"Generation image manifest image_views must match image_paths: {manifest_path}")
    image_paths: list[Path] = []
    for raw_path in image_paths_raw:
        image_path = Path(str(raw_path)).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_paths.append(image_path.resolve())
    return image_paths, [str(view) for view in image_views_raw]


def _resolve_view_rgb_paths(frame: Any, episode_root: Path) -> list[Path]:
    if frame.rgb_paths_by_view:
        return [episode_root / path for path in frame.rgb_paths_by_view]
    return [episode_root / frame.rgb_path]


def _resolve_mask_paths(frame: Any, episode_root: Path, mask_source: str) -> tuple[list[Path], str]:
    if mask_source not in {"auto", "object", "part"}:
        raise ValueError("mask_source must be one of: auto, object, part")
    object_paths = [episode_root / path for path in frame.mask_paths_by_view]
    if not object_paths and frame.mask_path:
        object_paths = [episode_root / frame.mask_path]
    part_paths = [episode_root / path for path in frame.part_mask_paths_by_view]
    if not part_paths and frame.part_mask_path:
        part_paths = [episode_root / frame.part_mask_path]
    if mask_source == "object":
        return object_paths, "object-mask"
    if mask_source == "part":
        return part_paths, "part-mask-union"
    if object_paths:
        return object_paths, "object-mask"
    return part_paths, "part-mask-union"


def _resolve_image_views(image_views: Sequence[str] | None, count: int) -> list[str]:
    if image_views is not None:
        views = [str(view) for view in image_views]
        if len(views) != count:
            raise ValueError(f"image_views length ({len(views)}) must match selected view count ({count}).")
        return views
    if count > len(DEFAULT_HUNYUAN_VIEW_ORDER):
        raise ValueError(f"At most {len(DEFAULT_HUNYUAN_VIEW_ORDER)} views are supported by the Hunyuan3D client.")
    return list(DEFAULT_HUNYUAN_VIEW_ORDER[:count])


def _load_rgb(path: Path):
    if path.suffix.lower() == ".png":
        return _read_png_rgb(path)
    if path.suffix.lower() == ".ppm":
        from ..sim.mujoco_recorder import _read_ppm_rgb

        return _read_ppm_rgb(path)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RGB loading requires Pillow and NumPy.") from exc
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _padded_square_bbox(mask: Any, padding_ratio: float) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(mask)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    width = right - left
    height = bottom - top
    side = max(width, height)
    side = int(round(side * (1.0 + 2.0 * max(0.0, padding_ratio))))
    side = max(1, side)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    image_height, image_width = mask.shape
    left = int(round(center_x - 0.5 * side))
    top = int(round(center_y - 0.5 * side))
    left = max(0, min(left, max(0, image_width - side)))
    top = max(0, min(top, max(0, image_height - side)))
    right = min(image_width, left + side)
    bottom = min(image_height, top + side)
    return left, top, right, bottom
