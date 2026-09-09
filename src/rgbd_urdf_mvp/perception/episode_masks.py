from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..core.serialization import load_episode, load_json, save_json
from .generation_preprocess import _resolve_view_rgb_paths
from .pointcloud_fusion import _load_depth_u16


@dataclass(frozen=True, slots=True)
class FrameViewInput:
    frame_index: int
    view_index: int
    rgb_path: Path
    output_mask_path: Path


@dataclass(frozen=True, slots=True)
class MaskProviderResult:
    mask_path: Path
    metadata: dict[str, Any]


class EpisodeMaskProvider(Protocol):
    provider_name: str

    def predict(self, frame_view: FrameViewInput) -> MaskProviderResult:
        """Write or locate one mask for a frame/view pair."""


@dataclass(slots=True)
class EpisodeMaskWriteConfig:
    episode_path: Path
    output_episode_path: Path | None = None
    output_dir: Path | None = None
    provider: str = "mask-dir"
    mask_dir: Path | None = None
    command: str | None = None
    sam2_root: Path | None = None
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: Path | None = None
    sam2_device: str = "auto"
    sam2_prompt_mode: str = "full-image-box"
    sam2_box_xyxy: Sequence[float] | None = None
    sam2_center_box_scale: float = 0.75
    sam2_multimask: bool = True
    sam2_mask_selection: str = "best-iou"
    sam2_mask_index: int | None = None
    sam3_checkpoint: Path | None = None
    sam3_device: str = "auto"
    sam3_prompt_mode: str = "full-image-box"
    sam3_box_xyxy: Sequence[float] | None = None
    sam3_center_box_scale: float = 0.75
    sam3_mask_selection: str = "largest"
    sam3_mask_index: int | None = None
    sam3_conf: float = 0.05
    mask_kind: str = "object"
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    view_indices: Sequence[int] | None = None
    threshold: int = 0
    force: bool = False


@dataclass(slots=True)
class EpisodeMaskBatchConfig:
    manifest_path: Path
    jobs: int = 1
    provider: str = "sam2"
    output_root: Path | None = None
    sam2_root: Path | None = None
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: Path | None = None
    sam2_device: str = "auto"
    sam2_prompt_mode: str = "full-image-box"
    sam2_box_xyxy: Sequence[float] | None = None
    sam2_center_box_scale: float = 0.75
    sam2_multimask: bool = True
    sam2_mask_selection: str = "best-iou"
    sam2_mask_index: int | None = None
    sam3_checkpoint: Path | None = None
    sam3_device: str = "auto"
    sam3_prompt_mode: str = "full-image-box"
    sam3_box_xyxy: Sequence[float] | None = None
    sam3_center_box_scale: float = 0.75
    sam3_mask_selection: str = "largest"
    sam3_mask_index: int | None = None
    sam3_conf: float = 0.05
    mask_kind: str = "object"
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    view_indices: Sequence[int] | None = None
    threshold: int = 0
    force: bool = False


@dataclass(slots=True)
class EpisodeMaskPropagationConfig:
    episode_path: Path
    output_episode_path: Path | None = None
    output_dir: Path | None = None
    backend: str = "sam2-video"
    mask_kind: str = "object"
    sam2_root: Path | None = None
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: Path | None = None
    sam2_device: str = "auto"
    sam2_part_mode: str = "independent"
    sam2_offload_video_to_cpu: bool = False
    sam2_offload_state_to_cpu: bool = False
    reference_frame: int = 0
    start_frame: int = 0
    end_frame: int | None = None
    frame_stride: int = 1
    view_indices: Sequence[int] | None = None
    device: str = "auto"
    cotracker_repo: Path | None = None
    cotracker_checkpoint: Path | None = None
    cotracker_model: str = "cotracker3_offline"
    seed_stride_px: int = 12
    max_tracks_per_view: int = 512
    visibility_threshold: float = 0.5
    mask_radius_px: int = 8
    threshold: int = 0
    force: bool = False


@dataclass(slots=True)
class EpisodeMaskEvaluationConfig:
    predicted_episode_path: Path
    reference_episode_path: Path
    output_json: Path | None = None
    mask_kind: str = "object"
    threshold: int = 0


class MaskDirectoryProvider:
    provider_name = "mask-dir"

    def __init__(self, mask_dir: Path, threshold: int = 0) -> None:
        self.mask_dir = mask_dir.expanduser().resolve()
        self.threshold = int(threshold)

    def predict(self, frame_view: FrameViewInput) -> MaskProviderResult:
        source_path = self._find_mask(frame_view.frame_index, frame_view.view_index)
        _copy_mask_as_u16(source_path, frame_view.output_mask_path, threshold=self.threshold)
        return MaskProviderResult(
            mask_path=frame_view.output_mask_path,
            metadata={"source_mask_path": str(source_path)},
        )

    def _find_mask(self, frame_index: int, view_index: int) -> Path:
        names = [
            f"frame_{frame_index:04d}_view_{view_index}_mask.png",
            f"view_{view_index}/frame_{frame_index:04d}_mask.png",
            f"frame_{frame_index:04d}_mask.png" if view_index == 0 else "",
        ]
        for name in names:
            if not name:
                continue
            path = self.mask_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No mask found for frame {frame_index} view {view_index} under {self.mask_dir}"
        )


class ExternalCommandMaskProvider:
    provider_name = "external-command"

    def __init__(self, command_template: str, threshold: int = 0) -> None:
        self.command_template = command_template
        self.threshold = int(threshold)

    def predict(self, frame_view: FrameViewInput) -> MaskProviderResult:
        frame_view.output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command_template.format(
            rgb_path=str(frame_view.rgb_path),
            output_mask_path=str(frame_view.output_mask_path),
            frame_index=frame_view.frame_index,
            view_index=frame_view.view_index,
        )
        subprocess.run(shlex.split(command), check=True)
        if not frame_view.output_mask_path.exists():
            raise FileNotFoundError(f"Mask command did not write {frame_view.output_mask_path}")
        if self.threshold > 0:
            _copy_mask_as_u16(frame_view.output_mask_path, frame_view.output_mask_path, threshold=self.threshold)
        return MaskProviderResult(
            mask_path=frame_view.output_mask_path,
            metadata={"command": command},
        )


class SAM2MaskProvider:
    provider_name = "sam2"

    def __init__(
        self,
        *,
        repo_root: Path | None,
        model_config: str,
        checkpoint_path: Path,
        device: str,
        prompt_mode: str,
        box_xyxy: Sequence[float] | None,
        center_box_scale: float,
        multimask: bool,
        mask_selection: str,
        mask_index: int | None,
    ) -> None:
        self.repo_root = repo_root.expanduser().resolve() if repo_root is not None else None
        self.model_config = str(model_config)
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device_name = str(device)
        self.prompt_mode = str(prompt_mode)
        self.box_xyxy = tuple(float(value) for value in box_xyxy) if box_xyxy is not None else None
        self.center_box_scale = float(center_box_scale)
        self.multimask = bool(multimask)
        self.mask_selection = str(mask_selection)
        self.mask_index = mask_index
        self._predictor = None
        self._automatic_mask_generator = None
        self._torch = None

    def predict(self, frame_view: FrameViewInput) -> MaskProviderResult:
        import numpy as np
        from PIL import Image

        image = np.array(Image.open(frame_view.rgb_path).convert("RGB"), dtype=np.uint8, copy=True)
        if self.prompt_mode == "auto-masks":
            mask, metadata = self._predict_automatic(image)
        else:
            mask, metadata = self._predict_prompted(image)
        _write_mask_u16(mask.astype(np.uint16), frame_view.output_mask_path)
        return MaskProviderResult(
            mask_path=frame_view.output_mask_path,
            metadata={
                "sam2_config": self.model_config,
                "sam2_checkpoint": str(self.checkpoint_path),
                "sam2_device": self._resolved_device(),
                "sam2_prompt_mode": self.prompt_mode,
                **metadata,
            },
        )

    def _predict_prompted(self, image):
        import numpy as np

        predictor = self._load_predictor()
        torch = self._torch
        box = np.asarray(self._resolve_box(image.shape[1], image.shape[0]), dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(image)
            masks, iou_predictions, _low_res = predictor.predict(
                box=box,
                multimask_output=self.multimask,
            )
        areas = [int(mask.sum()) for mask in masks]
        best_index = self._select_mask_index(iou_predictions, areas)
        return masks[best_index].astype(bool), {
            "box_xyxy": [float(value) for value in box.tolist()],
            "predicted_iou": float(iou_predictions[best_index]) if len(iou_predictions) else None,
            "mask_choice": best_index,
            "mask_selection": self.mask_selection,
            "mask_candidates": [
                {"index": index, "predicted_iou": float(iou), "area": int(areas[index])}
                for index, iou in enumerate(iou_predictions)
            ],
        }

    def _predict_automatic(self, image):
        generator = self._load_automatic_mask_generator()
        anns = generator.generate(image)
        if not anns:
            raise RuntimeError("SAM2 automatic mask generator returned no masks.")
        best_index, best_ann = max(
            enumerate(anns),
            key=lambda item: (
                float(item[1].get("predicted_iou", 0.0)),
                int(item[1].get("area", 0)),
            ),
        )
        return best_ann["segmentation"].astype(bool), {
            "automatic_mask_count": len(anns),
            "automatic_mask_choice": int(best_index),
            "predicted_iou": float(best_ann.get("predicted_iou", 0.0)),
            "stability_score": float(best_ann.get("stability_score", 0.0)),
            "area": int(best_ann.get("area", 0)),
            "bbox_xywh": [float(value) for value in best_ann.get("bbox", [])],
        }

    def _load_predictor(self):
        if self._predictor is None:
            self._prepare_import_path()
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._torch = torch
            model = build_sam2(
                self.model_config,
                str(self.checkpoint_path),
                device=self._resolved_device(),
                apply_postprocessing=False,
            )
            self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def _load_automatic_mask_generator(self):
        if self._automatic_mask_generator is None:
            self._prepare_import_path()
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            self._torch = torch
            model = build_sam2(
                self.model_config,
                str(self.checkpoint_path),
                device=self._resolved_device(),
                apply_postprocessing=False,
            )
            self._automatic_mask_generator = SAM2AutomaticMaskGenerator(model, min_mask_region_area=0)
        return self._automatic_mask_generator

    def _prepare_import_path(self) -> None:
        if self.repo_root is None:
            return
        repo_root = str(self.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # Running from a parent directory named with a SAM2 checkout can shadow
        # the real package. If that happened earlier, drop the bad import and
        # let repo_root/sam2/__init__.py win.
        module = sys.modules.get("sam2")
        if module is not None:
            module_path = getattr(module, "__file__", None)
            if module_path is None or not str(module_path).startswith(repo_root):
                del sys.modules["sam2"]

    def _resolved_device(self) -> str:
        import torch

        if self.device_name == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if self.device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("SAM2 device requested as mps, but torch.backends.mps.is_available() is false.")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM2 device requested as cuda, but torch.cuda.is_available() is false.")
        return self.device_name

    def _resolve_box(self, width: int, height: int) -> tuple[float, float, float, float]:
        if self.prompt_mode == "box":
            if self.box_xyxy is None:
                raise ValueError("--sam2-box-xyxy is required when --sam2-prompt-mode box")
            if len(self.box_xyxy) != 4:
                raise ValueError("--sam2-box-xyxy must contain exactly four numbers")
            return self.box_xyxy
        if self.prompt_mode == "center-box":
            scale = min(max(self.center_box_scale, 0.05), 1.0)
            box_w = width * scale
            box_h = height * scale
            cx = width * 0.5
            cy = height * 0.5
            return (cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5)
        if self.prompt_mode == "full-image-box":
            return (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
        raise ValueError("sam2_prompt_mode must be one of: full-image-box, center-box, box, auto-masks")

    def _select_mask_index(self, iou_predictions, areas: Sequence[int]) -> int:
        import numpy as np

        if not len(iou_predictions):
            return 0
        if self.mask_selection == "best-iou":
            return int(np.argmax(iou_predictions))
        if self.mask_selection == "smallest":
            return int(np.argmin(np.asarray(areas)))
        if self.mask_selection == "largest":
            return int(np.argmax(np.asarray(areas)))
        if self.mask_selection == "index":
            if self.mask_index is None:
                raise ValueError("--sam2-mask-index is required when --sam2-mask-selection index")
            if self.mask_index < 0 or self.mask_index >= len(iou_predictions):
                raise ValueError(
                    f"--sam2-mask-index {self.mask_index} is out of range for {len(iou_predictions)} candidates"
                )
            return int(self.mask_index)
        raise ValueError("sam2_mask_selection must be one of: best-iou, smallest, largest, index")


class SAM3MaskProvider:
    provider_name = "sam3"

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str,
        prompt_mode: str,
        box_xyxy: Sequence[float] | None,
        center_box_scale: float,
        mask_selection: str,
        mask_index: int | None,
        conf: float,
    ) -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device_name = str(device)
        self.prompt_mode = str(prompt_mode)
        self.box_xyxy = tuple(float(value) for value in box_xyxy) if box_xyxy is not None else None
        self.center_box_scale = float(center_box_scale)
        self.mask_selection = str(mask_selection)
        self.mask_index = mask_index
        self.conf = float(conf)
        self._model = None
        self._torch = None

    def predict(self, frame_view: FrameViewInput) -> MaskProviderResult:
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(frame_view.rgb_path).convert("RGB"), dtype=np.uint8)
        box = np.asarray(self._resolve_box(image.shape[1], image.shape[0]), dtype=np.float32)
        model = self._load_model()
        results = model.predict(
            source=image,
            bboxes=[box.tolist()],
            device=self._resolved_device(),
            conf=self.conf,
            verbose=False,
        )
        if not results or results[0].masks is None:
            raise RuntimeError("SAM3 returned no masks.")
        masks = results[0].masks.data.detach().cpu().numpy().astype(bool)
        if masks.ndim != 3 or masks.shape[0] == 0:
            raise RuntimeError("SAM3 returned an empty mask tensor.")
        areas = [int(mask.sum()) for mask in masks]
        scores = _sam3_box_scores(results[0], len(areas))
        best_index = self._select_mask_index(scores, areas)
        _write_mask_u16(masks[best_index].astype(np.uint16), frame_view.output_mask_path)
        return MaskProviderResult(
            mask_path=frame_view.output_mask_path,
            metadata={
                "sam3_checkpoint": str(self.checkpoint_path),
                "sam3_device": self._resolved_device(),
                "sam3_prompt_mode": self.prompt_mode,
                "sam3_conf": self.conf,
                "box_xyxy": [float(value) for value in box.tolist()],
                "mask_choice": best_index,
                "mask_selection": self.mask_selection,
                "mask_candidates": [
                    {
                        "index": index,
                        "score": float(scores[index]) if index < len(scores) else None,
                        "area": int(areas[index]),
                    }
                    for index in range(len(areas))
                ],
            },
        )

    def _load_model(self):
        if self._model is None:
            import torch
            from ultralytics import SAM

            self._torch = torch
            self._model = SAM(str(self.checkpoint_path))
        return self._model

    def _resolved_device(self) -> str:
        import torch

        if self.device_name == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if self.device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("SAM3 device requested as mps, but torch.backends.mps.is_available() is false.")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM3 device requested as cuda, but torch.cuda.is_available() is false.")
        return self.device_name

    def _resolve_box(self, width: int, height: int) -> tuple[float, float, float, float]:
        if self.prompt_mode == "box":
            if self.box_xyxy is None:
                raise ValueError("--sam3-box-xyxy is required when --sam3-prompt-mode box")
            if len(self.box_xyxy) != 4:
                raise ValueError("--sam3-box-xyxy must contain exactly four numbers")
            return self.box_xyxy
        if self.prompt_mode == "center-box":
            scale = min(max(self.center_box_scale, 0.05), 1.0)
            box_w = width * scale
            box_h = height * scale
            cx = width * 0.5
            cy = height * 0.5
            return (cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5)
        if self.prompt_mode == "full-image-box":
            return (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
        raise ValueError("sam3_prompt_mode must be one of: full-image-box, center-box, box")

    def _select_mask_index(self, scores: Sequence[float], areas: Sequence[int]) -> int:
        import numpy as np

        if not areas:
            return 0
        if self.mask_selection == "best-score":
            return int(np.argmax(np.asarray(scores))) if scores else 0
        if self.mask_selection == "smallest":
            return int(np.argmin(np.asarray(areas)))
        if self.mask_selection == "largest":
            return int(np.argmax(np.asarray(areas)))
        if self.mask_selection == "index":
            if self.mask_index is None:
                raise ValueError("--sam3-mask-index is required when --sam3-mask-selection index")
            if self.mask_index < 0 or self.mask_index >= len(areas):
                raise ValueError(f"--sam3-mask-index {self.mask_index} is out of range for {len(areas)} candidates")
            return int(self.mask_index)
        raise ValueError("sam3_mask_selection must be one of: best-score, smallest, largest, index")


class EpisodeMaskWriter:
    def write(self, config: EpisodeMaskWriteConfig) -> Path:
        if config.mask_kind not in {"object", "part"}:
            raise ValueError("mask_kind must be one of: object, part")
        episode_path = config.episode_path.expanduser().resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)
        provider = _build_provider(config)
        output_root = (
            config.output_dir.expanduser().resolve()
            if config.output_dir is not None
            else episode_root / "assets" / "masks" / provider.provider_name
        )
        selected_frames = list(range(max(0, int(config.start_frame)), len(episode.frames), max(1, int(config.frame_stride))))
        if config.max_frames is not None:
            selected_frames = selected_frames[: max(0, int(config.max_frames))]

        predictions: list[dict[str, Any]] = []
        for frame_index in selected_frames:
            frame = episode.frames[frame_index]
            rgb_paths = _resolve_view_rgb_paths(frame, episode_root)
            view_indices = [int(index) for index in config.view_indices] if config.view_indices is not None else list(range(len(rgb_paths)))
            mask_paths_by_view = _current_mask_paths(frame, config.mask_kind)
            for view_index in view_indices:
                if view_index < 0 or view_index >= len(rgb_paths):
                    raise ValueError(f"View index {view_index} is out of range for frame {frame_index}.")
                output_mask_path = output_root / f"view_{view_index}" / f"frame_{frame_index:04d}_mask.png"
                if output_mask_path.exists() and not config.force:
                    result = MaskProviderResult(output_mask_path, {"reused_existing": True})
                else:
                    result = provider.predict(
                        FrameViewInput(
                            frame_index=frame_index,
                            view_index=view_index,
                            rgb_path=rgb_paths[view_index],
                            output_mask_path=output_mask_path,
                        )
                    )
                rel_path = result.mask_path.relative_to(episode_root).as_posix()
                if view_index > len(mask_paths_by_view):
                    raise ValueError(
                        f"Cannot write sparse mask_paths_by_view for frame {frame_index}: "
                        f"view {view_index} requested but only {len(mask_paths_by_view)} masks exist. "
                        "Run all lower view indices in the same command."
                    )
                if view_index == len(mask_paths_by_view):
                    mask_paths_by_view.append(rel_path)
                else:
                    mask_paths_by_view[view_index] = rel_path
                predictions.append(
                    {
                        "frame_index": frame_index,
                        "view_index": view_index,
                        "mask_path": rel_path,
                        **result.metadata,
                    }
                )
            _set_mask_paths(frame, config.mask_kind, mask_paths_by_view)

        episode.metadata["mask_provider"] = {
            "provider": provider.provider_name,
            "mask_kind": config.mask_kind,
            "output_dir": str(output_root),
            "prediction_count": len(predictions),
        }
        if config.mask_kind == "part" and "part_segmentation" not in episode.metadata:
            episode.metadata["part_segmentation"] = {
                "provider": provider.provider_name,
                "version": 1,
                "mask_encoding": "indexed-mask-u16",
                "background_part_id": 0,
                "parts": [],
                "notes": "Part ids are provider-defined. Add part metadata before track-part-pixels when names/roles matter.",
            }

        output_episode_path = (
            config.output_episode_path.expanduser().resolve()
            if config.output_episode_path is not None
            else episode_path
        )
        save_json(episode.to_dict(), output_episode_path)
        save_json(
            {
                "source": "episode-mask-writer",
                "episode_path": str(episode_path),
                "output_episode_path": str(output_episode_path),
                "provider": provider.provider_name,
                "mask_kind": config.mask_kind,
                "predictions": predictions,
            },
            output_root / "mask_predictions.json",
        )
        return output_episode_path


class EpisodeMaskBatchWriter:
    def __init__(self, config: EpisodeMaskBatchConfig) -> None:
        self.config = config
        self._console_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        rows = _read_mask_batch_manifest(self.config.manifest_path)
        successes: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []

        def process(row: dict[str, str]) -> dict[str, str]:
            episode_path = _manifest_path(row["episode_path"], self.config.manifest_path.parent)
            object_id = row.get("object_id") or episode_path.parent.name
            output_episode = _optional_manifest_path(row.get("output_episode"), self.config.manifest_path.parent)
            output_dir = _optional_manifest_path(row.get("output_dir"), self.config.manifest_path.parent)
            if output_dir is None and self.config.output_root is not None:
                output_dir = self.config.output_root.expanduser().resolve() / object_id
            episode_out = EpisodeMaskWriter().write(
                EpisodeMaskWriteConfig(
                    episode_path=episode_path,
                    output_episode_path=output_episode,
                    output_dir=output_dir,
                    provider=row.get("provider") or self.config.provider,
                    mask_dir=_optional_manifest_path(row.get("mask_dir"), self.config.manifest_path.parent),
                    command=row.get("command") or None,
                    sam2_root=_optional_manifest_path(row.get("sam2_root"), self.config.manifest_path.parent)
                    or self.config.sam2_root,
                    sam2_config=row.get("sam2_config") or self.config.sam2_config,
                    sam2_checkpoint=_optional_manifest_path(row.get("sam2_checkpoint"), self.config.manifest_path.parent)
                    or self.config.sam2_checkpoint,
                    sam2_device=row.get("sam2_device") or self.config.sam2_device,
                    sam2_prompt_mode=row.get("sam2_prompt_mode") or self.config.sam2_prompt_mode,
                    sam2_box_xyxy=_optional_box(row.get("sam2_box_xyxy")) or self.config.sam2_box_xyxy,
                    sam2_center_box_scale=_float_value(row, "sam2_center_box_scale", self.config.sam2_center_box_scale),
                    sam2_multimask=_bool_value(row, "sam2_multimask", self.config.sam2_multimask),
                    sam2_mask_selection=row.get("sam2_mask_selection") or self.config.sam2_mask_selection,
                    sam2_mask_index=_optional_int_value(row, "sam2_mask_index", self.config.sam2_mask_index),
                    sam3_checkpoint=_optional_manifest_path(row.get("sam3_checkpoint"), self.config.manifest_path.parent)
                    or self.config.sam3_checkpoint,
                    sam3_device=row.get("sam3_device") or self.config.sam3_device,
                    sam3_prompt_mode=row.get("sam3_prompt_mode") or self.config.sam3_prompt_mode,
                    sam3_box_xyxy=_optional_box(row.get("sam3_box_xyxy")) or self.config.sam3_box_xyxy,
                    sam3_center_box_scale=_float_value(row, "sam3_center_box_scale", self.config.sam3_center_box_scale),
                    sam3_mask_selection=row.get("sam3_mask_selection") or self.config.sam3_mask_selection,
                    sam3_mask_index=_optional_int_value(row, "sam3_mask_index", self.config.sam3_mask_index),
                    sam3_conf=_float_value(row, "sam3_conf", self.config.sam3_conf),
                    mask_kind=row.get("mask_kind") or self.config.mask_kind,
                    frame_stride=_int_value(row, "frame_stride", self.config.frame_stride),
                    start_frame=_int_value(row, "start_frame", self.config.start_frame),
                    max_frames=_optional_int_value(row, "max_frames", self.config.max_frames),
                    view_indices=_optional_int_list(row.get("view_indices")) or self.config.view_indices,
                    threshold=_int_value(row, "threshold", self.config.threshold),
                    force=_bool_value(row, "force", self.config.force),
                )
            )
            return {"object_id": object_id, "episode_path": str(episode_out)}

        jobs = max(1, int(self.config.jobs))
        if jobs == 1:
            for row in rows:
                try:
                    result = process(row)
                    successes.append(result)
                    self._console(f"[{len(successes) + len(failures)}/{len(rows)}] segmented {result['object_id']}")
                except Exception as exc:
                    failures.append({"episode_path": row.get("episode_path", ""), "error": str(exc)})
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {executor.submit(process, row): row for row in rows}
                for future in as_completed(future_map):
                    row = future_map[future]
                    try:
                        result = future.result()
                        successes.append(result)
                        self._console(f"[{len(successes) + len(failures)}/{len(rows)}] segmented {result['object_id']}")
                    except Exception as exc:
                        failures.append({"episode_path": row.get("episode_path", ""), "error": str(exc)})

        if failures:
            raise RuntimeError(f"Mask batch finished with failures: {failures}")
        return {
            "manifest_path": str(self.config.manifest_path.expanduser().resolve()),
            "segmented": len(successes),
            "episodes": successes,
        }

    def _console(self, message: str) -> None:
        with self._console_lock:
            print(message)


class EpisodeMaskPropagator:
    def propagate(self, config: EpisodeMaskPropagationConfig) -> Path:
        if config.mask_kind not in {"object", "part"}:
            raise ValueError("mask_kind must be one of: object, part")
        if config.backend == "sam2-video":
            return self._propagate_sam2_video(config)
        if config.backend == "cotracker-sparse":
            return self._propagate_cotracker(config)
        raise ValueError("backend must be one of: sam2-video, cotracker-sparse")

    def _propagate_sam2_video(self, config: EpisodeMaskPropagationConfig) -> Path:
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SAM2 video propagation requires NumPy and PyTorch.") from exc

        if config.sam2_checkpoint is None:
            raise ValueError("--sam2-checkpoint is required when --backend sam2-video")
        checkpoint_path = config.sam2_checkpoint.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        _prepare_sam2_import_path(config.sam2_root)
        from sam2.build_sam import build_sam2_video_predictor

        episode_path = config.episode_path.expanduser().resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)
        if not episode.frames:
            raise ValueError("Episode has no frames to propagate.")
        start_frame = min(max(0, int(config.start_frame)), len(episode.frames) - 1)
        end_frame = len(episode.frames) - 1 if config.end_frame is None else min(int(config.end_frame), len(episode.frames) - 1)
        if end_frame < start_frame:
            raise ValueError("Propagation end frame must not precede start frame")
        reference_frame = min(max(start_frame, int(config.reference_frame)), end_frame)
        sampled_frame_indices = list(range(start_frame, end_frame + 1, max(1, int(config.frame_stride))))
        if reference_frame not in sampled_frame_indices:
            sampled_frame_indices = sorted({reference_frame, *sampled_frame_indices})
        sampled_frames = [episode.frames[index] for index in sampled_frame_indices]
        local_reference_frame = sampled_frame_indices.index(reference_frame)
        view_count = len(_resolve_view_rgb_paths(episode.frames[reference_frame], episode_root))
        view_indices = [int(index) for index in config.view_indices] if config.view_indices is not None else list(range(view_count))
        output_root = (
            config.output_dir.expanduser().resolve()
            if config.output_dir is not None
            else episode_root / "assets" / "masks" / f"sam2-video-{config.mask_kind}"
        )
        device = _resolve_sam2_device(config.sam2_device)
        predictor = build_sam2_video_predictor(
            config.sam2_config,
            str(checkpoint_path),
            device=device,
            apply_postprocessing=False,
        )

        predictions: list[dict[str, Any]] = []
        reference_part_ids: set[int] = set()
        with torch.inference_mode():
            for view_index in view_indices:
                if view_index < 0 or view_index >= view_count:
                    raise ValueError(f"View index {view_index} is out of range for reference frame {reference_frame}.")
                video_dir = _write_sam2_video_frame_dir(
                    sampled_frames,
                    episode_root,
                    view_index,
                    output_root / "_sam2_video_frames" / f"view_{view_index}",
                )
                reference_mask_path = _reference_mask_path(
                    episode.frames[reference_frame],
                    episode_root,
                    view_index,
                    mask_kind=config.mask_kind,
                )
                reference_mask = np.asarray(_load_depth_u16(reference_mask_path), dtype=np.uint16)
                part_ids = _mask_object_ids(reference_mask, mask_kind=config.mask_kind, threshold=int(config.threshold))
                ignored_part_ids: list[int] = []
                if config.mask_kind == "part":
                    declared_part_ids = _declared_part_ids(episode.metadata)
                    if declared_part_ids:
                        unknown_part_ids = [part_id for part_id in part_ids if part_id not in declared_part_ids]
                        ignored_part_ids.extend(unknown_part_ids)
                        part_ids = [part_id for part_id in part_ids if part_id in declared_part_ids]
                    part_ids, small_part_ids = _filter_reference_part_ids_by_area(reference_mask, part_ids)
                    ignored_part_ids.extend(small_part_ids)
                if not part_ids:
                    raise ValueError(f"Reference mask has no foreground ids: {reference_mask_path}")
                if config.mask_kind == "part":
                    reference_part_ids.update(part_ids)
                part_mode = str(config.sam2_part_mode or "independent")
                if config.mask_kind == "part" and part_mode not in {"independent", "joint"}:
                    raise ValueError("--sam2-part-mode must be independent or joint")

                if config.mask_kind == "part" and part_mode == "independent":
                    output_logits: dict[int, list[tuple[int, Any]]] = {}
                    for part_id in part_ids:
                        state = predictor.init_state(
                            str(video_dir),
                            offload_video_to_cpu=bool(config.sam2_offload_video_to_cpu),
                            offload_state_to_cpu=bool(config.sam2_offload_state_to_cpu),
                        )
                        predictor.add_new_mask(
                            state,
                            frame_idx=local_reference_frame,
                            obj_id=int(part_id),
                            mask=reference_mask == int(part_id),
                        )
                        part_outputs: dict[int, tuple[Sequence[int], Any]] = {}
                        for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                            state,
                            start_frame_idx=local_reference_frame,
                        ):
                            part_outputs[int(local_idx)] = (list(obj_ids), mask_logits)
                            if int(local_idx) < len(sampled_frame_indices):
                                _write_sam2_independent_preview_mask(
                                    output_root / f"view_{view_index}" / f"frame_{sampled_frame_indices[int(local_idx)]:04d}_mask.png",
                                    int(part_id),
                                    mask_logits,
                                )
                        if local_reference_frame > 0:
                            for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                                state,
                                start_frame_idx=local_reference_frame,
                                reverse=True,
                            ):
                                part_outputs[int(local_idx)] = (list(obj_ids), mask_logits)
                                if int(local_idx) < len(sampled_frame_indices):
                                    _write_sam2_independent_preview_mask(
                                        output_root / f"view_{view_index}" / f"frame_{sampled_frame_indices[int(local_idx)]:04d}_mask.png",
                                        int(part_id),
                                        mask_logits,
                                    )
                        for local_idx, (_obj_ids, mask_logits) in part_outputs.items():
                            output_logits.setdefault(int(local_idx), []).append((int(part_id), mask_logits))
                    outputs: dict[int, tuple[Sequence[int], Any]] = {
                        local_idx: ([part_id for part_id, _ in entries], entries)
                        for local_idx, entries in output_logits.items()
                    }
                else:
                    state = predictor.init_state(
                        str(video_dir),
                        offload_video_to_cpu=bool(config.sam2_offload_video_to_cpu),
                        offload_state_to_cpu=bool(config.sam2_offload_state_to_cpu),
                    )
                    for part_id in part_ids:
                        predictor.add_new_mask(
                            state,
                            frame_idx=local_reference_frame,
                            obj_id=int(part_id),
                            mask=reference_mask == int(part_id) if config.mask_kind == "part" else reference_mask > int(config.threshold),
                        )

                    outputs = {}
                    for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=local_reference_frame):
                        outputs[int(local_idx)] = (list(obj_ids), mask_logits)
                        if int(local_idx) < len(sampled_frame_indices):
                            _write_mask_u16(
                                _sam2_video_logits_to_indexed_mask(mask_logits, list(obj_ids)),
                                output_root / f"view_{view_index}" / f"frame_{sampled_frame_indices[int(local_idx)]:04d}_mask.png",
                            )
                    if local_reference_frame > 0:
                        for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                            state,
                            start_frame_idx=local_reference_frame,
                            reverse=True,
                        ):
                            outputs[int(local_idx)] = (list(obj_ids), mask_logits)
                            if int(local_idx) < len(sampled_frame_indices):
                                _write_mask_u16(
                                    _sam2_video_logits_to_indexed_mask(mask_logits, list(obj_ids)),
                                    output_root / f"view_{view_index}" / f"frame_{sampled_frame_indices[int(local_idx)]:04d}_mask.png",
                                )

                for local_frame_index, frame_index in enumerate(sampled_frame_indices):
                    if local_frame_index not in outputs:
                        continue
                    obj_ids, mask_logits = outputs[local_frame_index]
                    if config.mask_kind == "part" and part_mode == "independent":
                        mask = _sam2_video_independent_logits_to_indexed_mask(mask_logits)
                    else:
                        mask = _sam2_video_logits_to_indexed_mask(mask_logits, obj_ids)
                    frame = episode.frames[frame_index]
                    output_mask_path = output_root / f"view_{view_index}" / f"frame_{frame_index:04d}_mask.png"
                    if output_mask_path.exists() and not config.force:
                        pass
                    else:
                        _write_mask_u16(mask, output_mask_path)
                    rel_path = output_mask_path.relative_to(episode_root).as_posix()
                    mask_paths = _current_mask_paths(frame, config.mask_kind)
                    while len(mask_paths) <= view_index:
                        mask_paths.append("")
                    mask_paths[view_index] = rel_path
                    _set_mask_paths(frame, config.mask_kind, mask_paths)
                    predictions.append(
                        {
                            "frame_index": frame_index,
                            "view_index": view_index,
                            "mask_path": rel_path,
                            "object_ids": [int(obj_id) for obj_id in obj_ids],
                            "ignored_reference_part_ids": sorted(set(int(part_id) for part_id in ignored_part_ids)),
                            "foreground_pixels": int((mask > 0).sum()),
                        }
                    )

        episode.metadata["mask_provider"] = {
            "provider": "sam2-video",
            "mask_kind": config.mask_kind,
            "reference_frame": reference_frame,
            "frame_range": [start_frame, end_frame],
            "sam2_part_mode": config.sam2_part_mode,
            "output_dir": str(output_root),
            "prediction_count": len(predictions),
        }
        if config.mask_kind == "part" and "part_segmentation" not in episode.metadata:
            episode.metadata["part_segmentation"] = _part_segmentation_from_part_ids(sorted(reference_part_ids))
        output_episode_path = (
            config.output_episode_path.expanduser().resolve()
            if config.output_episode_path is not None
            else episode_path
        )
        save_json(episode.to_dict(), output_episode_path)
        save_json(
            {
                "source": "episode-mask-propagator",
                "episode_path": str(episode_path),
                "output_episode_path": str(output_episode_path),
                "provider": "sam2-video",
                "mask_kind": config.mask_kind,
                "reference_frame": reference_frame,
                "sampled_frame_indices": sampled_frame_indices,
                "device": device,
                "sam2_config": config.sam2_config,
                "sam2_checkpoint": str(checkpoint_path),
                "sam2_part_mode": config.sam2_part_mode,
                "predictions": predictions,
                "config": {
                    "offload_video_to_cpu": config.sam2_offload_video_to_cpu,
                    "offload_state_to_cpu": config.sam2_offload_state_to_cpu,
                },
            },
            output_root / "mask_propagation.json",
        )
        return output_episode_path

    def _propagate_cotracker(self, config: EpisodeMaskPropagationConfig) -> Path:
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Mask propagation requires NumPy, PyTorch, and CoTracker.") from exc

        from .part_tracking import _load_cotracker_model, _load_video_tensor, _resolve_tracking_device
        from .part_tracking import PartPixelTrackingConfig

        episode_path = config.episode_path.expanduser().resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)
        if not episode.frames:
            raise ValueError("Episode has no frames to propagate.")
        reference_frame = min(max(0, int(config.reference_frame)), len(episode.frames) - 1)
        sampled_frame_indices = list(range(0, len(episode.frames), max(1, int(config.frame_stride))))
        if reference_frame not in sampled_frame_indices:
            sampled_frame_indices = sorted({reference_frame, *sampled_frame_indices})
        sampled_frames = [episode.frames[index] for index in sampled_frame_indices]
        local_reference_frame = sampled_frame_indices.index(reference_frame)
        view_count = len(_resolve_view_rgb_paths(episode.frames[reference_frame], episode_root))
        view_indices = [int(index) for index in config.view_indices] if config.view_indices is not None else list(range(view_count))
        output_root = (
            config.output_dir.expanduser().resolve()
            if config.output_dir is not None
            else episode_root / "assets" / "masks" / f"cotracker-propagated-{config.mask_kind}"
        )

        tracking_config = PartPixelTrackingConfig(
            episode_path=episode_path,
            device=config.device,
            cotracker_repo=config.cotracker_repo,
            cotracker_checkpoint=config.cotracker_checkpoint,
            cotracker_model=config.cotracker_model,
            reference_frame=reference_frame,
            frame_stride=config.frame_stride,
            seed_stride_px=config.seed_stride_px,
            max_tracks_per_part_view=config.max_tracks_per_view,
            visibility_threshold=config.visibility_threshold,
            show_progress=False,
        )
        device = _resolve_tracking_device(torch, tracking_config)
        model = _load_cotracker_model(tracking_config, torch, device)

        predictions: list[dict[str, Any]] = []
        reference_part_ids: set[int] = set()
        with torch.no_grad():
            for view_index in view_indices:
                if view_index < 0 or view_index >= view_count:
                    raise ValueError(f"View index {view_index} is out of range for reference frame {reference_frame}.")
                reference_mask_path = _reference_mask_path(
                    episode.frames[reference_frame],
                    episode_root,
                    view_index,
                    mask_kind=config.mask_kind,
                )
                reference_mask = np.asarray(_load_depth_u16(reference_mask_path), dtype=np.uint16)
                if config.mask_kind == "part":
                    reference_part_ids.update(int(value) for value in np.unique(reference_mask) if int(value) > 0)
                seeds = _sample_labeled_seed_pixels(
                    reference_mask,
                    mask_kind=config.mask_kind,
                    threshold=int(config.threshold),
                    stride_px=int(config.seed_stride_px),
                    max_points=int(config.max_tracks_per_view),
                )
                if not seeds:
                    raise ValueError(f"Reference mask has no foreground pixels: {reference_mask_path}")
                rgb_paths = [
                    _resolve_view_rgb_paths(frame, episode_root)[view_index]
                    for frame in sampled_frames
                ]
                video = _load_video_tensor(rgb_paths, torch, device)
                queries = torch.tensor(
                    [[[float(local_reference_frame), float(u), float(v)] for u, v, _part_id in seeds]],
                    dtype=torch.float32,
                    device=device,
                )
                pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=True)
                pred_tracks = pred_tracks.detach().cpu().numpy()
                pred_visibility = pred_visibility.detach().cpu().numpy()
                for local_frame_index, frame_index in enumerate(sampled_frame_indices):
                    frame = episode.frames[frame_index]
                    rgb_paths_for_frame = _resolve_view_rgb_paths(frame, episode_root)
                    height, width = _image_size_hw(rgb_paths_for_frame[view_index])
                    mask = _rasterize_labeled_tracked_points(
                        pred_tracks[0, local_frame_index],
                        pred_visibility[0, local_frame_index],
                        part_ids=[part_id for _u, _v, part_id in seeds],
                        width=width,
                        height=height,
                        visibility_threshold=float(config.visibility_threshold),
                        radius_px=int(config.mask_radius_px),
                    )
                    output_mask_path = output_root / f"view_{view_index}" / f"frame_{frame_index:04d}_mask.png"
                    if output_mask_path.exists() and not config.force:
                        pass
                    else:
                        _write_mask_u16(mask, output_mask_path)
                    rel_path = output_mask_path.relative_to(episode_root).as_posix()
                    mask_paths = _current_mask_paths(frame, config.mask_kind)
                    while len(mask_paths) <= view_index:
                        mask_paths.append("")
                    mask_paths[view_index] = rel_path
                    _set_mask_paths(frame, config.mask_kind, mask_paths)
                    predictions.append(
                        {
                            "frame_index": frame_index,
                            "view_index": view_index,
                            "mask_path": rel_path,
                            "seed_count": len(seeds),
                            "visible_track_count": int((pred_visibility[0, local_frame_index] >= float(config.visibility_threshold)).sum()),
                        }
                    )

        episode.metadata["mask_provider"] = {
            "provider": "cotracker-propagated",
            "mask_kind": config.mask_kind,
            "reference_frame": reference_frame,
            "output_dir": str(output_root),
            "prediction_count": len(predictions),
        }
        if config.mask_kind == "part" and "part_segmentation" not in episode.metadata:
            episode.metadata["part_segmentation"] = _part_segmentation_from_part_ids(sorted(reference_part_ids))
        output_episode_path = (
            config.output_episode_path.expanduser().resolve()
            if config.output_episode_path is not None
            else episode_path
        )
        save_json(episode.to_dict(), output_episode_path)
        save_json(
            {
                "source": "episode-mask-propagator",
                "episode_path": str(episode_path),
                "output_episode_path": str(output_episode_path),
                "provider": "cotracker-propagated",
                "mask_kind": config.mask_kind,
                "reference_frame": reference_frame,
                "sampled_frame_indices": sampled_frame_indices,
                "device": device,
                "predictions": predictions,
                "config": {
                    "seed_stride_px": config.seed_stride_px,
                    "max_tracks_per_view": config.max_tracks_per_view,
                    "visibility_threshold": config.visibility_threshold,
                    "mask_radius_px": config.mask_radius_px,
                    "threshold": config.threshold,
                },
            },
            output_root / "mask_propagation.json",
        )
        return output_episode_path


class EpisodeMaskEvaluator:
    def evaluate(self, config: EpisodeMaskEvaluationConfig) -> Path:
        if config.mask_kind not in {"object", "part"}:
            raise ValueError("mask_kind must be one of: object, part")
        predicted_episode_path = config.predicted_episode_path.expanduser().resolve()
        reference_episode_path = config.reference_episode_path.expanduser().resolve()
        predicted = load_episode(predicted_episode_path)
        reference = load_episode(reference_episode_path)
        predicted_root = predicted_episode_path.parent
        reference_root = reference_episode_path.parent
        records: list[dict[str, Any]] = []
        for frame_index, (pred_frame, ref_frame) in enumerate(zip(predicted.frames, reference.frames)):
            pred_paths = _resolve_eval_mask_paths(pred_frame, predicted_root, config.mask_kind)
            ref_paths = _resolve_eval_mask_paths(ref_frame, reference_root, config.mask_kind)
            for view_index, (pred_path, ref_path) in enumerate(zip(pred_paths, ref_paths)):
                records.append(
                    {
                        "frame_index": frame_index,
                        "view_index": view_index,
                        **_mask_metrics(pred_path, ref_path, threshold=int(config.threshold)),
                    }
                )
        if not records:
            raise ValueError("No overlapping mask paths found for evaluation.")
        summary = _summarize_metrics(records)
        output_json = (
            config.output_json.expanduser().resolve()
            if config.output_json is not None
            else predicted_episode_path.parent / f"{config.mask_kind}_mask_evaluation.json"
        )
        save_json(
            {
                "source": "episode-mask-evaluation",
                "predicted_episode_path": str(predicted_episode_path),
                "reference_episode_path": str(reference_episode_path),
                "mask_kind": config.mask_kind,
                "summary": summary,
                "records": records,
            },
            output_json,
        )
        return output_json


def _build_provider(config: EpisodeMaskWriteConfig) -> EpisodeMaskProvider:
    if config.provider == "mask-dir":
        if config.mask_dir is None:
            raise ValueError("--mask-dir is required when --provider mask-dir")
        return MaskDirectoryProvider(config.mask_dir, threshold=int(config.threshold))
    if config.provider == "external-command":
        if not config.command:
            raise ValueError("--command is required when --provider external-command")
        return ExternalCommandMaskProvider(config.command, threshold=int(config.threshold))
    if config.provider == "sam2":
        if config.mask_kind != "object":
            raise ValueError("Native --provider sam2 currently writes object masks only. Use external-command for part masks.")
        if config.sam2_checkpoint is None:
            raise ValueError("--sam2-checkpoint is required when --provider sam2")
        if not config.sam2_checkpoint.expanduser().exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {config.sam2_checkpoint}")
        return SAM2MaskProvider(
            repo_root=config.sam2_root,
            model_config=config.sam2_config,
            checkpoint_path=config.sam2_checkpoint,
            device=config.sam2_device,
            prompt_mode=config.sam2_prompt_mode,
            box_xyxy=config.sam2_box_xyxy,
            center_box_scale=float(config.sam2_center_box_scale),
            multimask=bool(config.sam2_multimask),
            mask_selection=str(config.sam2_mask_selection),
            mask_index=config.sam2_mask_index,
        )
    if config.provider == "sam3":
        if config.mask_kind != "object":
            raise ValueError("Native --provider sam3 currently writes object masks only. Use external-command for part masks.")
        if config.sam3_checkpoint is None:
            raise ValueError("--sam3-checkpoint is required when --provider sam3")
        if not config.sam3_checkpoint.expanduser().exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {config.sam3_checkpoint}")
        return SAM3MaskProvider(
            checkpoint_path=config.sam3_checkpoint,
            device=config.sam3_device,
            prompt_mode=config.sam3_prompt_mode,
            box_xyxy=config.sam3_box_xyxy,
            center_box_scale=float(config.sam3_center_box_scale),
            mask_selection=str(config.sam3_mask_selection),
            mask_index=config.sam3_mask_index,
            conf=float(config.sam3_conf),
        )
    raise ValueError(f"Unsupported mask provider: {config.provider}")


def _read_mask_batch_manifest(path: Path) -> list[dict[str, str]]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Mask batch manifest not found: {manifest_path}")
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {str(key): str(value).strip() for key, value in row.items() if key is not None and value is not None}
            for row in csv.DictReader(
                (line for line in handle if line.strip() and not line.lstrip().startswith("#")),
                delimiter="\t",
            )
        ]
    if not rows:
        raise ValueError(f"Mask batch manifest is empty: {manifest_path}")
    for row in rows:
        if not row.get("episode_path"):
            raise ValueError("Mask batch manifest must include an episode_path column.")
    return rows


def _sam3_box_scores(result: Any, count: int) -> list[float]:
    boxes = getattr(result, "boxes", None)
    conf = getattr(boxes, "conf", None) if boxes is not None else None
    if conf is None:
        return [0.0] * int(count)
    try:
        values = conf.detach().cpu().numpy().tolist()
    except AttributeError:
        values = list(conf)
    scores = [float(value) for value in values[:count]]
    if len(scores) < count:
        scores.extend([0.0] * (count - len(scores)))
    return scores


def _manifest_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _optional_manifest_path(raw: str | None, base_dir: Path) -> Path | None:
    if raw in (None, ""):
        return None
    return _manifest_path(raw, base_dir)


def _int_value(row: dict[str, str], key: str, default: int) -> int:
    raw = row.get(key)
    return int(default if raw in (None, "") else raw)


def _optional_int_value(row: dict[str, str], key: str, default: int | None) -> int | None:
    raw = row.get(key)
    return default if raw in (None, "") else int(raw)


def _float_value(row: dict[str, str], key: str, default: float) -> float:
    raw = row.get(key)
    return float(default if raw in (None, "") else raw)


def _bool_value(row: dict[str, str], key: str, default: bool) -> bool:
    raw = row.get(key)
    if raw in (None, ""):
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_box(raw: str | None) -> tuple[float, float, float, float] | None:
    if raw in (None, ""):
        return None
    values = [float(value) for value in str(raw).replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError("sam2_box_xyxy manifest value must contain four numbers.")
    return (values[0], values[1], values[2], values[3])


def _optional_int_list(raw: str | None) -> list[int] | None:
    if raw in (None, ""):
        return None
    return [int(value) for value in str(raw).replace(",", " ").split()]


def _prepare_sam2_import_path(repo_root: Path | None) -> None:
    if repo_root is None:
        return
    resolved = repo_root.expanduser().resolve()
    repo_root_text = str(resolved)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    module = sys.modules.get("sam2")
    if module is not None:
        module_path = getattr(module, "__file__", None)
        if module_path is None or not str(module_path).startswith(repo_root_text):
            del sys.modules["sam2"]


def _resolve_sam2_device(device_name: str) -> str:
    import torch

    if device_name == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("SAM2 device requested as mps, but torch.backends.mps.is_available() is false.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM2 device requested as cuda, but torch.cuda.is_available() is false.")
    if device_name not in {"cpu", "mps", "cuda"}:
        raise ValueError("sam2_device must be one of: auto, mps, cpu, cuda")
    return device_name


def _write_sam2_video_frame_dir(frames: Sequence[Any], episode_root: Path, view_index: int, output_dir: Path) -> Path:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for local_index, frame in enumerate(frames):
        rgb_paths = _resolve_view_rgb_paths(frame, episode_root)
        if view_index >= len(rgb_paths):
            raise ValueError(f"View index {view_index} is out of range while preparing SAM2 video frames.")
        output_path = output_dir / f"{local_index:05d}.jpg"
        if output_path.exists():
            continue
        Image.open(rgb_paths[view_index]).convert("RGB").save(output_path, quality=95)
    return output_dir


def _mask_object_ids(mask, *, mask_kind: str, threshold: int) -> list[int]:
    import numpy as np

    mask_np = np.asarray(mask)
    if mask_kind == "object":
        return [1] if bool((mask_np > int(threshold)).any()) else []
    return sorted(int(value) for value in np.unique(mask_np) if int(value) > int(threshold))


def _declared_part_ids(metadata: dict[str, Any]) -> set[int]:
    parts = metadata.get("part_segmentation", {}).get("parts") if isinstance(metadata, dict) else None
    if not isinstance(parts, list):
        return set()
    part_ids: set[int] = set()
    for part in parts:
        if not isinstance(part, dict):
            continue
        try:
            part_id = int(part.get("part_id") or 0)
        except (TypeError, ValueError):
            continue
        if part_id > 0:
            part_ids.add(part_id)
    return part_ids


def _filter_reference_part_ids_by_area(mask: Any, part_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    import numpy as np

    mask_np = np.asarray(mask)
    min_pixels = max(128, int(mask_np.size * 0.0005))
    kept: list[int] = []
    ignored: list[int] = []
    for part_id in part_ids:
        pixels = int((mask_np == int(part_id)).sum())
        if pixels >= min_pixels:
            kept.append(int(part_id))
        else:
            ignored.append(int(part_id))
    return kept, ignored


def _sam2_video_logits_to_indexed_mask(mask_logits, obj_ids: Sequence[int]):
    import numpy as np

    logits = mask_logits.detach().cpu().float().numpy()
    if logits.ndim == 4:
        logits = logits[:, 0, :, :]
    if logits.ndim == 2:
        logits = logits[None, :, :]
    if logits.shape[0] != len(obj_ids):
        raise ValueError(f"SAM2 returned {logits.shape[0]} masks for {len(obj_ids)} object ids.")
    positive = logits > 0.0
    output = np.zeros(logits.shape[1:], dtype=np.uint16)
    if not bool(positive.any()):
        return output
    best_indices = np.argmax(logits, axis=0)
    for mask_index, obj_id in enumerate(obj_ids):
        keep = positive[mask_index] & (best_indices == mask_index)
        output[keep] = np.uint16(max(1, int(obj_id)))
    return _keep_largest_component_per_label(output)


def _sam2_video_independent_logits_to_indexed_mask(entries: Sequence[tuple[int, Any]]):
    import numpy as np

    logits_by_id: list[tuple[int, Any]] = []
    for obj_id, mask_logits in entries:
        logits = mask_logits.detach().cpu().float().numpy()
        if logits.ndim == 4:
            logits = logits[:, 0, :, :]
        if logits.ndim == 2:
            logits = logits[None, :, :]
        if logits.shape[0] < 1:
            continue
        logits_by_id.append((int(obj_id), logits[0]))
    if not logits_by_id:
        return np.zeros((1, 1), dtype=np.uint16)
    stacked = np.stack([logits for _obj_id, logits in logits_by_id], axis=0)
    positive = stacked > 0.0
    output = np.zeros(stacked.shape[1:], dtype=np.uint16)
    if not bool(positive.any()):
        return output
    best_indices = np.argmax(stacked, axis=0)
    for mask_index, (obj_id, _logits) in enumerate(logits_by_id):
        keep = positive[mask_index] & (best_indices == mask_index)
        output[keep] = np.uint16(max(1, int(obj_id)))
    return _keep_largest_component_per_label(output)


def _write_sam2_independent_preview_mask(output_path: Path, part_id: int, mask_logits: Any) -> None:
    import numpy as np
    from PIL import Image

    logits = mask_logits.detach().cpu().float().numpy()
    if logits.ndim == 4:
        logits = logits[:, 0, :, :]
    if logits.ndim == 2:
        logits = logits[None, :, :]
    if logits.shape[0] < 1:
        return
    positive = _largest_connected_component(logits[0] > 0.0)
    if output_path.exists():
        output = np.asarray(Image.open(output_path), dtype=np.uint16).copy()
        if output.shape != positive.shape:
            output = np.zeros(positive.shape, dtype=np.uint16)
    else:
        output = np.zeros(positive.shape, dtype=np.uint16)
    output[output == int(part_id)] = 0
    output[positive] = np.uint16(max(1, int(part_id)))
    output = _keep_largest_component_per_label(output)
    _write_mask_u16(output, output_path)


def _keep_largest_component_per_label(mask: Any):
    import numpy as np

    mask_np = np.asarray(mask, dtype=np.uint16)
    output = np.zeros(mask_np.shape, dtype=np.uint16)
    for raw_label in sorted(int(value) for value in np.unique(mask_np) if int(value) > 0):
        component = _largest_connected_component(mask_np == raw_label)
        output[component] = np.uint16(raw_label)
    return output


def _largest_connected_component(mask: Any):
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not bool(binary.any()):
        return np.zeros(binary.shape, dtype=bool)

    height, width = binary.shape
    seen = np.zeros(binary.shape, dtype=bool)
    best_coords: list[tuple[int, int]] = []
    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=True):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        coords: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and binary[next_y, next_x] and not seen[next_y, next_x]:
                    seen[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(coords) > len(best_coords):
            best_coords = coords

    output = np.zeros(binary.shape, dtype=bool)
    if best_coords:
        yy, xx = zip(*best_coords, strict=True)
        output[np.asarray(yy), np.asarray(xx)] = True
    return output


def _reference_mask_path(frame: Any, episode_root: Path, view_index: int, *, mask_kind: str) -> Path:
    if mask_kind == "part":
        raw_paths = list(frame.part_mask_paths_by_view)
        if not raw_paths and frame.part_mask_path:
            raw_paths = [frame.part_mask_path]
    else:
        raw_paths = list(frame.mask_paths_by_view)
        if not raw_paths and frame.mask_path:
            raw_paths = [frame.mask_path]
    if view_index < 0 or view_index >= len(raw_paths) or not raw_paths[view_index]:
        raise ValueError(f"Reference frame does not contain a {mask_kind} mask for view {view_index}.")
    return episode_root / raw_paths[view_index]


def _sample_labeled_seed_pixels(
    mask,
    *,
    mask_kind: str,
    threshold: int,
    stride_px: int,
    max_points: int,
) -> list[tuple[int, int, int]]:
    import numpy as np

    mask_np = np.asarray(mask)
    if mask_kind == "object":
        return [
            (u_coord, v_coord, 1)
            for u_coord, v_coord in _sample_foreground_seed_pixels(
                mask_np,
                threshold=threshold,
                stride_px=stride_px,
                max_points=max_points,
            )
        ]

    seeds: list[tuple[int, int, int]] = []
    for part_id in sorted(int(value) for value in np.unique(mask_np) if int(value) > int(threshold)):
        part_mask = mask_np == int(part_id)
        seeds.extend(
            (u_coord, v_coord, part_id)
            for u_coord, v_coord in _sample_foreground_seed_pixels(
                part_mask,
                threshold=0,
                stride_px=stride_px,
                max_points=max_points,
            )
        )
    return seeds


def _sample_foreground_seed_pixels(mask, *, threshold: int, stride_px: int, max_points: int) -> list[tuple[int, int]]:
    import numpy as np

    foreground = np.asarray(mask) > int(threshold)
    height, width = foreground.shape
    stride = max(1, int(stride_px))
    candidates = [
        (u_coord, v_coord)
        for v_coord in range(0, height, stride)
        for u_coord in range(0, width, stride)
        if bool(foreground[v_coord, u_coord])
    ]
    max_points = max(1, int(max_points))
    if len(candidates) <= max_points:
        return candidates
    step = len(candidates) / float(max_points)
    return [candidates[min(len(candidates) - 1, int(round(index * step)))] for index in range(max_points)]


def _image_size_hw(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _rasterize_labeled_tracked_points(
    points,
    visibility,
    *,
    part_ids: Sequence[int],
    width: int,
    height: int,
    visibility_threshold: float,
    radius_px: int,
):
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint16)
    radius = max(1, int(radius_px))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    for point, visible, part_id in zip(points, visibility, part_ids):
        if float(visible) < float(visibility_threshold):
            continue
        u_coord = int(round(float(point[0])))
        v_coord = int(round(float(point[1])))
        if u_coord < 0 or v_coord < 0 or u_coord >= width or v_coord >= height:
            continue
        x0 = max(0, u_coord - radius)
        x1 = min(width, u_coord + radius + 1)
        y0 = max(0, v_coord - radius)
        y1 = min(height, v_coord + radius + 1)
        dx0 = x0 - (u_coord - radius)
        dy0 = y0 - (v_coord - radius)
        mask[y0:y1, x0:x1][disk[dy0 : dy0 + (y1 - y0), dx0 : dx0 + (x1 - x0)]] = max(1, int(part_id))
    return mask


def _part_segmentation_from_part_ids(part_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "provider": "cotracker-propagated-reference-part-mask",
        "version": 1,
        "mask_encoding": "indexed-mask-u16",
        "background_part_id": 0,
        "parts": [
            {
                "part_id": int(part_id),
                "name": f"part_{int(part_id)}",
                "role": "unknown",
                "metadata": {"source": "reference_part_mask"},
            }
            for part_id in sorted({int(part_id) for part_id in part_ids if int(part_id) > 0})
        ],
    }


def _current_mask_paths(frame: Any, mask_kind: str) -> list[str]:
    return list(frame.part_mask_paths_by_view if mask_kind == "part" else frame.mask_paths_by_view)


def _set_mask_paths(frame: Any, mask_kind: str, paths: list[str]) -> None:
    if mask_kind == "part":
        frame.part_mask_paths_by_view = paths
        frame.part_mask_path = paths[0] if len(paths) == 1 else (paths[0] if not frame.part_mask_path else frame.part_mask_path)
    else:
        frame.mask_paths_by_view = paths
        frame.mask_path = paths[0] if len(paths) == 1 else (paths[0] if not frame.mask_path else frame.mask_path)


def _resolve_eval_mask_paths(frame: Any, episode_root: Path, mask_kind: str) -> list[Path]:
    if mask_kind == "part":
        raw_paths = list(frame.part_mask_paths_by_view)
        if not raw_paths and frame.part_mask_path:
            raw_paths = [frame.part_mask_path]
    else:
        raw_paths = list(frame.mask_paths_by_view)
        if not raw_paths and frame.mask_path:
            raw_paths = [frame.mask_path]
    return [episode_root / path for path in raw_paths if path]


def _copy_mask_as_u16(source_path: Path, output_path: Path, threshold: int = 0) -> None:
    import numpy as np
    from PIL import Image

    source = np.asarray(Image.open(source_path))
    if source.ndim == 3:
        source = source[:, :, 0]
    if threshold > 0:
        source = (source > int(threshold)).astype(np.uint16)
    else:
        source = source.astype(np.uint16)
    _write_mask_u16(source.astype(np.uint16), output_path)


def _write_mask_u16(mask, output_path: Path) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="I;16").save(output_path)


def _mask_metrics(pred_path: Path, ref_path: Path, threshold: int = 0) -> dict[str, Any]:
    import numpy as np

    pred = np.asarray(_load_depth_u16(pred_path), dtype=np.uint16)
    ref = np.asarray(_load_depth_u16(ref_path), dtype=np.uint16)
    if pred.shape != ref.shape:
        raise ValueError(f"Mask shape mismatch: {pred_path} {pred.shape} vs {ref_path} {ref.shape}")
    pred_fg = pred > threshold
    ref_fg = ref > threshold
    tp = int(np.logical_and(pred_fg, ref_fg).sum())
    fp = int(np.logical_and(pred_fg, ~ref_fg).sum())
    fn = int(np.logical_and(~pred_fg, ref_fg).sum())
    intersection = tp
    union = int(np.logical_or(pred_fg, ref_fg).sum())
    return {
        "pred_mask_path": str(pred_path),
        "ref_mask_path": str(ref_path),
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "pred_pixels": int(pred_fg.sum()),
        "ref_pixels": int(ref_fg.sum()),
    }


def _summarize_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        f"mean_{key}": float(sum(float(record[key]) for record in records) / len(records))
        for key in ("iou", "precision", "recall")
    }
