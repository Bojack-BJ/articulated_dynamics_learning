from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .part_pose import _centroid, _matvec3, _pose_payload, _relative_pose, _rotation_angle_from_matrix
from .part_segmentation import part_name_lookup
from .quality_weights import observation_weight
from .pointcloud_fusion import (
    _camera_to_world_point,
    _depth_convention,
    _load_depth_u16,
    _resolve_view_camera_poses,
    _resolve_view_depth_paths,
    _resolve_view_intrinsics,
    _resolve_view_mask_paths,
    _resolve_view_part_mask_paths,
)
from ..core.serialization import load_episode, load_json, save_json


@dataclass(slots=True)
class PartPixelTrackingConfig:
    episode_path: str | Path
    output_json: str | Path | None = None
    device: str = "auto"
    cotracker_repo: str | Path | None = None
    cotracker_checkpoint: str | Path | None = None
    cotracker_model: str = "cotracker3_offline"
    unsafe_force_mps: bool = False
    reference_frame: int = 0
    frame_stride: int = 1
    seed_stride_px: int = 16
    max_tracks_per_part_view: int = 128
    max_queries_per_forward: int = 512
    min_depth_m: float = 0.05
    max_depth_m: float = 6.0
    visibility_threshold: float = 0.5
    depth_consistency_window_radius_px: int = 0
    depth_consistency_max_delta_m: float = 0.08
    repair_temporal_depth_spikes: bool = False
    depth_spike_jump_threshold_m: float = 0.08
    depth_spike_neighbor_tolerance_m: float = 0.03
    depth_spike_max_run_frames: int = 2
    require_part_mask_consistency: bool = True
    strict_object_mask_consistency: bool = False
    allow_backward_tracking: bool = True
    show_progress: bool = True
    export_cotracker_features: bool = False
    cotracker_features_output: str | Path | None = None
    dynamic_reseeding: bool = False
    dynamic_reseed_bidirectional: bool = False
    reseed_interval_frames: int = 5
    reseed_coverage_radius_px: float = 12.0
    reseed_bbox_scale: float = 1.2
    reseed_max_tracks_per_frame_view: int = 64
    reseed_max_tracks_per_view: int = 256


@dataclass(slots=True)
class TrackPartPoseEstimationConfig:
    input_path: str | Path
    output_json: str | Path | None = None
    min_tracks_per_part: int = 4
    anchor_part_id: int | None = None
    anchor_selection: str = "metadata"
    quality_weighted: bool = False
    quality_weight_field: str = "timestep_quality_score"
    track_quality_field: str = "track_quality_score"
    min_timestep_weight: float = 0.2


def _resolve_view_rgb_paths(frame: Any, episode_root: Path) -> list[Path]:
    if frame.rgb_paths_by_view:
        return [episode_root / rgb_path for rgb_path in frame.rgb_paths_by_view]
    return [episode_root / frame.rgb_path]


def _resolve_device(torch_module: Any, device: str) -> str:
    if device == "auto":
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "mps":
        if not (hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps, but PyTorch MPS is not available.")
        return "mps"
    if device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported tracking device: {device}")


def _resolve_tracking_device(torch_module: Any, config: PartPixelTrackingConfig) -> str:
    if config.device == "mps" and bool(config.unsafe_force_mps):
        return "mps"
    return _resolve_device(torch_module, config.device)


def _sync_torch_device(torch_module: Any, device: str) -> None:
    try:
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        elif str(device) == "mps" and hasattr(torch_module, "mps") and hasattr(torch_module.mps, "synchronize"):
            torch_module.mps.synchronize()
    except Exception:
        return


def _reset_torch_peak_stats(torch_module: Any, device: str) -> None:
    try:
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()
        elif str(device) == "mps" and hasattr(torch_module, "mps") and hasattr(torch_module.mps, "reset_peak_memory_stats"):
            torch_module.mps.reset_peak_memory_stats()
    except Exception:
        return


def _torch_resource_snapshot(torch_module: Any, device: str, label: str) -> dict[str, Any]:
    _sync_torch_device(torch_module, device)
    snapshot: dict[str, Any] = {
        "label": label,
        "device": str(device),
        "timestamp_s": time.time(),
    }
    try:
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            snapshot.update(
                {
                    "backend": "cuda",
                    "current_allocated_bytes": int(torch_module.cuda.memory_allocated()),
                    "driver_or_reserved_bytes": int(torch_module.cuda.memory_reserved()),
                    "max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                    "max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
                    "device_name": str(torch_module.cuda.get_device_name()),
                }
            )
        elif str(device) == "mps" and hasattr(torch_module, "mps"):
            mps = torch_module.mps
            if hasattr(mps, "current_allocated_memory"):
                snapshot["current_allocated_bytes"] = int(mps.current_allocated_memory())
            if hasattr(mps, "driver_allocated_memory"):
                snapshot["driver_or_reserved_bytes"] = int(mps.driver_allocated_memory())
            if hasattr(mps, "recommended_max_memory"):
                snapshot["recommended_max_bytes"] = int(mps.recommended_max_memory())
            snapshot["backend"] = "mps"
        else:
            snapshot["backend"] = "cpu"
    except Exception as exc:
        snapshot["error"] = str(exc)
    return snapshot


def _summarize_torch_resource_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def max_numeric(key: str) -> int | None:
        values = [
            int(sample[key])
            for sample in samples
            if isinstance(sample, dict) and isinstance(sample.get(key), (int, float))
        ]
        return max(values) if values else None

    def mib(value: int | None) -> float | None:
        return None if value is None else float(value) / (1024.0 * 1024.0)

    peak_current = max_numeric("current_allocated_bytes")
    peak_driver = max_numeric("driver_or_reserved_bytes")
    peak_max_allocated = max_numeric("max_memory_allocated_bytes")
    return {
        "sample_count": len(samples),
        "device": samples[-1].get("device") if samples else None,
        "backend": samples[-1].get("backend") if samples else None,
        "peak_current_allocated_bytes": peak_current,
        "peak_current_allocated_mib": mib(peak_current),
        "peak_driver_or_reserved_bytes": peak_driver,
        "peak_driver_or_reserved_mib": mib(peak_driver),
        "peak_max_memory_allocated_bytes": peak_max_allocated,
        "peak_max_memory_allocated_mib": mib(peak_max_allocated),
        "recommended_max_bytes": max_numeric("recommended_max_bytes"),
        "samples": samples,
    }


class _ProgressPrinter:
    def __init__(self, total_steps: int, enabled: bool) -> None:
        self.total_steps = max(1, int(total_steps))
        self.enabled = bool(enabled)
        self.start_time = time.monotonic()
        self.last_line_length = 0

    def update(self, completed_steps: int, message: str) -> None:
        if not self.enabled:
            return
        elapsed_s = max(1e-6, time.monotonic() - self.start_time)
        completed = min(max(0, int(completed_steps)), self.total_steps)
        rate = completed / elapsed_s if completed > 0 else 0.0
        remaining = max(0, self.total_steps - completed)
        eta_s = (remaining / rate) if rate > 0.0 else None
        eta_text = f"{eta_s:5.1f}s" if eta_s is not None else "  n/a"
        line = (
            f"\r[track-part-pixels] {completed:>3}/{self.total_steps:<3} "
            f"elapsed={elapsed_s:5.1f}s eta={eta_text} {message}"
        )
        padding = max(0, self.last_line_length - len(line))
        sys.stderr.write(line + (" " * padding))
        sys.stderr.flush()
        self.last_line_length = len(line)

    def finish(self, message: str) -> None:
        if not self.enabled:
            return
        self.update(self.total_steps, message)
        sys.stderr.write("\n")
        sys.stderr.flush()


def _load_rgb_frame(path: Path) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CoTracker RGB loading requires Pillow and NumPy.") from exc
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def _load_video_tensor(rgb_paths: list[Path], torch_module: Any, device: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CoTracker video loading requires NumPy.") from exc
    frames = [_load_rgb_frame(path) for path in rgb_paths]
    if not frames:
        raise ValueError("No RGB frames were found for CoTracker.")
    first_shape = frames[0].shape
    for path, frame in zip(rgb_paths, frames):
        if frame.shape != first_shape:
            raise ValueError(f"RGB frame shape mismatch for {path}: {frame.shape} != {first_shape}")
    video_np = np.stack(frames, axis=0)
    video = torch_module.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    return video.to(device)


def _sampled_frame_indices(frame_count: int, frame_stride: int) -> list[int]:
    stride = max(1, int(frame_stride))
    if frame_count <= 0:
        return []
    return list(range(0, frame_count, stride))


def _effective_fps(frames: list[Any]) -> float | None:
    deltas = [
        float(current.timestamp_s) - float(previous.timestamp_s)
        for previous, current in zip(frames[:-1], frames[1:])
        if float(current.timestamp_s) > float(previous.timestamp_s)
    ]
    if not deltas:
        return None
    mean_dt = sum(deltas) / float(len(deltas))
    if mean_dt <= 0.0:
        return None
    return 1.0 / mean_dt


def _load_cotracker_model(config: PartPixelTrackingConfig, torch_module: Any, device: str) -> Any:
    checkpoint = (
        Path(config.cotracker_checkpoint).expanduser().resolve()
        if config.cotracker_checkpoint is not None
        else None
    )
    if config.cotracker_repo is not None:
        repo = str(Path(config.cotracker_repo).expanduser().resolve())
        if checkpoint is not None:
            if not checkpoint.exists():
                raise FileNotFoundError(f"CoTracker checkpoint does not exist: {checkpoint}")
            model = torch_module.hub.load(repo, config.cotracker_model, source="local", pretrained=False)
            state_dict = torch_module.load(str(checkpoint), map_location="cpu", weights_only=True)
            model.model.load_state_dict(state_dict)
        else:
            model = torch_module.hub.load(repo, config.cotracker_model, source="local")
    else:
        if checkpoint is not None:
            raise ValueError("--cotracker-checkpoint requires --cotracker-repo so the local CoTracker code can be loaded.")
        model = torch_module.hub.load("facebookresearch/co-tracker", config.cotracker_model)
    return model.to(device).eval()


def _resolve_cotracker_updateformer(model: Any) -> Any:
    """Resolve CoTracker's update transformer without modifying the vendored repository."""
    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        updateformer = getattr(candidate, "updateformer", None)
        if updateformer is not None:
            return updateformer
    raise RuntimeError(
        "Unable to find CoTracker updateformer. Feature export currently supports "
        "CoTracker models exposing model.updateformer or updateformer."
    )


class _CoTrackerFeatureCapture:
    """Capture final per-query hidden tokens entering CoTracker's flow head."""

    def __init__(self, model: Any) -> None:
        updateformer = _resolve_cotracker_updateformer(model)
        flow_head = getattr(updateformer, "flow_head", None)
        if flow_head is None or not hasattr(flow_head, "register_forward_pre_hook"):
            raise RuntimeError("CoTracker updateformer does not expose a hookable flow_head.")
        self.latest: Any = None
        self._handle = flow_head.register_forward_pre_hook(self._capture)

    def _capture(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs:
            self.latest = inputs[0].detach()

    def reset(self) -> None:
        self.latest = None

    def pooled_seed_features(self, seed_count: int) -> Any:
        if self.latest is None:
            raise RuntimeError("CoTracker feature hook did not capture hidden tokens.")
        hidden = self.latest
        if getattr(hidden, "ndim", 0) != 4 or int(hidden.shape[0]) != 1:
            raise RuntimeError(f"Unexpected CoTracker hidden-token shape: {tuple(hidden.shape)}")
        if int(hidden.shape[1]) < int(seed_count):
            raise RuntimeError(
                f"CoTracker returned {int(hidden.shape[1])} hidden tracks for {int(seed_count)} seed queries."
            )
        pooled = hidden[0, :seed_count].mean(dim=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return pooled.detach().cpu()

    def close(self) -> None:
        self._handle.remove()


def _part_metadata(part_segmentation: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(part_segmentation, dict):
        return {}
    parts: dict[int, dict[str, Any]] = {}
    for raw_part in part_segmentation.get("parts", []):
        if isinstance(raw_part, dict):
            part_id = int(raw_part.get("part_id", 0))
            if part_id > 0:
                parts[part_id] = raw_part
    return parts


def _count_part_pixels(mask_u16: list[list[int]], part_id: int) -> int:
    return sum(1 for row in mask_u16 for value in row if int(value) == part_id)


def _count_foreground_pixels(mask_u16: list[list[int]]) -> int:
    return sum(1 for row in mask_u16 for value in row if int(value) > 0)


def _sample_part_seed_pixels(mask_u16: list[list[int]], part_id: int, stride_px: int, max_points: int) -> list[tuple[int, int]]:
    height = len(mask_u16)
    width = len(mask_u16[0]) if height else 0
    stride = max(1, int(stride_px))
    candidates: list[tuple[int, int]] = []
    for v_coord in range(0, height, stride):
        row = mask_u16[v_coord]
        for u_coord in range(0, width, stride):
            if int(row[u_coord]) == part_id:
                candidates.append((u_coord, v_coord))
    if len(candidates) <= max_points:
        return candidates

    # Deterministic uniform downsampling keeps tests reproducible and avoids a random seed knob.
    step = len(candidates) / float(max_points)
    return [candidates[min(len(candidates) - 1, int(round(index * step)))] for index in range(max_points)]


def _sample_foreground_seed_pixels(mask_u16: list[list[int]], stride_px: int, max_points: int) -> list[tuple[int, int]]:
    height = len(mask_u16)
    width = len(mask_u16[0]) if height else 0
    stride = max(1, int(stride_px))
    candidates: list[tuple[int, int]] = []
    for v_coord in range(0, height, stride):
        row = mask_u16[v_coord]
        for u_coord in range(0, width, stride):
            if int(row[u_coord]) > 0:
                candidates.append((u_coord, v_coord))
    if len(candidates) <= max_points:
        return candidates
    step = len(candidates) / float(max_points)
    return [candidates[min(len(candidates) - 1, int(round(index * step)))] for index in range(max_points)]


def _sample_uncovered_foreground_seed_pixels(
    mask_u16: list[list[int]],
    depth_u16: list[list[int]],
    covered_uv: list[tuple[float, float]],
    stride_px: int,
    coverage_radius_px: float,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
    expected_part_id: int = -1,
) -> list[tuple[int, int]]:
    """Sample valid foreground pixels not already explained by visible tracks."""
    height = min(len(mask_u16), len(depth_u16))
    width = min(len(mask_u16[0]), len(depth_u16[0])) if height else 0
    stride = max(1, int(stride_px))
    radius_sq = max(0.0, float(coverage_radius_px)) ** 2
    candidates: list[tuple[int, int]] = []
    for v_coord in range(0, height, stride):
        for u_coord in range(0, width, stride):
            mask_value = int(mask_u16[v_coord][u_coord])
            if (expected_part_id < 0 and mask_value <= 0) or (
                expected_part_id >= 0 and mask_value != expected_part_id
            ):
                continue
            depth_m = float(depth_u16[v_coord][u_coord]) / 1000.0
            if depth_m < float(min_depth_m) or depth_m > float(max_depth_m):
                continue
            if any((u_coord - u) ** 2 + (v_coord - v) ** 2 <= radius_sq for u, v in covered_uv):
                continue
            candidates.append((u_coord, v_coord))
    if len(candidates) <= max_points:
        return candidates
    step = len(candidates) / float(max_points)
    return [candidates[min(len(candidates) - 1, int(round(index * step)))] for index in range(max_points)]


def _expanded_robust_bbox(
    points: list[list[float]],
    scale: float,
) -> tuple[list[float], list[float]] | None:
    """Return a quantile-trimmed AABB expanded around its center."""
    valid = [point for point in points if len(point) == 3 and all(math.isfinite(float(value)) for value in point)]
    if not valid:
        return None
    lower: list[float] = []
    upper: list[float] = []
    expansion = max(1.0, float(scale))
    for axis in range(3):
        values = sorted(float(point[axis]) for point in valid)
        low_index = min(len(values) - 1, int(math.floor(0.01 * (len(values) - 1))))
        high_index = min(len(values) - 1, int(math.ceil(0.99 * (len(values) - 1))))
        low = values[low_index]
        high = values[high_index]
        center = 0.5 * (low + high)
        half_extent = max(0.025, 0.5 * (high - low) * expansion)
        lower.append(center - half_extent)
        upper.append(center + half_extent)
    return lower, upper


def _point_in_bbox(point: list[float], bbox: tuple[list[float], list[float]] | None) -> bool:
    if bbox is None or len(point) != 3:
        return False
    lower, upper = bbox
    return all(float(lower[axis]) <= float(point[axis]) <= float(upper[axis]) for axis in range(3))


def _visibility_value(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0 if bool(raw) else 0.0


def _backproject_track_sample(
    u_float: float,
    v_float: float,
    depth_u16: list[list[int]],
    part_mask_u16: list[list[int]] | None,
    expected_part_id: int,
    intrinsics: dict[str, float],
    camera_pose: list[list[float]],
    depth_convention: str,
    min_depth_m: float,
    max_depth_m: float,
    require_part_mask_consistency: bool,
) -> tuple[list[float] | None, bool, bool]:
    height = len(depth_u16)
    width = len(depth_u16[0]) if height else 0
    u_coord = int(round(float(u_float)))
    v_coord = int(round(float(v_float)))
    if u_coord < 0 or v_coord < 0 or u_coord >= width or v_coord >= height:
        return None, False, False

    mask_consistent = True
    if part_mask_u16 is not None:
        if int(expected_part_id) < 0:
            mask_consistent = int(part_mask_u16[v_coord][u_coord]) > 0
        else:
            mask_consistent = int(part_mask_u16[v_coord][u_coord]) == int(expected_part_id)
        if require_part_mask_consistency and not mask_consistent:
            return None, False, mask_consistent

    depth_m = depth_u16[v_coord][u_coord] / 1000.0
    if depth_m < min_depth_m or depth_m > max_depth_m:
        return None, False, mask_consistent

    xyz_world = _camera_to_world_point(
        u_coord=u_coord,
        v_coord=v_coord,
        depth_m=depth_m,
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        depth_convention=depth_convention,
    )
    return [float(value) for value in xyz_world], True, mask_consistent


def _backproject_track_sample_depth_consistent(
    u_float: float,
    v_float: float,
    depth_u16: list[list[int]],
    part_mask_u16: list[list[int]] | None,
    expected_part_id: int,
    intrinsics: dict[str, float],
    camera_pose: list[list[float]],
    depth_convention: str,
    min_depth_m: float,
    max_depth_m: float,
    require_part_mask_consistency: bool,
    window_radius_px: int,
    previous_depth_m: float | None,
    max_delta_m: float,
) -> tuple[list[float] | None, bool, bool, float | None]:
    radius = max(0, int(window_radius_px))
    if radius == 0:
        xyz, valid, mask_consistent = _backproject_track_sample(
            u_float,
            v_float,
            depth_u16,
            part_mask_u16,
            expected_part_id,
            intrinsics,
            camera_pose,
            depth_convention,
            min_depth_m,
            max_depth_m,
            require_part_mask_consistency,
        )
        u_coord = int(round(float(u_float)))
        v_coord = int(round(float(v_float)))
        depth_m = float(depth_u16[v_coord][u_coord]) / 1000.0 if valid else None
        return xyz, valid, mask_consistent, depth_m

    height = len(depth_u16)
    width = len(depth_u16[0]) if height else 0
    center_u = int(round(float(u_float)))
    center_v = int(round(float(v_float)))
    if center_u < 0 or center_v < 0 or center_u >= width or center_v >= height:
        return None, False, False, None

    def mask_matches(u_coord: int, v_coord: int) -> bool:
        if part_mask_u16 is None:
            return True
        value = int(part_mask_u16[v_coord][u_coord])
        return value > 0 if int(expected_part_id) < 0 else value == int(expected_part_id)

    center_mask_consistent = mask_matches(center_u, center_v)
    if require_part_mask_consistency and not center_mask_consistent:
        return None, False, False, None

    candidates: list[float] = []
    for v_coord in range(max(0, center_v - radius), min(height, center_v + radius + 1)):
        for u_coord in range(max(0, center_u - radius), min(width, center_u + radius + 1)):
            if not mask_matches(u_coord, v_coord):
                continue
            depth_m = float(depth_u16[v_coord][u_coord]) / 1000.0
            if min_depth_m <= depth_m <= max_depth_m:
                candidates.append(depth_m)
    if previous_depth_m is not None:
        candidates = [value for value in candidates if abs(value - previous_depth_m) <= max_delta_m]
    if not candidates:
        return None, False, center_mask_consistent, None

    candidates.sort()
    selected_depth_m = candidates[len(candidates) // 2]
    xyz_world = _camera_to_world_point(
        u_coord=center_u,
        v_coord=center_v,
        depth_m=selected_depth_m,
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        depth_convention=depth_convention,
    )
    return [float(value) for value in xyz_world], True, center_mask_consistent, selected_depth_m


def _repair_temporal_depth_spikes(
    samples: list[dict[str, Any]],
    *,
    intrinsics: dict[str, float],
    camera_poses: list[list[list[float]]],
    depth_convention: str,
    jump_threshold_m: float,
    neighbor_tolerance_m: float,
    max_run_frames: int,
) -> int:
    """Repair short A-B-A depth toggles without constraining persistent motion."""
    if len(samples) < 3 or max_run_frames < 1:
        return 0
    repaired = 0
    index = 1
    while index < len(samples) - 1:
        before_depth = samples[index - 1].get("selected_depth_m")
        if before_depth is None:
            index += 1
            continue
        accepted_end: int | None = None
        for run_length in range(1, max_run_frames + 1):
            after_index = index + run_length
            if after_index >= len(samples):
                break
            after_depth = samples[after_index].get("selected_depth_m")
            run_depths = [samples[offset].get("selected_depth_m") for offset in range(index, after_index)]
            if after_depth is None or any(value is None for value in run_depths):
                continue
            if abs(float(before_depth) - float(after_depth)) > neighbor_tolerance_m:
                continue
            baseline = 0.5 * (float(before_depth) + float(after_depth))
            if all(abs(float(value) - baseline) > jump_threshold_m for value in run_depths):
                accepted_end = after_index
                break
        if accepted_end is None:
            index += 1
            continue
        before_value = float(before_depth)
        after_value = float(samples[accepted_end]["selected_depth_m"])
        interpolation_span = accepted_end - index + 1
        for offset in range(index, accepted_end):
            alpha = (offset - index + 1) / interpolation_span
            replacement_depth = (1.0 - alpha) * before_value + alpha * after_value
            sample = samples[offset]
            uv = sample.get("uv")
            frame_index = int(sample.get("frame_index", offset))
            if not isinstance(uv, list) or len(uv) != 2 or not 0 <= frame_index < len(camera_poses):
                continue
            sample["raw_selected_depth_m"] = sample.get("selected_depth_m")
            sample["selected_depth_m"] = replacement_depth
            sample["xyz_world"] = [
                float(value)
                for value in _camera_to_world_point(
                    u_coord=int(round(float(uv[0]))),
                    v_coord=int(round(float(uv[1]))),
                    depth_m=replacement_depth,
                    intrinsics=intrinsics,
                    camera_pose=camera_poses[frame_index],
                    depth_convention=depth_convention,
                )
            ]
            sample["depth_spike_repaired"] = True
            repaired += 1
        index = accepted_end
    return repaired


def _choose_reference_frame_for_part(
    part_id: int,
    requested_reference_frame: int,
    part_masks_by_frame_view: list[list[Path]],
) -> int:
    if requested_reference_frame >= 0:
        return min(requested_reference_frame, max(0, len(part_masks_by_frame_view) - 1))
    best_frame_index = 0
    best_count = -1
    for frame_index, paths_by_view in enumerate(part_masks_by_frame_view):
        count = 0
        for mask_path in paths_by_view:
            count += _count_part_pixels(_load_depth_u16(mask_path), part_id)
        if count > best_count:
            best_count = count
            best_frame_index = frame_index
    return best_frame_index


def _choose_reference_frame_for_object(requested_reference_frame: int, masks_by_frame_view: list[list[Path]]) -> int:
    if requested_reference_frame >= 0:
        return min(requested_reference_frame, max(0, len(masks_by_frame_view) - 1))
    best_frame_index = 0
    best_count = -1
    for frame_index, paths_by_view in enumerate(masks_by_frame_view):
        count = 0
        for mask_path in paths_by_view:
            count += _count_foreground_pixels(_load_depth_u16(mask_path))
        if count > best_count:
            best_count = count
            best_frame_index = frame_index
    return best_frame_index


def _fit_rigid_transform(
    source_points: list[list[float]],
    target_points: list[list[float]],
    weights: list[float] | None = None,
) -> tuple[list[list[float]], list[float], float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Track-based pose estimation requires NumPy.") from exc

    if len(source_points) != len(target_points):
        raise ValueError("Source and target point counts must match.")
    if len(source_points) < 3:
        raise ValueError("At least three point correspondences are required.")

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if weights is None:
        weight = np.ones((source.shape[0],), dtype=float)
    else:
        weight = np.asarray(weights, dtype=float)
    weight = np.maximum(weight, 1e-6)
    weight = weight / np.sum(weight)

    source_center = np.sum(source * weight[:, None], axis=0)
    target_center = np.sum(target * weight[:, None], axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ (target_centered * weight[:, None])
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    translation = target_center - rotation @ source_center
    residuals = target - ((rotation @ source.T).T + translation)
    rms = float(math.sqrt(float(np.mean(np.sum(residuals * residuals, axis=1)))))
    return rotation.tolist(), translation.tolist(), rms


def _choose_anchor_part_id(part_metadata: dict[int, dict[str, Any]], track_counts: dict[int, int], override: int | None) -> int:
    if override is not None and override in track_counts:
        return int(override)
    for part_id, raw_part in sorted(part_metadata.items()):
        if str(raw_part.get("role")) in {"base", "static", "fixed_child"} and part_id in track_counts:
            return part_id
    if not track_counts:
        return 0
    return max(sorted(track_counts), key=lambda part_id: track_counts[part_id])


def _track_cluster_motion_range(tracks: list[dict[str, Any]], min_tracks_per_frame: int) -> float:
    positions_by_frame: dict[int, list[list[float]]] = {}
    for track in tracks:
        for sample in track.get("samples", []):
            if not bool(sample.get("visible", False)) or not bool(sample.get("depth_valid", True)):
                continue
            xyz_world = sample.get("xyz_world")
            if not isinstance(xyz_world, list) or len(xyz_world) != 3:
                continue
            positions_by_frame.setdefault(int(sample.get("frame_index", 0)), []).append(
                [float(value) for value in xyz_world]
            )
    centroids = [
        [float(median(point[axis] for point in positions_by_frame[frame_index])) for axis in range(3)]
        for frame_index in sorted(positions_by_frame)
        if len(positions_by_frame[frame_index]) >= min_tracks_per_frame
    ]
    if len(centroids) < 2:
        return math.inf
    return max(
        math.sqrt(sum((first[axis] - second[axis]) ** 2 for axis in range(3)))
        for first in centroids
        for second in centroids
    )


class PartPixelTracker:
    def __init__(self, config: PartPixelTrackingConfig) -> None:
        self.config = config

    def track(self) -> Path:
        total_started = time.perf_counter()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Part pixel tracking requires PyTorch and CoTracker. Install the tracking extra and CoTracker first."
            ) from exc

        setup_started = time.perf_counter()
        episode_path = Path(self.config.episode_path).resolve()
        episode_root = episode_path.parent
        episode = load_episode(episode_path)
        if not episode.frames:
            raise ValueError("Episode has no frames to track.")
        sampled_frame_indices = _sampled_frame_indices(len(episode.frames), self.config.frame_stride)
        sampled_frames = [episode.frames[frame_index] for frame_index in sampled_frame_indices]
        if not sampled_frames:
            raise ValueError("Temporal sampling removed every frame. Check --frame-stride.")

        output_json = (
            Path(self.config.output_json).resolve()
            if self.config.output_json is not None
            else episode_root / "part_tracks.json"
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)

        device = _resolve_tracking_device(torch, self.config)
        resource_samples: list[dict[str, Any]] = []
        _reset_torch_peak_stats(torch, device)
        resource_samples.append(_torch_resource_snapshot(torch, device, "before_model_load"))
        model_started = time.perf_counter()
        model = _load_cotracker_model(self.config, torch, device)
        model_load_s = time.perf_counter() - model_started
        resource_samples.append(_torch_resource_snapshot(torch, device, "after_model_load"))
        depth_convention = _depth_convention(episode.metadata)
        part_segmentation = (
            dict(episode.metadata.get("part_segmentation", {}))
            if isinstance(episode.metadata.get("part_segmentation"), dict)
            else None
        )
        part_names = part_name_lookup(part_segmentation)
        part_metadata = _part_metadata(part_segmentation)
        object_mask_mode = False
        if not part_metadata:
            object_mask_mode = True
            part_segmentation = {
                "source": "object-mask-tracking-fallback",
                "background_part_id": 0,
                "parts": [{"part_id": 1, "name": "object", "role": "object"}],
            }
            part_names = {1: "object"}
            part_metadata = {1: {"part_id": 1, "name": "object", "role": "object"}}

        rgb_paths_by_frame_view = [_resolve_view_rgb_paths(frame, episode_root) for frame in sampled_frames]
        depth_paths_by_frame_view = [_resolve_view_depth_paths(frame, episode_root) for frame in sampled_frames]
        part_masks_by_frame_view = [
            (
                _resolve_view_mask_paths(frame, episode_root)
                if object_mask_mode
                else _resolve_view_part_mask_paths(frame, episode_root)
            )
            for frame in sampled_frames
        ]
        if not all(part_masks_by_frame_view):
            if object_mask_mode:
                raise ValueError("Episode does not contain per-view object masks. Run segment-episode-masks first.")
            raise ValueError("Episode does not contain per-view part masks. Re-record with --part-segmentation-masks.")

        view_count = min(len(paths) for paths in rgb_paths_by_frame_view)
        if view_count <= 0:
            raise ValueError("Episode does not contain RGB views.")

        camera_poses_by_frame_view: list[list[list[list[float]]]] = []
        pose_sources: set[str] = set()
        for frame, depth_paths in zip(sampled_frames, depth_paths_by_frame_view):
            camera_poses, pose_source = _resolve_view_camera_poses(frame, episode.metadata, len(depth_paths))
            camera_poses_by_frame_view.append(camera_poses)
            pose_sources.add(pose_source)

        reference_frame_by_part = {
            part_id: (
                _choose_reference_frame_for_object(self.config.reference_frame, part_masks_by_frame_view)
                if object_mask_mode
                else _choose_reference_frame_for_part(part_id, self.config.reference_frame, part_masks_by_frame_view)
            )
            for part_id in sorted(part_metadata)
        }
        depth_cache: dict[Path, list[list[int]]] = {}
        setup_s = time.perf_counter() - setup_started - model_load_s
        video_load_s = 0.0
        cotracker_forward_s = 0.0
        track_postprocess_s = 0.0
        progress = _ProgressPrinter(
            total_steps=view_count * max(1, len(part_metadata)),
            enabled=bool(self.config.show_progress),
        )

        def load_depth_cached(path: Path) -> list[list[int]]:
            if path not in depth_cache:
                depth_cache[path] = _load_depth_u16(path)
            return depth_cache[path]

        tracks: list[dict[str, Any]] = []
        feature_track_ids: list[int] = []
        feature_embeddings: list[Any] = []
        feature_capture = _CoTrackerFeatureCapture(model) if self.config.export_cotracker_features else None
        track_id = 0
        completed_steps = 0
        with torch.no_grad():
            for view_index in range(view_count):
                progress.update(
                    completed_steps,
                    f"loading RGB video for view {view_index + 1}/{view_count}",
                )
                view_rgb_paths = [paths[view_index] for paths in rgb_paths_by_frame_view]
                view_intrinsics = _resolve_view_intrinsics(
                    episode.camera_intrinsics, episode.metadata, view_index
                )
                video_started = time.perf_counter()
                video = _load_video_tensor(view_rgb_paths, torch, device)
                video_load_s += time.perf_counter() - video_started
                resource_samples.append(_torch_resource_snapshot(torch, device, f"view_{view_index}_video_loaded"))

                def run_query_batch(
                    query_records: list[dict[str, Any]],
                    stage: str,
                ) -> int:
                    nonlocal track_id, cotracker_forward_s, track_postprocess_s
                    if not query_records:
                        return 0
                    query_limit = max(1, int(self.config.max_queries_per_forward))
                    if len(query_records) > query_limit:
                        return sum(
                            run_query_batch(query_records[start : start + query_limit], f"{stage}_chunk_{start // query_limit}")
                            for start in range(0, len(query_records), query_limit)
                        )
                    query_payload = [
                        [float(record["frame_index"]), float(record["u"]), float(record["v"])]
                        for record in query_records
                    ]
                    queries = torch.tensor([query_payload], dtype=torch.float32, device=device)
                    if feature_capture is not None:
                        feature_capture.reset()
                    forward_started = time.perf_counter()
                    pred_tracks, pred_visibility = model(
                        video,
                        queries=queries,
                        backward_tracking=bool(self.config.allow_backward_tracking),
                    )
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif str(device).startswith("mps") and hasattr(torch, "mps"):
                        torch.mps.synchronize()
                    cotracker_forward_s += time.perf_counter() - forward_started
                    postprocess_started = time.perf_counter()
                    resource_samples.append(
                        _torch_resource_snapshot(torch, device, f"view_{view_index}_{stage}_cotracker_done")
                    )
                    pred_tracks = pred_tracks.detach().cpu()
                    pred_visibility = pred_visibility.detach().cpu()
                    query_features = (
                        feature_capture.pooled_seed_features(len(query_records))
                        if feature_capture is not None
                        else None
                    )
                    emitted = 0
                    for query_index, record in enumerate(query_records):
                        part_id = int(record["part_id"])
                        reference_frame = int(record["frame_index"])
                        seed_u = int(record["u"])
                        seed_v = int(record["v"])
                        dynamic_birth = bool(record.get("dynamic", False))
                        samples: list[dict[str, Any]] = []
                        reference_xyz: list[float] | None = None
                        previous_depth_m: float | None = None
                        for frame_index, frame in enumerate(sampled_frames):
                            if frame_index >= pred_tracks.shape[1]:
                                continue
                            uv = pred_tracks[0, frame_index, query_index].tolist()
                            tracker_visibility = _visibility_value(pred_visibility[0, frame_index, query_index].item())
                            source_frame_index = sampled_frame_indices[frame_index]
                            depth_path = depth_paths_by_frame_view[frame_index][view_index]
                            part_mask_path = (
                                part_masks_by_frame_view[frame_index][view_index]
                                if view_index < len(part_masks_by_frame_view[frame_index])
                                else None
                            )
                            xyz, depth_valid, mask_consistent, selected_depth_m = (
                                _backproject_track_sample_depth_consistent(
                                u_float=float(uv[0]),
                                v_float=float(uv[1]),
                                depth_u16=load_depth_cached(depth_path),
                                part_mask_u16=load_depth_cached(part_mask_path) if part_mask_path is not None else None,
                                expected_part_id=-1 if object_mask_mode else part_id,
                                intrinsics=view_intrinsics,
                                camera_pose=camera_poses_by_frame_view[frame_index][view_index],
                                depth_convention=depth_convention,
                                min_depth_m=float(self.config.min_depth_m),
                                max_depth_m=float(self.config.max_depth_m),
                                # Object masks are soft evidence: an opening door may
                                # legitimately expand beyond a propagated mask.
                                require_part_mask_consistency=(
                                    bool(self.config.require_part_mask_consistency)
                                    and (
                                        not object_mask_mode
                                        or bool(self.config.strict_object_mask_consistency)
                                    )
                                ),
                                window_radius_px=int(self.config.depth_consistency_window_radius_px),
                                previous_depth_m=previous_depth_m,
                                max_delta_m=float(self.config.depth_consistency_max_delta_m),
                            )
                            )
                            if depth_valid and selected_depth_m is not None:
                                previous_depth_m = selected_depth_m
                            after_birth = (
                                not dynamic_birth
                                or bool(self.config.dynamic_reseed_bidirectional)
                                or frame_index >= reference_frame
                            )
                            visible = (
                                after_birth
                                and tracker_visibility >= float(self.config.visibility_threshold)
                                and depth_valid
                                and xyz is not None
                            )
                            if frame_index == reference_frame and xyz is not None:
                                reference_xyz = xyz
                            samples.append(
                                {
                                    "frame_index": frame_index,
                                    "source_frame_index": source_frame_index,
                                    "timestamp_s": float(frame.timestamp_s),
                                    "uv": [float(uv[0]), float(uv[1])],
                                    "xyz_world": xyz,
                                    "visible": bool(visible),
                                    "tracker_visibility": float(tracker_visibility),
                                    "depth_valid": bool(depth_valid),
                                    "selected_depth_m": selected_depth_m,
                                    "mask_consistent": bool(mask_consistent),
                                    "confidence": float(tracker_visibility if visible else 0.0),
                                }
                            )
                        if self.config.repair_temporal_depth_spikes:
                            _repair_temporal_depth_spikes(
                                samples,
                                intrinsics=view_intrinsics,
                                camera_poses=[poses[view_index] for poses in camera_poses_by_frame_view],
                                depth_convention=depth_convention,
                                jump_threshold_m=float(self.config.depth_spike_jump_threshold_m),
                                neighbor_tolerance_m=float(self.config.depth_spike_neighbor_tolerance_m),
                                max_run_frames=int(self.config.depth_spike_max_run_frames),
                            )
                            reference_sample = next(
                                (sample for sample in samples if int(sample["frame_index"]) == reference_frame),
                                None,
                            )
                            if reference_sample is not None and reference_sample.get("xyz_world") is not None:
                                reference_xyz = [float(value) for value in reference_sample["xyz_world"]]
                        if reference_xyz is None:
                            continue
                        tracks.append(
                            {
                                "track_id": track_id,
                                "part_id": part_id,
                                "part_name": part_names.get(part_id, f"part_{part_id}"),
                                "view_index": view_index,
                                "query_frame_index": reference_frame,
                                "query_source_frame_index": sampled_frame_indices[reference_frame],
                                "query_uv": [float(seed_u), float(seed_v)],
                                "reference_xyz_world": reference_xyz,
                                "seed_source": "dynamic_reseed" if dynamic_birth else "reference_frame",
                                "samples": samples,
                            }
                        )
                        if query_features is not None:
                            feature_track_ids.append(track_id)
                            feature_embeddings.append(query_features[query_index])
                        track_id += 1
                        emitted += 1
                    track_postprocess_s += time.perf_counter() - postprocess_started
                    return emitted

                view_track_start = len(tracks)
                initial_queries: list[dict[str, Any]] = []
                for part_id in sorted(part_metadata):
                    reference_frame = reference_frame_by_part[part_id]
                    if reference_frame >= len(part_masks_by_frame_view):
                        continue
                    mask_paths_at_reference = part_masks_by_frame_view[reference_frame]
                    if view_index >= len(mask_paths_at_reference):
                        continue
                    reference_mask = load_depth_cached(mask_paths_at_reference[view_index])
                    if object_mask_mode:
                        seed_pixels = _sample_foreground_seed_pixels(
                            reference_mask,
                            stride_px=self.config.seed_stride_px,
                            max_points=max(1, self.config.max_tracks_per_part_view),
                        )
                    else:
                        seed_pixels = _sample_part_seed_pixels(
                            reference_mask,
                            part_id=part_id,
                            stride_px=self.config.seed_stride_px,
                            max_points=max(1, self.config.max_tracks_per_part_view),
                        )
                    part_name = part_names.get(part_id, f"part_{part_id}")
                    if not seed_pixels:
                        completed_steps += 1
                        progress.update(
                            completed_steps,
                            f"view {view_index + 1}/{view_count} {part_name}: no seed pixels",
                        )
                        continue

                    initial_queries.extend(
                        {
                            "part_id": part_id,
                            "frame_index": reference_frame,
                            "u": u_coord,
                            "v": v_coord,
                            "dynamic": False,
                        }
                        for u_coord, v_coord in seed_pixels
                    )
                    completed_steps += 1
                    progress.update(
                        completed_steps,
                        f"view {view_index + 1}/{view_count} {part_name}: queued {len(seed_pixels)} seeds",
                    )
                if initial_queries:
                    progress.update(
                        completed_steps,
                        (
                            f"view {view_index + 1}/{view_count}: running one CoTracker forward "
                            f"for {len(initial_queries)} seeds across {len(part_metadata)} parts"
                        ),
                    )
                    run_query_batch(initial_queries, "all_parts")

                if bool(self.config.dynamic_reseeding):
                    initial_view_tracks = tracks[view_track_start:]
                    envelope_points = [
                        sample["xyz_world"]
                        for track in initial_view_tracks
                        for sample in track.get("samples", [])
                        if bool(sample.get("visible", False))
                        and isinstance(sample.get("xyz_world"), list)
                    ]
                    bbox = _expanded_robust_bbox(envelope_points, self.config.reseed_bbox_scale)
                    interval = max(1, int(self.config.reseed_interval_frames))
                    all_dynamic_queries: list[dict[str, Any]] = []
                    for part_id in sorted(part_metadata):
                        part_view_tracks = [
                            track for track in initial_view_tracks if int(track.get("part_id", -1)) == part_id
                        ]
                        dynamic_queries: list[dict[str, Any]] = []
                        accepted_xyz: list[list[float]] = []
                        for frame_index in range(interval, len(sampled_frames), interval):
                            if len(dynamic_queries) >= max(1, int(self.config.reseed_max_tracks_per_view)):
                                break
                            mask_path = part_masks_by_frame_view[frame_index][view_index]
                            depth_path = depth_paths_by_frame_view[frame_index][view_index]
                            frame_mask = load_depth_cached(mask_path)
                            frame_depth = load_depth_cached(depth_path)
                            covered_uv: list[tuple[float, float]] = []
                            for track in part_view_tracks:
                                sample = next(
                                    (
                                        item
                                        for item in track.get("samples", [])
                                        if int(item.get("frame_index", -1)) == frame_index
                                        and bool(item.get("visible", False))
                                    ),
                                    None,
                                )
                                if sample is not None and isinstance(sample.get("uv"), list):
                                    covered_uv.append((float(sample["uv"][0]), float(sample["uv"][1])))
                            remaining = max(0, int(self.config.reseed_max_tracks_per_view) - len(dynamic_queries))
                            frame_limit = min(max(1, int(self.config.reseed_max_tracks_per_frame_view)), remaining)
                            expected_part_id = -1 if object_mask_mode else part_id
                            candidates = _sample_uncovered_foreground_seed_pixels(
                                frame_mask,
                                frame_depth,
                                covered_uv,
                                stride_px=self.config.seed_stride_px,
                                coverage_radius_px=self.config.reseed_coverage_radius_px,
                                min_depth_m=self.config.min_depth_m,
                                max_depth_m=self.config.max_depth_m,
                                max_points=frame_limit,
                                expected_part_id=expected_part_id,
                            )
                            for u_coord, v_coord in candidates:
                                xyz, depth_valid, _ = _backproject_track_sample(
                                    u_float=float(u_coord),
                                    v_float=float(v_coord),
                                    depth_u16=frame_depth,
                                    part_mask_u16=frame_mask,
                                    expected_part_id=expected_part_id,
                                    intrinsics=view_intrinsics,
                                    camera_pose=camera_poses_by_frame_view[frame_index][view_index],
                                    depth_convention=depth_convention,
                                    min_depth_m=float(self.config.min_depth_m),
                                    max_depth_m=float(self.config.max_depth_m),
                                    require_part_mask_consistency=True,
                                )
                                if not depth_valid or xyz is None or not _point_in_bbox(xyz, bbox):
                                    continue
                                if any(
                                    sum((xyz[axis] - other[axis]) ** 2 for axis in range(3)) < 0.0004
                                    for other in accepted_xyz
                                ):
                                    continue
                                accepted_xyz.append(xyz)
                                dynamic_queries.append(
                                    {
                                        "part_id": part_id,
                                        "frame_index": frame_index,
                                        "u": u_coord,
                                        "v": v_coord,
                                        "dynamic": True,
                                    }
                                )
                        if dynamic_queries:
                            all_dynamic_queries.extend(dynamic_queries)
                    if all_dynamic_queries:
                        progress.update(
                            completed_steps,
                            (
                                f"view {view_index + 1}/{view_count}: tracking "
                                f"{len(all_dynamic_queries)} dynamic seeds across all parts"
                            ),
                        )
                        run_query_batch(all_dynamic_queries, "all_parts_dynamic_reseed")
        progress.finish(f"done; emitted {len(tracks)} total tracks")
        if feature_capture is not None:
            feature_capture.close()
        resource_samples.append(_torch_resource_snapshot(torch, device, "finished"))

        feature_save_started = time.perf_counter()
        feature_artifact: dict[str, Any] | None = None
        if feature_capture is not None:
            if not feature_embeddings:
                raise ValueError("CoTracker feature export was requested, but no valid 3D tracks were emitted.")
            import numpy as np

            feature_output = (
                Path(self.config.cotracker_features_output).expanduser().resolve()
                if self.config.cotracker_features_output is not None
                else output_json.with_name("cotracker_features.npz")
            )
            feature_output.parent.mkdir(parents=True, exist_ok=True)
            embedding_array = np.stack([value.numpy() for value in feature_embeddings]).astype(np.float32)
            np.savez_compressed(
                feature_output,
                track_ids=np.asarray(feature_track_ids, dtype=np.int64),
                embeddings=embedding_array,
            )
            feature_artifact = {
                "path": str(feature_output),
                "track_count": int(embedding_array.shape[0]),
                "feature_dim": int(embedding_array.shape[1]),
                "pooling": "final-updateformer-token-temporal-mean-l2",
                "diagnostic_only": True,
            }
        feature_serialization_s = time.perf_counter() - feature_save_started

        track_counts: dict[int, int] = {}
        for track in tracks:
            part_id = int(track["part_id"])
            track_counts[part_id] = track_counts.get(part_id, 0) + 1

        payload = {
                "input_episode_path": str(episode_path),
                "estimator": "cotracker-depth-backprojection",
                "frame_count": len(sampled_frames),
                "source_frame_count": len(episode.frames),
                "sampled_frame_indices": sampled_frame_indices,
                "effective_tracking_fps_hz": _effective_fps(sampled_frames),
                "view_count": view_count,
                "device": device,
                "resource_usage": {
                    "torch": _summarize_torch_resource_samples(resource_samples),
                },
                "runtime_profile": {
                    "scope": "cotracker_tracking_feature_generation_and_rgbd_lifting",
                    "episode_and_camera_setup_s": setup_s,
                    "model_load_s": model_load_s,
                    "video_load_s": video_load_s,
                    "cotracker_accelerator_forward_s": cotracker_forward_s,
                    "rgbd_lifting_and_track_postprocess_s": track_postprocess_s,
                    "feature_serialization_s": feature_serialization_s,
                    "artifact_serialization_s": None,
                    "total_wall_s": None,
                    "device": device,
                    "view_count": view_count,
                    "source_frame_count": len(episode.frames),
                    "tracked_frame_count": len(sampled_frames),
                    "track_count": len(tracks),
                    "max_queries_per_forward": self.config.max_queries_per_forward,
                    "inter_object_parallelism": 1,
                },
                "cotracker_model": self.config.cotracker_model,
                "cotracker_repo": None if self.config.cotracker_repo is None else str(self.config.cotracker_repo),
                "cotracker_checkpoint": None if self.config.cotracker_checkpoint is None else str(self.config.cotracker_checkpoint),
                "cotracker_features": feature_artifact,
                "depth_convention": depth_convention,
                "pose_sources_used": sorted(pose_sources),
                "part_segmentation": part_segmentation,
                "object_mask_tracking_mode": bool(object_mask_mode),
                "part_reference_frames": {str(part_id): frame for part_id, frame in sorted(reference_frame_by_part.items())},
                "config": {
                    "reference_frame": self.config.reference_frame,
                    "frame_stride": self.config.frame_stride,
                    "seed_stride_px": self.config.seed_stride_px,
                    "max_tracks_per_part_view": self.config.max_tracks_per_part_view,
                    "min_depth_m": self.config.min_depth_m,
                    "max_depth_m": self.config.max_depth_m,
                    "visibility_threshold": self.config.visibility_threshold,
                    "depth_consistency_window_radius_px": self.config.depth_consistency_window_radius_px,
                    "depth_consistency_max_delta_m": self.config.depth_consistency_max_delta_m,
                    "repair_temporal_depth_spikes": self.config.repair_temporal_depth_spikes,
                    "depth_spike_jump_threshold_m": self.config.depth_spike_jump_threshold_m,
                    "depth_spike_neighbor_tolerance_m": self.config.depth_spike_neighbor_tolerance_m,
                    "depth_spike_max_run_frames": self.config.depth_spike_max_run_frames,
                    "require_part_mask_consistency": self.config.require_part_mask_consistency,
                    "strict_object_mask_consistency": self.config.strict_object_mask_consistency,
                    "allow_backward_tracking": self.config.allow_backward_tracking,
                    "export_cotracker_features": self.config.export_cotracker_features,
                    "dynamic_reseeding": self.config.dynamic_reseeding,
                    "dynamic_reseed_bidirectional": self.config.dynamic_reseed_bidirectional,
                    "reseed_interval_frames": self.config.reseed_interval_frames,
                    "reseed_coverage_radius_px": self.config.reseed_coverage_radius_px,
                    "reseed_bbox_scale": self.config.reseed_bbox_scale,
                    "reseed_max_tracks_per_frame_view": self.config.reseed_max_tracks_per_frame_view,
                    "reseed_max_tracks_per_view": self.config.reseed_max_tracks_per_view,
                },
                "part_track_counts": {
                    str(part_id): {
                        "name": part_names.get(part_id, f"part_{part_id}"),
                        "count": count,
                    }
                    for part_id, count in sorted(track_counts.items())
                },
                "tracks": tracks,
            }
        artifact_save_started = time.perf_counter()
        payload["runtime_profile"]["total_wall_s"] = time.perf_counter() - total_started
        save_json(payload, output_json)
        runtime_profile = dict(payload["runtime_profile"])
        runtime_profile["artifact_serialization_s"] = time.perf_counter() - artifact_save_started
        runtime_profile["total_wall_s"] = time.perf_counter() - total_started
        save_json(
            runtime_profile,
            output_json.with_name(f"{output_json.stem}.runtime.json"),
        )
        return output_json


class TrackPartPoseEstimator:
    def __init__(self, config: TrackPartPoseEstimationConfig) -> None:
        self.config = config

    def estimate(self) -> Path:
        input_path = Path(self.config.input_path).resolve()
        artifact = load_json(input_path)
        tracks = [track for track in artifact.get("tracks", []) if isinstance(track, dict)]
        if not tracks:
            raise ValueError("No 3D tracks found. Run track-part-pixels first.")

        frame_count = int(artifact.get("frame_count", 0))
        source_frame_count = int(artifact.get("source_frame_count", frame_count))
        sampled_frame_indices = [int(frame_index) for frame_index in artifact.get("sampled_frame_indices", list(range(frame_count)))]
        if len(sampled_frame_indices) != frame_count:
            sampled_frame_indices = list(range(frame_count))
        part_segmentation = (
            dict(artifact.get("part_segmentation", {}))
            if isinstance(artifact.get("part_segmentation"), dict)
            else None
        )
        part_names = part_name_lookup(part_segmentation)
        part_metadata = _part_metadata(part_segmentation)
        tracks_by_part: dict[int, list[dict[str, Any]]] = {}
        for track in tracks:
            part_id = int(track.get("part_id", 0))
            if part_id <= 0:
                continue
            tracks_by_part.setdefault(part_id, []).append(track)
        if not tracks_by_part:
            raise ValueError("No positive part_id tracks found.")

        track_counts = {part_id: len(items) for part_id, items in tracks_by_part.items()}
        anchor_part_id = _choose_anchor_part_id(part_metadata, track_counts, self.config.anchor_part_id)
        anchor_motion_scores: dict[int, float] = {}
        if self.config.anchor_part_id is None and self.config.anchor_selection == "lowest-motion":
            anchor_motion_scores = {
                part_id: _track_cluster_motion_range(items, max(3, self.config.min_tracks_per_part))
                for part_id, items in tracks_by_part.items()
            }
            finite_scores = {part_id: score for part_id, score in anchor_motion_scores.items() if math.isfinite(score)}
            if finite_scores:
                anchor_part_id = min(
                    sorted(finite_scores),
                    key=lambda part_id: (finite_scores[part_id], -track_counts[part_id], part_id),
                )
        part_tracks: dict[int, dict[str, Any]] = {}
        frame_times = self._frame_times(tracks)
        source_frame_indices = self._source_frame_indices(tracks, sampled_frame_indices)

        for part_id, part_track_items in sorted(tracks_by_part.items()):
            valid_reference_points = [
                track["reference_xyz_world"]
                for track in part_track_items
                if isinstance(track.get("reference_xyz_world"), list)
            ]
            if len(valid_reference_points) < max(3, self.config.min_tracks_per_part):
                continue
            reference_centroid_world = _centroid(
                [[float(value) for value in point] for point in valid_reference_points]
            )

            samples: list[dict[str, Any]] = []
            missing_frame_indices: list[int] = []
            for frame_index in range(frame_count):
                source_points: list[list[float]] = []
                target_points: list[list[float]] = []
                weights: list[float] = []
                for track in part_track_items:
                    reference_xyz = track.get("reference_xyz_world")
                    if not isinstance(reference_xyz, list):
                        continue
                    sample = self._sample_for_frame(track, frame_index)
                    if sample is None or not bool(sample.get("visible", False)):
                        continue
                    xyz_world = sample.get("xyz_world")
                    if not isinstance(xyz_world, list):
                        continue
                    source_points.append([float(value) for value in reference_xyz])
                    target_points.append([float(value) for value in xyz_world])
                    if self.config.quality_weighted:
                        weights.append(
                            observation_weight(
                                track,
                                sample,
                                timestep_field=self.config.quality_weight_field,
                                track_field=self.config.track_quality_field,
                                min_weight=float(self.config.min_timestep_weight),
                            )
                        )
                    else:
                        weights.append(float(sample.get("confidence", 1.0)))

                timestamp_s = frame_times.get(frame_index, 0.0)
                source_frame_index = source_frame_indices.get(frame_index, sampled_frame_indices[frame_index] if frame_index < len(sampled_frame_indices) else frame_index)
                if len(source_points) < max(3, self.config.min_tracks_per_part):
                    missing_frame_indices.append(frame_index)
                    samples.append(
                        {
                            "frame_index": frame_index,
                            "source_frame_index": source_frame_index,
                            "timestamp_s": timestamp_s,
                            "valid": False,
                            "track_count": len(source_points),
                            "confidence": 0.0,
                        }
                    )
                    continue

                rotation, translation, rms = _fit_rigid_transform(source_points, target_points, weights)
                centroid_world = [
                    rotated + shift
                    for rotated, shift in zip(_matvec3(rotation, reference_centroid_world), translation)
                ]
                confidence = max(0.0, min(1.0, len(source_points) / max(1, len(part_track_items)))) * (
                    1.0 / (1.0 + 20.0 * rms)
                )
                pose = _pose_payload(rotation, translation)
                pose.update(
                    {
                        "frame_index": frame_index,
                        "source_frame_index": source_frame_index,
                        "timestamp_s": timestamp_s,
                        "valid": True,
                        "track_count": len(source_points),
                        "confidence": float(confidence),
                        "registration_rms_m": float(rms),
                        "quality_weighted": bool(self.config.quality_weighted),
                        "centroid_world": [float(value) for value in centroid_world],
                    }
                )
                samples.append(pose)

            track = {
                "part_id": part_id,
                "name": part_names.get(part_id, f"part_{part_id}"),
                "role": self._part_role(part_metadata, part_id, anchor_part_id),
                "reference_frame_index": self._dominant_reference_frame(part_track_items),
                "reference_source_frame_index": source_frame_indices.get(
                    self._dominant_reference_frame(part_track_items),
                    self._dominant_reference_frame(part_track_items),
                ),
                "reference_timestamp_s": frame_times.get(self._dominant_reference_frame(part_track_items), 0.0),
                "reference_track_count": len(valid_reference_points),
                "reference_centroid_world": [float(value) for value in reference_centroid_world],
                "canonical_frame": _pose_payload(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [0.0, 0.0, 0.0],
                ),
                "samples": samples,
                "missing_frame_indices": missing_frame_indices,
            }
            part_tracks[part_id] = track

        if not part_tracks:
            raise ValueError("No part had enough visible 3D tracks for pose estimation.")
        if anchor_part_id not in part_tracks:
            anchor_part_id = max(sorted(part_tracks), key=lambda part_id: part_tracks[part_id]["reference_track_count"])

        anchor_track = part_tracks[anchor_part_id]
        anchor_samples = {
            int(sample["frame_index"]): sample
            for sample in anchor_track["samples"]
            if bool(sample.get("valid", False))
        }

        for part_id, track in part_tracks.items():
            relative_translations: list[list[float]] = []
            relative_rotation_angles: list[float] = []
            for sample in track["samples"]:
                frame_index = int(sample["frame_index"])
                if not bool(sample.get("valid", False)):
                    sample["relative_to_anchor"] = None
                    continue
                anchor_sample = anchor_samples.get(frame_index)
                if anchor_sample is None:
                    sample["relative_to_anchor"] = None
                    continue
                relative_rotation, relative_translation = _relative_pose(
                    anchor_rotation=anchor_sample["rotation_matrix"],
                    anchor_translation=anchor_sample["translation"],
                    child_rotation=sample["rotation_matrix"],
                    child_translation=sample["translation"],
                )
                relative_pose = _pose_payload(relative_rotation, relative_translation)
                sample["relative_to_anchor"] = relative_pose
                relative_translations.append(relative_translation)
                relative_rotation_angles.append(_rotation_angle_from_matrix(relative_rotation))

            track["relative_motion_summary"] = self._motion_summary(relative_translations, relative_rotation_angles)

        output_json = (
            Path(self.config.output_json).resolve()
            if self.config.output_json is not None
            else input_path.with_name("part_poses.json")
        )
        save_json(
            {
                "input_path": str(input_path),
                "pointcloud_path": None,
                "estimator": "cotracker-depth-rigid-registration",
                "frame_count": frame_count,
                "source_frame_count": source_frame_count,
                "sampled_frame_indices": sampled_frame_indices,
                "quality_weighting": {
                    "enabled": bool(self.config.quality_weighted),
                    "quality_weight_field": self.config.quality_weight_field,
                    "track_quality_field": self.config.track_quality_field,
                    "min_timestep_weight": float(self.config.min_timestep_weight),
                },
                "anchor_part_id": anchor_part_id,
                "anchor_part_name": part_tracks[anchor_part_id]["name"],
                "anchor_selection": self.config.anchor_selection,
                "anchor_motion_range_m": {
                    str(part_id): float(score)
                    for part_id, score in sorted(anchor_motion_scores.items())
                    if math.isfinite(score)
                },
                "parts": [part_tracks[part_id] for part_id in sorted(part_tracks)],
            },
            output_json,
        )
        return output_json

    def _sample_for_frame(self, track: dict[str, Any], frame_index: int) -> dict[str, Any] | None:
        for sample in track.get("samples", []):
            if isinstance(sample, dict) and int(sample.get("frame_index", -1)) == frame_index:
                return sample
        return None

    def _frame_times(self, tracks: list[dict[str, Any]]) -> dict[int, float]:
        times: dict[int, float] = {}
        for track in tracks:
            for sample in track.get("samples", []):
                if isinstance(sample, dict):
                    times[int(sample.get("frame_index", 0))] = float(sample.get("timestamp_s", 0.0))
        return times

    def _source_frame_indices(self, tracks: list[dict[str, Any]], fallback: list[int]) -> dict[int, int]:
        source_indices = {frame_index: frame_index for frame_index in range(len(fallback))}
        for local_frame_index, source_frame_index in enumerate(fallback):
            source_indices[local_frame_index] = int(source_frame_index)
        for track in tracks:
            for sample in track.get("samples", []):
                if isinstance(sample, dict) and "source_frame_index" in sample:
                    source_indices[int(sample.get("frame_index", 0))] = int(sample["source_frame_index"])
        return source_indices

    def _dominant_reference_frame(self, tracks: list[dict[str, Any]]) -> int:
        counts: dict[int, int] = {}
        for track in tracks:
            frame_index = int(track.get("query_frame_index", 0))
            counts[frame_index] = counts.get(frame_index, 0) + 1
        if not counts:
            return 0
        return max(sorted(counts), key=lambda frame_index: counts[frame_index])

    def _part_role(self, part_metadata: dict[int, dict[str, Any]], part_id: int, anchor_part_id: int) -> str:
        raw_part = part_metadata.get(part_id)
        if raw_part is not None:
            return str(raw_part.get("role", "unknown"))
        return "base" if part_id == anchor_part_id else "moving"

    def _motion_summary(
        self,
        relative_translations: list[list[float]],
        relative_rotation_angles: list[float],
    ) -> dict[str, Any]:
        if not relative_translations:
            return {
                "translation_range": [0.0, 0.0, 0.0],
                "rotation_angle_range_rad": 0.0,
            }
        translation_range = [
            max(item[axis] for item in relative_translations) - min(item[axis] for item in relative_translations)
            for axis in range(3)
        ]
        rotation_range = (
            max(relative_rotation_angles) - min(relative_rotation_angles)
            if relative_rotation_angles
            else 0.0
        )
        return {
            "translation_range": [float(value) for value in translation_range],
            "rotation_angle_range_rad": float(rotation_range),
        }
