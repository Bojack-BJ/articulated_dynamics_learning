from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.categories import SUPPORTED_CATEGORIES, normalize_category
from ..core.models import EpisodeInput, FrameObservation
from ..core.serialization import save_json


_TIMESTAMP_RE = re.compile(r"(\d+(?:\.\d+)?)(?=\.[^.]+$)")


@dataclass(slots=True)
class RBORecordingImportConfig:
    input_dir: Path
    output_dir: Path
    object_id: str | None = None
    category: str | None = None
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    max_sync_delta_s: float = 0.05
    mask_mode: str = "none"
    mask_depth_percentile: float = 35.0
    mask_depth_margin_m: float = 0.15
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    force: bool = False


@dataclass(frozen=True, slots=True)
class _TimedPath:
    timestamp_s: float
    path: Path


@dataclass(frozen=True, slots=True)
class _JointState:
    timestamp_s: float
    positions: dict[str, float]


class RBORecordingImporter:
    """Convert RBO/ROS-export RGB-D folders into the project episode format."""

    def import_recording(self, config: RBORecordingImportConfig) -> Path:
        input_dir = config.input_dir.expanduser()
        output_dir = config.output_dir.expanduser()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"RBO input directory does not exist: {input_dir}")
        if output_dir.exists() and any(output_dir.iterdir()) and not config.force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force to overwrite files.")

        assets_dir = output_dir / "assets" / "view_0"
        assets_dir.mkdir(parents=True, exist_ok=True)

        rgb_frames = _list_timed_files(input_dir / "camera_rgb", "*.png")
        depth_frames = _list_timed_files(input_dir / "camera_depth_registered", "*.txt")
        if not rgb_frames:
            raise FileNotFoundError(f"No RGB PNG files found under {input_dir / 'camera_rgb'}")
        if not depth_frames:
            raise FileNotFoundError(f"No registered depth TXT files found under {input_dir / 'camera_depth_registered'}")

        intrinsics = _read_camera_intrinsics(input_dir / "camera_depth_registered_camera_info.csv")
        joint_states = _read_joint_states(_find_joint_state_csv(input_dir))
        selected = rgb_frames[config.start_frame :: max(1, config.frame_stride)]
        if config.max_frames is not None:
            selected = selected[: max(0, int(config.max_frames))]

        object_id = config.object_id or input_dir.name
        category = _infer_category(input_dir.name, config.category)
        frames: list[FrameObservation] = []
        base_time = selected[0].timestamp_s if selected else 0.0
        sync_deltas: list[float] = []

        for output_index, rgb in enumerate(selected):
            depth, depth_delta = _nearest(rgb.timestamp_s, depth_frames)
            if depth_delta > config.max_sync_delta_s:
                continue
            sync_deltas.append(depth_delta)

            rgb_name = f"frame_{output_index:04d}_rgb.png"
            depth_name = f"frame_{output_index:04d}_depth.png"
            mask_name = f"frame_{output_index:04d}_mask.png"
            rgb_out = assets_dir / rgb_name
            depth_out = assets_dir / depth_name
            mask_out = assets_dir / mask_name

            shutil.copyfile(rgb.path, rgb_out)
            depth_m = _load_depth_txt(depth.path)
            _write_depth_png(depth_m, depth_out, config.min_depth_m, config.max_depth_m)

            mask_rel: str | None = None
            if config.mask_mode == "depth-near":
                _write_depth_near_mask(
                    depth_m,
                    mask_out,
                    percentile=float(config.mask_depth_percentile),
                    margin_m=float(config.mask_depth_margin_m),
                    min_depth_m=float(config.min_depth_m),
                    max_depth_m=float(config.max_depth_m),
                )
                mask_rel = _rel(mask_out, output_dir)

            joint_state, _joint_delta = _nearest_joint_state(rgb.timestamp_s, joint_states)
            action_log: dict[str, Any] = {}
            joint_position_hint = None
            if joint_state is not None:
                action_log["joint_positions"] = joint_state.positions
                if joint_state.positions:
                    joint_position_hint = next(iter(joint_state.positions.values()))

            frames.append(
                FrameObservation(
                    timestamp_s=rgb.timestamp_s - base_time,
                    rgb_path=_rel(rgb_out, output_dir),
                    depth_path=_rel(depth_out, output_dir),
                    mask_path=mask_rel,
                    camera_pose=_identity_pose(),
                    rgb_paths_by_view=[_rel(rgb_out, output_dir)],
                    depth_paths_by_view=[_rel(depth_out, output_dir)],
                    mask_paths_by_view=[] if mask_rel is None else [mask_rel],
                    camera_poses_by_view=[_identity_pose()],
                    action_log=action_log,
                    joint_position_hint=joint_position_hint,
                )
            )

        if not frames:
            raise RuntimeError("No synchronized RGB/depth frames were imported. Increase --max-sync-delta-s.")

        episode = EpisodeInput(
            object_instance_id=object_id,
            category=category,
            camera_intrinsics=intrinsics,
            frames=frames,
            metadata={
                "source": "rbo_rosbag_export",
                "source_dir": str(input_dir),
                "camera_pose_source": "identity_camera_frame",
                "depth_units": "meters_to_uint16_millimeters",
                "mask_mode": config.mask_mode,
                "rgb_frame_count": len(rgb_frames),
                "depth_frame_count": len(depth_frames),
                "imported_frame_count": len(frames),
                "max_rgb_depth_sync_delta_s": max(sync_deltas) if sync_deltas else None,
                "joint_state_csv": str(_find_joint_state_csv(input_dir)) if _find_joint_state_csv(input_dir) else None,
            },
        )
        episode_path = output_dir / "episode.json"
        save_json(episode.to_dict(), episode_path)
        return episode_path


def _list_timed_files(directory: Path, pattern: str) -> list[_TimedPath]:
    if not directory.is_dir():
        return []
    items = []
    for path in directory.glob(pattern):
        timestamp = _timestamp_from_path(path)
        if timestamp is not None:
            items.append(_TimedPath(timestamp_s=timestamp, path=path))
    return sorted(items, key=lambda item: item.timestamp_s)


def _timestamp_from_path(path: Path) -> float | None:
    match = _TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    return float(match.group(1))


def _nearest(timestamp_s: float, items: list[_TimedPath]) -> tuple[_TimedPath, float]:
    best = min(items, key=lambda item: abs(item.timestamp_s - timestamp_s))
    return best, abs(best.timestamp_s - timestamp_s)


def _nearest_joint_state(timestamp_s: float, items: list[_JointState]) -> tuple[_JointState | None, float | None]:
    if not items:
        return None, None
    best = min(items, key=lambda item: abs(item.timestamp_s - timestamp_s))
    return best, abs(best.timestamp_s - timestamp_s)


def _read_camera_intrinsics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Camera info CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return {
        "fx": float(row.get("field.P0") or row["field.K0"]),
        "fy": float(row.get("field.P5") or row["field.K4"]),
        "cx": float(row.get("field.P2") or row["field.K2"]),
        "cy": float(row.get("field.P6") or row["field.K5"]),
    }


def _find_joint_state_csv(input_dir: Path) -> Path | None:
    candidates = sorted(input_dir.glob("*_joint_states.csv"))
    return candidates[0] if candidates else None


def _read_joint_states(path: Path | None) -> list[_JointState]:
    if path is None or not path.exists():
        return []
    states: list[_JointState] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = _timestamp_from_ros_row(row)
            positions: dict[str, float] = {}
            for key, value in row.items():
                if not key.startswith("field.position") or value in (None, ""):
                    continue
                suffix = key.removeprefix("field.position")
                name = row.get(f"field.name{suffix}") or f"joint_{suffix}"
                positions[name] = float(value)
            states.append(_JointState(timestamp_s=timestamp, positions=positions))
    return sorted(states, key=lambda item: item.timestamp_s)


def _timestamp_from_ros_row(row: dict[str, str]) -> float:
    raw = row.get("field.header.stamp") or row.get("%time")
    if raw is None:
        raise ValueError("CSV row does not contain field.header.stamp or %time")
    value = float(raw)
    return value / 1_000_000_000.0 if value > 1_000_000_000.0 else value


def _load_depth_txt(path: Path):
    import numpy as np

    return np.loadtxt(path, dtype=np.float32)


def _write_depth_png(depth_m, output_path: Path, min_depth_m: float, max_depth_m: float) -> None:
    import numpy as np
    from PIL import Image

    finite = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[finite] = np.clip(depth_m[finite] * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(output_path)


def _write_depth_near_mask(
    depth_m,
    output_path: Path,
    *,
    percentile: float,
    margin_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> None:
    import numpy as np
    from PIL import Image

    finite = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    if not np.any(finite):
        mask = np.zeros(depth_m.shape, dtype=np.uint8)
    else:
        height, width = depth_m.shape
        y0, y1 = height // 4, height - height // 4
        x0, x1 = width // 4, width - width // 4
        center_values = depth_m[y0:y1, x0:x1]
        center_finite = np.isfinite(center_values) & (center_values >= min_depth_m) & (center_values <= max_depth_m)
        sample = center_values[center_finite] if np.any(center_finite) else depth_m[finite]
        threshold = float(np.percentile(sample, percentile)) + margin_m
        mask = (finite & (depth_m <= threshold)).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(output_path)


def _infer_category(sequence_name: str, explicit: str | None) -> str:
    if explicit is not None:
        category = normalize_category(explicit)
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")
        return category
    prefix = re.sub(r"\d+_?o?$", "", sequence_name.lower())
    category = normalize_category(prefix)
    if category in SUPPORTED_CATEGORIES:
        return category
    if category == "cabinet":
        return "door"
    if category == "cardboardbox":
        return "drawer"
    return "door"


def _identity_pose() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
