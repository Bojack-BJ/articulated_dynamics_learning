from __future__ import annotations

import csv
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    target_fps: float | None = None
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
class _SelectedFrame:
    source: _TimedPath
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class _JointState:
    timestamp_s: float
    positions: dict[str, float]


@dataclass(slots=True)
class RBORecordingBatchConfig:
    manifest_path: Path
    output_root: Path
    jobs: int = 1
    frame_stride: int = 1
    start_frame: int = 0
    max_frames: int | None = None
    target_fps: float | None = None
    max_sync_delta_s: float = 0.05
    mask_mode: str = "none"
    mask_depth_percentile: float = 35.0
    mask_depth_margin_m: float = 0.15
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    force: bool = False


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
        selected = _select_rgb_frames(rgb_frames, config)

        object_id = config.object_id or input_dir.name
        category = _infer_category(input_dir.name, config.category)
        frames: list[FrameObservation] = []
        sync_deltas: list[float] = []

        for output_index, selected_rgb in enumerate(selected):
            rgb = selected_rgb.source
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
            action_log["source_timestamp_s"] = float(rgb.timestamp_s)
            joint_position_hint = None
            if joint_state is not None:
                action_log["joint_positions"] = joint_state.positions
                if joint_state.positions:
                    joint_position_hint = next(iter(joint_state.positions.values()))

            frames.append(
                FrameObservation(
                    timestamp_s=float(selected_rgb.timestamp_s),
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
                "target_fps": config.target_fps,
                "rgb_frame_count": len(rgb_frames),
                "depth_frame_count": len(depth_frames),
                "candidate_rgb_frame_count": len(
                    rgb_frames[config.start_frame :: max(1, config.frame_stride)]
                ),
                "imported_frame_count": len(frames),
                "max_rgb_depth_sync_delta_s": max(sync_deltas) if sync_deltas else None,
                "joint_state_csv": str(_find_joint_state_csv(input_dir)) if _find_joint_state_csv(input_dir) else None,
            },
        )
        episode_path = output_dir / "episode.json"
        save_json(episode.to_dict(), episode_path)
        return episode_path


class RBORecordingBatchImporter:
    def __init__(self, config: RBORecordingBatchConfig) -> None:
        self.config = config
        self._console_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        specs = _read_rbo_batch_manifest(self.config.manifest_path)
        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        successes: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []

        def process(spec: dict[str, str]) -> dict[str, str]:
            input_dir = Path(spec["input_dir"]).expanduser()
            if not input_dir.is_absolute():
                input_dir = (self.config.manifest_path.parent / input_dir).resolve()
            object_id = spec.get("object_id") or input_dir.name
            output_dir = _optional_path(spec.get("output_dir"))
            if output_dir is None:
                output_dir = output_root / object_id
            elif not output_dir.is_absolute():
                output_dir = (self.config.manifest_path.parent / output_dir).resolve()
            episode_path = RBORecordingImporter().import_recording(
                RBORecordingImportConfig(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    object_id=object_id,
                    category=spec.get("category") or None,
                    frame_stride=_int_value(spec, "frame_stride", self.config.frame_stride),
                    start_frame=_int_value(spec, "start_frame", self.config.start_frame),
                    max_frames=_optional_int_value(spec, "max_frames", self.config.max_frames),
                    target_fps=_optional_float_value(spec, "target_fps", self.config.target_fps),
                    max_sync_delta_s=_float_value(spec, "max_sync_delta_s", self.config.max_sync_delta_s),
                    mask_mode=spec.get("mask_mode") or self.config.mask_mode,
                    mask_depth_percentile=_float_value(
                        spec,
                        "mask_depth_percentile",
                        self.config.mask_depth_percentile,
                    ),
                    mask_depth_margin_m=_float_value(
                        spec,
                        "mask_depth_margin_m",
                        self.config.mask_depth_margin_m,
                    ),
                    min_depth_m=_float_value(spec, "min_depth_m", self.config.min_depth_m),
                    max_depth_m=_float_value(spec, "max_depth_m", self.config.max_depth_m),
                    force=bool(self.config.force),
                )
            )
            return {"object_id": object_id, "episode_path": str(episode_path)}

        jobs = max(1, int(self.config.jobs))
        if jobs == 1:
            for spec in specs:
                try:
                    result = process(spec)
                    successes.append(result)
                    self._console(f"[{len(successes) + len(failures)}/{len(specs)}] imported {result['object_id']}")
                except Exception as exc:
                    failures.append({"input_dir": spec.get("input_dir", ""), "error": str(exc)})
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {executor.submit(process, spec): spec for spec in specs}
                for future in as_completed(future_map):
                    spec = future_map[future]
                    try:
                        result = future.result()
                        successes.append(result)
                        self._console(f"[{len(successes) + len(failures)}/{len(specs)}] imported {result['object_id']}")
                    except Exception as exc:
                        failures.append({"input_dir": spec.get("input_dir", ""), "error": str(exc)})

        manifest_out = output_root / "imported_rbo_episodes.tsv"
        _write_imported_manifest(successes, manifest_out)
        if failures:
            raise RuntimeError(f"RBO batch import finished with failures: {failures}")
        return {
            "manifest_path": str(self.config.manifest_path.expanduser().resolve()),
            "output_root": str(output_root),
            "imported": len(successes),
            "episodes_manifest": str(manifest_out),
            "episodes": successes,
        }

    def _console(self, message: str) -> None:
        with self._console_lock:
            print(message)


def _select_rgb_frames(rgb_frames: list[_TimedPath], config: RBORecordingImportConfig) -> list[_SelectedFrame]:
    candidates = rgb_frames[max(0, int(config.start_frame)) :: max(1, int(config.frame_stride))]
    if not candidates:
        return []
    if config.target_fps is None:
        base_time = candidates[0].timestamp_s
        selected = [
            _SelectedFrame(source=item, timestamp_s=float(item.timestamp_s - base_time))
            for item in candidates
        ]
        if config.max_frames is not None:
            selected = selected[: max(0, int(config.max_frames))]
        return selected

    fps = float(config.target_fps)
    if fps <= 0.0:
        raise ValueError("target_fps must be positive when provided.")
    base_time = candidates[0].timestamp_s
    end_time = candidates[-1].timestamp_s
    dt = 1.0 / fps
    target_count = max(1, int((end_time - base_time) / dt) + 1)
    selected: list[_SelectedFrame] = []
    used_paths: set[Path] = set()
    for target_index in range(target_count):
        if config.max_frames is not None and len(selected) >= int(config.max_frames):
            break
        target_time = base_time + target_index * dt
        nearest = min(candidates, key=lambda item: abs(item.timestamp_s - target_time))
        if nearest.path in used_paths:
            continue
        used_paths.add(nearest.path)
        selected.append(_SelectedFrame(source=nearest, timestamp_s=float(target_index * dt)))
    return selected


def _read_rbo_batch_manifest(path: Path) -> list[dict[str, str]]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"RBO batch manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {str(key): str(value).strip() for key, value in row.items() if key is not None and value is not None}
            for row in csv.DictReader(
                (line for line in handle if line.strip() and not line.lstrip().startswith("#")),
                delimiter="\t",
            )
        ]
    if not rows:
        raise ValueError(f"RBO batch manifest is empty: {manifest_path}")
    for row in rows:
        if not row.get("input_dir"):
            raise ValueError("RBO batch manifest must include an input_dir column.")
    return rows


def _write_imported_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object_id", "episode_path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _optional_path(raw: str | None) -> Path | None:
    return None if raw in (None, "") else Path(raw)


def _int_value(row: dict[str, str], key: str, default: int) -> int:
    raw = row.get(key)
    return int(default if raw in (None, "") else raw)


def _optional_int_value(row: dict[str, str], key: str, default: int | None) -> int | None:
    raw = row.get(key)
    return default if raw in (None, "") else int(raw)


def _float_value(row: dict[str, str], key: str, default: float) -> float:
    raw = row.get(key)
    return float(default if raw in (None, "") else raw)


def _optional_float_value(row: dict[str, str], key: str, default: float | None) -> float | None:
    raw = row.get(key)
    return default if raw in (None, "") else float(raw)


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
