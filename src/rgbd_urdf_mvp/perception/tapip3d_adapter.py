"""Adapters between recorded episodes and the CUDA-only TAPIP3D reference code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_episode, load_json, save_json
from .pointcloud_fusion import _load_depth_u16, _resolve_view_camera_poses, _resolve_view_depth_paths


@dataclass(slots=True)
class TAPIP3DInputConfig:
    episode_path: str | Path
    output_npz: str | Path
    view_index: int = 0
    frame_stride: int = 1
    seed_tracks: str | Path | None = None
    compressed: bool = True


@dataclass(slots=True)
class TAPIP3DImportConfig:
    tapip_input_npz: str | Path
    tapip_result_npz: str | Path
    seed_tracks: str | Path
    output_json: str | Path
    visibility_threshold: float = 0.5


@dataclass(slots=True)
class TAPIP3DMergeConfig:
    track_paths: list[str | Path]
    output_tracks: str | Path
    feature_paths: list[str | Path] | None = None
    output_features: str | Path | None = None


class TAPIP3DInputPreparer:
    """Write TAPIP3D's documented RGB-D NPZ format from one recorded view."""

    def __init__(self, config: TAPIP3DInputConfig) -> None:
        self.config = config

    def prepare(self) -> Path:
        np = _require_numpy()
        episode_path = Path(self.config.episode_path).expanduser().resolve()
        episode = load_episode(episode_path)
        if not episode.frames:
            raise ValueError("Cannot prepare TAPIP3D input from an episode without frames.")
        root = episode_path.parent
        indices = list(range(0, len(episode.frames), max(1, int(self.config.frame_stride))))
        rgb_frames: list[Any] = []
        depth_frames: list[Any] = []
        extrinsics: list[Any] = []
        for frame_index in indices:
            frame = episode.frames[frame_index]
            rgb_paths = _resolve_rgb_paths(frame, root)
            depth_paths = _resolve_view_depth_paths(frame, root)
            camera_poses, _ = _resolve_view_camera_poses(frame, episode.metadata, len(depth_paths))
            view_index = int(self.config.view_index)
            if view_index < 0 or view_index >= len(rgb_paths) or view_index >= len(depth_paths):
                raise ValueError(f"view_index={view_index} is unavailable for episode frame {frame_index}.")
            rgb_frames.append(_load_rgb_u8(rgb_paths[view_index]))
            depth_frames.append(np.asarray(_load_depth_u16(depth_paths[view_index]), dtype=np.float32) / 1000.0)
            extrinsics.append(_tapip_world_to_camera(camera_poses[view_index]))

        video = np.stack(rgb_frames, axis=0)
        depths = np.stack(depth_frames, axis=0)
        if video.shape[:3] != depths.shape:
            raise ValueError(f"RGB/depth shape mismatch: video={video.shape}, depths={depths.shape}.")
        height, width = depths.shape[1:]
        intrinsics = np.repeat(_intrinsics_matrix(episode.camera_intrinsics, width, height)[None, ...], len(indices), axis=0)
        output_path = Path(self.config.output_npz).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "video": video.astype(np.uint8, copy=False),
            "depths": depths.astype(np.float32, copy=False),
            "intrinsics": intrinsics.astype(np.float32, copy=False),
            "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32, copy=False),
            "source_frame_indices": np.asarray(indices, dtype=np.int64),
            "timestamps_s": np.asarray(
                [float(episode.frames[index].timestamp_s) for index in indices], dtype=np.float64
            ),
            "view_index": np.asarray([int(self.config.view_index)], dtype=np.int64),
        }
        if self.config.seed_tracks is not None:
            payload.update(_query_payload(self.config.seed_tracks, int(self.config.view_index), indices))
        save = np.savez_compressed if self.config.compressed else np.savez
        save(output_path, **payload)
        return output_path


class TAPIP3DTrackImporter:
    """Convert TAPIP3D world trajectories back to the project's track artifact."""

    def __init__(self, config: TAPIP3DImportConfig) -> None:
        self.config = config

    def import_tracks(self) -> Path:
        np = _require_numpy()
        input_path = Path(self.config.tapip_input_npz).expanduser().resolve()
        result_path = Path(self.config.tapip_result_npz).expanduser().resolve()
        seed_path = Path(self.config.seed_tracks).expanduser().resolve()
        with np.load(input_path) as input_payload, np.load(result_path) as result_payload:
            if "track_ids" not in input_payload:
                raise ValueError("TAPIP input NPZ has no track_ids. Recreate it with --seed-tracks.")
            track_ids = np.asarray(input_payload["track_ids"], dtype=np.int64)
            source_frame_indices = np.asarray(input_payload["source_frame_indices"], dtype=np.int64)
            timestamps_s = (
                np.asarray(input_payload["timestamps_s"], dtype=np.float64)
                if "timestamps_s" in input_payload
                else None
            )
            view_index = int(np.asarray(input_payload["view_index"]).reshape(-1)[0])
            coords = np.asarray(result_payload["coords"], dtype=np.float64)
            visibs = np.asarray(result_payload["visibs"], dtype=bool)
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError(f"Expected TAPIP3D coords [T,N,3], got {coords.shape}.")
        if visibs.shape != coords.shape[:2] or coords.shape[:2] != (len(source_frame_indices), len(track_ids)):
            raise ValueError("TAPIP3D result dimensions do not match input query/frame metadata.")
        seed_artifact = load_json(seed_path)
        seed_by_id = {int(track["track_id"]): track for track in seed_artifact.get("tracks", []) if isinstance(track, dict) and "track_id" in track}
        tracks: list[dict[str, Any]] = []
        for query_index, track_id in enumerate(track_ids.tolist()):
            source = seed_by_id.get(int(track_id))
            if source is None:
                raise ValueError(f"Seed track {track_id} is missing from {seed_path}.")
            samples_by_source_frame = {int(sample.get("source_frame_index", sample.get("frame_index", -1))): sample for sample in source.get("samples", []) if isinstance(sample, dict)}
            samples: list[dict[str, Any]] = []
            for frame_index, source_frame_index in enumerate(source_frame_indices.tolist()):
                seed_sample = samples_by_source_frame.get(int(source_frame_index), {})
                visible = bool(visibs[frame_index, query_index])
                samples.append({
                    "frame_index": int(frame_index), "source_frame_index": int(source_frame_index),
                    "timestamp_s": float(
                        timestamps_s[frame_index]
                        if timestamps_s is not None
                        else seed_sample.get("timestamp_s", frame_index)
                    ),
                    "xyz_world": [float(value) for value in coords[frame_index, query_index]],
                    "visible": visible, "tracker_visibility": float(visible), "depth_valid": True,
                    "mask_consistent": True, "confidence": float(visible),
                })
            reference = next((sample["xyz_world"] for sample in samples if sample["visible"]), None)
            if reference is None:
                continue
            output_track = {key: value for key, value in source.items() if key != "samples"}
            output_track.update({"view_index": view_index, "reference_xyz_world": reference, "samples": samples})
            tracks.append(output_track)
        output_path = Path(self.config.output_json).expanduser().resolve()
        effective_fps = None
        if timestamps_s is not None and len(timestamps_s) > 1:
            deltas = np.diff(timestamps_s)
            positive = deltas[deltas > 1e-9]
            if len(positive):
                effective_fps = float(1.0 / np.median(positive))
        payload = {
            **{key: value for key, value in seed_artifact.items() if key != "tracks"},
            "estimator": "tapip3d-world-track-import", "tapip_input_npz": str(input_path),
            "tapip_result_npz": str(result_path), "frame_count": int(coords.shape[0]),
            "source_frame_count": int(seed_artifact.get("source_frame_count", int(source_frame_indices[-1]) + 1)),
            "sampled_frame_indices": [int(value) for value in source_frame_indices.tolist()],
            "effective_tracking_fps_hz": effective_fps,
            "view_count": 1, "tapip_visibility_threshold": float(self.config.visibility_threshold), "tracks": tracks,
        }
        save_json(payload, output_path)
        return output_path


class TAPIP3DMultiViewMerger:
    """Merge independently tracked camera views in their shared world frame."""

    def __init__(self, config: TAPIP3DMergeConfig) -> None:
        self.config = config

    def merge(self) -> tuple[Path, Path | None]:
        if len(self.config.track_paths) < 2:
            raise ValueError("Multi-view TAPIP merge requires at least two track artifacts.")
        artifacts = [load_json(Path(path).expanduser().resolve()) for path in self.config.track_paths]
        tracks: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for artifact in artifacts:
            for track in artifact.get("tracks", []):
                track_id = int(track["track_id"])
                if track_id in seen_ids:
                    raise ValueError(f"Duplicate TAPIP track_id across views: {track_id}")
                seen_ids.add(track_id)
                tracks.append(track)
        output_tracks = Path(self.config.output_tracks).expanduser().resolve()
        payload = {
            **{key: value for key, value in artifacts[0].items() if key != "tracks"},
            "estimator": "tapip3d-multiview-world-tracks",
            "view_count": len(artifacts),
            "source_track_artifacts": [str(Path(path).expanduser().resolve()) for path in self.config.track_paths],
            "tracks": sorted(tracks, key=lambda track: int(track["track_id"])),
        }
        save_json(payload, output_tracks)

        output_features: Path | None = None
        if self.config.feature_paths:
            if len(self.config.feature_paths) != len(artifacts) or self.config.output_features is None:
                raise ValueError("Feature merge requires one feature NPZ per view and --output-features.")
            output_features = _merge_feature_npz(self.config.feature_paths, self.config.output_features, seen_ids)
        return output_tracks, output_features


def _query_payload(seed_tracks_path: str | Path, view_index: int, source_frame_indices: list[int]) -> dict[str, Any]:
    np = _require_numpy()
    artifact = load_json(seed_tracks_path)
    frame_to_local = {int(source_index): local_index for local_index, source_index in enumerate(source_frame_indices)}
    track_ids: list[int] = []
    queries: list[list[float]] = []
    for track in artifact.get("tracks", []):
        if not isinstance(track, dict) or int(track.get("view_index", -1)) != view_index:
            continue
        source_frame = int(track.get("query_source_frame_index", source_frame_indices[0]))
        reference = track.get("reference_xyz_world")
        if source_frame not in frame_to_local or not isinstance(reference, list) or len(reference) != 3:
            continue
        track_ids.append(int(track["track_id"]))
        queries.append([float(frame_to_local[source_frame]), *[float(value) for value in reference]])
    if not queries:
        raise ValueError(f"No seed tracks from view {view_index} overlap the selected frames.")
    return {"query_point": np.asarray(queries, dtype=np.float32), "track_ids": np.asarray(track_ids, dtype=np.int64)}


def _resolve_rgb_paths(frame: Any, root: Path) -> list[Path]:
    return [root / path for path in frame.rgb_paths_by_view] if frame.rgb_paths_by_view else [root / frame.rgb_path]


def _load_rgb_u8(path: Path) -> Any:
    np = _require_numpy()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Preparing TAPIP3D input requires Pillow. Install with `.[tracking]`.") from exc
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _intrinsics_matrix(intrinsics: dict[str, float], width: int, height: int) -> Any:
    np = _require_numpy()
    return np.asarray([[float(intrinsics["fx"]), 0.0, float(intrinsics.get("cx", (width - 1) / 2.0))], [0.0, float(intrinsics["fy"]), float(intrinsics.get("cy", (height - 1) / 2.0))], [0.0, 0.0, 1.0]], dtype=np.float32)


def _tapip_world_to_camera(camera_to_world: list[list[float]]) -> Any:
    """Convert recorder OpenGL camera-to-world into TAPIP3D's CV world-to-camera."""
    np = _require_numpy()
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera pose must be 4x4, got {c2w.shape}.")
    return np.diag([1.0, -1.0, 1.0, 1.0]) @ np.linalg.inv(c2w)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TAPIP3D adapter requires numpy. Install with `.[tracking]`.") from exc
    return np


def _merge_feature_npz(
    feature_paths: list[str | Path], output_path: str | Path, expected_track_ids: set[int]
) -> Path:
    np = _require_numpy()
    arrays: dict[str, list[Any]] = {}
    feature_sources: list[str] = []
    for path in feature_paths:
        with np.load(Path(path).expanduser().resolve()) as payload:
            for key in ("track_ids", "embeddings", "temporal_tokens", "valid_timestep_count"):
                if key in payload:
                    arrays.setdefault(key, []).append(np.asarray(payload[key]))
            if "feature_source" in payload:
                feature_sources.extend(str(value) for value in np.asarray(payload["feature_source"]).reshape(-1))
    if "track_ids" not in arrays or "embeddings" not in arrays:
        raise ValueError("Each TAPIP feature artifact must contain track_ids and embeddings.")
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    ids = [int(value) for value in merged["track_ids"].tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError("Merged TAPIP features contain duplicate track IDs.")
    missing = expected_track_ids - set(ids)
    if missing:
        raise ValueError(f"Merged TAPIP features are missing {len(missing)} imported world tracks.")
    keep = np.asarray([track_id in expected_track_ids for track_id in ids], dtype=bool)
    for key, values in list(merged.items()):
        if values.shape[0] == len(ids):
            merged[key] = values[keep]
    merged["feature_source"] = np.asarray(sorted(set(feature_sources)) or ["tapip3d_updateformer"])
    merged["filtered_invisible_feature_count"] = np.asarray([int((~keep).sum())], dtype=np.int64)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return output
