from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class MotionPartSegmentationConfig:
    input_tracks: str | Path
    output_json: str | Path | None = None
    rigidity_threshold_m: float = 0.015
    max_neighbor_distance_m: float = 0.18
    min_common_frames: int = 3
    min_tracks_per_part: int = 4
    min_motion_m: float = 0.01
    static_motion_threshold_m: float = 0.01


class MotionPartSegmenter:
    """Discover rigid parts from 3D point trajectories.

    This is the object-mask-first path: CoTracker provides point trajectories
    without reliable part ids, then this module relabels tracks by motion.
    """

    def __init__(self, config: MotionPartSegmentationConfig) -> None:
        self.config = config

    def segment(self) -> Path:
        input_path = Path(self.config.input_tracks).expanduser().resolve()
        artifact = load_json(input_path)
        tracks = [track for track in artifact.get("tracks", []) if _valid_track(track)]
        if not tracks:
            raise ValueError("No valid 3D tracks found for motion segmentation.")

        graph = self._build_graph(tracks)
        components = _connected_components(graph, len(tracks))
        components = self._merge_small_components(tracks, components)
        components = sorted(components, key=lambda indices: (-len(indices), min(indices)))
        part_ids = self._assign_part_ids(tracks, components)
        relabeled_tracks = [self._relabel_track(track, part_ids[index]) for index, track in enumerate(tracks)]
        part_counts: dict[int, int] = {}
        for part_id in part_ids.values():
            part_counts[part_id] = part_counts.get(part_id, 0) + 1

        anchor_part_id = self._choose_anchor_part_id(relabeled_tracks)
        part_segmentation = _part_segmentation(part_counts, anchor_part_id)
        output_json = (
            Path(self.config.output_json).expanduser().resolve()
            if self.config.output_json is not None
            else input_path.with_name("motion_part_tracks.json")
        )
        payload = dict(artifact)
        payload.update(
            {
                "input_track_path": str(input_path),
                "estimator": "motion-rigidity-cotracker-clustering",
                "part_segmentation": part_segmentation,
                "motion_segmentation": {
                    "source_estimator": artifact.get("estimator"),
                    "track_count": len(tracks),
                    "part_count": len(part_counts),
                    "anchor_part_id": anchor_part_id,
                    "config": {
                        "rigidity_threshold_m": float(self.config.rigidity_threshold_m),
                        "max_neighbor_distance_m": float(self.config.max_neighbor_distance_m),
                        "min_common_frames": int(self.config.min_common_frames),
                        "min_tracks_per_part": int(self.config.min_tracks_per_part),
                        "min_motion_m": float(self.config.min_motion_m),
                        "static_motion_threshold_m": float(self.config.static_motion_threshold_m),
                    },
                    "components": [
                        {
                            "part_id": part_id,
                            "track_count": int(count),
                            "mean_motion_m": self._component_motion(relabeled_tracks, part_id),
                        }
                        for part_id, count in sorted(part_counts.items())
                    ],
                },
                "part_track_counts": {
                    str(part_id): {"name": _part_name(part_id, anchor_part_id), "count": count}
                    for part_id, count in sorted(part_counts.items())
                },
                "tracks": relabeled_tracks,
            }
        )
        save_json(payload, output_json)
        return output_json

    def _build_graph(self, tracks: list[dict[str, Any]]) -> list[set[int]]:
        graph = [set() for _ in tracks]
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                metrics = _pair_metrics(trajectories[i], trajectories[j], self.config.min_common_frames)
                if metrics is None:
                    continue
                if metrics["rigidity_rmse_m"] > self.config.rigidity_threshold_m:
                    continue
                if metrics["median_distance_m"] > self.config.max_neighbor_distance_m and (
                    metrics["motion_disagreement_m"] > self.config.rigidity_threshold_m
                ):
                    continue
                graph[i].add(j)
                graph[j].add(i)
        return graph

    def _merge_small_components(self, tracks: list[dict[str, Any]], components: list[list[int]]) -> list[list[int]]:
        min_size = max(1, int(self.config.min_tracks_per_part))
        large = [component for component in components if len(component) >= min_size]
        small = [component for component in components if len(component) < min_size]
        if not large:
            return components
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        for component in small:
            best_index = None
            best_score = float("inf")
            for large_index, candidate in enumerate(large):
                scores = []
                for i in component:
                    for j in candidate:
                        metrics = _pair_metrics(trajectories[i], trajectories[j], self.config.min_common_frames)
                        if metrics is None:
                            continue
                        scores.append(
                            metrics["rigidity_rmse_m"]
                            + 0.25 * min(metrics["median_distance_m"], self.config.max_neighbor_distance_m)
                        )
                if not scores:
                    continue
                score = min(scores)
                if score < best_score:
                    best_score = score
                    best_index = large_index
            if best_index is None or best_score > (self.config.rigidity_threshold_m + 0.25 * self.config.max_neighbor_distance_m):
                large.append(list(component))
            else:
                large[best_index].extend(component)
        return [sorted(component) for component in large]

    def _assign_part_ids(self, tracks: list[dict[str, Any]], components: list[list[int]]) -> dict[int, int]:
        static_components = []
        moving_components = []
        for component in components:
            mean_motion = fmean(_track_motion_m(tracks[index]) for index in component)
            if mean_motion <= self.config.static_motion_threshold_m:
                static_components.append(component)
            else:
                moving_components.append(component)
        ordered = sorted(static_components, key=lambda items: (-len(items), min(items))) + sorted(
            moving_components,
            key=lambda items: (-fmean(_track_motion_m(tracks[index]) for index in items), -len(items), min(items)),
        )
        out: dict[int, int] = {}
        for part_id, component in enumerate(ordered, start=1):
            for index in component:
                out[index] = part_id
        return out

    def _choose_anchor_part_id(self, tracks: list[dict[str, Any]]) -> int:
        by_part: dict[int, list[dict[str, Any]]] = {}
        for track in tracks:
            by_part.setdefault(int(track["part_id"]), []).append(track)
        return min(
            sorted(by_part),
            key=lambda part_id: (
                fmean(_track_motion_m(track) for track in by_part[part_id]),
                -len(by_part[part_id]),
                part_id,
            ),
        )

    def _relabel_track(self, track: dict[str, Any], part_id: int) -> dict[str, Any]:
        original_part_id = int(track.get("part_id", 0))
        relabeled = dict(track)
        relabeled["original_part_id"] = original_part_id
        relabeled["part_id"] = int(part_id)
        relabeled["part_name"] = _part_name(part_id, None)
        return relabeled

    def _component_motion(self, tracks: list[dict[str, Any]], part_id: int) -> float:
        values = [_track_motion_m(track) for track in tracks if int(track.get("part_id", 0)) == part_id]
        return float(fmean(values)) if values else 0.0


def _valid_track(track: dict[str, Any]) -> bool:
    return isinstance(track.get("reference_xyz_world"), list) and len(_trajectory_by_frame(track)) >= 2


def _trajectory_by_frame(track: dict[str, Any]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for sample in track.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if not bool(sample.get("visible", False)) or not bool(sample.get("depth_valid", True)):
            continue
        point = sample.get("xyz_world")
        if not isinstance(point, list) or len(point) != 3:
            continue
        out[int(sample.get("frame_index", 0))] = [float(value) for value in point]
    return out


def _pair_metrics(
    a_by_frame: dict[int, list[float]],
    b_by_frame: dict[int, list[float]],
    min_common_frames: int,
) -> dict[str, float] | None:
    common = sorted(set(a_by_frame) & set(b_by_frame))
    if len(common) < min_common_frames:
        return None
    distances = [_distance(a_by_frame[frame], b_by_frame[frame]) for frame in common]
    mean_distance = fmean(distances)
    rigidity_rmse = math.sqrt(fmean((distance - mean_distance) ** 2 for distance in distances))
    start = common[0]
    end = common[-1]
    motion_a = _subtract(a_by_frame[end], a_by_frame[start])
    motion_b = _subtract(b_by_frame[end], b_by_frame[start])
    return {
        "common_frames": float(len(common)),
        "rigidity_rmse_m": float(rigidity_rmse),
        "median_distance_m": float(sorted(distances)[len(distances) // 2]),
        "motion_disagreement_m": float(_distance(motion_a, motion_b)),
    }


def _connected_components(graph: list[set[int]], count: int) -> list[list[int]]:
    seen = [False] * count
    components: list[list[int]] = []
    for start in range(count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(graph[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _track_motion_m(track: dict[str, Any]) -> float:
    trajectory = _trajectory_by_frame(track)
    if len(trajectory) < 2:
        return 0.0
    first = trajectory[min(trajectory)]
    last = trajectory[max(trajectory)]
    return _distance(first, last)


def _part_segmentation(part_counts: dict[int, int], anchor_part_id: int) -> dict[str, Any]:
    return {
        "source": "motion-rigidity-cotracker-clustering",
        "background_part_id": 0,
        "parts": [
            {
                "part_id": int(part_id),
                "name": _part_name(part_id, anchor_part_id),
                "role": "base" if int(part_id) == int(anchor_part_id) else "moving",
                "track_count": int(part_counts[part_id]),
            }
            for part_id in sorted(part_counts)
        ],
    }


def _part_name(part_id: int, anchor_part_id: int | None) -> str:
    if anchor_part_id is not None and int(part_id) == int(anchor_part_id):
        return "base"
    return f"motion_part_{int(part_id)}"


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]
