from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.serialization import load_json, save_json
from .quality_weights import (
    articulation_pair_compatibility_metrics,
    articulation_track_score,
    cluster_quality_summary,
    observation_weight,
    track_quality_score,
)


@dataclass(slots=True)
class MotionPartSegmentationConfig:
    input_tracks: str | Path
    output_json: str | Path | None = None
    mode: str = "connected"
    diagnostics_json: str | Path | None = None
    sweep_output_dir: str | Path | None = None
    rigidity_threshold_m: float = 0.015
    max_neighbor_distance_m: float = 0.18
    min_common_frames: int = 3
    min_tracks_per_part: int = 4
    min_motion_m: float = 0.01
    static_motion_threshold_m: float = 0.01
    moving_motion_threshold_m: float = 0.03
    quality_filter: bool = False
    min_visible_frames: int = 2
    max_depth_jump_m: float = 0.0
    max_trajectory_jump_m: float = 0.0
    knn_k: int = 12
    k_min: int = 2
    k_max: int = 8
    spectral_k: int | None = None
    edge_ablation: str = "B"
    quality_weighted_affinity: bool = False
    quality_weighted_affinity_time_only: bool = False
    quality_weighted_affinity_edge_prior: bool = False
    quality_affinity_min_pair_weight: float = 0.0
    articulation_compatible_affinity: bool = False
    articulation_type_mismatch_penalty: float = 0.65
    articulation_static_mismatch_penalty: float = 1.0
    articulation_min_motion_for_type_penalty: float = 0.03
    articulation_min_confidence_for_type_penalty: float = 0.75
    cotracker_features_npz: str | Path | None = None
    learned_affinity_floor: float = 0.7
    pairwise_affinity_model: str | Path | None = None
    pairwise_affinity_floor: float = 0.5
    pairwise_affinity_device: str = "cpu"
    pairwise_connect_threshold: float = 0.9
    skip_base_bridge_checks: bool = False
    ransac_iterations: int = 128
    ransac_sample_size: int = 4
    ransac_inlier_threshold_m: float = 0.025
    ransac_min_inliers: int = 12
    ransac_max_models: int = 8
    ransac_spatial_link_m: float = 0.35
    ransac_assignment_threshold_m: float = 0.05
    ransac_seed: int = 0
    ransac_base_stabilize: bool = True
    ransac_learned_seed: bool = False


@dataclass(slots=True)
class LocalSplitDiagnosticsConfig:
    input_tracks: str | Path
    evaluation_json: str | Path | None = None
    output_dir: str | Path | None = None
    split_cluster_ids: list[int] | None = None
    local_k_min: int = 2
    local_k_max: int = 3
    knn_k: int = 8
    edge_ablation: str = "B"
    rigidity_threshold_m: float = 0.015
    max_neighbor_distance_m: float = 0.18
    min_common_frames: int = 3


class MotionPartSegmenter:
    """Discover rigid parts from object-mask 3D point trajectories."""

    def __init__(self, config: MotionPartSegmentationConfig) -> None:
        self.config = config
        self._learned_feature_map: dict[int, Any] = {}
        self._pairwise_affinity_predictor: Any = None

    def segment(self) -> Path:
        input_path = Path(self.config.input_tracks).expanduser().resolve()
        artifact = load_json(input_path)
        if self.config.cotracker_features_npz is not None:
            from .cotracker_features import load_cotracker_feature_map

            self._learned_feature_map = load_cotracker_feature_map(self.config.cotracker_features_npz)
        if self.config.pairwise_affinity_model is not None:
            if not self._learned_feature_map:
                raise ValueError("--pairwise-affinity-model requires --cotracker-features-npz.")
            from .pairwise_affinity import PairwiseAffinityPredictor

            self._pairwise_affinity_predictor = PairwiseAffinityPredictor(
                self.config.pairwise_affinity_model,
                device=self.config.pairwise_affinity_device,
            )
        tracks, filter_report = self._load_tracks(artifact)
        if not tracks:
            raise ValueError("No valid 3D tracks found for motion segmentation.")

        if self.config.mode == "connected":
            output_path, diagnostics = self._segment_connected(input_path, artifact, tracks, filter_report)
        elif self.config.mode == "knn-spectral":
            output_path, diagnostics = self._segment_knn_spectral(input_path, artifact, tracks, filter_report)
        elif self.config.mode == "sequential-ransac":
            output_path, diagnostics = self._segment_sequential_ransac(input_path, artifact, tracks, filter_report)
        elif self.config.mode == "learned-connected":
            output_path, diagnostics = self._segment_learned_connected(input_path, artifact, tracks, filter_report)
        else:
            raise ValueError(f"Unsupported motion segmentation mode: {self.config.mode}")

        if self.config.diagnostics_json is not None:
            save_json(diagnostics, Path(self.config.diagnostics_json).expanduser().resolve())
        return output_path

    def _load_tracks(self, artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tracks = [track for track in artifact.get("tracks", []) if isinstance(track, dict)]
        kept = []
        dropped: dict[str, int] = {
            "invalid_track": 0,
            "too_few_visible_frames": 0,
            "depth_jump": 0,
            "trajectory_jump": 0,
        }
        dropped_gt: dict[str, dict[str, int]] = {}
        for track in tracks:
            reason = self._drop_reason(track)
            if reason is None:
                kept.append(track)
                continue
            dropped[reason] = dropped.get(reason, 0) + 1
            original_part_id = str(track.get("original_part_id", track.get("part_id", "unknown")))
            dropped_gt.setdefault(reason, {})
            dropped_gt[reason][original_part_id] = dropped_gt[reason].get(original_part_id, 0) + 1
        return kept, {
            "enabled": bool(self.config.quality_filter),
            "input_track_count": len(tracks),
            "kept_track_count": len(kept),
            "dropped_track_count": len(tracks) - len(kept),
            "dropped_by_reason": dropped,
            "dropped_gt_composition": dropped_gt,
        }

    def _drop_reason(self, track: dict[str, Any]) -> str | None:
        trajectory = _trajectory_by_frame(track)
        if not isinstance(track.get("reference_xyz_world"), list) or len(trajectory) < 2:
            return "invalid_track"
        if not self.config.quality_filter:
            return None
        if len(trajectory) < max(2, int(self.config.min_visible_frames)):
            return "too_few_visible_frames"
        if self.config.max_depth_jump_m > 0.0 and _max_depth_jump(track) > self.config.max_depth_jump_m:
            return "depth_jump"
        if self.config.max_trajectory_jump_m > 0.0 and _max_trajectory_jump_m(track) > self.config.max_trajectory_jump_m:
            return "trajectory_jump"
        return None

    def _segment_connected(
        self,
        input_path: Path,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        filter_report: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        graph, edge_metrics = self._build_connected_graph(tracks)
        components = _connected_components(graph, len(tracks))
        components = self._merge_small_components(tracks, components)
        components = sorted(components, key=lambda indices: (-len(indices), min(indices)))
        part_ids = self._assign_part_ids(tracks, components)
        output_path = self._write_assignment(
            input_path=input_path,
            artifact=artifact,
            tracks=tracks,
            part_ids=part_ids,
            estimator="motion-rigidity-cotracker-clustering",
            segmentation_extra={
                "mode": "connected",
                "source_estimator": artifact.get("estimator"),
                "track_count": len(tracks),
                "part_count": len(set(part_ids.values())),
                "config": self._config_payload(),
            },
        )
        diagnostics = self._diagnostics(
            mode="connected",
            artifact=artifact,
            tracks=tracks,
            graph=graph,
            edge_metrics=edge_metrics,
            part_ids=part_ids,
            filter_report=filter_report,
            extra={"base_bridge_checks": self._base_bridge_checks(tracks)},
        )
        return output_path, diagnostics

    def _segment_learned_connected(
        self,
        input_path: Path,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        filter_report: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        """Discover moving components directly from dense learned pair affinities."""
        if self._pairwise_affinity_predictor is None:
            raise ValueError("learned-connected requires --pairwise-affinity-model.")
        from .pairwise_affinity import pair_feature_vector

        static_indices, moving_indices, ambiguous_indices = self._motion_split(tracks)
        graph = [set() for _ in tracks]
        edge_metrics: list[dict[str, Any]] = []
        vectors: list[Any] = []
        pair_rows: list[tuple[int, int, dict[str, Any]]] = []
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        for offset, global_i in enumerate(moving_indices):
            for global_j in moving_indices[offset + 1 :]:
                metrics = _pair_metrics(
                    trajectories[global_i],
                    trajectories[global_j],
                    self.config.min_common_frames,
                )
                if metrics is None:
                    continue
                track_i = tracks[global_i]
                track_j = tracks[global_j]
                feature_i = self._learned_feature_map.get(int(track_i.get("track_id", -1)))
                feature_j = self._learned_feature_map.get(int(track_j.get("track_id", -1)))
                if feature_i is None or feature_j is None:
                    continue
                vectors.append(
                    pair_feature_vector(
                        feature_i,
                        feature_j,
                        track_i,
                        track_j,
                        metrics,
                        reference_distance_m=float(metrics.get("median_distance_m", 0.0)),
                    )
                )
                pair_rows.append((global_i, global_j, metrics))

        probabilities = self._pairwise_affinity_predictor.predict_many(vectors)
        threshold = float(self.config.pairwise_connect_threshold)
        for (global_i, global_j, metrics), probability in zip(pair_rows, probabilities):
            row = dict(metrics)
            row.update(
                {
                    "i": global_i,
                    "j": global_j,
                    "pairwise_affinity_available": True,
                    "pairwise_same_part_probability": probability,
                    "pairwise_affinity_multiplier": probability,
                    "connected": probability >= threshold,
                }
            )
            if probability >= threshold:
                graph[global_i].add(global_j)
                graph[global_j].add(global_i)
            edge_metrics.append(row)

        moving_graph = [set() for _ in moving_indices]
        local_index = {global_index: index for index, global_index in enumerate(moving_indices)}
        for global_i in moving_indices:
            for global_j in graph[global_i]:
                moving_graph[local_index[global_i]].add(local_index[global_j])
        components = [
            [moving_indices[local_index_value] for local_index_value in component]
            for component in _connected_components(moving_graph, len(moving_indices))
        ]
        components = self._merge_small_components(tracks, components)
        components = sorted(components, key=lambda values: (-len(values), min(values)))
        part_ids = {index: 1 for index in static_indices}
        for part_id, component in enumerate(components, start=2):
            for index in component:
                part_ids[index] = part_id
        for index in ambiguous_indices:
            part_ids[index] = _nearest_assigned_part_id(tracks[index], tracks, part_ids)

        output_path = self._write_assignment(
            input_path=input_path,
            artifact=artifact,
            tracks=tracks,
            part_ids=part_ids,
            estimator="motion-learned-connected-cotracker-clustering",
            segmentation_extra={
                "mode": "learned-connected",
                "pairwise_connect_threshold": threshold,
                "source_estimator": artifact.get("estimator"),
                "track_count": len(tracks),
                "part_count": len(set(part_ids.values())),
                "config": self._config_payload(),
            },
        )
        diagnostics = self._diagnostics(
            mode="learned-connected",
            artifact=artifact,
            tracks=tracks,
            graph=graph,
            edge_metrics=edge_metrics,
            part_ids=part_ids,
            filter_report=filter_report,
            extra={
                "pairwise_connect_threshold": threshold,
                "all_pair_candidate_count": len(pair_rows),
                "accepted_edge_count": sum(bool(row["connected"]) for row in edge_metrics),
                "motion_split": {
                    "static_confident": len(static_indices),
                    "moving_confident": len(moving_indices),
                    "ambiguous": len(ambiguous_indices),
                },
            },
        )
        return output_path, diagnostics

    def _segment_sequential_ransac(
        self,
        input_path: Path,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        filter_report: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        static_indices, moving_indices, ambiguous_indices = self._motion_split(tracks)
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        references = [_reference_position(track) for track in tracks]
        base_transforms = (
            _fit_frame_transforms(static_indices, references, trajectories)
            if self.config.ransac_base_stabilize and len(static_indices) >= 3
            else {}
        )
        stabilized = _stabilize_trajectories(trajectories, base_transforms)
        learned_seed_neighbors: dict[int, list[tuple[float, int]]] | None = None
        if self.config.ransac_learned_seed:
            if self._pairwise_affinity_predictor is None:
                raise ValueError("--ransac-learned-seed requires --pairwise-affinity-model.")
            _, _, learned_edges = self._build_knn_weighted_graph(tracks, moving_indices, "B")
            learned_seed_neighbors = {index: [] for index in moving_indices}
            for edge in learned_edges:
                probability = edge.get("pairwise_same_part_probability")
                if not isinstance(probability, (int, float)):
                    continue
                index_i = int(edge["i"])
                index_j = int(edge["j"])
                learned_seed_neighbors[index_i].append((float(probability), index_j))
                learned_seed_neighbors[index_j].append((float(probability), index_i))
            for neighbors in learned_seed_neighbors.values():
                neighbors.sort(reverse=True)
        rng = random.Random(int(self.config.ransac_seed))
        remaining = set(moving_indices)
        models: list[dict[str, Any]] = []

        for model_index in range(max(1, int(self.config.ransac_max_models))):
            if len(remaining) < max(3, int(self.config.ransac_min_inliers)):
                break
            model = _best_rigid_ransac_model(
                candidate_indices=sorted(remaining),
                references=references,
                trajectories=stabilized,
                iterations=max(1, int(self.config.ransac_iterations)),
                sample_size=max(3, int(self.config.ransac_sample_size)),
                min_common_frames=max(2, int(self.config.min_common_frames)),
                inlier_threshold_m=max(1e-6, float(self.config.ransac_inlier_threshold_m)),
                spatial_link_m=max(1e-6, float(self.config.ransac_spatial_link_m)),
                rng=rng,
                learned_seed_neighbors=learned_seed_neighbors,
            )
            if model is None or len(model["inlier_indices"]) < max(3, int(self.config.ransac_min_inliers)):
                break
            model["model_index"] = model_index
            models.append(model)
            remaining.difference_update(model["inlier_indices"])

        if not models:
            raise ValueError(
                "Sequential RANSAC found no rigid moving-part model. "
                "Lower --ransac-min-inliers or increase --ransac-inlier-threshold-m."
            )

        part_ids = {index: 1 for index in static_indices}
        for model_index, model in enumerate(models, start=2):
            for index in model["inlier_indices"]:
                part_ids[int(index)] = model_index

        unassigned = sorted(set(ambiguous_indices) | remaining)
        assignment_rows = []
        for index in unassigned:
            best_part_id, residual = _best_model_assignment(
                index,
                models,
                references,
                stabilized,
            )
            motion = _track_motion_m(tracks[index])
            if motion <= float(self.config.static_motion_threshold_m):
                assigned = 1
            elif best_part_id is not None and residual is not None and residual <= float(
                self.config.ransac_assignment_threshold_m
            ):
                assigned = 2 + best_part_id
            else:
                assigned = _nearest_assigned_part_id(tracks[index], tracks, part_ids)
            part_ids[index] = int(assigned)
            assignment_rows.append(
                {
                    "track_index": int(index),
                    "assigned_part_id": int(assigned),
                    "best_model_residual_m": residual,
                    "fallback": best_part_id is None
                    or residual is None
                    or residual > float(self.config.ransac_assignment_threshold_m),
                }
            )

        model_rows = [
            {
                "model_index": int(model["model_index"]),
                "part_id": int(model["model_index"]) + 2,
                "inlier_count": len(model["inlier_indices"]),
                "inlier_ratio_at_extraction": model["inlier_ratio"],
                "mean_inlier_residual_m": model["mean_inlier_residual_m"],
                "median_inlier_residual_m": model["median_inlier_residual_m"],
                "model_frame_count": len(model["transforms"]),
                "sample_pool_count": int(model["sample_pool_count"]),
                "minimum_sample_support_frames": int(model["minimum_sample_support_frames"]),
                "sample_indices": [int(value) for value in model["sample_indices"]],
            }
            for model in models
        ]
        output_path = self._write_assignment(
            input_path=input_path,
            artifact=artifact,
            tracks=tracks,
            part_ids=part_ids,
            estimator="motion-sequential-ransac-cotracker-clustering",
            segmentation_extra={
                "mode": "sequential-ransac",
                "source_estimator": artifact.get("estimator"),
                "track_count": len(tracks),
                "part_count": len(set(part_ids.values())),
                "config": self._config_payload(),
                "motion_split": {
                    "static_confident": len(static_indices),
                    "moving_confident": len(moving_indices),
                    "ambiguous": len(ambiguous_indices),
                },
                "base_stabilization_frame_count": len(base_transforms),
                "models": model_rows,
                "learned_seed_enabled": bool(self.config.ransac_learned_seed),
                "unassigned_reassignment": assignment_rows,
            },
        )
        diagnostics = self._diagnostics(
            mode="sequential-ransac",
            artifact=artifact,
            tracks=tracks,
            graph=[set() for _ in tracks],
            edge_metrics=[],
            part_ids=part_ids,
            filter_report=filter_report,
            extra={
                "motion_split": {
                    "static_confident": len(static_indices),
                    "moving_confident": len(moving_indices),
                    "ambiguous": len(ambiguous_indices),
                },
                "base_stabilization_frame_count": len(base_transforms),
                "models": model_rows,
                "unassigned_reassignment": assignment_rows,
            },
        )
        return output_path, diagnostics

    def _segment_knn_spectral(
        self,
        input_path: Path,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        filter_report: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        static_indices, moving_indices, ambiguous_indices = self._motion_split(tracks)
        if len(moving_indices) < max(2, self.config.k_min):
            raise ValueError("Not enough moving-confident tracks for kNN spectral segmentation.")

        ablations = ["A", "B", "C"] if self.config.edge_ablation == "all" else [self.config.edge_ablation]
        k_values = list(range(max(2, self.config.k_min), max(self.config.k_min, self.config.k_max) + 1))
        sweep: list[dict[str, Any]] = []
        selected_part_ids: dict[int, int] | None = None
        selected_ablation = "B" if self.config.edge_ablation == "all" else self.config.edge_ablation
        selected_k = self.config.spectral_k if self.config.spectral_k is not None else k_values[0]
        sweep_output_dir = (
            Path(self.config.sweep_output_dir).expanduser().resolve()
            if self.config.sweep_output_dir is not None
            else None
        )

        for ablation in ablations:
            weights, graph, edge_metrics = self._build_knn_weighted_graph(tracks, moving_indices, ablation)
            for k_value in k_values:
                labels = _spectral_cluster(weights, k_value)
                part_ids = self._part_ids_from_spectral(
                    tracks,
                    static_indices=static_indices,
                    moving_indices=moving_indices,
                    ambiguous_indices=ambiguous_indices,
                    labels=labels,
                )
                diagnostics = self._diagnostics(
                    mode="knn-spectral",
                    artifact=artifact,
                    tracks=tracks,
                    graph=graph,
                    edge_metrics=edge_metrics,
                    part_ids=part_ids,
                    filter_report=filter_report,
                    extra={
                        "ablation": ablation,
                        "k": k_value,
                        "motion_split": {
                            "static_confident": len(static_indices),
                            "moving_confident": len(moving_indices),
                            "ambiguous": len(ambiguous_indices),
                        },
                        "base_bridge_checks": self._base_bridge_checks(tracks),
                    },
                )
                sweep.append(
                    {
                        "ablation": ablation,
                        "k": k_value,
                        "diagnostics": diagnostics,
                    }
                )
                if sweep_output_dir is not None:
                    self._write_assignment(
                        input_path=input_path,
                        artifact=artifact,
                        tracks=tracks,
                        part_ids=part_ids,
                        estimator="motion-knn-spectral-cotracker-clustering",
                        output_path=sweep_output_dir / f"motion_part_tracks_{ablation}_K{k_value}.json",
                        segmentation_extra={
                            "mode": "knn-spectral",
                            "ablation": ablation,
                            "k": k_value,
                            "source_estimator": artifact.get("estimator"),
                            "track_count": len(tracks),
                            "part_count": len(set(part_ids.values())),
                            "config": self._config_payload(),
                        },
                    )
                if ablation == selected_ablation and k_value == selected_k:
                    selected_part_ids = part_ids

        if selected_part_ids is None:
            selected_part_ids = self._part_ids_from_spectral(
                tracks,
                static_indices=static_indices,
                moving_indices=moving_indices,
                ambiguous_indices=ambiguous_indices,
                labels=_spectral_cluster(
                    self._build_knn_weighted_graph(tracks, moving_indices, selected_ablation)[0],
                    selected_k,
                ),
            )

        output_path = self._write_assignment(
            input_path=input_path,
            artifact=artifact,
            tracks=tracks,
            part_ids=selected_part_ids,
            estimator="motion-knn-spectral-cotracker-clustering",
            segmentation_extra={
                "mode": "knn-spectral",
                "selected_ablation": selected_ablation,
                "selected_k": selected_k,
                "source_estimator": artifact.get("estimator"),
                "track_count": len(tracks),
                "part_count": len(set(selected_part_ids.values())),
                "config": self._config_payload(),
                "sweep": [
                    {
                        "ablation": item["ablation"],
                        "k": item["k"],
                        "largest_cluster_ratio": item["diagnostics"]["cluster_summary"]["largest_cluster_ratio"],
                        "mean_gt_purity": item["diagnostics"]["cluster_summary"].get("mean_gt_purity"),
                        "mean_rigid_rmse_m": item["diagnostics"]["cluster_summary"].get("mean_rigid_rmse_m"),
                    }
                    for item in sweep
                ],
            },
        )
        diagnostics = {
            "mode": "knn-spectral",
            "selected_ablation": selected_ablation,
            "selected_k": selected_k,
            "filter_report": filter_report,
            "sweep": sweep,
        }
        return output_path, diagnostics

    def _write_assignment(
        self,
        *,
        input_path: Path,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        part_ids: dict[int, int],
        estimator: str,
        segmentation_extra: dict[str, Any],
        output_path: Path | None = None,
    ) -> Path:
        relabeled_tracks = [self._relabel_track(track, part_ids[index]) for index, track in enumerate(tracks)]
        part_counts: dict[int, int] = {}
        for part_id in part_ids.values():
            part_counts[part_id] = part_counts.get(part_id, 0) + 1
        anchor_part_id = self._choose_anchor_part_id(relabeled_tracks)
        output_json = (
            output_path
            if output_path is not None
            else (
                Path(self.config.output_json).expanduser().resolve()
                if self.config.output_json is not None
                else input_path.with_name("motion_part_tracks.json")
            )
        )
        payload = dict(artifact)
        segmentation_payload = dict(segmentation_extra)
        segmentation_payload["anchor_part_id"] = anchor_part_id
        payload.update(
            {
                "input_track_path": str(input_path),
                "estimator": estimator,
                "part_segmentation": _part_segmentation(part_counts, anchor_part_id, estimator),
                "motion_segmentation": segmentation_payload,
                "part_track_counts": {
                    str(part_id): {"name": _part_name(part_id, anchor_part_id), "count": count}
                    for part_id, count in sorted(part_counts.items())
                },
                "tracks": relabeled_tracks,
            }
        )
        save_json(payload, output_json)
        return output_json

    def _build_connected_graph(self, tracks: list[dict[str, Any]]) -> tuple[list[set[int]], list[dict[str, Any]]]:
        graph = [set() for _ in tracks]
        edge_metrics = []
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        quality_by_track = [_quality_by_frame(track) for track in tracks] if self._use_time_quality_affinity() else None
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                metrics = _pair_metrics(
                    trajectories[i],
                    trajectories[j],
                    self.config.min_common_frames,
                    quality_by_track[i] if quality_by_track is not None else None,
                    quality_by_track[j] if quality_by_track is not None else None,
                    self._pair_quality_prior(tracks[i], tracks[j]) if self._use_edge_quality_prior() else None,
                )
                if metrics is None:
                    continue
                self._add_articulation_metrics(metrics, tracks[i], tracks[j])
                accepted = True
                if metrics["rigidity_rmse_m"] > self.config.rigidity_threshold_m:
                    accepted = False
                if metrics["median_distance_m"] > self.config.max_neighbor_distance_m and (
                    metrics["motion_disagreement_m"] > self.config.rigidity_threshold_m
                ):
                    accepted = False
                if accepted:
                    graph[i].add(j)
                    graph[j].add(i)
                    edge_metrics.append({"i": i, "j": j, **metrics})
        return graph, edge_metrics

    def _build_knn_weighted_graph(
        self,
        tracks: list[dict[str, Any]],
        indices: list[int],
        ablation: str,
    ) -> tuple[Any, list[set[int]], list[dict[str, Any]]]:
        np = _require_numpy()
        count = len(indices)
        weights = np.zeros((count, count), dtype=float)
        graph = [set() for _ in tracks]
        edge_metrics: list[dict[str, Any]] = []
        reference_points = [_reference_position(tracks[index]) for index in indices]
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        quality_by_track = [_quality_by_frame(track) for track in tracks] if self._use_time_quality_affinity() else None

        for local_i, global_i in enumerate(indices):
            distances = [
                (_distance(reference_points[local_i], reference_points[local_j]), local_j)
                for local_j in range(count)
                if local_j != local_i
            ]
            for _, local_j in sorted(distances)[: max(1, self.config.knn_k)]:
                if local_i >= local_j:
                    continue
                global_j = indices[local_j]
                metrics = _pair_metrics(
                    trajectories[global_i],
                    trajectories[global_j],
                    self.config.min_common_frames,
                    quality_by_track[global_i] if quality_by_track is not None else None,
                    quality_by_track[global_j] if quality_by_track is not None else None,
                    self._pair_quality_prior(tracks[global_i], tracks[global_j]) if self._use_edge_quality_prior() else None,
                )
                if metrics is None:
                    continue
                self._add_articulation_metrics(metrics, tracks[global_i], tracks[global_j])
                self._add_learned_feature_metrics(metrics, tracks[global_i], tracks[global_j])
                self._add_pairwise_affinity_metrics(metrics, tracks[global_i], tracks[global_j])
                weight = self._edge_weight(metrics, ablation)
                if weight <= 0.0:
                    continue
                weights[local_i, local_j] = max(weights[local_i, local_j], weight)
                weights[local_j, local_i] = max(weights[local_j, local_i], weight)
                graph[global_i].add(global_j)
                graph[global_j].add(global_i)
                edge_metrics.append({"i": global_i, "j": global_j, "weight": weight, "ablation": ablation, **metrics})
        return weights, graph, edge_metrics

    def _edge_weight(self, metrics: dict[str, float], ablation: str) -> float:
        rigidity = math.exp(-metrics["rigidity_rmse_m"] / max(1e-6, self.config.rigidity_threshold_m))
        spatial = math.exp(-metrics["median_distance_m"] / max(1e-6, self.config.max_neighbor_distance_m))
        endpoint = math.exp(-metrics["motion_disagreement_m"] / max(1e-6, self.config.moving_motion_threshold_m))
        visibility = min(1.0, metrics["common_frames"] / max(1.0, float(self.config.min_common_frames * 2)))
        velocity = max(0.0, metrics.get("velocity_cosine", 0.0))
        curve = math.exp(-metrics.get("displacement_curve_rmse_m", 0.0) / max(1e-6, self.config.moving_motion_threshold_m))
        if ablation == "A":
            score = rigidity * spatial
        elif ablation == "B":
            score = 0.55 * rigidity + 0.30 * endpoint + 0.15 * visibility
            score *= spatial
        elif ablation == "C":
            score = 0.35 * rigidity + 0.20 * curve + 0.20 * velocity + 0.15 * endpoint + 0.10 * visibility
            score *= spatial
        else:
            raise ValueError(f"Unsupported edge ablation: {ablation}")
        if bool(metrics.get("quality_edge_prior_enabled", False)):
            score *= float(metrics.get("pair_quality_prior", 1.0) or 1.0)
        if bool(metrics.get("articulation_affinity_enabled", False)):
            score *= float(metrics.get("articulation_compatibility", 1.0) or 1.0)
        if bool(metrics.get("learned_affinity_enabled", False)):
            score *= float(metrics.get("learned_affinity_multiplier", 1.0) or 1.0)
        if bool(metrics.get("pairwise_affinity_enabled", False)):
            score *= float(metrics.get("pairwise_affinity_multiplier", 1.0) or 1.0)
        return float(max(0.0, min(1.0, score)))

    def _merge_small_components(self, tracks: list[dict[str, Any]], components: list[list[int]]) -> list[list[int]]:
        min_size = max(1, int(self.config.min_tracks_per_part))
        large = [component for component in components if len(component) >= min_size]
        small = [component for component in components if len(component) < min_size]
        if not large:
            return components
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        quality_by_track = [_quality_by_frame(track) for track in tracks] if self._use_time_quality_affinity() else None
        for component in small:
            best_index = None
            best_score = float("inf")
            for large_index, candidate in enumerate(large):
                scores = []
                for i in component:
                    for j in candidate:
                        metrics = _pair_metrics(
                            trajectories[i],
                            trajectories[j],
                            self.config.min_common_frames,
                            quality_by_track[i] if quality_by_track is not None else None,
                            quality_by_track[j] if quality_by_track is not None else None,
                            self._pair_quality_prior(tracks[i], tracks[j]) if self._use_edge_quality_prior() else None,
                        )
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

    def _part_ids_from_spectral(
        self,
        tracks: list[dict[str, Any]],
        *,
        static_indices: list[int],
        moving_indices: list[int],
        ambiguous_indices: list[int],
        labels: list[int],
    ) -> dict[int, int]:
        part_ids: dict[int, int] = {}
        for index in static_indices:
            part_ids[index] = 1
        for index, label in zip(moving_indices, labels):
            part_ids[index] = 2 + int(label)
        for index in ambiguous_indices:
            part_ids[index] = _nearest_assigned_part_id(tracks[index], tracks, part_ids)
        return part_ids

    def _motion_split(self, tracks: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
        static_indices = []
        moving_indices = []
        ambiguous_indices = []
        for index, track in enumerate(tracks):
            motion = _track_motion_m(track)
            if motion <= self.config.static_motion_threshold_m:
                static_indices.append(index)
            elif motion >= self.config.moving_motion_threshold_m:
                moving_indices.append(index)
            else:
                ambiguous_indices.append(index)
        return static_indices, moving_indices, ambiguous_indices

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
        original_part_id = int(track.get("original_part_id", track.get("part_id", 0)))
        relabeled = dict(track)
        relabeled["original_part_id"] = original_part_id
        relabeled["part_id"] = int(part_id)
        relabeled["part_name"] = _part_name(part_id, None)
        return relabeled

    def _component_motion(self, tracks: list[dict[str, Any]], part_id: int) -> float:
        values = [_track_motion_m(track) for track in tracks if int(track.get("part_id", 0)) == part_id]
        return float(fmean(values)) if values else 0.0

    def _config_payload(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "rigidity_threshold_m": float(self.config.rigidity_threshold_m),
            "max_neighbor_distance_m": float(self.config.max_neighbor_distance_m),
            "min_common_frames": int(self.config.min_common_frames),
            "min_tracks_per_part": int(self.config.min_tracks_per_part),
            "min_motion_m": float(self.config.min_motion_m),
            "static_motion_threshold_m": float(self.config.static_motion_threshold_m),
            "moving_motion_threshold_m": float(self.config.moving_motion_threshold_m),
            "quality_filter": bool(self.config.quality_filter),
            "min_visible_frames": int(self.config.min_visible_frames),
            "max_depth_jump_m": float(self.config.max_depth_jump_m),
            "max_trajectory_jump_m": float(self.config.max_trajectory_jump_m),
            "knn_k": int(self.config.knn_k),
            "k_min": int(self.config.k_min),
            "k_max": int(self.config.k_max),
            "spectral_k": self.config.spectral_k,
            "edge_ablation": self.config.edge_ablation,
            "quality_weighted_affinity": bool(self.config.quality_weighted_affinity),
            "quality_weighted_affinity_time_only": bool(self.config.quality_weighted_affinity_time_only),
            "quality_weighted_affinity_edge_prior": bool(self.config.quality_weighted_affinity_edge_prior),
            "quality_affinity_min_pair_weight": float(self.config.quality_affinity_min_pair_weight),
            "articulation_compatible_affinity": bool(self.config.articulation_compatible_affinity),
            "articulation_type_mismatch_penalty": float(self.config.articulation_type_mismatch_penalty),
            "articulation_static_mismatch_penalty": float(self.config.articulation_static_mismatch_penalty),
            "articulation_min_motion_for_type_penalty": float(self.config.articulation_min_motion_for_type_penalty),
            "articulation_min_confidence_for_type_penalty": float(
                self.config.articulation_min_confidence_for_type_penalty
            ),
            "cotracker_features_npz": (
                None
                if self.config.cotracker_features_npz is None
                else str(Path(self.config.cotracker_features_npz).expanduser().resolve())
            ),
            "learned_affinity_floor": float(self.config.learned_affinity_floor),
            "pairwise_affinity_model": (
                None
                if self.config.pairwise_affinity_model is None
                else str(Path(self.config.pairwise_affinity_model).expanduser().resolve())
            ),
            "pairwise_affinity_floor": float(self.config.pairwise_affinity_floor),
            "pairwise_affinity_device": self.config.pairwise_affinity_device,
            "pairwise_connect_threshold": float(self.config.pairwise_connect_threshold),
            "skip_base_bridge_checks": bool(self.config.skip_base_bridge_checks),
            "ransac_iterations": int(self.config.ransac_iterations),
            "ransac_sample_size": int(self.config.ransac_sample_size),
            "ransac_inlier_threshold_m": float(self.config.ransac_inlier_threshold_m),
            "ransac_min_inliers": int(self.config.ransac_min_inliers),
            "ransac_max_models": int(self.config.ransac_max_models),
            "ransac_spatial_link_m": float(self.config.ransac_spatial_link_m),
            "ransac_assignment_threshold_m": float(self.config.ransac_assignment_threshold_m),
            "ransac_seed": int(self.config.ransac_seed),
            "ransac_base_stabilize": bool(self.config.ransac_base_stabilize),
            "ransac_learned_seed": bool(self.config.ransac_learned_seed),
        }

    def _use_time_quality_affinity(self) -> bool:
        return bool(self.config.quality_weighted_affinity or self.config.quality_weighted_affinity_time_only)

    def _use_edge_quality_prior(self) -> bool:
        return bool(self.config.quality_weighted_affinity or self.config.quality_weighted_affinity_edge_prior)

    def _pair_quality_prior(self, a_track: dict[str, Any], b_track: dict[str, Any]) -> float:
        raw = math.sqrt(track_quality_score(a_track) * track_quality_score(b_track))
        return max(float(self.config.quality_affinity_min_pair_weight), float(raw))

    def _add_articulation_metrics(
        self,
        metrics: dict[str, Any],
        a_track: dict[str, Any],
        b_track: dict[str, Any],
    ) -> None:
        enabled = bool(self.config.articulation_compatible_affinity)
        metrics["articulation_affinity_enabled"] = enabled
        if not enabled:
            return
        articulation_metrics = articulation_pair_compatibility_metrics(
            a_track,
            b_track,
            type_mismatch_penalty=float(self.config.articulation_type_mismatch_penalty),
            static_mismatch_penalty=float(self.config.articulation_static_mismatch_penalty),
            min_motion_for_type_penalty=float(self.config.articulation_min_motion_for_type_penalty),
            min_confidence_for_type_penalty=float(self.config.articulation_min_confidence_for_type_penalty),
        )
        metrics["articulation_compatibility"] = articulation_metrics["compatibility"]
        metrics["articulation_penalty"] = articulation_metrics["penalty"]
        metrics["articulation_penalty_reason"] = articulation_metrics["penalty_reason"]
        metrics["articulation_type_pair"] = (
            f"{articulation_metrics['a_best_motion_type']}:{articulation_metrics['b_best_motion_type']}"
        )

    def _add_learned_feature_metrics(
        self,
        metrics: dict[str, Any],
        a_track: dict[str, Any],
        b_track: dict[str, Any],
    ) -> None:
        enabled = bool(self._learned_feature_map) and self._pairwise_affinity_predictor is None
        metrics["learned_affinity_enabled"] = enabled
        if not enabled:
            return
        a_feature = self._learned_feature_map.get(int(a_track.get("track_id", -1)))
        b_feature = self._learned_feature_map.get(int(b_track.get("track_id", -1)))
        if a_feature is None or b_feature is None:
            metrics["learned_feature_pair_available"] = False
            metrics["learned_affinity_multiplier"] = 1.0
            return
        cosine = float(max(-1.0, min(1.0, float(a_feature @ b_feature))))
        similarity = 0.5 * (cosine + 1.0)
        floor = max(0.0, min(1.0, float(self.config.learned_affinity_floor)))
        metrics["learned_feature_pair_available"] = True
        metrics["learned_feature_cosine"] = cosine
        metrics["learned_affinity_multiplier"] = floor + (1.0 - floor) * similarity

    def _add_pairwise_affinity_metrics(
        self,
        metrics: dict[str, Any],
        a_track: dict[str, Any],
        b_track: dict[str, Any],
    ) -> None:
        metrics["pairwise_affinity_enabled"] = self._pairwise_affinity_predictor is not None
        if self._pairwise_affinity_predictor is None:
            return
        a_feature = self._learned_feature_map.get(int(a_track.get("track_id", -1)))
        b_feature = self._learned_feature_map.get(int(b_track.get("track_id", -1)))
        if a_feature is None or b_feature is None:
            metrics["pairwise_affinity_available"] = False
            metrics["pairwise_affinity_multiplier"] = 1.0
            return
        probability = self._pairwise_affinity_predictor.predict(
            a_feature,
            b_feature,
            a_track,
            b_track,
            metrics,
        )
        floor = max(0.0, min(1.0, float(self.config.pairwise_affinity_floor)))
        metrics["pairwise_affinity_available"] = True
        metrics["pairwise_same_part_probability"] = probability
        metrics["pairwise_affinity_multiplier"] = floor + (1.0 - floor) * probability

    def _base_bridge_checks(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config.skip_base_bridge_checks:
            return {"skipped": True, "reason": "disabled by --skip-base-bridge-checks"}
        out = {}
        ordered = sorted(range(len(tracks)), key=lambda index: _track_motion_m(tracks[index]))
        for fraction in (0.4, 0.6):
            remove_count = int(round(len(ordered) * fraction))
            keep = sorted(set(range(len(tracks))) - set(ordered[:remove_count]))
            if len(keep) < 2:
                out[f"remove_lowest_{int(fraction * 100)}pct"] = {"kept_track_count": len(keep)}
                continue
            subset = [tracks[index] for index in keep]
            graph, edge_metrics = self._build_connected_graph(subset)
            components = _connected_components(graph, len(subset))
            largest = max((len(component) for component in components), default=0)
            out[f"remove_lowest_{int(fraction * 100)}pct"] = {
                "kept_track_count": len(keep),
                "component_count": len(components),
                "largest_cluster_size": largest,
                "largest_cluster_ratio": largest / max(1, len(keep)),
                "edge_count": len(edge_metrics),
            }
        return out

    def _diagnostics(
        self,
        *,
        mode: str,
        artifact: dict[str, Any],
        tracks: list[dict[str, Any]],
        graph: list[set[int]],
        edge_metrics: list[dict[str, Any]],
        part_ids: dict[int, int],
        filter_report: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        cluster_items: dict[int, list[int]] = {}
        for index, part_id in part_ids.items():
            cluster_items.setdefault(part_id, []).append(index)
        cluster_diagnostics = [
            _cluster_diagnostics(part_id, indices, tracks, artifact)
            for part_id, indices in sorted(cluster_items.items())
        ]
        largest = max((item["track_count"] for item in cluster_diagnostics), default=0)
        mean_gt_purity = _mean_optional(item.get("gt_purity") for item in cluster_diagnostics)
        mean_rigid_rmse = _mean_optional(item.get("rigid_rmse_m") for item in cluster_diagnostics)
        return {
            "mode": mode,
            "track_count": len(tracks),
            "filter_report": filter_report,
            "edge_summary": _edge_summary(edge_metrics, tracks, artifact),
            "graph_component_summary": _graph_component_summary(graph),
            "cluster_summary": {
                "cluster_count": len(cluster_diagnostics),
                "largest_cluster_size": largest,
                "largest_cluster_ratio": largest / max(1, len(tracks)),
                "mean_gt_purity": mean_gt_purity,
                "mean_rigid_rmse_m": mean_rigid_rmse,
            },
            "clusters": cluster_diagnostics,
            **extra,
        }


class LocalMotionClusterSplitter:
    """Generate diagnostic local split candidates for bad motion clusters."""

    def __init__(self, config: LocalSplitDiagnosticsConfig) -> None:
        self.config = config

    def split(self) -> Path:
        input_path = Path(self.config.input_tracks).expanduser().resolve()
        artifact = load_json(input_path)
        tracks = [track for track in artifact.get("tracks", []) if isinstance(track, dict) and _valid_track(track)]
        if not tracks:
            raise ValueError("No valid 3D tracks found for local split diagnostics.")

        cluster_ids = self._target_cluster_ids()
        if not cluster_ids:
            raise ValueError("No split cluster ids were provided or recommended by the evaluation artifact.")

        output_dir = (
            Path(self.config.output_dir).expanduser().resolve()
            if self.config.output_dir is not None
            else input_path.with_name("local_split_diagnostics")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        helper = MotionPartSegmenter(
            MotionPartSegmentationConfig(
                input_tracks=input_path,
                mode="knn-spectral",
                rigidity_threshold_m=float(self.config.rigidity_threshold_m),
                max_neighbor_distance_m=float(self.config.max_neighbor_distance_m),
                min_common_frames=max(2, int(self.config.min_common_frames)),
                knn_k=max(1, int(self.config.knn_k)),
                edge_ablation=self.config.edge_ablation,
            )
        )

        candidates = []
        for cluster_id in cluster_ids:
            indices = [
                index
                for index, track in enumerate(tracks)
                if int(track.get("part_id", 0)) == int(cluster_id)
            ]
            before = _cluster_diagnostics(int(cluster_id), indices, tracks, artifact)
            if len(indices) < 4:
                candidates.append(
                    {
                        "split_cluster_id": int(cluster_id),
                        "status": "skipped",
                        "reason": "too_few_tracks",
                        "before": before,
                    }
                )
                continue
            local_tracks = [tracks[index] for index in indices]
            local_indices = list(range(len(local_tracks)))
            ablations = ["A", "B", "C"] if self.config.edge_ablation == "all" else [self.config.edge_ablation]
            for ablation in ablations:
                weights, graph, edge_metrics = helper._build_knn_weighted_graph(local_tracks, local_indices, ablation)
                for k_value in range(max(2, int(self.config.local_k_min)), max(int(self.config.local_k_min), int(self.config.local_k_max)) + 1):
                    labels = _spectral_cluster(weights, k_value)
                    part_ids = self._candidate_part_ids(tracks, indices, labels, int(cluster_id))
                    output_path = output_dir / f"motion_part_tracks_split_cluster_{int(cluster_id)}_{ablation}_K{k_value}.json"
                    helper._write_assignment(
                        input_path=input_path,
                        artifact=artifact,
                        tracks=tracks,
                        part_ids=part_ids,
                        estimator="motion-local-split-diagnostic",
                        output_path=output_path,
                        segmentation_extra={
                            "mode": "local-split-diagnostic",
                            "source_estimator": artifact.get("estimator"),
                            "split_cluster_id": int(cluster_id),
                            "local_ablation": ablation,
                            "local_k": int(k_value),
                            "track_count": len(tracks),
                            "part_count": len(set(part_ids.values())),
                            "config": self._config_payload(),
                        },
                    )
                    affected_part_ids = sorted({part_ids[index] for index in indices})
                    after = self._candidate_after_diagnostics(part_ids, tracks, artifact, affected_part_ids)
                    candidates.append(
                        {
                            "split_cluster_id": int(cluster_id),
                            "status": "written",
                            "ablation": ablation,
                            "local_k": int(k_value),
                            "output_json": str(output_path),
                            "before": before,
                            "after": after,
                            "edge_summary": _edge_summary(edge_metrics, local_tracks, artifact),
                        }
                    )

        summary_path = output_dir / "local_split_summary.json"
        save_json(
            {
                "source": "local-motion-cluster-split-diagnostics",
                "input_tracks": str(input_path),
                "evaluation_json": str(Path(self.config.evaluation_json).expanduser().resolve())
                if self.config.evaluation_json is not None
                else None,
                "target_cluster_ids": [int(value) for value in cluster_ids],
                "candidate_count": sum(1 for item in candidates if item.get("status") == "written"),
                "candidates": candidates,
                "notes": [
                    "This command only writes diagnostic candidate track artifacts.",
                    "Run estimate-part-poses, infer-joints, and evaluate-object-mask-kinematics on a candidate to compare cleanup quality.",
                ],
            },
            summary_path,
        )
        return summary_path

    def _target_cluster_ids(self) -> list[int]:
        explicit = [int(value) for value in (self.config.split_cluster_ids or []) if int(value) > 0]
        if explicit:
            return sorted(set(explicit))
        if self.config.evaluation_json is None:
            return []
        payload = load_json(Path(self.config.evaluation_json).expanduser().resolve())
        cluster_ids = []
        for joint in payload.get("per_joint", []):
            if not isinstance(joint, dict):
                continue
            reason = str(joint.get("failure_reason_guess") or "")
            recommendation = str(joint.get("cleanup_recommendation") or "")
            if reason not in {"child_cluster_mixed_or_nonrigid", "pivot_sensitive_to_child_cluster_outliers"}:
                continue
            if "split" not in recommendation:
                continue
            try:
                cluster_ids.append(int(joint.get("child_part_id")))
            except (TypeError, ValueError):
                continue
        return sorted(set(cluster_ids))

    def _candidate_part_ids(
        self,
        tracks: list[dict[str, Any]],
        split_indices: list[int],
        labels: list[int],
        split_cluster_id: int,
    ) -> dict[int, int]:
        original_ids = sorted({int(track.get("part_id", 0)) for track in tracks if int(track.get("part_id", 0)) > 0})
        next_part_id = max(original_ids, default=split_cluster_id) + 1
        label_counts: dict[int, int] = {}
        for label in labels:
            label_counts[int(label)] = label_counts.get(int(label), 0) + 1
        ordered_labels = sorted(label_counts, key=lambda label: (-label_counts[label], label))
        label_to_part_id = {ordered_labels[0]: int(split_cluster_id)}
        for label in ordered_labels[1:]:
            label_to_part_id[label] = next_part_id
            next_part_id += 1
        out = {
            index: int(track.get("part_id", 0))
            for index, track in enumerate(tracks)
        }
        for global_index, label in zip(split_indices, labels):
            out[global_index] = int(label_to_part_id[int(label)])
        return out

    def _candidate_after_diagnostics(
        self,
        part_ids: dict[int, int],
        tracks: list[dict[str, Any]],
        artifact: dict[str, Any],
        affected_part_ids: list[int],
    ) -> dict[str, Any]:
        out = []
        for part_id in affected_part_ids:
            indices = [index for index, value in part_ids.items() if value == part_id]
            if not indices:
                continue
            out.append(_cluster_diagnostics(part_id, indices, tracks, artifact))
        scored = _score_local_split_clusters(out)
        return {
            "clusters": scored,
            "best_cluster_by_no_gt_score": _best_scored_cluster(scored, "selection_score_no_gt"),
            "best_cluster_by_diagnostic_gt_score": _best_scored_cluster(scored, "diagnostic_score_with_gt"),
            "mean_gt_purity": _mean_optional(item.get("gt_purity") for item in scored),
            "mean_rigid_rmse_m": _mean_optional(item.get("rigid_rmse_m") for item in scored),
            "max_rigid_rmse_m": max((float(item["rigid_rmse_m"]) for item in scored if item.get("rigid_rmse_m") is not None), default=None),
        }

    def _config_payload(self) -> dict[str, Any]:
        return {
            "local_k_min": int(self.config.local_k_min),
            "local_k_max": int(self.config.local_k_max),
            "knn_k": int(self.config.knn_k),
            "edge_ablation": self.config.edge_ablation,
            "rigidity_threshold_m": float(self.config.rigidity_threshold_m),
            "max_neighbor_distance_m": float(self.config.max_neighbor_distance_m),
            "min_common_frames": int(self.config.min_common_frames),
        }


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


def _quality_by_frame(track: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for sample in track.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if not bool(sample.get("visible", False)) or not bool(sample.get("depth_valid", True)):
            continue
        if not isinstance(sample.get("xyz_world"), list) or len(sample.get("xyz_world", [])) != 3:
            continue
        out[int(sample.get("frame_index", 0))] = observation_weight(track, sample)
    return out


def _pair_quality_prior(a_track: dict[str, Any], b_track: dict[str, Any]) -> float:
    return math.sqrt(track_quality_score(a_track) * track_quality_score(b_track))


def _pair_metrics(
    a_by_frame: dict[int, list[float]],
    b_by_frame: dict[int, list[float]],
    min_common_frames: int,
    a_quality_by_frame: dict[int, float] | None = None,
    b_quality_by_frame: dict[int, float] | None = None,
    pair_quality_prior: float | None = None,
) -> dict[str, float] | None:
    common = sorted(set(a_by_frame) & set(b_by_frame))
    if len(common) < min_common_frames:
        return None
    distances = [_distance(a_by_frame[frame], b_by_frame[frame]) for frame in common]
    weights = [
        math.sqrt(
            max(1e-9, (a_quality_by_frame or {}).get(frame, 1.0))
            * max(1e-9, (b_quality_by_frame or {}).get(frame, 1.0))
        )
        for frame in common
    ]
    weighted = a_quality_by_frame is not None or b_quality_by_frame is not None
    mean_distance = _weighted_mean(distances, weights) if weighted else fmean(distances)
    rigidity_rmse = (
        math.sqrt(_weighted_mean([(distance - mean_distance) ** 2 for distance in distances], weights))
        if weighted
        else math.sqrt(fmean((distance - mean_distance) ** 2 for distance in distances))
    )
    start = common[0]
    end = common[-1]
    motion_a = _subtract(a_by_frame[end], a_by_frame[start])
    motion_b = _subtract(b_by_frame[end], b_by_frame[start])
    velocity_cosine = _cosine(motion_a, motion_b)
    displacement_curve_rmse = math.sqrt(
        fmean(
            (
                _distance(a_by_frame[frame], a_by_frame[start])
                - _distance(b_by_frame[frame], b_by_frame[start])
            )
            ** 2
            for frame in common
        )
    )
    return {
        "common_frames": float(len(common)),
        "quality_weighted": bool(weighted),
        "quality_time_weighted": bool(weighted),
        "quality_edge_prior_enabled": pair_quality_prior is not None,
        "weighted_pair_count": float(sum(weights)) if weighted else float(len(common)),
        "pair_quality_prior": float(pair_quality_prior) if pair_quality_prior is not None else 1.0,
        "effective_pair_observation_count": float(sum(weights)) if weighted else float(len(common)),
        "effective_pair_observation_ratio": float(sum(weights) / len(common)) if weighted else 1.0,
        "low_quality_pair_ratio": (
            float(sum(weight < 0.35 for weight in weights) / len(weights))
            if weighted and weights
            else None
        ),
        "rigidity_rmse_m": float(rigidity_rmse),
        "median_distance_m": float(sorted(distances)[len(distances) // 2]),
        "motion_disagreement_m": float(_distance(motion_a, motion_b)),
        "velocity_cosine": float(velocity_cosine),
        "displacement_curve_rmse_m": float(displacement_curve_rmse),
    }


def _fit_rigid_transform(
    source_points: list[list[float]],
    target_points: list[list[float]],
) -> tuple[list[list[float]], list[float]] | None:
    np = _optional_numpy()
    if np is None or len(source_points) < 3 or len(source_points) != len(target_points):
        return None
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    if np.linalg.matrix_rank(source_centered) < 2:
        return None
    try:
        covariance = source_centered.T @ target_centered
        u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_centroid - rotation @ source_centroid
    return rotation.tolist(), translation.tolist()


def _apply_rigid_transform(
    point: list[float],
    transform: tuple[list[list[float]], list[float]],
) -> list[float]:
    rotation, translation = transform
    return [
        sum(float(rotation[row][col]) * float(point[col]) for col in range(3)) + float(translation[row])
        for row in range(3)
    ]


def _apply_inverse_rigid_transform(
    point: list[float],
    transform: tuple[list[list[float]], list[float]],
) -> list[float]:
    rotation, translation = transform
    centered = [float(point[axis]) - float(translation[axis]) for axis in range(3)]
    return [sum(float(rotation[row][col]) * centered[row] for row in range(3)) for col in range(3)]


def _fit_frame_transforms(
    indices: list[int],
    references: list[list[float]],
    trajectories: list[dict[int, list[float]]],
) -> dict[int, tuple[list[list[float]], list[float]]]:
    frames = sorted({frame for index in indices for frame in trajectories[index]})
    transforms: dict[int, tuple[list[list[float]], list[float]]] = {}
    for frame in frames:
        visible = [index for index in indices if frame in trajectories[index]]
        transform = _fit_rigid_transform(
            [references[index] for index in visible],
            [trajectories[index][frame] for index in visible],
        )
        if transform is not None:
            transforms[int(frame)] = transform
    return transforms


def _stabilize_trajectories(
    trajectories: list[dict[int, list[float]]],
    base_transforms: dict[int, tuple[list[list[float]], list[float]]],
) -> list[dict[int, list[float]]]:
    if not base_transforms:
        return trajectories
    return [
        {
            frame: (
                _apply_inverse_rigid_transform(point, base_transforms[frame])
                if frame in base_transforms
                else list(point)
            )
            for frame, point in trajectory.items()
        }
        for trajectory in trajectories
    ]


def _model_track_residual(
    index: int,
    transforms: dict[int, tuple[list[list[float]], list[float]]],
    references: list[list[float]],
    trajectories: list[dict[int, list[float]]],
    min_common_frames: int,
) -> float | None:
    frames = sorted(set(transforms) & set(trajectories[index]))
    if len(frames) < min_common_frames:
        return None
    errors = [
        _distance(
            _apply_rigid_transform(references[index], transforms[frame]),
            trajectories[index][frame],
        )
        for frame in frames
    ]
    errors.sort()
    keep = errors[: max(min_common_frames, int(math.ceil(0.8 * len(errors))))]
    return math.sqrt(fmean(error * error for error in keep))


def _spatial_consensus_component(
    inliers: list[int],
    sample_indices: list[int],
    references: list[list[float]],
    spatial_link_m: float,
) -> list[int]:
    if not inliers:
        return []
    local_graph = [set() for _ in inliers]
    for local_i, index_i in enumerate(inliers):
        distances = sorted(
            (
                _distance(references[index_i], references[index_j]),
                local_j,
            )
            for local_j, index_j in enumerate(inliers)
            if local_j != local_i
        )
        for distance, local_j in distances[:8]:
            if distance <= spatial_link_m:
                local_graph[local_i].add(local_j)
                local_graph[local_j].add(local_i)
    components = _connected_components(local_graph, len(inliers))
    sample_set = set(sample_indices)
    best = max(
        components,
        key=lambda component: (
            sum(inliers[local_index] in sample_set for local_index in component),
            len(component),
        ),
    )
    return sorted(inliers[local_index] for local_index in best)


def _best_rigid_ransac_model(
    *,
    candidate_indices: list[int],
    references: list[list[float]],
    trajectories: list[dict[int, list[float]]],
    iterations: int,
    sample_size: int,
    min_common_frames: int,
    inlier_threshold_m: float,
    spatial_link_m: float,
    rng: random.Random,
    learned_seed_neighbors: dict[int, list[tuple[float, int]]] | None = None,
) -> dict[str, Any] | None:
    if len(candidate_indices) < max(3, sample_size):
        return None
    maximum_support = max(len(trajectories[index]) for index in candidate_indices)
    minimum_sample_support = max(min_common_frames, int(math.ceil(0.5 * maximum_support)))
    sample_pool = [
        index
        for index in candidate_indices
        if len(trajectories[index]) >= minimum_sample_support
    ]
    if len(sample_pool) < max(3, sample_size):
        sample_pool = candidate_indices
    best: dict[str, Any] | None = None
    for _ in range(iterations):
        sample = _ransac_sample(
            sample_pool,
            sample_size,
            rng,
            learned_seed_neighbors=learned_seed_neighbors,
        )
        transforms = _fit_frame_transforms(sample, references, trajectories)
        if len(transforms) < min_common_frames:
            continue
        residuals = {
            index: residual
            for index in candidate_indices
            if (
                residual := _model_track_residual(
                    index,
                    transforms,
                    references,
                    trajectories,
                    min_common_frames,
                )
            )
            is not None
        }
        inliers = sorted(index for index, residual in residuals.items() if residual <= inlier_threshold_m)
        inliers = _spatial_consensus_component(inliers, sample, references, spatial_link_m)
        if len(inliers) < 3:
            continue
        inlier_residuals = [residuals[index] for index in inliers]
        key = (len(inliers), -fmean(inlier_residuals), len(transforms))
        if best is None or key > best["selection_key"]:
            best = {
                "sample_indices": sample,
                "inlier_indices": inliers,
                "transforms": transforms,
                "residuals": residuals,
                "selection_key": key,
            }
    if best is None:
        return None

    refined_transforms = _fit_frame_transforms(best["inlier_indices"], references, trajectories)
    if len(refined_transforms) >= min_common_frames:
        refined_residuals = {
            index: residual
            for index in candidate_indices
            if (
                residual := _model_track_residual(
                    index,
                    refined_transforms,
                    references,
                    trajectories,
                    min_common_frames,
                )
            )
            is not None
        }
        refined_inliers = sorted(
            index for index, residual in refined_residuals.items() if residual <= inlier_threshold_m
        )
        refined_inliers = _spatial_consensus_component(
            refined_inliers,
            best["sample_indices"],
            references,
            spatial_link_m,
        )
        # The minimal hypothesis may only span the few frames shared by its sampled
        # tracks. Always prefer a valid consensus refit so the model is evaluated
        # over the full temporal support, even when stricter replay drops inliers.
        if len(refined_inliers) >= 3:
            best["inlier_indices"] = refined_inliers
            best["transforms"] = refined_transforms
            best["residuals"] = refined_residuals

    values = [best["residuals"][index] for index in best["inlier_indices"]]
    best["inlier_ratio"] = len(best["inlier_indices"]) / max(1, len(candidate_indices))
    best["sample_pool_count"] = len(sample_pool)
    best["minimum_sample_support_frames"] = minimum_sample_support
    best["mean_inlier_residual_m"] = float(fmean(values))
    best["median_inlier_residual_m"] = float(_median(values))
    best.pop("selection_key", None)
    return best


def _ransac_sample(
    sample_pool: list[int],
    sample_size: int,
    rng: random.Random,
    *,
    learned_seed_neighbors: dict[int, list[tuple[float, int]]] | None,
) -> list[int]:
    count = min(sample_size, len(sample_pool))
    if learned_seed_neighbors is None or count <= 1:
        return sorted(rng.sample(sample_pool, count))
    anchor = rng.choice(sample_pool)
    pool_set = set(sample_pool)
    ranked = [
        index
        for probability, index in learned_seed_neighbors.get(anchor, [])
        if probability >= 0.5 and index in pool_set and index != anchor
    ]
    candidate_window = ranked[: max(count - 1, 3 * (count - 1))]
    chosen = rng.sample(candidate_window, min(count - 1, len(candidate_window)))
    if len(chosen) < count - 1:
        fallback = [index for index in sample_pool if index != anchor and index not in chosen]
        chosen.extend(rng.sample(fallback, count - 1 - len(chosen)))
    return sorted([anchor, *chosen])


def _best_model_assignment(
    index: int,
    models: list[dict[str, Any]],
    references: list[list[float]],
    trajectories: list[dict[int, list[float]]],
) -> tuple[int | None, float | None]:
    candidates = []
    for model_index, model in enumerate(models):
        residual = _model_track_residual(
            index,
            model["transforms"],
            references,
            trajectories,
            min_common_frames=2,
        )
        if residual is not None:
            candidates.append((float(residual), model_index))
    if not candidates:
        return None, None
    residual, model_index = min(candidates)
    return int(model_index), float(residual)


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


def _spectral_cluster(weights: Any, k_value: int) -> list[int]:
    np = _require_numpy()
    count = int(weights.shape[0])
    if count == 0:
        return []
    k = max(1, min(int(k_value), count))
    if k == 1:
        return [0 for _ in range(count)]
    degrees = weights.sum(axis=1)
    isolated = degrees <= 1e-12
    safe_degrees = np.where(isolated, 1.0, degrees)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe_degrees))
    laplacian = np.eye(count) - inv_sqrt @ weights @ inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    embedding = eigenvectors[:, np.argsort(eigenvalues)[:k]]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-12)
    labels = _kmeans(embedding, k)
    return [int(label) for label in labels]


def _kmeans(points: Any, k_value: int, iterations: int = 50) -> list[int]:
    np = _require_numpy()
    count = int(points.shape[0])
    k = max(1, min(int(k_value), count))
    centers = [points[0]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(points[int(np.argmax(distances))])
    centers_array = np.stack(centers, axis=0)
    labels = np.zeros(count, dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.sum((points - center) ** 2, axis=1) for center in centers_array], axis=1)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = points[labels == cluster_id]
            if len(members):
                centers_array[cluster_id] = members.mean(axis=0)
    return [int(label) for label in labels]


def _track_motion_m(track: dict[str, Any]) -> float:
    trajectory = _trajectory_by_frame(track)
    if len(trajectory) < 2:
        return 0.0
    first = trajectory[min(trajectory)]
    last = trajectory[max(trajectory)]
    return _distance(first, last)


def _max_trajectory_jump_m(track: dict[str, Any]) -> float:
    trajectory = _trajectory_by_frame(track)
    frames = sorted(trajectory)
    if len(frames) < 2:
        return 0.0
    return max(_distance(trajectory[a], trajectory[b]) for a, b in zip(frames[:-1], frames[1:]))


def _max_depth_jump(track: dict[str, Any]) -> float:
    samples = [
        sample
        for sample in track.get("samples", [])
        if isinstance(sample, dict)
        and bool(sample.get("visible", False))
        and bool(sample.get("depth_valid", True))
        and isinstance(sample.get("depth_m"), (int, float))
    ]
    if len(samples) < 2:
        return 0.0
    samples = sorted(samples, key=lambda item: int(item.get("frame_index", 0)))
    return max(abs(float(a["depth_m"]) - float(b["depth_m"])) for a, b in zip(samples[:-1], samples[1:]))


def _reference_position(track: dict[str, Any]) -> list[float]:
    point = track.get("reference_xyz_world")
    if isinstance(point, list) and len(point) == 3:
        return [float(value) for value in point]
    trajectory = _trajectory_by_frame(track)
    if trajectory:
        return trajectory[min(trajectory)]
    return [0.0, 0.0, 0.0]


def _nearest_assigned_part_id(track: dict[str, Any], tracks: list[dict[str, Any]], part_ids: dict[int, int]) -> int:
    if not part_ids:
        return 1
    point = _reference_position(track)
    best_index = min(part_ids, key=lambda index: _distance(point, _reference_position(tracks[index])))
    return int(part_ids[best_index])


def _part_segmentation(part_counts: dict[int, int], anchor_part_id: int, source: str) -> dict[str, Any]:
    return {
        "source": source,
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


def _edge_summary(edge_metrics: list[dict[str, Any]], tracks: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    same_gt = 0
    cross_gt = 0
    unknown = 0
    type_counts: dict[str, int] = {}
    same_gt_learned_cosines: list[float] = []
    cross_gt_learned_cosines: list[float] = []
    same_gt_pairwise_probabilities: list[float] = []
    cross_gt_pairwise_probabilities: list[float] = []
    part_meta = _original_part_metadata(artifact)
    for edge in edge_metrics:
        left = _original_part_id(tracks[int(edge["i"])])
        right = _original_part_id(tracks[int(edge["j"])])
        if left is None or right is None:
            unknown += 1
            continue
        if left == right:
            same_gt += 1
            if isinstance(edge.get("learned_feature_cosine"), (int, float)):
                same_gt_learned_cosines.append(float(edge["learned_feature_cosine"]))
            if isinstance(edge.get("pairwise_same_part_probability"), (int, float)):
                same_gt_pairwise_probabilities.append(float(edge["pairwise_same_part_probability"]))
        else:
            cross_gt += 1
            if isinstance(edge.get("learned_feature_cosine"), (int, float)):
                cross_gt_learned_cosines.append(float(edge["learned_feature_cosine"]))
            if isinstance(edge.get("pairwise_same_part_probability"), (int, float)):
                cross_gt_pairwise_probabilities.append(float(edge["pairwise_same_part_probability"]))
            edge_type = _cross_edge_type(left, right, part_meta)
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
    total_known = same_gt + cross_gt
    weights = [float(edge["weight"]) for edge in edge_metrics if isinstance(edge.get("weight"), (int, float))]
    pair_priors = [
        float(edge["pair_quality_prior"])
        for edge in edge_metrics
        if isinstance(edge.get("pair_quality_prior"), (int, float))
    ]
    effective_counts = [
        float(edge["effective_pair_observation_count"])
        for edge in edge_metrics
        if isinstance(edge.get("effective_pair_observation_count"), (int, float))
    ]
    effective_ratios = [
        float(edge["effective_pair_observation_ratio"])
        for edge in edge_metrics
        if isinstance(edge.get("effective_pair_observation_ratio"), (int, float))
    ]
    articulation_compatibilities = [
        float(edge["articulation_compatibility"])
        for edge in edge_metrics
        if isinstance(edge.get("articulation_compatibility"), (int, float))
    ]
    learned_cosines = [
        float(edge["learned_feature_cosine"])
        for edge in edge_metrics
        if isinstance(edge.get("learned_feature_cosine"), (int, float))
    ]
    learned_multipliers = [
        float(edge["learned_affinity_multiplier"])
        for edge in edge_metrics
        if bool(edge.get("learned_feature_pair_available"))
        and isinstance(edge.get("learned_affinity_multiplier"), (int, float))
    ]
    pairwise_probabilities = [
        float(edge["pairwise_same_part_probability"])
        for edge in edge_metrics
        if bool(edge.get("pairwise_affinity_available"))
        and isinstance(edge.get("pairwise_same_part_probability"), (int, float))
    ]
    pairwise_multipliers = [
        float(edge["pairwise_affinity_multiplier"])
        for edge in edge_metrics
        if bool(edge.get("pairwise_affinity_available"))
        and isinstance(edge.get("pairwise_affinity_multiplier"), (int, float))
    ]
    static_articulated_pair_count = 0
    static_articulated_penalized_count = 0
    static_articulated_neutralized_count = 0
    confident_type_mismatch_pair_count = 0
    for edge in edge_metrics:
        reason = str(edge.get("articulation_penalty_reason") or "")
        type_pair = str(edge.get("articulation_type_pair") or "")
        if "static" in type_pair and ":" in type_pair and len(set(type_pair.split(":"))) > 1:
            static_articulated_pair_count += 1
            if reason == "static_mismatch_penalized":
                static_articulated_penalized_count += 1
            elif reason == "static_mismatch_neutralized":
                static_articulated_neutralized_count += 1
        if reason == "confident_nonstatic_type_mismatch":
            confident_type_mismatch_pair_count += 1
    low_weight_edges = sum(1 for edge in edge_metrics if float(edge.get("pair_quality_prior", 1.0) or 1.0) < 0.35)
    return {
        "edge_count": len(edge_metrics),
        "weighted_graph_edge_count": len(weights),
        "mean_edge_weight": _mean_optional(weights),
        "median_edge_weight": _median(weights) if weights else None,
        "mean_pair_quality_prior": _mean_optional(pair_priors),
        "median_pair_quality_prior": _median(pair_priors) if pair_priors else None,
        "low_weight_edge_ratio": low_weight_edges / len(edge_metrics) if edge_metrics else None,
        "effective_pair_observation_count_mean": _mean_optional(effective_counts),
        "effective_pair_observation_count_median": _median(effective_counts) if effective_counts else None,
        "effective_pair_observation_ratio_mean": _mean_optional(effective_ratios),
        "effective_pair_observation_ratio_median": _median(effective_ratios) if effective_ratios else None,
        "mean_articulation_compatibility": _mean_optional(articulation_compatibilities),
        "median_articulation_compatibility": (
            _median(articulation_compatibilities) if articulation_compatibilities else None
        ),
        "learned_feature_pair_count": len(learned_cosines),
        "learned_feature_pair_coverage": len(learned_cosines) / len(edge_metrics) if edge_metrics else None,
        "mean_learned_feature_cosine": _mean_optional(learned_cosines),
        "median_learned_feature_cosine": _median(learned_cosines) if learned_cosines else None,
        "mean_learned_affinity_multiplier": _mean_optional(learned_multipliers),
        "median_learned_affinity_multiplier": _median(learned_multipliers) if learned_multipliers else None,
        "mean_same_gt_learned_feature_cosine": _mean_optional(same_gt_learned_cosines),
        "mean_cross_gt_learned_feature_cosine": _mean_optional(cross_gt_learned_cosines),
        "pairwise_affinity_pair_count": len(pairwise_probabilities),
        "mean_pairwise_same_part_probability": _mean_optional(pairwise_probabilities),
        "median_pairwise_same_part_probability": _median(pairwise_probabilities) if pairwise_probabilities else None,
        "mean_pairwise_affinity_multiplier": _mean_optional(pairwise_multipliers),
        "mean_same_gt_pairwise_probability": _mean_optional(same_gt_pairwise_probabilities),
        "mean_cross_gt_pairwise_probability": _mean_optional(cross_gt_pairwise_probabilities),
        "static_articulated_pair_count": static_articulated_pair_count,
        "static_articulated_penalized_count": static_articulated_penalized_count,
        "static_articulated_neutralized_count": static_articulated_neutralized_count,
        "confident_type_mismatch_pair_count": confident_type_mismatch_pair_count,
        "same_gt_edge_count": same_gt,
        "cross_gt_edge_count": cross_gt,
        "unknown_gt_edge_count": unknown,
        "same_gt_edge_ratio": same_gt / total_known if total_known else None,
        "cross_gt_edge_ratio": cross_gt / total_known if total_known else None,
        "cross_edge_type_breakdown": type_counts,
    }


def _graph_component_summary(graph: list[set[int]]) -> dict[str, Any]:
    components = _connected_components(graph, len(graph))
    sizes = sorted((len(component) for component in components), reverse=True)
    return {
        "component_count": len(sizes),
        "largest_component_size": sizes[0] if sizes else 0,
        "largest_component_ratio": sizes[0] / max(1, len(graph)) if sizes else 0.0,
    }


def _cluster_diagnostics(
    part_id: int,
    indices: list[int],
    tracks: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    composition: dict[str, int] = {}
    for index in indices:
        original = _original_part_id(tracks[index])
        key = str(original) if original is not None else "unknown"
        composition[key] = composition.get(key, 0) + 1
    dominant_count = max(composition.values(), default=0)
    residual = _cluster_rigid_residual([tracks[index] for index in indices])
    cluster_tracks = [tracks[index] for index in indices]
    rigid_rmse = residual.get("rigid_rmse_m")
    cluster_articulation_score = (
        math.exp(-float(rigid_rmse) / 0.03)
        if isinstance(rigid_rmse, (int, float)) and math.isfinite(float(rigid_rmse))
        else None
    )
    return {
        "part_id": int(part_id),
        "track_count": len(indices),
        "mean_motion_m": float(fmean(_track_motion_m(tracks[index]) for index in indices)) if indices else 0.0,
        "median_motion_m": _median([_track_motion_m(tracks[index]) for index in indices]),
        "bbox_diag_m": _cluster_bbox_diag_m(cluster_tracks),
        "visible_frame_ratio": _cluster_visible_frame_ratio(cluster_tracks),
        "gt_composition": composition,
        "dominant_gt_part": max(composition, key=composition.get) if composition else None,
        "gt_purity": dominant_count / max(1, len(indices)) if composition and "unknown" not in composition else None,
        "cluster_articulation_model": "shared_rigid_se3_proxy",
        "cluster_articulation_rmse_m": rigid_rmse,
        "cluster_articulation_score": cluster_articulation_score,
        "cluster_articulation_track_score_mean": _mean_optional(
            [articulation_track_score(tracks[index]) for index in indices]
        ),
        **cluster_quality_summary(cluster_tracks),
        **residual,
    }


def _score_local_split_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not clusters:
        return []
    max_track_count = max((int(item.get("track_count", 0)) for item in clusters), default=1)
    max_bbox = max((float(item.get("bbox_diag_m", 0.0) or 0.0) for item in clusters), default=0.0)
    max_motion = max((float(item.get("mean_motion_m", 0.0) or 0.0) for item in clusters), default=0.0)
    scored = []
    for cluster in clusters:
        track_count = int(cluster.get("track_count", 0))
        bbox_diag = float(cluster.get("bbox_diag_m", 0.0) or 0.0)
        motion = float(cluster.get("mean_motion_m", 0.0) or 0.0)
        rigid_rmse = cluster.get("rigid_rmse_m")
        inlier_ratio = cluster.get("inlier_ratio")
        visible_ratio = cluster.get("visible_frame_ratio")
        purity = cluster.get("gt_purity")

        # These are soft proposal scores, not rejection thresholds. Small but
        # real parts should remain visible in the table instead of being
        # discarded by aggressive bbox or motion filters.
        track_score = math.sqrt(_safe_ratio(track_count, max_track_count))
        bbox_score = math.sqrt(_safe_ratio(bbox_diag, max_bbox))
        motion_score = math.sqrt(_safe_ratio(motion, max_motion))
        rigid_score = _rigid_score(rigid_rmse)
        inlier_score = _bounded_score(inlier_ratio, default=0.5)
        visibility_score = _bounded_score(visible_ratio, default=0.5)
        base_like_low_motion = motion < 0.03
        selection_score = (
            0.15 * rigid_score
            + 0.15 * inlier_score
            + 0.10 * visibility_score
            + 0.30 * motion_score
            + 0.15 * bbox_score
            + 0.15 * track_score
        )
        if base_like_low_motion:
            selection_score *= 0.75
        diagnostic_score = selection_score
        if purity is not None:
            diagnostic_score = 0.80 * selection_score + 0.20 * _bounded_score(purity, default=0.0)

        item = dict(cluster)
        item["candidate_selection"] = {
            "selection_score_no_gt": float(selection_score),
            "diagnostic_score_with_gt": float(diagnostic_score),
            "track_count_score": float(track_score),
            "bbox_score": float(bbox_score),
            "motion_score": float(motion_score),
            "rigid_score": float(rigid_score),
            "inlier_score": float(inlier_score),
            "visibility_score": float(visibility_score),
            "selection_score_uses_gt": False,
            "diagnostic_score_uses_gt": purity is not None,
            "base_like_low_motion": bool(base_like_low_motion),
            "notes": [
                "Scores are soft ranking signals, not hard filters.",
                "Low-motion candidates are downweighted, not rejected.",
                "diagnostic_score_with_gt is for simulation analysis only.",
            ],
        }
        scored.append(item)
    return scored


def _best_scored_cluster(clusters: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not clusters:
        return None
    best = max(
        clusters,
        key=lambda item: float(item.get("candidate_selection", {}).get(key, -1.0)),
    )
    return {
        "part_id": best.get("part_id"),
        "track_count": best.get("track_count"),
        "gt_purity": best.get("gt_purity"),
        "rigid_rmse_m": best.get("rigid_rmse_m"),
        "mean_motion_m": best.get("mean_motion_m"),
        "bbox_diag_m": best.get("bbox_diag_m"),
        key: best.get("candidate_selection", {}).get(key),
    }


def _rigid_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(numeric):
        return 0.0
    return 1.0 / (1.0 + max(0.0, numeric) / 0.03)


def _bounded_score(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(max(0.0, min(1.0, numeric)))


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, float(value) / float(denominator)))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(max(1e-9, float(weight)) for weight in weights[: len(values)])
    if total_weight <= 1e-12:
        return float(fmean(values))
    return sum(float(value) * max(1e-9, float(weight)) for value, weight in zip(values, weights)) / total_weight


def _cluster_bbox_diag_m(tracks: list[dict[str, Any]]) -> float:
    points = []
    for track in tracks:
        points.append(_reference_position(track))
    if not points:
        return 0.0
    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    return _distance(mins, maxs)


def _cluster_visible_frame_ratio(tracks: list[dict[str, Any]]) -> float | None:
    if not tracks:
        return None
    frame_sets = [set(_trajectory_by_frame(track)) for track in tracks]
    all_frames = set().union(*frame_sets)
    if not all_frames:
        return None
    ratios = [len(frames) / len(all_frames) for frames in frame_sets if frames]
    return float(fmean(ratios)) if ratios else None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _cluster_rigid_residual(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    np = _optional_numpy()
    if np is None or len(tracks) < 3:
        return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
    reference_points = []
    for track in tracks:
        reference = track.get("reference_xyz_world")
        if not isinstance(reference, list) or len(reference) != 3:
            return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
        reference_points.append([float(value) for value in reference])
    frame_indices = sorted(set.intersection(*[set(_trajectory_by_frame(track)) for track in tracks]))
    if not frame_indices:
        return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
    per_frame = {}
    rms_values = []
    for frame_index in frame_indices:
        target_points = [_trajectory_by_frame(track)[frame_index] for track in tracks]
        rms = _fit_rigid_rms(reference_points, target_points)
        per_frame[str(frame_index)] = rms
        rms_values.append(rms)
    rigid_rmse = float(fmean(rms_values)) if rms_values else None
    inlier_ratio = (
        sum(1 for value in rms_values if value <= 0.03) / len(rms_values)
        if rms_values
        else None
    )
    return {
        "rigid_rmse_m": rigid_rmse,
        "per_frame_rmse_m": per_frame,
        "inlier_ratio": inlier_ratio,
    }


def _fit_rigid_rms(source_points: list[list[float]], target_points: list[list[float]]) -> float:
    np = _require_numpy()
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape[0] < 3:
        return float("inf")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_centroid - rotation @ source_centroid
    predicted = (rotation @ source.T).T + translation
    return float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))


def _original_part_metadata(artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    candidates = [
        artifact.get("original_part_segmentation"),
        artifact.get("part_segmentation"),
    ]
    out: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for part in candidate.get("parts", []):
            if not isinstance(part, dict):
                continue
            try:
                part_id = int(part.get("part_id", 0))
            except (TypeError, ValueError):
                continue
            out[part_id] = part
        if out:
            break
    return out


def _original_part_id(track: dict[str, Any]) -> int | None:
    value = track.get("original_part_id", track.get("part_id"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cross_edge_type(left: int, right: int, part_meta: dict[int, dict[str, Any]]) -> str:
    left_role = _part_kind(left, part_meta.get(left, {}))
    right_role = _part_kind(right, part_meta.get(right, {}))
    pair = sorted([left_role, right_role])
    if pair == ["base", "door"]:
        return "base-door"
    if pair == ["door", "door"]:
        return "left-right door"
    if pair == ["door", "drawer"]:
        return "drawer-door"
    if pair == ["base", "drawer"]:
        return "drawer-base"
    return "other"


def _part_kind(part_id: int, meta: dict[str, Any]) -> str:
    name = str(meta.get("name", "")).lower()
    role = str(meta.get("role", "")).lower()
    if "door" in name:
        return "door"
    if "drawer" in name or "freezer" in name or "fridge" in name:
        return "drawer"
    if part_id == 1 or role == "base" or "refrigerator" in name:
        return "base"
    return "other"


def _mean_optional(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(fmean(clean)) if clean else None


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _cosine(a: list[float], b: list[float]) -> float:
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _optional_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


def _require_numpy() -> Any:
    np = _optional_numpy()
    if np is None:
        raise RuntimeError("kNN spectral motion segmentation requires NumPy.")
    return np
