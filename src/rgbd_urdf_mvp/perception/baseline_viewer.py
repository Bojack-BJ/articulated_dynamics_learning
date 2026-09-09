from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from ..benchmarks.aim_pointcloud_iou import read_ascii_labeled_ply
from .object_mask_flow_html import _mjcf_replay_payload


_SEGMENTATION_COUNTS = re.compile(
    r"^\s*(?P<static>\d+)\s+(?P<dynamic>\d+)(?:\s+\[[^\n]+\])?\s*$",
    re.MULTILINE,
)


@dataclass(slots=True)
class BaselineViewerConfig:
    output_html: str | Path
    reference_ply: str | Path
    aim_run: str | Path | None = None
    aim_metrics: str | Path | None = None
    hybrid_tracks: str | Path | None = None
    hybrid_metrics: str | Path | None = None
    reart_prediction: str | Path | None = None
    reart_metrics: str | Path | None = None
    reart_sequence_manifest: str | Path | None = None
    reart_status: str | Path | None = None
    gt_frames_dir: str | Path | None = None
    gt_mesh_episode: str | Path | None = None
    gt_mesh_opacity: float = 0.28
    max_points: int = 20_000
    gt_geometry_max_points: int = 4_000
    max_trajectories: int = 100
    timeline_steps: int = 31
    gt_mapping_distance_ratio: float = 0.02
    random_seed: int = 17
    axis_remap: str = "x,y,z"


class BaselineComparisonViewerBuilder:
    """Build a self-contained multi-baseline diagnostic viewer."""

    def build(self, config: BaselineViewerConfig) -> Path:
        reference = Path(config.reference_ply).expanduser().resolve()
        reference_points, reference_labels = read_ascii_labeled_ply(
            reference, label_mode="part_id"
        )
        transform = _AxisRemap.from_raw(config.axis_remap)
        gt_frames = _load_gt_geometry_frames(
            _optional_path(config.gt_frames_dir),
            transform=transform,
            max_points=max(1, int(config.gt_geometry_max_points)),
            timeline_steps=max(2, int(config.timeline_steps)),
            seed=config.random_seed,
        )
        gt_mesh = _load_gt_mesh_replay(
            _optional_path(config.gt_mesh_episode),
            transform=transform,
            timeline_steps=max(2, int(config.timeline_steps)),
            opacity=float(config.gt_mesh_opacity),
        )
        methods: dict[str, dict[str, Any]] = {
            "gt": _gt_adapter(
                reference_points,
                reference_labels,
                transform=transform,
                max_points=config.max_points,
                seed=config.random_seed,
                geometry_frames=gt_frames,
                timeline_steps=max(2, int(config.timeline_steps)),
            )
        }
        if config.hybrid_tracks:
            methods["hybrid"] = _hybrid_adapter(
                Path(config.hybrid_tracks).expanduser().resolve(),
                reference_points,
                reference_labels,
                metrics_path=_optional_path(config.hybrid_metrics),
                transform=transform,
                max_points=config.max_points,
                timeline_steps=max(2, int(config.timeline_steps)),
                seed=config.random_seed,
            )
        if config.aim_run:
            methods["aim"] = _aim_adapter(
                Path(config.aim_run).expanduser().resolve(),
                reference_points,
                reference_labels,
                metrics_path=_optional_path(config.aim_metrics),
                transform=transform,
                max_points=config.max_points,
                timeline_steps=max(2, int(config.timeline_steps)),
                mapping_distance_ratio=config.gt_mapping_distance_ratio,
                seed=config.random_seed,
            )
        if config.reart_prediction:
            methods["reart"] = _reart_adapter(
                Path(config.reart_prediction).expanduser().resolve(),
                reference_points,
                reference_labels,
                metrics_path=_optional_path(config.reart_metrics),
                transform=transform,
                max_points=config.max_points,
                timeline_steps=max(2, int(config.timeline_steps)),
                seed=config.random_seed,
            )
        elif config.reart_sequence_manifest:
            methods["reart"] = _reart_failure_adapter(
                Path(config.reart_sequence_manifest).expanduser().resolve(),
                reference_points,
                reference_labels,
                status_path=_optional_path(config.reart_status),
                transform=transform,
                max_points=config.max_points,
                timeline_steps=max(2, int(config.timeline_steps)),
                seed=config.random_seed,
            )
        bounds = _combined_bounds(methods, gt_mesh=gt_mesh)
        for data in methods.values():
            data["bounds"] = bounds
        payload = {
            "schema": "baseline-comparison-viewer-v1",
            "reference_ply": str(reference),
            "axis_remap": config.axis_remap,
            "default_max_trajectories": max(1, int(config.max_trajectories)),
            "methods": methods,
            "bounds": bounds,
            "gt_mesh": gt_mesh,
        }
        output = Path(config.output_html).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_build_html(payload), encoding="utf-8")
        viewer_data = output.parent / "viewer_data"
        viewer_data.mkdir(exist_ok=True)
        for method, data in methods.items():
            (viewer_data / f"{method}.json").write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        if gt_mesh is not None:
            (viewer_data / "gt_mesh.json").write_text(
                json.dumps(gt_mesh, indent=2) + "\n", encoding="utf-8"
            )
        (viewer_data / "metrics.json").write_text(
            json.dumps(
                {method: data.get("metrics", {}) for method, data in methods.items()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output


class _AxisRemap:
    def __init__(self, axes: list[tuple[int, float]]) -> None:
        self.axes = axes

    @classmethod
    def from_raw(cls, raw: str) -> "_AxisRemap":
        lookup = {"x": 0, "y": 1, "z": 2}
        tokens = [value.strip() for value in raw.split(",") if value.strip()]
        if len(tokens) != 3:
            raise ValueError("axis_remap must contain three axes")
        axes: list[tuple[int, float]] = []
        seen: set[str] = set()
        for token in tokens:
            sign = -1.0 if token.startswith("-") else 1.0
            name = token[1:] if token.startswith("-") else token
            if name not in lookup or name in seen:
                raise ValueError(f"Invalid axis remap token: {token}")
            seen.add(name)
            axes.append((lookup[name], sign))
        return cls(axes)

    def points(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [sign * values[:, index] for index, sign in self.axes]
        )

    def point(self, value: list[float]) -> list[float]:
        return [
            float(sign * value[index]) for index, sign in self.axes
        ]

    def vector(self, value: list[float]) -> list[float]:
        return self.point(value)


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _load_gt_geometry_frames(
    frames_dir: Path | None,
    *,
    transform: _AxisRemap,
    max_points: int,
    timeline_steps: int,
    seed: int,
) -> list[dict[str, Any]] | None:
    if frames_dir is None:
        return None
    all_paths = [
        path
        for path in sorted(frames_dir.glob("frame_*.ply"))
        if _ply_vertex_count(path) > 0
    ]
    dense_paths = [
        path
        for path in all_paths
        if _ply_vertex_count(path) >= min(500, max_points)
    ]
    paths = dense_paths or all_paths
    if not paths:
        raise FileNotFoundError(f"No frame_*.ply files found under {frames_dir}")
    frame_indices = np.linspace(0, len(paths) - 1, timeline_steps).round().astype(int)
    output = []
    for timeline_index, source_index in enumerate(frame_indices):
        points, _, extra = _read_ply(paths[int(source_index)])
        labels_raw = extra.get("part_id", extra.get("original_part_id"))
        labels = (
            np.asarray(labels_raw, dtype=int)
            if labels_raw is not None
            else np.full(len(points), -1, dtype=int)
        )
        selected = _deterministic_indices(
            len(points),
            max_points=max_points,
            seed=seed + timeline_index,
        )
        output.append(
            {
                "timeline_index": timeline_index,
                "source_frame_index": int(
                    re.search(r"(\d+)$", paths[int(source_index)].stem).group(1)
                ),
                "points": transform.points(points[selected]).tolist(),
                "gt_part_id": labels[selected].tolist(),
            }
        )
    return output


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    return 0


def _load_gt_mesh_replay(
    episode_path: Path | None,
    *,
    transform: _AxisRemap,
    timeline_steps: int,
    opacity: float,
) -> dict[str, Any] | None:
    if episode_path is None:
        return None
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    frame_count = len(episode.get("frames", []))
    if frame_count <= 0:
        raise ValueError(f"GT mesh replay episode has no frames: {episode_path}")
    viewer_frames = list(range(timeline_steps))
    source_frames = (
        np.linspace(0, frame_count - 1, timeline_steps).round().astype(int).tolist()
    )
    return _mjcf_replay_payload(
        episode_path,
        frames=viewer_frames,
        sampled_source_frames=source_frames,
        transform=transform,
        opacity=opacity,
    )


def _gt_adapter(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    transform: _AxisRemap,
    max_points: int,
    seed: int,
    geometry_frames: list[dict[str, Any]] | None,
    timeline_steps: int,
) -> dict[str, Any]:
    transformed = transform.points(points)
    overlap = _overlap_payload(labels, labels)
    selected = _deterministic_indices(len(points), max_points=max_points, seed=seed)
    static_trajectory = np.repeat(
        transformed[selected, None, :], timeline_steps, axis=1
    )
    return {
        "source_method": "gt",
        "display_name": "GT",
        "points": transformed[selected].tolist(),
        "gt_part_id": labels[selected].astype(int).tolist(),
        "pred_part_id": labels[selected].astype(int).tolist(),
        "trajectory": static_trajectory.tolist(),
        "trajectory_times": np.linspace(0.0, 1.0, timeline_steps).tolist(),
        "geometry_frames": geometry_frames,
        "gt_mapping_valid": [True] * len(selected),
        "gt_match_distance": [0.0] * len(selected),
        "full_point_count": len(points),
        "visualized_point_count": len(selected),
        "available_fields": ["gt_part", "pred_part", "overlap", "trajectory"],
        "overlap": overlap,
        "metrics": {
            "status": "ground_truth",
            "predicted_part_count": int(len(np.unique(labels))),
            "gt_part_count": int(len(np.unique(labels))),
            "point_iou": 1.0,
            "ari": 1.0,
            "rand_index": 1.0,
        },
        "statistics": _part_statistics(labels, labels),
    }


def _hybrid_adapter(
    path: Path,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    metrics_path: Path | None,
    transform: _AxisRemap,
    max_points: int,
    timeline_steps: int,
    seed: int,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    tracks = [row for row in artifact.get("tracks", []) if isinstance(row, dict)]
    trajectories: list[list[list[float]]] = []
    points: list[list[float]] = []
    predicted: list[int] = []
    gt: list[int] = []
    confidence: list[float | None] = []
    for track in tracks:
        samples = [
            sample
            for sample in track.get("samples", [])
            if sample.get("visible") and sample.get("xyz_world") is not None
        ]
        if not samples:
            continue
        samples.sort(key=lambda row: int(row.get("frame_index", 0)))
        xyz = np.asarray([sample["xyz_world"] for sample in samples], dtype=float)
        trajectories.append(
            _resample_point_trajectory(transform.points(xyz), timeline_steps).tolist()
        )
        points.append(transform.points(xyz[-1:])[0].tolist())
        predicted.append(int(track.get("part_id", -1)))
        gt.append(int(track.get("original_part_id", -1)))
        quality = track.get("track_quality") or {}
        value = quality.get("track_quality_score", track.get("assignment_confidence"))
        confidence.append(float(value) if value is not None else None)
    arrays = {
        "points": np.asarray(points, dtype=float),
        "pred": np.asarray(predicted, dtype=int),
        "gt": np.asarray(gt, dtype=int),
        "trajectory": trajectories,
        "confidence": confidence,
    }
    selected = _deterministic_indices(len(points), max_points=max_points, seed=seed)
    overlap = _overlap_payload(arrays["pred"], arrays["gt"], valid_gt=arrays["gt"] >= 0)
    return {
        "source_method": "hybrid",
        "display_name": "Ours / Hybrid",
        "points": arrays["points"][selected].tolist(),
        "gt_part_id": arrays["gt"][selected].tolist(),
        "pred_part_id": arrays["pred"][selected].tolist(),
        "trajectory": [trajectories[index] for index in selected],
        "trajectory_times": np.linspace(0.0, 1.0, timeline_steps).tolist(),
        "slot_confidence": [confidence[index] for index in selected],
        "gt_mapping_valid": (arrays["gt"][selected] >= 0).tolist(),
        "gt_match_distance": [0.0 if arrays["gt"][index] >= 0 else None for index in selected],
        "full_point_count": len(points),
        "visualized_point_count": len(selected),
        "available_fields": [
            "gt_part",
            "pred_part",
            "overlap",
            "trajectory",
            "slot_confidence",
        ],
        "overlap": overlap,
        "metrics": _metrics_payload(metrics_path),
        "statistics": _part_statistics(arrays["pred"], arrays["gt"]),
    }


def _aim_adapter(
    run_dir: Path,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    metrics_path: Path | None,
    transform: _AxisRemap,
    max_points: int,
    timeline_steps: int,
    mapping_distance_ratio: float,
    seed: int,
) -> dict[str, Any]:
    initial_points, initial_rgb, initial_extra = _read_ply(
        run_dir / "seg_init_dual" / "segmented_point.ply"
    )
    final_path = run_dir / "motion_seg_final" / "segmented_point.ply"
    if final_path.is_file():
        final_points, final_rgb, final_extra = _read_ply(final_path)
    else:
        final_points, final_rgb, final_extra = initial_points, initial_rgb, initial_extra
    trajectories = _aim_trajectories(run_dir)
    count = min(
        len(initial_points),
        len(final_points),
        *(len(values) for values in trajectories),
    )
    initial_points = initial_points[:count]
    final_points = final_points[:count]
    trajectories = [values[:count] for values in trajectories]
    static_count, dynamic_count = _aim_counts(run_dir, count=count)
    dynamic = np.arange(count) >= static_count
    postmerge = _rgb_labels(final_rgb[:count]) if final_rgb is not None else np.full(count, -1)
    premerge = _aim_premerge_components(run_dir, final_points)

    bbox = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
    distance, nearest = cKDTree(reference_points).query(final_points, k=1)
    valid = distance <= max(mapping_distance_ratio * bbox, 1e-9)
    gt = np.full(count, -1, dtype=int)
    gt[valid] = reference_labels[nearest[valid]]
    trajectory_array = np.stack(trajectories, axis=1)
    if trajectory_array.shape[1] != timeline_steps:
        trajectory_array = np.stack(
            [
                _resample_point_trajectory(values, timeline_steps)
                for values in trajectory_array
            ],
            axis=0,
        )
    deformation = np.linalg.norm(trajectory_array[:, -1] - trajectory_array[:, 0], axis=1)
    rigid_residual = _component_rigid_residual(trajectory_array, postmerge)
    joint_axes = _aim_joint_axes(
        run_dir,
        transform=transform,
        component_labels=_motion_to_component_labels(run_dir),
        component_points=final_points,
        predicted=postmerge,
    )

    opacity = _optional_column(initial_extra, "opacity", count)
    scale = _gaussian_scale(initial_extra, count)
    selected = _deterministic_indices(count, max_points=max_points, seed=seed)
    transformed_trajectories = [
        transform.points(trajectory_array[index]).tolist() for index in selected
    ]
    overlap = _overlap_payload(postmerge, gt, valid_gt=valid)
    return {
        "source_method": "aim",
        "display_name": "AiM",
        "points": transform.points(final_points[selected]).tolist(),
        "gt_part_id": gt[selected].tolist(),
        "pred_part_id": postmerge[selected].tolist(),
        "dynamic_flag": dynamic[selected].astype(int).tolist(),
        "premerge_component_id": premerge[selected].tolist(),
        "postmerge_component_id": postmerge[selected].tolist(),
        "deformation_magnitude": deformation[selected].tolist(),
        "rigid_residual": rigid_residual[selected].tolist(),
        "joint_axes": joint_axes,
        "trajectory": transformed_trajectories,
        "opacity": opacity[selected].tolist() if opacity is not None else None,
        "scale": scale[selected].tolist() if scale is not None else None,
        "gt_match_distance": distance[selected].tolist(),
        "gt_mapping_valid": valid[selected].tolist(),
        "full_point_count": count,
        "visualized_point_count": len(selected),
        "available_fields": [
            "gt_part",
            "pred_part",
            "overlap",
            "dynamic_static",
            "premerge_component",
            "postmerge_component",
            "deformation_magnitude",
            "rigid_residual",
            "trajectory",
            *(["joint_axes"] if joint_axes else []),
            *(["opacity"] if opacity is not None else []),
            *(["scale"] if scale is not None else []),
        ],
        "trajectory_times": np.linspace(0.0, 1.0, timeline_steps).tolist(),
        "overlap": overlap,
        "metrics": {
            **_metrics_payload(metrics_path),
            "initial_static_gaussian_count": static_count,
            "initial_dynamic_gaussian_count": dynamic_count,
            "accepted_premerge_component_ids": [
                int(value) for value in np.unique(premerge[premerge >= 0])
            ],
        },
        "statistics": _part_statistics(
            postmerge,
            gt,
            dynamic=dynamic,
            deformation=deformation,
            rigid_residual=rigid_residual,
        ),
    }


def _reart_adapter(
    path: Path,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    metrics_path: Path | None,
    transform: _AxisRemap,
    max_points: int,
    timeline_steps: int,
    seed: int,
) -> dict[str, Any]:
    trajectory_array: np.ndarray | None = None
    joint_axes: list[dict[str, Any]] = []
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        points = np.asarray(payload["cano_pc"], dtype=float)
        predicted = np.asarray(payload["pred_cano_part"], dtype=int)
        trajectory_array = _reart_pose_trajectory(
            points,
            predicted,
            np.asarray(payload.get("pred_pose_list"), dtype=float),
            canonical_index=int(payload.get("cano_idx", 0)),
            frame_count=len(payload.get("complete_pc_list", [])),
            output_frame_count=timeline_steps,
        )
        joint_axes = _reart_joint_axes(
            np.asarray(payload.get("pred_pose_list"), dtype=float),
            payload.get("joint_connection") or [],
            points=points,
            predicted=predicted,
            transform=transform,
        )
    elif path.suffix.lower() == ".npz":
        payload = np.load(path)
        raw_points = np.asarray(payload["points"], dtype=float)
        predicted = np.asarray(
            payload["predicted_part_id"]
            if "predicted_part_id" in payload
            else payload["labels"],
            dtype=int,
        )
        if "trajectory" in payload:
            trajectory_array = _trajectory_point_major(
                np.asarray(payload["trajectory"], dtype=float)
            )
            points = trajectory_array[:, -1]
        elif raw_points.ndim == 3:
            # ReArt point sequences are serialized as [time, point, xyz].
            trajectory_array = np.swapaxes(raw_points, 0, 1)
            points = trajectory_array[:, -1]
        else:
            points = raw_points
    else:
        points, predicted = read_ascii_labeled_ply(path, label_mode="auto")
    predicted = np.asarray(predicted, dtype=int).reshape(-1)
    if len(predicted) != len(points):
        raise ValueError(
            f"ReArt labels ({len(predicted)}) do not match points ({len(points)}): {path}"
        )
    bbox = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
    distance, nearest = cKDTree(reference_points).query(points, k=1)
    valid = distance <= max(0.02 * bbox, 1e-9)
    gt = np.full(len(points), -1, dtype=int)
    gt[valid] = reference_labels[nearest[valid]]
    selected = _deterministic_indices(len(points), max_points=max_points, seed=seed)
    transformed_trajectory = None
    if trajectory_array is not None:
        if trajectory_array.shape[1] != timeline_steps:
            trajectory_array = np.stack(
                [
                    _resample_point_trajectory(values, timeline_steps)
                    for values in trajectory_array
                ],
                axis=0,
            )
        transformed_trajectory = [
            transform.points(trajectory_array[index]).tolist() for index in selected
        ]
    return {
        "source_method": "reart",
        "display_name": "ReArt",
        "points": transform.points(points[selected]).tolist(),
        "gt_part_id": gt[selected].tolist(),
        "pred_part_id": predicted[selected].tolist(),
        "trajectory": transformed_trajectory,
        "joint_axes": joint_axes,
        "trajectory_times": (
            np.linspace(0.0, 1.0, timeline_steps).tolist()
            if transformed_trajectory is not None
            else None
        ),
        "gt_match_distance": distance[selected].tolist(),
        "gt_mapping_valid": valid[selected].tolist(),
        "full_point_count": len(points),
        "visualized_point_count": len(selected),
        "available_fields": [
            "gt_part",
            "pred_part",
            "overlap",
            *(["trajectory"] if transformed_trajectory is not None else []),
            *(["joint_axes"] if joint_axes else []),
        ],
        "overlap": _overlap_payload(predicted, gt, valid_gt=valid),
        "metrics": _metrics_payload(metrics_path),
        "statistics": _part_statistics(predicted, gt),
    }


def _motion_to_component_labels(run_dir: Path) -> dict[int, int]:
    """Map AiM motion indices to the labels written in its final segmented PLY."""
    final_path = run_dir / "motion_seg_final" / "segmented_point.ply"
    if not final_path.is_file():
        return {}
    points, rgb, _ = _read_ply(final_path)
    if rgb is None:
        return {}
    labels = _rgb_labels(rgb)
    tree = cKDTree(points)
    output = {}
    for path in sorted(run_dir.glob("sub_*_point_cloud_end.ply")):
        motion_id = int(path.name.split("_")[1])
        component_points, _, _ = _read_ply(path)
        if not len(component_points):
            continue
        sample = component_points[:: max(1, len(component_points) // 2048)]
        distances, indices = tree.query(sample, k=1)
        cutoff = max(1e-5, float(np.quantile(distances, 0.9)) + 1e-8)
        matched = labels[indices[distances <= cutoff]]
        if len(matched):
            values, counts = np.unique(matched, return_counts=True)
            output[motion_id] = int(values[int(np.argmax(counts))])
    return output


def _aim_joint_axes(
    run_dir: Path,
    *,
    transform: _AxisRemap,
    component_labels: dict[int, int],
    component_points: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    path = run_dir / "motion.json"
    if not path.is_file():
        return []
    motions = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for motion_id, motion in enumerate(motions):
        # AiM reserves index zero for the static identity component.
        if motion_id == 0:
            continue
        axis = np.asarray(motion.get("axis"), dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            continue
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            continue
        axis /= norm
        joint_type = "prismatic" if int(motion.get("motion_type", 0)) == 0 else "revolute"
        child = component_labels.get(motion_id, motion_id)
        center = np.asarray(motion.get("center", [0.0, 0.0, 0.0]), dtype=float)
        if joint_type == "prismatic":
            child_points = component_points[predicted == child]
            if len(child_points):
                center = np.median(child_points, axis=0)
        output.append(
            {
                "joint_id": f"aim_motion_{motion_id}",
                "parent_part_id": None,
                "child_part_id": int(child),
                "joint_type": joint_type,
                "origin": transform.point(center.tolist()),
                "axis": transform.vector(axis.tolist()),
                "source": "native_motion_json",
                "confidence": None,
            }
        )
    return output


def _reart_joint_axes(
    poses: np.ndarray,
    connections: list[Any],
    *,
    points: np.ndarray,
    predicted: np.ndarray,
    transform: _AxisRemap,
) -> list[dict[str, Any]]:
    """Convert ReArt's native part-pose graph into diagnostic joint axes.

    ReArt does not emit joint type/axis in its released non-GT result. These
    axes are analytic fits to its predicted relative SE(3) trajectories and
    are therefore explicitly marked as converted, not native predictions.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        return []
    output = []
    for edge_index, edge in enumerate(connections):
        if not isinstance(edge, (list, tuple, np.ndarray)) or len(edge) < 2:
            continue
        parent, child = int(edge[0]), int(edge[1])
        if parent >= poses.shape[1] or child >= poses.shape[1]:
            continue
        relative = np.stack(
            [np.linalg.inv(frame[parent]) @ frame[child] for frame in poses],
            axis=0,
        )
        reference_inv = np.linalg.inv(relative[0])
        motion = np.stack([value @ reference_inv for value in relative], axis=0)
        rotation_vectors = Rotation.from_matrix(motion[:, :3, :3]).as_rotvec()
        rotation_magnitudes = np.linalg.norm(rotation_vectors, axis=1)
        translation = motion[:, :3, 3] - motion[0, :3, 3]
        translation_extent = float(
            np.max(np.linalg.norm(translation, axis=1))
        )
        rotating = rotation_magnitudes > np.deg2rad(1.0)
        child_points = points[predicted == child]
        child_center = (
            np.median(child_points, axis=0)
            if len(child_points)
            else np.median(points, axis=0)
        )
        if int(rotating.sum()) >= 2 and float(rotation_magnitudes.max()) >= np.deg2rad(3.0):
            axes = rotation_vectors[rotating] / rotation_magnitudes[rotating, None]
            reference = axes[int(np.argmax(rotation_magnitudes[rotating]))]
            axes *= np.where(axes @ reference >= 0.0, 1.0, -1.0)[:, None]
            weights = rotation_magnitudes[rotating]
            axis = np.sum(axes * weights[:, None], axis=0)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            matrices = []
            targets = []
            for value in motion[rotating]:
                matrices.append(np.eye(3) - value[:3, :3])
                targets.append(
                    value[:3, 3] - axis * float(axis @ value[:3, 3])
                )
            matrices.append(axis[None, :])
            targets.append(np.zeros(1, dtype=float))
            matrix = np.concatenate(matrices, axis=0)
            target = np.concatenate(targets, axis=0)
            origin, *_ = np.linalg.lstsq(matrix, target, rcond=None)
            residual = float(np.sqrt(np.mean((matrix @ origin - target) ** 2)))
            joint_type = "revolute"
            total_motion = float(rotation_magnitudes.max())
        elif translation_extent > 1e-6:
            centered = translation - translation.mean(axis=0)
            _, singular, vt_mat = np.linalg.svd(centered, full_matrices=False)
            axis = vt_mat[0]
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            projection = centered @ axis
            orthogonal = centered - projection[:, None] * axis
            residual = float(np.sqrt(np.mean(np.sum(orthogonal * orthogonal, axis=1))))
            origin = child_center
            joint_type = "prismatic"
            total_motion = translation_extent
        else:
            continue
        confidence = float(
            np.clip(total_motion / max(total_motion + residual, 1e-12), 0.0, 1.0)
        )
        output.append(
            {
                "joint_id": f"reart_edge_{edge_index}",
                "parent_part_id": parent,
                "child_part_id": child,
                "joint_type": joint_type,
                "origin": transform.point(origin.tolist()),
                "axis": transform.vector(axis.tolist()),
                "source": "converted_from_native_part_poses",
                "confidence": confidence,
                "fit_residual": residual,
                "total_motion": total_motion,
            }
        )
    return output


def _reart_failure_adapter(
    manifest_path: Path,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    status_path: Path | None,
    transform: _AxisRemap,
    max_points: int,
    timeline_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Expose failed ReArt inputs without inventing segmentation predictions."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [
        Path(frame["output_path"]).expanduser().resolve()
        for frame in manifest.get("frames", [])
    ]
    if len(frame_paths) < 2:
        raise ValueError(f"ReArt failure viewer needs at least two input frames: {manifest_path}")
    source_frames = []
    for frame_index, frame_path in enumerate(frame_paths):
        points, _, _ = _read_ply(frame_path)
        selected = _deterministic_indices(
            len(points), max_points=max_points, seed=seed + frame_index
        )
        source_frames.append(transform.points(points[selected]))
    target_indices = (
        np.linspace(0, len(source_frames) - 1, timeline_steps).round().astype(int)
    )
    geometry_frames = [
        {
            "timeline_index": timeline_index,
            "source_frame_index": int(
                manifest.get("source_frame_indices", list(range(len(source_frames))))[
                    source_index
                ]
            ),
            "points": source_frames[source_index].tolist(),
            "gt_part_id": [-1] * len(source_frames[source_index]),
        }
        for timeline_index, source_index in enumerate(target_indices)
    ]
    canonical = source_frames[0]
    static_trajectory = np.repeat(canonical[:, None, :], timeline_steps, axis=1)
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path is not None and status_path.is_file()
        else {"status": "failed", "failure_stage": "unknown"}
    )
    metrics = {
        "status": status.get("status", "failed"),
        "failure_stage": status.get("failure_stage"),
        "exception": status.get("exception"),
        "predicted_part_count": None,
        "gt_part_count": int(len(np.unique(reference_labels))),
    }
    return {
        "source_method": "reart",
        "display_name": "ReArt (failed; inputs only)",
        "points": canonical.tolist(),
        "gt_part_id": [-1] * len(canonical),
        "pred_part_id": [-1] * len(canonical),
        "trajectory": static_trajectory.tolist(),
        "trajectory_times": np.linspace(0.0, 1.0, timeline_steps).tolist(),
        "geometry_frames": geometry_frames,
        "gt_mapping_valid": [False] * len(canonical),
        "gt_match_distance": [None] * len(canonical),
        "full_point_count": len(canonical),
        "visualized_point_count": len(canonical),
        "available_fields": ["trajectory"],
        "overlap": _overlap_payload(
            np.full(len(canonical), -1, dtype=int),
            np.full(len(canonical), -1, dtype=int),
            valid_gt=np.zeros(len(canonical), dtype=bool),
        ),
        "metrics": metrics,
        "statistics": {"pred_parts": [], "gt_parts": []},
        "diagnostic_note": (
            "Official ReArt failed before producing segmentation. The viewer shows "
            "the four native input point clouds resampled onto the common timeline."
        ),
    }


def _reart_pose_trajectory(
    canonical_points: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_poses: np.ndarray,
    *,
    canonical_index: int,
    frame_count: int,
    output_frame_count: int | None = None,
) -> np.ndarray | None:
    """Replay ReArt's per-part poses as point-major trajectories."""
    if (
        predicted_poses.ndim != 4
        or predicted_poses.shape[-2:] != (4, 4)
        or frame_count <= 0
    ):
        return None
    noncanonical = [index for index in range(frame_count) if index != canonical_index]
    if len(noncanonical) != predicted_poses.shape[0]:
        return None
    trajectory = np.repeat(
        canonical_points[:, None, :], frame_count, axis=1
    )
    for pose_index, frame_index in enumerate(noncanonical):
        frame_poses = predicted_poses[pose_index]
        for part_id in np.unique(predicted_labels):
            if part_id < 0 or part_id >= len(frame_poses):
                continue
            selected = np.where(predicted_labels == part_id)[0]
            rotation = frame_poses[part_id, :3, :3]
            translation = frame_poses[part_id, :3, 3]
            trajectory[selected, frame_index] = (
                canonical_points[selected] @ rotation.T + translation
            )
    if output_frame_count is None or output_frame_count == frame_count:
        return trajectory
    return _interpolate_part_pose_trajectory(
        canonical_points,
        predicted_labels,
        trajectory,
        output_frame_count=output_frame_count,
    )


def _interpolate_part_pose_trajectory(
    canonical_points: np.ndarray,
    predicted_labels: np.ndarray,
    sparse_trajectory: np.ndarray,
    *,
    output_frame_count: int,
) -> np.ndarray:
    source_times = np.linspace(0.0, 1.0, sparse_trajectory.shape[1])
    target_times = np.linspace(0.0, 1.0, output_frame_count)
    output = np.repeat(
        canonical_points[:, None, :], output_frame_count, axis=1
    )
    for part_id in np.unique(predicted_labels):
        selected = np.where(predicted_labels == part_id)[0]
        if len(selected) < 3:
            for point_index in selected:
                output[point_index] = _resample_point_trajectory(
                    sparse_trajectory[point_index], output_frame_count
                )
            continue
        rotations = []
        translations = []
        for frame_index in range(sparse_trajectory.shape[1]):
            rotation, translation = _fit_rigid(
                canonical_points[selected],
                sparse_trajectory[selected, frame_index],
            )
            rotations.append(rotation)
            translations.append(translation)
        interpolated_rotations = Slerp(
            source_times, Rotation.from_matrix(np.asarray(rotations))
        )(target_times).as_matrix()
        interpolated_translations = np.column_stack(
            [
                np.interp(target_times, source_times, np.asarray(translations)[:, axis])
                for axis in range(3)
            ]
        )
        for frame_index in range(output_frame_count):
            output[selected, frame_index] = (
                canonical_points[selected] @ interpolated_rotations[frame_index].T
                + interpolated_translations[frame_index]
            )
    return output


def _resample_point_trajectory(
    trajectory: np.ndarray, output_frame_count: int
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) == output_frame_count:
        return trajectory
    source = np.linspace(0.0, 1.0, len(trajectory))
    target = np.linspace(0.0, 1.0, output_frame_count)
    return np.column_stack(
        [np.interp(target, source, trajectory[:, axis]) for axis in range(3)]
    )


def _trajectory_point_major(values: np.ndarray) -> np.ndarray:
    """Normalize a trajectory to [point, time, xyz]."""
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectory shaped [N,T,3] or [T,N,3], got {values.shape}"
        )
    # ReArt native inputs have very few timesteps and many points.
    return np.swapaxes(values, 0, 1) if values.shape[0] < values.shape[1] else values


def _aim_trajectories(run_dir: Path) -> list[np.ndarray]:
    dense_path = run_dir / "dense_trajectory.npz"
    if dense_path.is_file():
        payload = np.load(dense_path)
        trajectory = np.asarray(payload["trajectory"], dtype=float)
        times = np.asarray(payload["times"]).reshape(-1) if "times" in payload else None
        if times is not None and trajectory.shape[0] == len(times):
            trajectory = np.swapaxes(trajectory, 0, 1)
        elif times is None or trajectory.shape[1] != len(times):
            trajectory = _trajectory_point_major(trajectory)
        return [
            trajectory[:, time_index]
            for time_index in range(trajectory.shape[1])
        ]
    values = []
    for time in (0.0, 0.5, 1.0):
        path = run_dir / f"motion_traj_t={time:.1f}" / "point_cloud_seq_0.ply"
        if path.is_file():
            values.append(_read_ply(path)[0])
    if not values:
        values.append(_read_ply(run_dir / "seg_init_dual" / "segmented_point.ply")[0])
    return values


def _aim_trajectory_times(run_dir: Path, count: int) -> list[float]:
    dense_path = run_dir / "dense_trajectory.npz"
    if dense_path.is_file():
        payload = np.load(dense_path)
        if "times" in payload:
            values = np.asarray(payload["times"], dtype=float).reshape(-1)
            if len(values) == count:
                return values.tolist()
    return np.linspace(0.0, 1.0, count).tolist()


def _aim_counts(run_dir: Path, *, count: int) -> tuple[int, int]:
    log = run_dir / "segmentation.log"
    if log.is_file():
        matches = list(
            _SEGMENTATION_COUNTS.finditer(
                log.read_text(encoding="utf-8", errors="replace")
            )
        )
        if matches:
            static = int(matches[-1].group("static"))
            dynamic = int(matches[-1].group("dynamic"))
            if static + dynamic == count:
                return static, dynamic
    rgb = _read_ply(run_dir / "seg_init_dual" / "segmented_point.ply")[1]
    if rgb is None:
        return count, 0
    labels = _rgb_labels(rgb[:count])
    counts = np.bincount(labels)
    static_label = int(np.argmax(counts))
    static = int(np.sum(labels == static_label))
    return static, count - static


def _aim_premerge_components(run_dir: Path, points: np.ndarray) -> np.ndarray:
    labels = np.full(len(points), -1, dtype=int)
    tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 1e-4, 1e-6)
    for path in sorted(run_dir.glob("sub_*_point_cloud_end.ply")):
        match = re.search(r"sub_(\d+)_", path.name)
        if not match:
            continue
        component = int(match.group(1))
        subset = _read_ply(path)[0]
        distance, index = cKDTree(points).query(subset, k=1)
        labels[index[distance <= tolerance]] = component
    return labels


def _component_rigid_residual(
    trajectory: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    residual = np.full(len(trajectory), np.nan, dtype=float)
    for label in np.unique(labels):
        selected = np.where(labels == label)[0]
        if len(selected) < 3:
            continue
        source = trajectory[selected, 0]
        values = np.zeros((len(selected), trajectory.shape[1]), dtype=float)
        for time_index in range(trajectory.shape[1]):
            rotation, translation = _fit_rigid(source, trajectory[selected, time_index])
            prediction = source @ rotation.T + translation
            values[:, time_index] = np.linalg.norm(
                prediction - trajectory[selected, time_index], axis=1
            )
        residual[selected] = np.mean(values, axis=1)
    return residual


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return rotation, target_center - rotation @ source_center


def _overlap_payload(
    predicted: np.ndarray,
    gt: np.ndarray,
    *,
    valid_gt: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=int)
    gt = np.asarray(gt, dtype=int)
    valid = gt >= 0 if valid_gt is None else np.asarray(valid_gt, dtype=bool)
    pred_ids = sorted(int(value) for value in np.unique(predicted))
    gt_ids = sorted(int(value) for value in np.unique(gt[valid]))
    matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=int)
    for row, pred_id in enumerate(pred_ids):
        for column, gt_id in enumerate(gt_ids):
            matrix[row, column] = int(
                np.sum(valid & (predicted == pred_id) & (gt == gt_id))
            )
    pred_sizes = np.asarray([np.sum(predicted == value) for value in pred_ids])
    gt_sizes = np.asarray([np.sum(valid & (gt == value)) for value in gt_ids])
    purity = matrix / np.maximum(pred_sizes[:, None], 1)
    recall = matrix / np.maximum(gt_sizes[None, :], 1)
    union = pred_sizes[:, None] + gt_sizes[None, :] - matrix
    iou = matrix / np.maximum(union, 1)
    return {
        "pred_ids": pred_ids,
        "gt_ids": gt_ids,
        "count": matrix.tolist(),
        "pred_purity": purity.tolist(),
        "gt_recall": recall.tolist(),
        "iou": iou.tolist(),
        "unmatched_count": int(np.sum(~valid)),
    }


def _part_statistics(
    predicted: np.ndarray,
    gt: np.ndarray,
    *,
    dynamic: np.ndarray | None = None,
    deformation: np.ndarray | None = None,
    rigid_residual: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    predicted = np.asarray(predicted, dtype=int)
    gt = np.asarray(gt, dtype=int)
    gt_rows = []
    for gt_id in sorted(int(value) for value in np.unique(gt[gt >= 0])):
        selected = gt == gt_id
        components, counts = np.unique(predicted[selected], return_counts=True)
        order = np.argsort(counts)[::-1]
        gt_rows.append(
            {
                "gt_part_id": gt_id,
                "point_count": int(np.sum(selected)),
                "static_fraction": (
                    float(np.mean(~dynamic[selected])) if dynamic is not None else None
                ),
                "median_motion": (
                    float(np.nanmedian(deformation[selected]))
                    if deformation is not None
                    else None
                ),
                "rigid_rmse": (
                    float(np.sqrt(np.nanmean(rigid_residual[selected] ** 2)))
                    if rigid_residual is not None
                    and np.any(np.isfinite(rigid_residual[selected]))
                    else None
                ),
                "components": [
                    {
                        "pred_part_id": int(components[index]),
                        "count": int(counts[index]),
                        "fraction": float(counts[index] / max(np.sum(selected), 1)),
                    }
                    for index in order
                ],
            }
        )
    pred_rows = []
    for pred_id in sorted(int(value) for value in np.unique(predicted)):
        selected = predicted == pred_id
        valid = selected & (gt >= 0)
        parts, counts = np.unique(gt[valid], return_counts=True)
        dominant = int(parts[np.argmax(counts)]) if len(parts) else None
        pred_rows.append(
            {
                "pred_part_id": pred_id,
                "point_count": int(np.sum(selected)),
                "dominant_gt_part": dominant,
                "purity": float(np.max(counts) / max(np.sum(valid), 1))
                if len(counts)
                else None,
                "gt_parts": [
                    {
                        "gt_part_id": int(part),
                        "count": int(count),
                        "fraction": float(count / max(np.sum(valid), 1)),
                    }
                    for part, count in zip(parts, counts, strict=True)
                ],
            }
        )
    return {"gt_parts": gt_rows, "pred_parts": pred_rows}


def _metrics_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "primary" in payload:
        primary = payload["primary"].get("covered_only_metrics") or {}
        threshold = next(
            (
                row
                for row in payload.get("thresholds", [])
                if abs(float(row.get("distance_ratio_bbox", -1)) - 0.02) < 1e-9
            ),
            {},
        )
        return {
            "status": "success",
            "predicted_part_count": payload.get("predicted_part_count"),
            "gt_part_count": payload.get("gt_part_count"),
            "point_iou": primary.get("one_to_one_mean_iou"),
            "ari": primary.get("adjusted_rand_index"),
            "rand_index": primary.get("rand_index"),
            "geometry_coverage": threshold.get("geometry_coverage"),
            "largest_cluster_ratio": primary.get("largest_cluster_ratio"),
            "unmatched_gt_part_count": primary.get("unmatched_gt_part_count"),
            "undersegmented_gt_part_count": primary.get(
                "undersegmented_gt_part_count"
            ),
        }
    return payload


def _deterministic_indices(count: int, *, max_points: int, seed: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False))


def _rgb_labels(rgb: np.ndarray | None) -> np.ndarray:
    if rgb is None:
        raise ValueError("RGB labels are required")
    colors = [tuple(int(value) for value in row) for row in rgb]
    palette = {color: index for index, color in enumerate(sorted(set(colors)))}
    return np.asarray([palette[color] for color in colors], dtype=int)


def _read_ply(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    try:
        from plyfile import PlyData

        vertex = PlyData.read(str(path))["vertex"].data
        names = list(vertex.dtype.names or ())
        points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
            float
        )
        rgb = (
            np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(
                np.uint8
            )
            if {"red", "green", "blue"}.issubset(names)
            else None
        )
        extra = {
            name: np.asarray(vertex[name])
            for name in names
            if name not in {"x", "y", "z", "red", "green", "blue"}
        }
        return points, rgb, extra
    except ImportError:
        return _read_ascii_ply_columns(path)


def _read_ascii_ply_columns(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Read all scalar vertex properties without requiring plyfile."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    marker_index = raw.find(marker)
    if marker_index < 0:
        marker = b"end_header\r\n"
        marker_index = raw.find(marker)
    if marker_index < 0:
        raise ValueError(f"PLY has no end_header: {path}")
    header_end = marker_index + len(marker)
    lines = raw[:header_end].decode("ascii").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Expected a PLY file: {path}")
    properties: list[str] = []
    vertex_count = 0
    data_format = "ascii"
    in_vertices = False
    property_types: list[str] = []
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "format":
            data_format = fields[1]
        if fields[0] == "element":
            in_vertices = fields[1] == "vertex"
            if in_vertices:
                vertex_count = int(fields[2])
        elif fields[0] == "property" and in_vertices:
            if fields[1] == "list":
                raise ValueError(f"List-valued vertex properties are unsupported: {path}")
            property_types.append(fields[1])
            properties.append(fields[-1])
    if not {"x", "y", "z"}.issubset(properties):
        raise ValueError(f"PLY is missing a valid vertex header: {path}")
    if data_format == "ascii":
        rows = [
            line.split()
            for line in raw[header_end:].decode("utf-8").splitlines()[:vertex_count]
            if line.strip()
        ]
        if len(rows) != vertex_count or any(
            len(row) < len(properties) for row in rows
        ):
            raise ValueError(f"PLY vertex data is incomplete: {path}")
        values = np.asarray(
            [[float(value) for value in row[: len(properties)]] for row in rows],
            dtype=float,
        )
        columns = {name: values[:, index] for index, name in enumerate(properties)}
    else:
        endian = "<" if data_format == "binary_little_endian" else ">"
        type_lookup = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "i2",
            "int16": "i2",
            "ushort": "u2",
            "uint16": "u2",
            "int": "i4",
            "int32": "i4",
            "uint": "u4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }
        if data_format not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format {data_format}: {path}")
        try:
            dtype = np.dtype(
                [
                    (name, endian + type_lookup[property_type])
                    for name, property_type in zip(
                        properties, property_types, strict=True
                    )
                ]
            )
        except KeyError as error:
            raise ValueError(f"Unsupported PLY property type {error.args[0]}") from error
        vertices = np.frombuffer(
            raw,
            dtype=dtype,
            count=vertex_count,
            offset=header_end,
        )
        if len(vertices) != vertex_count:
            raise ValueError(f"PLY vertex data is incomplete: {path}")
        columns = {name: np.asarray(vertices[name]) for name in properties}
    points = np.column_stack([columns[name] for name in ("x", "y", "z")])
    rgb = None
    if {"red", "green", "blue"}.issubset(columns):
        rgb = np.column_stack(
            [columns[name] for name in ("red", "green", "blue")]
        ).astype(np.uint8)
    excluded = {"x", "y", "z", "red", "green", "blue"}
    return points, rgb, {
        name: value for name, value in columns.items() if name not in excluded
    }


def _optional_column(
    extra: dict[str, np.ndarray],
    name: str,
    count: int,
) -> np.ndarray | None:
    value = extra.get(name)
    return np.asarray(value[:count], dtype=float) if value is not None else None


def _gaussian_scale(
    extra: dict[str, np.ndarray],
    count: int,
) -> np.ndarray | None:
    names = [name for name in ("scale_0", "scale_1", "scale_2") if name in extra]
    if not names:
        return None
    values = np.column_stack([extra[name][:count] for name in names]).astype(float)
    return np.exp(values)


def _combined_bounds(
    methods: dict[str, dict[str, Any]],
    *,
    gt_mesh: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    arrays = []
    for method in methods.values():
        if method.get("points"):
            arrays.append(np.asarray(method["points"], dtype=float))
        if method.get("trajectory"):
            arrays.append(
                np.asarray(method["trajectory"], dtype=float).reshape(-1, 3)
            )
        for frame in method.get("geometry_frames") or []:
            if frame.get("points"):
                arrays.append(np.asarray(frame["points"], dtype=float))
    if gt_mesh:
        geometries = {
            int(geometry["payload_id"]): np.asarray(geometry["vertices"], dtype=float)
            for geometry in gt_mesh.get("geometries", [])
        }
        for transforms in gt_mesh.get("frames", {}).values():
            for item in transforms:
                vertices = geometries.get(int(item["payload_id"]))
                if vertices is None:
                    continue
                rotation = np.asarray(item["rotation"], dtype=float)
                position = np.asarray(item["position"], dtype=float)
                arrays.append(vertices @ rotation.T + position)
    points = np.concatenate(arrays, axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 1e-6) * 1.08
    half = 0.5 * span
    return {
        "min": (center - half).tolist(),
        "max": (center + half).tolist(),
        "center": center.tolist(),
        "span": span,
    }


def _build_html(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"))
    try:
        from plotly.offline.offline import get_plotlyjs

        plotly_script = f"<script>{get_plotlyjs()}</script>"
    except ImportError:
        plotly_script = (
            '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        )
    return (
        _HTML_TEMPLATE.replace("__PAYLOAD__", encoded)
        .replace("__PLOTLY_SCRIPT__", plotly_script)
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Baseline Motion Decomposition Viewer</title>
__PLOTLY_SCRIPT__
<style>
:root { --bg:#0a1118; --panel:#111d28; --line:#263847; --text:#e7f0f6; --muted:#91a6b5; --accent:#6ee7c2; }
* { box-sizing:border-box; }
body { margin:0; color:var(--text); background:var(--bg); font:14px/1.4 "IBM Plex Sans","Avenir Next",sans-serif; }
#app { display:grid; grid-template-columns:330px 1fr; min-height:100vh; }
aside { padding:18px; background:#0e1923; border-right:1px solid var(--line); overflow:auto; max-height:100vh; }
h1 { font-size:20px; margin:0 0 16px; } h2 { font-size:14px; margin:18px 0 8px; color:var(--accent); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:12px; }
label { display:block; color:var(--muted); margin:8px 0 4px; }
select,input[type=range] { width:100%; } select { background:#0b141d; color:var(--text); border:1px solid var(--line); padding:7px; border-radius:7px; }
.check { display:flex; gap:8px; align-items:center; margin:7px 0; } .check input { width:auto; }
#plot { width:100%; height:100vh; } #main { position:relative; min-width:0; }
.metric { display:grid; grid-template-columns:1fr auto; gap:5px; } .metric span:nth-child(odd){color:var(--muted);}
table { border-collapse:collapse; width:100%; font-size:11px; } th,td { border:1px solid var(--line); padding:4px; text-align:center; }
td.cell { cursor:pointer; } td.cell:hover { outline:2px solid var(--accent); }
.palette { display:flex; flex-wrap:wrap; gap:6px; } .pill { padding:3px 7px; border-radius:999px; color:#061015; font-weight:700; }
#selection { white-space:pre-wrap; font-family:ui-monospace,SFMono-Regular,monospace; font-size:11px; }
.warn { color:#ffb86b; }
</style>
</head>
<body><div id="app"><aside>
<h1>Baseline Motion Decomposition</h1>
<div class="card">
<label>Method</label><select id="method"></select>
<label>Pipeline stage</label><select id="stage"></select>
<label>Color by</label><select id="color"></select>
<label>Overlap matrix value</label><select id="matrixMode"><option value="count">Count</option><option value="pred_purity">Pred purity</option><option value="gt_recall">GT recall</option><option value="iou">IoU</option></select>
</div>
<div class="card">
<label>Time <span id="timeValue"></span></label><input id="time" type="range" min="0" max="0" value="0" />
<label>Max trajectories <span id="trajectoryValue"></span></label><input id="trajectoryCount" type="range" min="0" max="200" step="10" value="100" />
<label>Point size <span id="pointSizeValue"></span></label><input id="pointSize" type="range" min="1" max="12" step="0.5" value="5" />
<label>Point opacity <span id="pointOpacityValue"></span></label><input id="pointOpacity" type="range" min="0.1" max="1" step="0.05" value="0.9" />
<label>Color palette</label><select id="palette"><option value="vivid">Vivid</option><option value="colorblind">Colorblind</option><option value="pastel">Pastel</option></select>
<div class="check"><input id="showCurrent" type="checkbox" checked /><span>Show current positions</span></div>
<div class="check"><input id="showTrajectories" type="checkbox" /><span>Show full trajectories</span></div>
<div class="check"><input id="showJointAxes" type="checkbox" checked /><span>Show joint axes</span></div>
<div class="check"><input id="showGTMesh" type="checkbox" checked /><span>Show GT mesh replay</span></div>
<div class="check"><input id="showGTGeometry" type="checkbox" /><span>Show GT point-cloud overlap</span></div>
<div class="check"><input id="mappedOnly" type="checkbox" /><span>GT-mapped only</span></div>
<div class="check" id="smallPartsRow"><input id="smallParts" type="checkbox" /><span>Small-part diagnostic (GT 6/7/8)</span></div>
</div>
<div class="card"><h2>Metrics</h2><div id="metrics" class="metric"></div><div id="sampleInfo"></div></div>
<div class="card"><h2>Overlap matrix</h2><div id="matrix"></div></div>
<div class="card"><h2>Selection</h2><div id="selection">Click a point or overlap cell.</div></div>
</aside><main id="main"><div id="plot"></div></main></div>
<script>
const payload=__PAYLOAD__;
const methods=payload.methods;
const gtMesh=payload.gt_mesh;
const methodSelect=document.getElementById("method"), stageSelect=document.getElementById("stage"), colorSelect=document.getElementById("color");
const timeSlider=document.getElementById("time"), trajectorySlider=document.getElementById("trajectoryCount");
const PALETTES={
 vivid:["#00e5ff","#ff1744","#76ff03","#ffea00","#d500f9","#ff9100","#00e676","#ff4081","#2979ff","#c6ff00","#f50057","#651fff"],
 colorblind:["#009e73","#d55e00","#0072b2","#cc79a7","#f0e442","#56b4e9","#e69f00","#000000"],
 pastel:["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"]
};
const preferredMethod=["hybrid","aim","reart","gt"].find(key=>methods[key]);
let currentMethod=preferredMethod, camera=null, pairFilter=null, handlersBound=false;
for(const [key,value] of Object.entries(methods)){const o=document.createElement("option");o.value=key;o.textContent=value.display_name||key;methodSelect.appendChild(o);}
methodSelect.value=currentMethod;
function has(data,key){return (data.available_fields||[]).includes(key);}
function optionsFor(data){
 const values=[["gt_part","GT part"],["pred_part","Predicted part"],["overlap","GT ↔ prediction overlap"]];
 if(has(data,"dynamic_static"))values.push(["dynamic_static","AiM dynamic/static"]);
 if(has(data,"premerge_component"))values.push(["premerge_component","AiM pre-merge RANSAC"]);
 if(has(data,"postmerge_component"))values.push(["postmerge_component","AiM post-merge RANSAC"]);
 if(has(data,"deformation_magnitude"))values.push(["deformation_magnitude","Deformation magnitude"]);
 if(has(data,"rigid_residual"))values.push(["rigid_residual","Rigid residual"]);
 if(has(data,"opacity"))values.push(["opacity","Gaussian opacity"]);
 if(has(data,"scale"))values.push(["scale","Gaussian scale"]);
 if(has(data,"slot_confidence"))values.push(["slot_confidence","Slot confidence"]);
 return values;
}
function stageOptions(data){
 const values=[["raw","Raw points"]];
 if(has(data,"dynamic_static"))values.push(["dynamic_static","Dynamic/static"]);
 if(has(data,"premerge_component"))values.push(["premerge","Pre-merge RANSAC"]);
 if(has(data,"postmerge_component"))values.push(["postmerge","Post-merge RANSAC"]);
 return values;
}
function fillSelect(select,values){select.innerHTML="";for(const [v,t] of values){const o=document.createElement("option");o.value=v;o.textContent=t;select.appendChild(o);}}
function activePalette(){return PALETTES[document.getElementById("palette").value]||PALETTES.vivid;}
function paletteColor(value,offset=0){const palette=activePalette();if(value===null||value===undefined||value<0)return "#64748b";return palette[(Math.abs(Number(value))+offset)%palette.length];}
function numericColors(values){return values.map(v=>Number.isFinite(v)?v:null);}
function pointColors(data,indices){
 const mode=colorSelect.value;
 if(mode==="gt_part")return indices.map(i=>paletteColor(data.gt_part_id[i]));
 if(mode==="pred_part"||mode==="postmerge_component")return indices.map(i=>paletteColor(data.pred_part_id[i],3));
 if(mode==="premerge_component")return indices.map(i=>paletteColor(data.premerge_component_id[i],3));
 if(mode==="dynamic_static")return indices.map(i=>data.dynamic_flag[i]?"#ff6b6b":"#7dd3fc");
 if(mode==="overlap")return indices.map(i=>data.gt_part_id[i]<0?"#64748b":paletteColor(data.gt_part_id[i]*31+data.pred_part_id[i]*7));
 const field={deformation_magnitude:"deformation_magnitude",rigid_residual:"rigid_residual",opacity:"opacity",scale:"scale",slot_confidence:"slot_confidence"}[mode];
 return numericColors(indices.map(i=>{const value=(data[field]||[])[i];return Array.isArray(value)?Math.hypot(...value):value;}));
}
function pointSizes(data,indices){
 const size=Number(document.getElementById("pointSize").value);
 return indices.map(()=>size);
}
function visibleIndices(data){
 let out=data.points.map((_,i)=>i);
 const stage=stageSelect.value;
 if(stage==="dynamic_static"&&data.dynamic_flag)out=out.filter(i=>data.dynamic_flag[i]===1);
 if(stage==="premerge"&&data.premerge_component_id)out=out.filter(i=>data.premerge_component_id[i]>=0);
 if(document.getElementById("mappedOnly").checked)out=out.filter(i=>data.gt_mapping_valid[i]);
 if(document.getElementById("smallParts").checked)out=out.filter(i=>[6,7,8].includes(data.gt_part_id[i]));
 if(pairFilter)out=out.filter(i=>data.pred_part_id[i]===pairFilter.pred&&data.gt_part_id[i]===pairFilter.gt);
 return out;
}
function currentPoints(data,indices){
 if(!data.trajectory)return indices.map(i=>data.points[i]);
 const t=Number(timeSlider.value);
 return indices.map(i=>data.trajectory[i][Math.min(t,data.trajectory[i].length-1)]);
}
function gtGeometryFrame(){
 const gt=methods.gt,frames=gt?.geometry_frames;
 if(!frames||!frames.length)return null;
 return frames[Math.min(Number(timeSlider.value),frames.length-1)];
}
function activeGeometryFrame(data){
 const frames=data?.geometry_frames;
 if(!frames||!frames.length)return null;
 return frames[Math.min(Number(timeSlider.value),frames.length-1)];
}
function meshVertices(vertices,transform){
 const r=transform.rotation,p=transform.position,x=[],y=[],z=[];
 for(const v of vertices){
  x.push(p[0]+r[0][0]*v[0]+r[0][1]*v[1]+r[0][2]*v[2]);
  y.push(p[1]+r[1][0]*v[0]+r[1][1]*v[1]+r[1][2]*v[2]);
  z.push(p[2]+r[2][0]*v[0]+r[2][1]*v[1]+r[2][2]*v[2]);
 }
 return {x,y,z};
}
function gtMeshTraces(){
 if(!gtMesh||!document.getElementById("showGTMesh").checked)return [];
 const transforms=gtMesh.frames[String(Number(timeSlider.value))]||[];
 const byId=new Map(transforms.map(item=>[Number(item.payload_id),item]));
 return (gtMesh.geometries||[]).flatMap(geometry=>{
  const transform=byId.get(Number(geometry.payload_id));
  if(!transform)return [];
  const vertices=meshVertices(geometry.vertices,transform);
  return [{type:"mesh3d",name:`GT mesh: ${geometry.name}`,x:vertices.x,y:vertices.y,z:vertices.z,
   i:geometry.faces.map(face=>face[0]),j:geometry.faces.map(face=>face[1]),k:geometry.faces.map(face=>face[2]),
   color:geometry.color,opacity:Number(gtMesh.opacity||0.28),flatshading:true,
   hovertemplate:`${geometry.name}<br>simulation-only GT mesh<extra></extra>`}];
 });
}
function jointAxisTraces(data){
 if(!document.getElementById("showJointAxes").checked||!data.joint_axes?.length)return [];
 const length=Math.max(Number(payload.bounds.span||1)*0.62,1e-4);
 return data.joint_axes.map(joint=>{
  const origin=joint.origin,axis=joint.axis,norm=Math.hypot(...axis)||1;
  const unit=axis.map(value=>value/norm),half=length*.5;
  const start=origin.map((value,k)=>value-half*unit[k]);
  const end=origin.map((value,k)=>value+half*unit[k]);
  const source=joint.source==="native_motion_json"?"native AiM motion":"analytic conversion from native ReArt poses";
  return {type:"scatter3d",mode:"lines",name:`${joint.joint_id}: ${joint.joint_type}`,
   x:[start[0],end[0]],y:[start[1],end[1]],z:[start[2],end[2]],
   line:{color:paletteColor(joint.child_part_id,3),width:8,dash:joint.source.startsWith("converted_")?"dash":"solid"},
   hovertemplate:`${joint.joint_id}<br>type=${joint.joint_type}<br>source=${source}<extra></extra>`};
 });
}
function metricsPanel(data){
 const m=data.metrics||{}, fields=[["Status",m.status],["Pred/GT",`${m.predicted_part_count??"?"}/${m.gt_part_count??"?"}`],["Point IoU",m.point_iou],["ARI",m.ari],["RI",m.rand_index],["Coverage",m.geometry_coverage],["Largest cluster",m.largest_cluster_ratio],["Unmatched GT",m.unmatched_gt_part_count]];
 document.getElementById("metrics").innerHTML=fields.filter(x=>x[1]!==undefined&&x[1]!==null).map(([k,v])=>`<span>${k}</span><span>${typeof v==="number"?v.toFixed(3):v}</span>`).join("");
 document.getElementById("sampleInfo").innerHTML=`<p>visualized ${data.visualized_point_count.toLocaleString()} / total ${data.full_point_count.toLocaleString()} points</p>`;
}
function matrixPanel(data){
 const o=data.overlap;if(!o||!o.gt_ids.length){document.getElementById("matrix").innerHTML="<span class=warn>No GT-mapped overlap.</span>";return;}
 const mode=document.getElementById("matrixMode").value, matrix=o[mode];
 let html="<table><tr><th>Pred \\ GT</th>"+o.gt_ids.map(v=>`<th>${v}</th>`).join("")+"</tr>";
 o.pred_ids.forEach((p,r)=>{html+=`<tr><th>${p}</th>`;o.gt_ids.forEach((g,c)=>{const value=matrix[r][c];html+=`<td class=cell data-pred=${p} data-gt=${g}>${mode==="count"?value:Number(value).toFixed(2)}</td>`;});html+="</tr>";});html+="</table>";
 document.getElementById("matrix").innerHTML=html;
 document.querySelectorAll("td.cell").forEach(cell=>cell.onclick=()=>{
  const pred=Number(cell.dataset.pred),gt=Number(cell.dataset.gt);
  pairFilter=pairFilter&&pairFilter.pred===pred&&pairFilter.gt===gt?null:{pred,gt};
  const predStats=(data.statistics?.pred_parts||[]).find(row=>row.pred_part_id===pred);
  const gtStats=(data.statistics?.gt_parts||[]).find(row=>row.gt_part_id===gt);
  document.getElementById("selection").textContent=JSON.stringify({intersection:pairFilter,predicted_part:predStats,gt_part:gtStats},null,2);
  render();
 });
}
function render(){
 const data=methods[currentMethod], indices=visibleIndices(data), traces=[];
 const opacity=Number(document.getElementById("pointOpacity").value);
 const gtFrame=gtGeometryFrame();
 const methodFrame=activeGeometryFrame(data);
 const showMethodGeometry=currentMethod!=="gt"||document.getElementById("showGTGeometry").checked;
 if(methodFrame&&document.getElementById("showCurrent").checked&&showMethodGeometry){
  const isGT=currentMethod==="gt";
  traces.push({type:"scatter3d",mode:"markers",name:isGT?"GT geometry":"method input geometry",x:methodFrame.points.map(p=>p[0]),y:methodFrame.points.map(p=>p[1]),z:methodFrame.points.map(p=>p[2]),marker:{size:Number(document.getElementById("pointSize").value),color:isGT?methodFrame.gt_part_id.map(v=>paletteColor(v)):"#00e5ff",opacity},hovertemplate:isGT?"GT part=%{text}<extra></extra>":"input geometry<extra></extra>",text:methodFrame.gt_part_id});
 } else if(document.getElementById("showCurrent").checked){
  const pts=currentPoints(data,indices), colors=pointColors(data,indices), numeric=colors.some(v=>typeof v==="number");
  traces.push({type:"scatter3d",mode:"markers",name:"current points",x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),customdata:indices,marker:{size:pointSizes(data,indices),color:colors,colorscale:numeric?"Turbo":undefined,showscale:numeric,opacity},hovertemplate:"index=%{customdata}<extra></extra>"});
 }
 if(currentMethod!=="gt"&&gtFrame&&document.getElementById("showGTGeometry").checked){
  traces.push({type:"scatter3d",mode:"markers",name:"GT geometry overlap",x:gtFrame.points.map(p=>p[0]),y:gtFrame.points.map(p=>p[1]),z:gtFrame.points.map(p=>p[2]),marker:{size:Math.max(1,Number(document.getElementById("pointSize").value)*0.55),color:gtFrame.gt_part_id.map(v=>paletteColor(v)),opacity:Math.min(0.32,opacity*0.4)},hovertemplate:"GT part=%{text}<extra></extra>",text:gtFrame.gt_part_id});
 }
 traces.push(...gtMeshTraces());
 traces.push(...jointAxisTraces(data));
 if(document.getElementById("showTrajectories").checked&&data.trajectory){
  const limit=Math.min(Number(trajectorySlider.value),indices.length), chosen=indices.slice(0,limit),x=[],y=[],z=[];
  for(const i of chosen){for(const p of data.trajectory[i]){x.push(p[0]);y.push(p[1]);z.push(p[2]);}x.push(null);y.push(null);z.push(null);}
  traces.push({type:"scatter3d",mode:"lines",name:"trajectories",x,y,z,line:{color:"#a7f3d0",width:2},opacity:.42,hoverinfo:"skip"});
 }
 const b=payload.bounds,layout={paper_bgcolor:"#0a1118",plot_bgcolor:"#0a1118",font:{color:"#e7f0f6"},margin:{l:0,r:0,t:36,b:0},uirevision:"baseline-viewer-fixed-cube",scene:{aspectmode:"cube",xaxis:{range:[b.min[0],b.max[0]],autorange:false,title:"X",gridcolor:"#263847"},yaxis:{range:[b.min[1],b.max[1]],autorange:false,title:"Y",gridcolor:"#263847"},zaxis:{range:[b.min[2],b.max[2]],autorange:false,title:"Z",gridcolor:"#263847"},camera:camera||undefined}};
 Plotly.react("plot",traces,layout,{responsive:true,displaylogo:false});
 const plot=document.getElementById("plot");
 if(!handlersBound){
  plot.on("plotly_relayout",e=>{if(e["scene.camera"])camera=e["scene.camera"];});
  plot.on("plotly_click",e=>{const i=e.points?.[0]?.customdata;if(i===undefined)return;const active=methods[currentMethod];const stat={index:i,gt_part:active.gt_part_id[i],pred_part:active.pred_part_id[i],dynamic:active.dynamic_flag?.[i],premerge:active.premerge_component_id?.[i],postmerge:active.postmerge_component_id?.[i],deformation:active.deformation_magnitude?.[i],rigid_residual:active.rigid_residual?.[i],gt_distance:active.gt_match_distance?.[i]};document.getElementById("selection").textContent=JSON.stringify(stat,null,2);});
  handlersBound=true;
 }
 document.getElementById("timeValue").textContent=data.trajectory?`${Number(timeSlider.value)+1}/${data.trajectory[0].length}`:"static reference";
 document.getElementById("trajectoryValue").textContent=trajectorySlider.value;
 document.getElementById("pointSizeValue").textContent=document.getElementById("pointSize").value;
 document.getElementById("pointOpacityValue").textContent=document.getElementById("pointOpacity").value;
}
function configure(){
 const data=methods[currentMethod];pairFilter=null;fillSelect(stageSelect,stageOptions(data));fillSelect(colorSelect,optionsFor(data));
 const T=data.trajectory?.[0]?.length||1;timeSlider.max=Math.max(0,T-1);timeSlider.value=T-1;
 document.getElementById("smallPartsRow").style.display=(data.gt_part_id||[]).some(value=>[6,7,8].includes(value))?"flex":"none";
 metricsPanel(data);matrixPanel(data);render();
}
methodSelect.onchange=()=>{currentMethod=methodSelect.value;configure();};
stageSelect.onchange=()=>{
 const preferred={dynamic_static:"dynamic_static",premerge:"premerge_component",postmerge:"postmerge_component"}[stageSelect.value];
 if(preferred&&Array.from(colorSelect.options).some(option=>option.value===preferred))colorSelect.value=preferred;
 render();
};
[colorSelect,timeSlider,trajectorySlider,document.getElementById("pointSize"),document.getElementById("pointOpacity"),document.getElementById("palette"),document.getElementById("showCurrent"),document.getElementById("showTrajectories"),document.getElementById("showJointAxes"),document.getElementById("showGTMesh"),document.getElementById("showGTGeometry"),document.getElementById("mappedOnly"),document.getElementById("smallParts")].forEach(e=>e.oninput=render);
document.getElementById("matrixMode").onchange=()=>matrixPanel(methods[currentMethod]);
configure();
</script></body></html>"""
