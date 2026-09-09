"""Point-domain segmentation evaluation for official AiM outputs."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import remap_part_labels


def read_ascii_labeled_ply(path: Path, *, label_mode: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Read xyz and either part_id or exact RGB labels from ASCII/binary PLY."""
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"PLY has no end_header: {path}")
    data_offset = raw.find(b"\n", header_end)
    if data_offset < 0:
        raise ValueError(f"PLY header has no terminating newline: {path}")
    header = raw[:data_offset].decode("ascii")
    lines = header.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Expected a PLY: {path}")
    properties: list[tuple[str, str]] = []
    vertex_count = 0
    in_vertices = False
    ply_format = ""
    for line_index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "format":
            ply_format = fields[1]
        if fields[0] == "element":
            in_vertices = fields[1] == "vertex"
            if in_vertices:
                vertex_count = int(fields[2])
        elif fields[0] == "property" and in_vertices:
            if fields[1] == "list":
                raise ValueError(f"List-valued vertex properties are unsupported: {path}")
            properties.append((fields[1], fields[-1]))
    property_names = [name for _, name in properties]
    property_index = {name: index for index, name in enumerate(property_names)}
    if not {"x", "y", "z"}.issubset(property_index):
        raise ValueError(f"PLY is missing xyz properties: {path}")
    if label_mode == "auto":
        if "part_id" in property_index:
            label_mode = "part_id"
        elif {"red", "green", "blue"}.issubset(property_index):
            label_mode = "rgb"
        elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(property_index):
            label_mode = "sh_dc"
        else:
            raise ValueError(f"PLY has no supported label properties: {path}")
    if label_mode == "part_id" and "part_id" not in property_index:
        raise ValueError(f"PLY has no part_id property: {path}")
    if label_mode == "rgb" and not {"red", "green", "blue"}.issubset(property_index):
        raise ValueError(f"PLY has no RGB label properties: {path}")
    if label_mode == "sh_dc" and not {
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }.issubset(property_index):
        raise ValueError(f"PLY has no spherical-harmonic DC label properties: {path}")

    if ply_format == "ascii":
        data_lines = raw[data_offset + 1 :].decode("utf-8").splitlines()
        rows = [
            line.split()
            for line in data_lines[:vertex_count]
            if len(line.split()) >= len(properties)
        ]
        columns = {
            name: np.asarray([float(row[index]) for row in rows])
            for index, name in enumerate(property_names)
        }
    elif ply_format in {"binary_little_endian", "binary_big_endian"}:
        byte_order = "<" if ply_format == "binary_little_endian" else ">"
        type_map = {
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
        try:
            dtype = np.dtype(
                [(name, byte_order + type_map[data_type]) for data_type, name in properties]
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY scalar type {exc.args[0]}: {path}") from exc
        records = np.frombuffer(
            raw, dtype=dtype, count=vertex_count, offset=data_offset + 1
        )
        columns = {name: records[name] for name in property_names}
    else:
        raise ValueError(f"Unsupported PLY format {ply_format!r}: {path}")
    if not len(columns["x"]):
        raise ValueError(f"PLY contains no vertices: {path}")
    points = np.column_stack([columns[axis] for axis in ("x", "y", "z")])
    if label_mode == "part_id":
        labels = np.asarray(columns["part_id"], dtype=np.int64)
    else:
        # Preserve one global component identity across independently exported
        # timestep PLYs. Per-file palette enumeration would collapse a
        # single-colored frame to label zero in every frame.
        if label_mode == "sh_dc":
            sh_c0 = 0.28209479177387814
            colors = np.clip(
                0.5
                + sh_c0
                * np.column_stack(
                    [columns[f"f_dc_{channel}"] for channel in range(3)]
                ),
                0.0,
                1.0,
            )
            colors = np.rint(colors * 255.0).astype(np.int64)
        else:
            colors = np.column_stack(
                [columns[channel] for channel in ("red", "green", "blue")]
            ).astype(np.int64)
        labels = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return np.asarray(points, dtype=np.float64), labels


def evaluate_aim_pointcloud_iou(
    predicted_ply: Path,
    reference_ply: Path,
    *,
    distance_ratios: Iterable[float] = (0.01, 0.02, 0.05),
    primary_distance_ratio: float = 0.02,
    reference_part_ids: Iterable[int] | None = None,
    reference_part_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate AiM labels on the observed GT point domain using nearest-neighbor transfer."""
    predicted_points, predicted_labels = read_ascii_labeled_ply(predicted_ply, label_mode="rgb")
    result = evaluate_labeled_points_on_reference(
        predicted_points,
        predicted_labels,
        reference_ply,
        distance_ratios=distance_ratios,
        primary_distance_ratio=primary_distance_ratio,
        reference_part_ids=reference_part_ids,
        reference_part_id_map=reference_part_id_map,
    )
    result.update({
        "prediction_source": "aim-segmented-point-ply",
        "predicted_ply": str(predicted_ply.expanduser().resolve()),
    })
    return result


def evaluate_aim_multiframe_union(
    frame_pairs: Iterable[tuple[Path, Path]],
    *,
    component_label_ply: Path | None = None,
    distance_ratios: Iterable[float] = (0.01, 0.02, 0.05),
    primary_distance_ratio: float = 0.02,
    reference_part_ids: Iterable[int] | None = None,
    reference_part_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate time-aligned AiM/GT frame pairs with one global label matching.

    Each prediction is transferred only to the GT points at the same articulation
    state. The transferred labels are then concatenated before Hungarian
    matching, so moving geometry is never compared across different timesteps.
    """
    pairs = [(Path(prediction), Path(reference)) for prediction, reference in frame_pairs]
    if not pairs:
        raise ValueError("At least one time-aligned prediction/reference pair is required")
    selected_part_ids = (
        sorted({int(value) for value in reference_part_ids})
        if reference_part_ids is not None
        else None
    )
    component_label_points: np.ndarray | None = None
    fixed_component_labels: np.ndarray | None = None
    if component_label_ply is not None:
        component_label_points, fixed_component_labels = read_ascii_labeled_ply(
            component_label_ply, label_mode="rgb"
        )
    prepared: list[dict[str, Any]] = []
    global_points: list[np.ndarray] = []
    component_trajectory_indices: np.ndarray | None = None
    propagated_component_labels: np.ndarray | None = None
    component_transfer: dict[str, Any] | None = None
    for prediction_path, reference_path in pairs:
        predicted_points, predicted_labels = read_ascii_labeled_ply(
            prediction_path, label_mode="auto"
        )
        if fixed_component_labels is not None:
            if component_trajectory_indices is None:
                (
                    component_trajectory_indices,
                    propagated_component_labels,
                    component_transfer,
                ) = _component_label_index_map(
                    predicted_points,
                    component_label_points,
                    fixed_component_labels,
                )
            if len(predicted_points) <= int(np.max(component_trajectory_indices)):
                raise ValueError(
                    "AiM trajectory exports do not preserve a common Gaussian "
                    f"index domain: {prediction_path} has {len(predicted_points)} points"
                )
            predicted_points = predicted_points[component_trajectory_indices]
            predicted_labels = propagated_component_labels
        else:
            component_transfer = None
        reference_points, reference_labels = read_ascii_labeled_ply(
            reference_path, label_mode="part_id"
        )
        if reference_part_id_map:
            reference_labels = remap_part_labels(reference_labels, reference_part_id_map)
        if selected_part_ids is not None:
            keep = np.isin(reference_labels, selected_part_ids)
            reference_points = reference_points[keep]
            reference_labels = reference_labels[keep]
        if not len(reference_points):
            continue
        prepared.append(
            {
                "prediction_path": prediction_path,
                "reference_path": reference_path,
                "predicted_points": predicted_points,
                "predicted_labels": predicted_labels,
                "reference_points": reference_points,
                "reference_labels": reference_labels,
                "component_label_transfer": component_transfer,
            }
        )
        global_points.append(reference_points)
    if not prepared:
        raise ValueError("No GT points remain in the selected multi-frame domain")

    all_reference_points = np.concatenate(global_points, axis=0)
    bbox_diagonal = float(np.linalg.norm(np.ptp(all_reference_points, axis=0)))
    if not math.isfinite(bbox_diagonal) or bbox_diagonal <= 0.0:
        raise ValueError("Multi-frame GT domain has a zero bounding-box diagonal")

    ratios = sorted({float(value) for value in distance_ratios} | {float(primary_distance_ratio)})
    threshold_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        threshold_m = ratio * bbox_diagonal
        transferred_chunks: list[np.ndarray] = []
        reference_chunks: list[np.ndarray] = []
        per_frame: list[dict[str, Any]] = []
        for frame in prepared:
            distances, indices = cKDTree(frame["predicted_points"]).query(
                frame["reference_points"], k=1
            )
            covered = distances <= threshold_m
            transferred = frame["predicted_labels"][np.asarray(indices, dtype=np.int64)]
            transferred_chunks.append(transferred[covered])
            reference_chunks.append(frame["reference_labels"][covered])
            per_frame.append(
                {
                    "predicted_ply": str(frame["prediction_path"].resolve()),
                    "reference_ply": str(frame["reference_path"].resolve()),
                    "covered_reference_point_count": int(np.sum(covered)),
                    "reference_point_count": int(len(covered)),
                    "geometry_coverage": float(np.mean(covered)),
                    "component_label_transfer": frame["component_label_transfer"],
                }
            )
        transferred_all = np.concatenate(transferred_chunks)
        reference_all = np.concatenate(reference_chunks)
        metrics = (
            evaluate_slot_assignments(transferred_all, reference_all)
            if len(reference_all)
            else None
        )
        if metrics is not None:
            metrics["rand_index"] = _rand_index(transferred_all, reference_all)
        threshold_rows.append(
            {
                "distance_ratio_bbox": ratio,
                "distance_threshold_m": threshold_m,
                "covered_reference_point_count": int(len(reference_all)),
                "reference_point_count": int(len(all_reference_points)),
                "geometry_coverage": float(len(reference_all) / len(all_reference_points)),
                "covered_only_metrics": metrics,
                "per_frame": per_frame,
            }
        )
    primary = next(
        row
        for row in threshold_rows
        if row["distance_ratio_bbox"] == float(primary_distance_ratio)
    )
    return {
        "metric": "time-aligned-multiframe-observed-point-hungarian-iou",
        "frame_count": len(prepared),
        "predicted_part_count": int(
            len(
                np.unique(
                    np.concatenate([frame["predicted_labels"] for frame in prepared])
                )
            )
        ),
        "gt_part_count": int(
            len(
                np.unique(
                    np.concatenate([frame["reference_labels"] for frame in prepared])
                )
            )
        ),
        "selected_reference_part_ids": selected_part_ids,
        "component_label_ply": (
            str(Path(component_label_ply).expanduser().resolve())
            if component_label_ply is not None
            else None
        ),
        "reference_bbox_diagonal_m": bbox_diagonal,
        "primary_distance_ratio_bbox": float(primary_distance_ratio),
        "primary": primary,
        "thresholds": threshold_rows,
        "comparability_note": (
            "Prediction and GT are paired at identical articulation states. "
            "One global Hungarian mapping is computed across all visible frames."
        ),
    }


def _component_label_index_map(
    trajectory_points: np.ndarray,
    component_points: np.ndarray | None,
    component_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Attach final AiM component labels without inventing missing labels.

    AiM may filter Gaussians between trajectory export and final segmentation.
    Equal-size exports preserve Gaussian order. Otherwise final component points
    are matched back to the source-time trajectory with a strict geometric
    tolerance, and unmatched trajectory Gaussians are excluded from evaluation.
    """
    if component_points is None:
        raise ValueError("Component label points are required")
    if len(component_labels) == len(trajectory_points):
        return np.arange(len(trajectory_points)), component_labels, {
            "mode": "gaussian-index",
            "trajectory_point_count": int(len(trajectory_points)),
            "labeled_trajectory_point_count": int(len(trajectory_points)),
            "labeled_fraction": 1.0,
        }

    bbox_diagonal = float(np.linalg.norm(np.ptp(component_points, axis=0)))
    tolerance_m = max(1e-6, bbox_diagonal * 1e-5)
    distances, indices = cKDTree(component_points).query(trajectory_points, k=1)
    matched = distances <= tolerance_m
    if not np.any(matched):
        raise ValueError(
            "AiM filtered component PLY cannot be aligned to the trajectory "
            f"(minimum distance {float(np.min(distances)):.6g} m, tolerance "
            f"{tolerance_m:.6g} m)"
        )
    return (
        np.flatnonzero(matched),
        component_labels[np.asarray(indices[matched], dtype=np.int64)],
        {
            "mode": "source-time-index-map-propagated",
            "trajectory_point_count": int(len(trajectory_points)),
            "component_point_count": int(len(component_points)),
            "labeled_trajectory_point_count": int(np.sum(matched)),
            "labeled_fraction": float(np.mean(matched)),
            "match_tolerance_m": tolerance_m,
            "matched_distance_max_m": float(np.max(distances[matched])),
        },
    )


def _evaluate_labeled_points(
    predicted_points: np.ndarray,
    predicted_labels: np.ndarray,
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    *,
    reference_ply: Path,
    distance_ratios: Iterable[float],
    primary_distance_ratio: float,
    selected_reference_part_ids: list[int] | None,
) -> dict[str, Any]:
    bounds = np.ptp(reference_points, axis=0)
    bbox_diagonal = float(np.linalg.norm(bounds))
    if not math.isfinite(bbox_diagonal) or bbox_diagonal <= 0.0:
        raise ValueError("Reference point cloud must have a nonzero bounding-box diagonal")
    tree = cKDTree(predicted_points)
    nearest_distances, nearest_indices = tree.query(reference_points, k=1)
    transferred = predicted_labels[np.asarray(nearest_indices, dtype=np.int64)]

    ratios = sorted({float(value) for value in distance_ratios} | {float(primary_distance_ratio)})
    if not ratios or ratios[0] <= 0.0:
        raise ValueError("Distance ratios must be positive")
    threshold_rows = []
    for ratio in ratios:
        threshold_m = ratio * bbox_diagonal
        covered = nearest_distances <= threshold_m
        covered_count = int(np.sum(covered))
        covered_metrics = (
            evaluate_slot_assignments(transferred[covered], reference_labels[covered])
            if covered_count
            else None
        )
        if covered_metrics is not None:
            covered_metrics["rand_index"] = _rand_index(
                transferred[covered],
                reference_labels[covered],
            )
        coverage_aware = _coverage_aware_iou(
            transferred[covered],
            reference_labels[covered],
            reference_labels,
            expected_gt_values=selected_reference_part_ids,
        )
        threshold_rows.append({
            "distance_ratio_bbox": ratio,
            "distance_threshold_m": threshold_m,
            "covered_reference_point_count": covered_count,
            "reference_point_count": int(len(reference_points)),
            "geometry_coverage": covered_count / len(reference_points),
            "nearest_distance_mean_m": float(np.mean(nearest_distances)),
            "nearest_distance_median_m": float(np.median(nearest_distances)),
            "covered_only_metrics": covered_metrics,
            "coverage_aware_metrics": coverage_aware,
        })
    primary = next(row for row in threshold_rows if row["distance_ratio_bbox"] == float(primary_distance_ratio))
    return {
        "metric": "observed-reference-point-hungarian-iou",
        "reference_ply": str(reference_ply.expanduser().resolve()),
        "predicted_point_count": int(len(predicted_points)),
        "reference_point_count": int(len(reference_points)),
        "predicted_part_count": int(len(np.unique(predicted_labels))),
        "gt_part_count": int(
            len(selected_reference_part_ids)
            if selected_reference_part_ids is not None
            else len(np.unique(reference_labels))
        ),
        "reference_domain_missing_part_ids": (
            sorted(set(selected_reference_part_ids).difference(np.unique(reference_labels)))
            if selected_reference_part_ids is not None
            else []
        ),
        "selected_reference_part_ids": selected_reference_part_ids,
        "reference_bbox_diagonal_m": bbox_diagonal,
        "primary_distance_ratio_bbox": float(primary_distance_ratio),
        "primary": primary,
        "thresholds": threshold_rows,
        "comparability_note": (
            "This evaluates labels on observed GT points and is directly comparable to this project's "
            "point-domain IoU. It is not AiM's published voxelized mesh IoU."
        ),
    }


def _rand_index(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Return the ordinary pairwise Rand Index without a sklearn dependency."""
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference labels must have matching shapes")
    sample_count = int(predicted.size)
    if sample_count < 2:
        return 1.0
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    _, reference_inverse = np.unique(reference, return_inverse=True)
    contingency = np.zeros(
        (int(predicted_inverse.max()) + 1, int(reference_inverse.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(contingency, (predicted_inverse, reference_inverse), 1)
    choose_two = lambda values: values * (values - 1) // 2
    true_positive = int(choose_two(contingency).sum())
    predicted_same = int(choose_two(contingency.sum(axis=1)).sum())
    reference_same = int(choose_two(contingency.sum(axis=0)).sum())
    total_pairs = sample_count * (sample_count - 1) // 2
    false_positive = predicted_same - true_positive
    false_negative = reference_same - true_positive
    true_negative = total_pairs - true_positive - false_positive - false_negative
    return float((true_positive + true_negative) / total_pairs)


def read_track_json_labeled_points(
    path: Path,
    *,
    source_frame_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read the latest visible 3D sample at or before a requested source frame."""
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"Track JSON has no tracks list: {path}")
    if source_frame_index is None:
        indices = [
            int(sample["source_frame_index"])
            for track in tracks
            for sample in track.get("samples", [])
            if sample.get("visible") and sample.get("xyz_world") is not None
        ]
        if not indices:
            raise ValueError(f"Track JSON has no visible 3D samples: {path}")
        source_frame_index = max(indices)

    points: list[list[float]] = []
    labels: list[int] = []
    for track in tracks:
        candidates = [
            sample
            for sample in track.get("samples", [])
            if sample.get("visible")
            and sample.get("xyz_world") is not None
            and int(sample.get("source_frame_index", -1)) <= source_frame_index
        ]
        if not candidates:
            continue
        sample = max(candidates, key=lambda row: int(row.get("source_frame_index", -1)))
        points.append([float(value) for value in sample["xyz_world"]])
        labels.append(int(track["part_id"]))
    if not points:
        raise ValueError(f"No tracks are visible by source frame {source_frame_index}: {path}")
    return np.asarray(points, dtype=np.float64), np.asarray(labels, dtype=np.int64), source_frame_index


def evaluate_track_pointcloud_iou(
    predicted_tracks_json: Path,
    reference_ply: Path,
    *,
    source_frame_index: int | None = None,
    distance_ratios: Iterable[float] = (0.01, 0.02, 0.05),
    primary_distance_ratio: float = 0.02,
    reference_part_ids: Iterable[int] | None = None,
    reference_part_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate predicted track labels on the same observed reference domain as AiM."""
    points, labels, selected_frame = read_track_json_labeled_points(
        predicted_tracks_json,
        source_frame_index=source_frame_index,
    )
    result = evaluate_labeled_points_on_reference(
        points,
        labels,
        reference_ply,
        distance_ratios=distance_ratios,
        primary_distance_ratio=primary_distance_ratio,
        reference_part_ids=reference_part_ids,
        reference_part_id_map=reference_part_id_map,
    )
    result.update({
        "prediction_source": "motion-part-track-json",
        "predicted_tracks_json": str(predicted_tracks_json.expanduser().resolve()),
        "source_frame_index": selected_frame,
    })
    return result


def evaluate_labeled_points_on_reference(
    predicted_points: np.ndarray,
    predicted_labels: np.ndarray,
    reference_ply: Path,
    *,
    distance_ratios: Iterable[float] = (0.01, 0.02, 0.05),
    primary_distance_ratio: float = 0.02,
    reference_part_ids: Iterable[int] | None = None,
    reference_part_id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate arbitrary labeled 3D points on a labeled observed reference PLY."""
    reference_points, reference_labels = read_ascii_labeled_ply(reference_ply, label_mode="part_id")
    if reference_part_id_map:
        reference_labels = remap_part_labels(reference_labels, reference_part_id_map)
    selected_reference_part_ids = (
        sorted({int(value) for value in reference_part_ids})
        if reference_part_ids is not None
        else None
    )
    if selected_reference_part_ids is not None:
        selected = np.isin(reference_labels, selected_reference_part_ids)
        reference_points = reference_points[selected]
        reference_labels = reference_labels[selected]
        if not len(reference_points):
            raise ValueError("No reference points remain after filtering reference part IDs")
    return _evaluate_labeled_points(
        np.asarray(predicted_points, dtype=np.float64),
        np.asarray(predicted_labels, dtype=np.int64),
        reference_points,
        reference_labels,
        reference_ply=reference_ply,
        distance_ratios=distance_ratios,
        primary_distance_ratio=primary_distance_ratio,
        selected_reference_part_ids=selected_reference_part_ids,
    )


def read_original_part_ids(path: Path) -> list[int]:
    """Return the simulation-only GT part IDs represented by a track artifact."""
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    values = {
        int(track["original_part_id"])
        for track in payload.get("tracks", [])
        if track.get("original_part_id") is not None
    }
    if (
        not values
        and payload.get("object_mask_tracking_mode") is False
        and payload.get("part_segmentation", {}).get("provider") == "mujoco-body-geom-prior"
    ):
        values = {
            int(track["part_id"])
            for track in payload.get("tracks", [])
            if track.get("part_id") is not None
        }
    if not values:
        raise ValueError(f"Track JSON has no simulation GT part labels: {path}")
    return sorted(values)


def _coverage_aware_iou(
    covered_predicted: np.ndarray,
    covered_reference: np.ndarray,
    all_reference: np.ndarray,
    *,
    expected_gt_values: Iterable[int] | None = None,
) -> dict[str, Any]:
    pred_values = sorted(int(value) for value in np.unique(covered_predicted))
    gt_values = (
        sorted({int(value) for value in expected_gt_values})
        if expected_gt_values is not None
        else sorted(int(value) for value in np.unique(all_reference))
    )
    overlap = np.zeros((len(pred_values), len(gt_values)), dtype=np.int64)
    for pred_index, pred_value in enumerate(pred_values):
        for gt_index, gt_value in enumerate(gt_values):
            overlap[pred_index, gt_index] = int(
                np.sum((covered_predicted == pred_value) & (covered_reference == gt_value))
            )
    pred_sizes = overlap.sum(axis=1)
    gt_sizes = np.asarray([np.sum(all_reference == value) for value in gt_values], dtype=np.int64)
    union = pred_sizes[:, None] + gt_sizes[None, :] - overlap
    iou = overlap / np.maximum(union, 1)
    if iou.size:
        rows, columns = linear_sum_assignment(-iou)
        pairs = list(zip(rows.tolist(), columns.tolist(), strict=True))
    else:
        pairs = []
    matching = [{
        "pred_part": pred_values[pred_index],
        "gt_part": gt_values[gt_index],
        "intersection": int(overlap[pred_index, gt_index]),
        "iou": float(iou[pred_index, gt_index]),
        "gt_coverage": float(overlap[pred_index, gt_index] / max(1, gt_sizes[gt_index])),
    } for pred_index, gt_index in pairs]
    return {
        "one_to_one_mean_iou": float(sum(row["iou"] for row in matching) / max(1, len(gt_values))),
        "mean_matched_gt_coverage": float(
            sum(row["gt_coverage"] for row in matching) / max(1, len(gt_values))
        ),
        "unmatched_gt_part_count": len(gt_values) - len({row["gt_part"] for row in matching}),
        "overlap_matrix": overlap.tolist(),
        "iou_matrix": iou.tolist(),
        "matching": matching,
    }
