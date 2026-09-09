from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from .joint_inference import (
    _dot,
    _joint_defs_from_mjcf,
    _load_mujoco_joint_priors,
    _matvec3,
    _norm,
    _normalize,
    _subtract,
    _transpose3,
)


@dataclass(slots=True)
class KinematicModelEvaluationConfig:
    joint_inference_path: str | Path
    part_pose_path: str | Path | None = None
    output_json: str | Path | None = None


@dataclass(slots=True)
class ObjectMaskKinematicEvaluationConfig:
    joint_inference_path: str | Path
    part_pose_path: str | Path | None = None
    output_json: str | Path | None = None
    output_csv: str | Path | None = None
    matching_metric: str = "iou"


class KinematicModelEvaluator:
    """Evaluate inferred kinematic joints against simulator joint priors.

    The evaluator is intentionally tied to the simulation-debug setting for now:
    it compares a `joint_inference.json` artifact against MuJoCo joint metadata
    recoverable through the corresponding `part_poses.json -> fusion_manifest ->
    episode.json -> model_path` chain. This avoids pretending that we have a
    solved cross-method part correspondence problem for generated/PARTICULATE
    outputs.
    """

    def evaluate(self, config: KinematicModelEvaluationConfig) -> Path:
        joint_path = Path(config.joint_inference_path).expanduser().resolve()
        joint_artifact = load_json(joint_path)
        part_pose_path = self._resolve_part_pose_path(config, joint_artifact, joint_path)
        part_pose_artifact = load_json(part_pose_path)
        priors = _load_mujoco_joint_priors(part_pose_artifact)

        predicted_joints = [
            item
            for item in joint_artifact.get("joints", [])
            if isinstance(item, dict)
        ]
        per_joint = [
            self._evaluate_joint(joint, priors.get(int(joint.get("child_part_id", -1))))
            for joint in predicted_joints
        ]
        matched = [item for item in per_joint if bool(item.get("matched", False))]

        summary = self._summary(
            predicted_joints=predicted_joints,
            priors=priors,
            matched=matched,
        )
        output_path = (
            Path(config.output_json).expanduser().resolve()
            if config.output_json is not None
            else joint_path.with_name("kinematic_evaluation.json")
        )
        save_json(
            {
                "source": "kinematic-model-evaluation",
                "joint_inference_path": str(joint_path),
                "part_pose_path": str(part_pose_path),
                "ground_truth_available": bool(priors),
                "summary": summary,
                "per_joint": per_joint,
                "notes": [
                    "Ground truth is loaded from the MuJoCo model referenced by the recording episode.",
                    "Axis angular error is sign-invariant because revolute/prismatic axes can be flipped with q sign.",
                    "Revolute pivot error is the shortest distance between predicted and GT joint axes.",
                    "q_rmse_offset_aligned removes a constant q offset and is usually more meaningful when the predicted q starts at zero.",
                ],
            },
            output_path,
        )
        return output_path

    def _resolve_part_pose_path(
        self,
        config: KinematicModelEvaluationConfig,
        joint_artifact: dict[str, Any],
        joint_path: Path,
    ) -> Path:
        if config.part_pose_path is not None:
            return Path(config.part_pose_path).expanduser().resolve()
        input_path = joint_artifact.get("input_path")
        if not isinstance(input_path, str) or not input_path:
            raise ValueError("Provide --part-poses because joint_inference.json does not contain input_path.")
        candidate = Path(input_path).expanduser()
        if not candidate.is_absolute():
            candidate = joint_path.parent / candidate
        return candidate.resolve()

    def _evaluate_joint(self, predicted: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
        child_part_id = int(predicted.get("child_part_id", -1))
        result: dict[str, Any] = {
            "name": predicted.get("name"),
            "child_part_id": child_part_id,
            "child_name": predicted.get("child_name"),
            "predicted_joint_type": predicted.get("joint_type"),
            "matched": prior is not None,
        }
        if prior is None:
            result["reason"] = "no_ground_truth_prior_for_child_part"
            return result

        predicted_type = _canonical_joint_type(str(predicted.get("joint_type", "unknown")))
        gt_type = _canonical_joint_type(str(prior.get("joint_type", "unknown")))
        pred_axis = _parse_vec3(predicted.get("axis"), [0.0, 0.0, 1.0])
        gt_axis = _parse_vec3(prior.get("axis"), [0.0, 0.0, 1.0])
        pred_pivot = _parse_vec3(predicted.get("pivot"), [0.0, 0.0, 0.0])
        gt_pivot = _parse_vec3(prior.get("pivot"), [0.0, 0.0, 0.0])
        pred_limits = _parse_limits(predicted.get("limits"))
        gt_limits = _parse_limits(prior.get("limits"))

        axis_error_deg = _axis_angle_error_deg(pred_axis, gt_axis)
        pivot_error_m = _joint_position_error(pred_pivot, pred_axis, predicted_type, gt_pivot, gt_axis, gt_type)
        q_metrics = _q_error_metrics(predicted.get("q_samples"), prior.get("q_by_frame"))
        result.update(
            {
                "ground_truth_joint_name": prior.get("joint_name"),
                "ground_truth_joint_type": gt_type,
                "joint_type_correct": predicted_type == gt_type,
                "predicted_axis": pred_axis,
                "ground_truth_axis": gt_axis,
                "axis_angle_error_deg": axis_error_deg,
                "predicted_pivot": pred_pivot,
                "ground_truth_pivot": gt_pivot,
                "pivot_error_m": pivot_error_m,
                "predicted_limits": pred_limits,
                "ground_truth_limits": gt_limits,
                "limit_error": _limit_error(pred_limits, gt_limits),
                "q_error": q_metrics,
            }
        )
        return result

    def _summary(
        self,
        predicted_joints: list[dict[str, Any]],
        priors: dict[int, dict[str, Any]],
        matched: list[dict[str, Any]],
    ) -> dict[str, Any]:
        type_correct = [
            bool(item.get("joint_type_correct", False))
            for item in matched
            if "joint_type_correct" in item
        ]
        geometry_rows = [item for item in matched if bool(item.get("joint_type_correct", False))]
        axis_errors = _finite_values(item.get("axis_angle_error_deg") for item in geometry_rows)
        pivot_errors = _finite_values(item.get("pivot_error_m") for item in geometry_rows)
        limit_abs_errors = []
        q_rmse = []
        q_rmse_aligned = []
        for item in matched:
            limit_error = item.get("limit_error")
            if isinstance(limit_error, dict):
                limit_abs_errors.extend(_finite_values(limit_error.get(key) for key in ("lower_abs_error", "upper_abs_error")))
            q_error = item.get("q_error")
            if isinstance(q_error, dict):
                q_rmse.extend(_finite_values([q_error.get("q_rmse")]))
                q_rmse_aligned.extend(_finite_values([q_error.get("q_rmse_offset_aligned")]))
        return {
            "predicted_joint_count": len(predicted_joints),
            "ground_truth_joint_count": len(priors),
            "matched_joint_count": len(matched),
            "joint_coverage": len(matched) / max(1, len(priors)),
            "joint_type_accuracy": (
                sum(1 for value in type_correct if value) / len(type_correct)
                if type_correct
                else None
            ),
            "axis_angle_error_deg_mean": _mean_or_none(axis_errors),
            "axis_angle_error_deg_median": _median_or_none(axis_errors),
            "axis_angle_error_deg_max": max(axis_errors) if axis_errors else None,
            "axis_evaluable_joint_count": len(axis_errors),
            "pivot_error_m_mean": _mean_or_none(pivot_errors),
            "pivot_error_m_median": _median_or_none(pivot_errors),
            "pivot_error_m_max": max(pivot_errors) if pivot_errors else None,
            "limit_abs_error_mean": _mean_or_none(limit_abs_errors),
            "q_rmse_mean": _mean_or_none(q_rmse),
            "q_rmse_offset_aligned_mean": _mean_or_none(q_rmse_aligned),
            "by_ground_truth_joint_type": _summary_by_joint_type(matched),
        }


class ObjectMaskKinematicEvaluator(KinematicModelEvaluator):
    """Simulation-only evaluator for object-mask clusters with arbitrary ids."""

    def evaluate(self, config: ObjectMaskKinematicEvaluationConfig) -> Path:
        joint_path = Path(config.joint_inference_path).expanduser().resolve()
        joint_artifact = load_json(joint_path)
        part_pose_path = self._resolve_part_pose_path(config, joint_artifact, joint_path)
        part_pose_artifact = load_json(part_pose_path)
        track_path = _resolve_track_path(part_pose_artifact, part_pose_path)
        track_artifact = load_json(track_path)

        overlap = _cluster_gt_overlap(track_artifact)
        matching = _match_clusters_to_gt(overlap, metric=config.matching_metric)
        gt_priors = _load_object_mask_gt_priors(part_pose_artifact, track_artifact)
        cluster_debug = _track_cluster_debug(track_artifact)
        cluster_motion = _cluster_motion_stats(track_artifact)
        predicted_joints = [
            item
            for item in joint_artifact.get("joints", [])
            if isinstance(item, dict)
        ]
        per_joint = [
            self._evaluate_remapped_joint(joint, matching["pred_to_gt"], gt_priors, cluster_motion, cluster_debug)
            for joint in predicted_joints
        ]
        directed_matches = [item for item in per_joint if bool(item.get("directed_matched", False))]
        undirected_matches = [item for item in per_joint if bool(item.get("undirected_matched", False))]
        reverse_matches = [item for item in per_joint if bool(item.get("reverse_directed_matched", False))]
        summary = _object_mask_summary(
            predicted_joints=predicted_joints,
            gt_priors=gt_priors,
            directed_matches=directed_matches,
            undirected_matches=undirected_matches,
            reverse_matches=reverse_matches,
            overlap=overlap,
        )

        output_path = (
            Path(config.output_json).expanduser().resolve()
            if config.output_json is not None
            else joint_path.with_name("object_mask_kinematic_evaluation.json")
        )
        payload = {
            "source": "object-mask-cluster-to-gt-kinematic-evaluation",
            "diagnostic_only": True,
            "joint_inference_path": str(joint_path),
            "part_pose_path": str(part_pose_path),
            "track_path": str(track_path),
            "matching_metric": config.matching_metric,
            "overlap": overlap,
            "cluster_motion": cluster_motion,
            "cluster_debug": cluster_debug,
            "matching": matching,
            "ground_truth_available": bool(gt_priors),
            "summary": summary,
            "per_joint": per_joint,
            "notes": [
                "This evaluator uses original_part_id only for simulation diagnostics.",
                "Cluster-to-GT matching is not used during inference.",
                "Directed coverage requires predicted parent and child clusters to map to the GT parent/child.",
                "Undirected coverage ignores joint direction after cluster-to-GT remapping.",
            ],
        }
        save_json(payload, output_path)
        csv_path = Path(config.output_csv).expanduser().resolve() if config.output_csv is not None else None
        if csv_path is not None:
            _write_object_mask_csv(csv_path, payload)
        return output_path

    def _evaluate_remapped_joint(
        self,
        predicted: dict[str, Any],
        pred_to_gt: dict[str, int],
        gt_priors: dict[int, dict[str, Any]],
        cluster_motion: dict[str, dict[str, Any]],
        cluster_debug: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        parent_cluster = int(predicted.get("parent_part_id", -1))
        child_cluster = int(predicted.get("child_part_id", -1))
        parent_gt = pred_to_gt.get(str(parent_cluster))
        child_gt = pred_to_gt.get(str(child_cluster))
        directed_prior = gt_priors.get(child_gt) if child_gt is not None else None
        directed = self._evaluate_joint(predicted, directed_prior)
        directed["parent_part_id"] = parent_cluster
        directed["child_part_id"] = child_cluster
        directed["mapped_parent_gt_part_id"] = parent_gt
        directed["mapped_child_gt_part_id"] = child_gt
        directed["pred_parent_motion_score"] = cluster_motion.get(str(parent_cluster), {}).get("median_motion_m")
        directed["pred_child_motion_score"] = cluster_motion.get(str(child_cluster), {}).get("median_motion_m")
        directed["pred_parent_track_count"] = cluster_motion.get(str(parent_cluster), {}).get("track_count")
        directed["pred_child_track_count"] = cluster_motion.get(str(child_cluster), {}).get("track_count")
        directed["parent_cluster_debug"] = cluster_debug.get(str(parent_cluster))
        directed["child_cluster_debug"] = cluster_debug.get(str(child_cluster))
        directed["directed_matched"] = directed_prior is not None and (
            parent_gt is None or int(directed_prior.get("parent_part_id", -999999)) == int(parent_gt)
        )

        undirected_prior = _find_undirected_prior(parent_gt, child_gt, gt_priors)
        reverse_prior = gt_priors.get(parent_gt) if parent_gt is not None else None
        undirected_eval = self._evaluate_joint(predicted, undirected_prior)
        directed["undirected_matched"] = undirected_prior is not None
        directed["reverse_directed_matched"] = reverse_prior is not None and child_gt is not None and (
            int(reverse_prior.get("parent_part_id", -999999)) == int(child_gt)
        )
        directed["undirected_ground_truth_joint_name"] = (
            undirected_prior.get("joint_name") if undirected_prior is not None else None
        )
        directed["reverse_ground_truth_joint_name"] = (
            reverse_prior.get("joint_name") if directed["reverse_directed_matched"] and reverse_prior is not None else None
        )
        if not bool(directed["directed_matched"]) and undirected_prior is not None:
            for key in (
                "ground_truth_joint_name",
                "ground_truth_joint_type",
                "joint_type_correct",
                "ground_truth_axis",
                "axis_angle_error_deg",
                "ground_truth_pivot",
                "pivot_error_m",
                "ground_truth_limits",
                "limit_error",
                "q_error",
            ):
                if key in undirected_eval:
                    directed[f"undirected_{key}"] = undirected_eval[key]
        directed["failure_reason_guess"] = _joint_failure_reason_guess(directed)
        directed["cleanup_recommendation"] = _joint_cleanup_recommendation(directed)
        return directed


def _resolve_track_path(part_pose_artifact: dict[str, Any], part_pose_path: Path) -> Path:
    input_path = part_pose_artifact.get("input_path")
    if not isinstance(input_path, str) or not input_path:
        raise ValueError("Object-mask evaluation requires part_poses.json with input_path pointing to part_tracks.json.")
    candidate = Path(input_path).expanduser()
    if not candidate.is_absolute():
        candidate = part_pose_path.parent / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Track artifact does not exist: {candidate}")
    return candidate.resolve()


def _cluster_gt_overlap(track_artifact: dict[str, Any]) -> dict[str, Any]:
    counts: dict[int, dict[int, int]] = {}
    pred_totals: dict[int, int] = {}
    gt_totals: dict[int, int] = {}
    for track in track_artifact.get("tracks", []):
        if not isinstance(track, dict):
            continue
        try:
            pred = int(track.get("part_id", 0))
            gt = int(track.get("original_part_id"))
        except (TypeError, ValueError):
            continue
        if pred <= 0 or gt <= 0:
            continue
        counts.setdefault(pred, {})
        counts[pred][gt] = counts[pred].get(gt, 0) + 1
        pred_totals[pred] = pred_totals.get(pred, 0) + 1
        gt_totals[gt] = gt_totals.get(gt, 0) + 1

    pred_ids = sorted(pred_totals)
    gt_ids = sorted(gt_totals)
    matrix = {
        str(pred): {str(gt): int(counts.get(pred, {}).get(gt, 0)) for gt in gt_ids}
        for pred in pred_ids
    }
    iou_matrix = {
        str(pred): {
            str(gt): _safe_div(
                counts.get(pred, {}).get(gt, 0),
                pred_totals[pred] + gt_totals[gt] - counts.get(pred, {}).get(gt, 0),
            )
            for gt in gt_ids
        }
        for pred in pred_ids
    }
    per_cluster = []
    for pred in pred_ids:
        row = counts.get(pred, {})
        best_gt, best_count = _best_item(row)
        per_cluster.append(
            {
                "pred_cluster_id": pred,
                "track_count": pred_totals[pred],
                "dominant_gt_part_id": best_gt,
                "dominant_gt_overlap": best_count,
                "purity": _safe_div(best_count, pred_totals[pred]),
            }
        )
    per_gt = []
    for gt in gt_ids:
        best_pred = None
        best_count = 0
        for pred in pred_ids:
            value = counts.get(pred, {}).get(gt, 0)
            if value > best_count:
                best_pred = pred
                best_count = value
        per_gt.append(
            {
                "gt_part_id": gt,
                "track_count": gt_totals[gt],
                "best_pred_cluster_id": best_pred,
                "best_overlap": best_count,
                "coverage": _safe_div(best_count, gt_totals[gt]),
            }
        )
    total_tracks = sum(pred_totals.values())
    return {
        "pred_cluster_ids": pred_ids,
        "gt_part_ids": gt_ids,
        "overlap_matrix": matrix,
        "iou_matrix": iou_matrix,
        "pred_cluster_totals": {str(key): value for key, value in pred_totals.items()},
        "gt_part_totals": {str(key): value for key, value in gt_totals.items()},
        "per_cluster": per_cluster,
        "per_gt": per_gt,
        "mean_purity": _mean_or_none([item["purity"] for item in per_cluster]),
        "mean_gt_coverage": _mean_or_none([item["coverage"] for item in per_gt]),
        "largest_cluster_ratio": max(pred_totals.values(), default=0) / max(1, total_tracks),
        "track_count": total_tracks,
    }


def _match_clusters_to_gt(overlap: dict[str, Any], metric: str) -> dict[str, Any]:
    pred_ids = [int(value) for value in overlap.get("pred_cluster_ids", [])]
    gt_ids = [int(value) for value in overlap.get("gt_part_ids", [])]
    if metric not in {"iou", "overlap"}:
        raise ValueError("--matching-metric must be 'iou' or 'overlap'.")
    score_matrix = overlap["iou_matrix"] if metric == "iou" else overlap["overlap_matrix"]
    pairs = _max_assignment(
        pred_ids,
        gt_ids,
        lambda pred, gt: float(score_matrix.get(str(pred), {}).get(str(gt), 0.0)),
    )
    pred_to_gt = {str(pred): gt for pred, gt, score in pairs if score > 0.0}
    gt_to_pred = {str(gt): pred for pred, gt, score in pairs if score > 0.0}
    return {
        "metric": metric,
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
        "pairs": [
            {"pred_cluster_id": pred, "gt_part_id": gt, "score": score}
            for pred, gt, score in pairs
        ],
    }


def _max_assignment(pred_ids: list[int], gt_ids: list[int], score_fn: Any) -> list[tuple[int, int, float]]:
    if not pred_ids or not gt_ids:
        return []
    if max(len(pred_ids), len(gt_ids)) <= 8:
        return _max_assignment_exhaustive(pred_ids, gt_ids, score_fn)
    remaining_gt = set(gt_ids)
    pairs = []
    for pred in sorted(pred_ids):
        if not remaining_gt:
            break
        gt = max(sorted(remaining_gt), key=lambda item: score_fn(pred, item))
        score = float(score_fn(pred, gt))
        pairs.append((pred, gt, score))
        remaining_gt.remove(gt)
    return pairs


def _max_assignment_exhaustive(pred_ids: list[int], gt_ids: list[int], score_fn: Any) -> list[tuple[int, int, float]]:
    from itertools import permutations

    left = pred_ids
    right = gt_ids
    transpose = False
    if len(left) > len(right):
        left, right = gt_ids, pred_ids
        transpose = True
    best_score = -1.0
    best_pairs: list[tuple[int, int, float]] = []
    for perm in permutations(right, len(left)):
        pairs = []
        total = 0.0
        for a, b in zip(left, perm):
            pred, gt = (b, a) if transpose else (a, b)
            score = float(score_fn(pred, gt))
            total += score
            pairs.append((pred, gt, score))
        if total > best_score:
            best_score = total
            best_pairs = pairs
    return best_pairs


def _load_object_mask_gt_priors(part_pose_artifact: dict[str, Any], track_artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episode_path = _episode_path_from_track_artifact(track_artifact)
    if episode_path is None:
        return {}
    episode = load_json(episode_path)
    model_path_raw = episode.get("metadata", {}).get("model_path")
    if not isinstance(model_path_raw, str):
        return {}
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = episode_path.parent / model_path
    if not model_path.exists():
        return {}
    joint_defs = _joint_defs_from_mjcf(model_path)
    part_segmentation = track_artifact.get("original_part_segmentation") or track_artifact.get("part_segmentation")
    if not isinstance(part_segmentation, dict):
        return {}
    part_id_by_body_id = _part_id_by_body_id(part_segmentation)

    parent_rotation, parent_translation = _anchor_parent_transform(part_pose_artifact)
    parent_inv = _transpose3(parent_rotation)
    frames = episode.get("frames", [])
    priors: dict[int, dict[str, Any]] = {}
    for raw_part in part_segmentation.get("parts", []):
        if not isinstance(raw_part, dict):
            continue
        part_id = int(raw_part.get("part_id", 0))
        joint_names = [str(name) for name in raw_part.get("joint_names", []) if isinstance(name, str)]
        if not joint_names:
            continue
        for joint_name in joint_names:
            joint_def = joint_defs.get(joint_name)
            if joint_def is None:
                continue
            axis_parent = _normalize(_matvec3(parent_inv, joint_def["axis_world"]), fallback=[0.0, 0.0, 1.0])
            pivot_parent = _matvec3(parent_inv, _subtract(joint_def["pivot_world"], parent_translation))
            q_by_frame: dict[int, float] = {}
            for frame_index, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    continue
                sample_frame_index = int(frame.get("frame_index", frame_index))
                joint_positions = frame.get("action_log", {}).get("joint_positions", {})
                if isinstance(joint_positions, dict) and joint_name in joint_positions:
                    q_by_frame[sample_frame_index] = float(joint_positions[joint_name])
            priors[part_id] = {
                **joint_def,
                "joint_name": joint_name,
                "parent_part_id": _gt_parent_part_id(raw_part, part_id_by_body_id),
                "axis": axis_parent,
                "pivot": pivot_parent,
                "q_by_frame": q_by_frame,
                "source": "mujoco-mjcf-object-mask-diagnostic",
                "model_path": str(model_path),
            }
            break
    return priors


def _part_id_by_body_id(part_segmentation: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for raw_part in part_segmentation.get("parts", []):
        if not isinstance(raw_part, dict):
            continue
        try:
            body_id = int(raw_part.get("body_id"))
            part_id = int(raw_part.get("part_id"))
        except (TypeError, ValueError):
            continue
        out[body_id] = part_id
    return out


def _gt_parent_part_id(raw_part: dict[str, Any], part_id_by_body_id: dict[int, int]) -> int:
    try:
        explicit = int(raw_part.get("parent_part_id"))
        if explicit > 0:
            return explicit
    except (TypeError, ValueError):
        pass
    try:
        parent_body_id = int(raw_part.get("parent_body_id"))
    except (TypeError, ValueError):
        return 0
    return int(part_id_by_body_id.get(parent_body_id, 0))


def _episode_path_from_track_artifact(track_artifact: dict[str, Any]) -> Path | None:
    raw = track_artifact.get("episode_path") or track_artifact.get("input_episode_path")
    if not isinstance(raw, str):
        return None
    path = Path(raw)
    return path if path.exists() else None


def _anchor_parent_transform(part_pose_artifact: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    anchor_id = int(part_pose_artifact.get("anchor_part_id", 0))
    for part in part_pose_artifact.get("parts", []):
        if not isinstance(part, dict) or int(part.get("part_id", -1)) != anchor_id:
            continue
        canonical = part.get("canonical_frame")
        if isinstance(canonical, dict):
            rotation = canonical.get("rotation_matrix")
            translation = canonical.get("translation")
            if isinstance(rotation, list) and isinstance(translation, list):
                return (
                    [[float(value) for value in row] for row in rotation],
                    [float(value) for value in translation],
                )
    return (
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
    )


def _find_undirected_prior(parent_gt: int | None, child_gt: int | None, priors: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if child_gt is None:
        return None
    direct = priors.get(child_gt)
    if direct is not None:
        if parent_gt is None or int(direct.get("parent_part_id", -999999)) == int(parent_gt):
            return direct
    if parent_gt is not None:
        reverse = priors.get(parent_gt)
        if reverse is not None and int(reverse.get("parent_part_id", -999999)) == int(child_gt):
            return reverse
    return direct


def _cluster_motion_stats(track_artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_part: dict[int, list[float]] = {}
    for track in track_artifact.get("tracks", []):
        if not isinstance(track, dict):
            continue
        try:
            part_id = int(track.get("part_id", 0))
        except (TypeError, ValueError):
            continue
        if part_id <= 0:
            continue
        motion = _track_endpoint_motion(track)
        by_part.setdefault(part_id, []).append(motion)
    return {
        str(part_id): {
            "track_count": len(values),
            "median_motion_m": _median(values),
            "mean_motion_m": sum(values) / len(values) if values else 0.0,
        }
        for part_id, values in sorted(by_part.items())
    }


def _track_cluster_debug(track_artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clusters: dict[int, list[dict[str, Any]]] = {}
    for track in track_artifact.get("tracks", []):
        if not isinstance(track, dict):
            continue
        try:
            part_id = int(track.get("part_id", 0))
        except (TypeError, ValueError):
            continue
        if part_id <= 0:
            continue
        clusters.setdefault(part_id, []).append(track)
    return {
        str(part_id): _single_cluster_debug(part_id, tracks)
        for part_id, tracks in sorted(clusters.items())
    }


def _single_cluster_debug(part_id: int, tracks: list[dict[str, Any]]) -> dict[str, Any]:
    composition: dict[str, int] = {}
    for track in tracks:
        key = str(track.get("original_part_id", "unknown"))
        composition[key] = composition.get(key, 0) + 1
    dominant_gt = max(composition, key=composition.get) if composition else None
    dominant_count = composition.get(dominant_gt, 0) if dominant_gt is not None else 0
    residual = _cluster_rigid_residual(tracks)
    return {
        "part_id": part_id,
        "track_count": len(tracks),
        "gt_composition": composition,
        "dominant_gt_part_id": int(dominant_gt) if dominant_gt is not None and dominant_gt.isdigit() else dominant_gt,
        "gt_purity": dominant_count / max(1, len(tracks)) if "unknown" not in composition else None,
        **residual,
    }


def _cluster_rigid_residual(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tracks) < 3:
        return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
    reference_points = []
    trajectories = []
    for track in tracks:
        reference = track.get("reference_xyz_world")
        if not isinstance(reference, list) or len(reference) != 3:
            return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
        trajectory = _track_trajectory_by_frame(track)
        if not trajectory:
            return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
        reference_points.append([float(value) for value in reference])
        trajectories.append(trajectory)
    common = sorted(set.intersection(*[set(trajectory) for trajectory in trajectories]))
    if not common:
        return {"rigid_rmse_m": None, "per_frame_rmse_m": {}, "inlier_ratio": None}
    per_frame = {}
    rms_values = []
    for frame_index in common:
        target_points = [trajectory[frame_index] for trajectory in trajectories]
        rms = _fit_rigid_rms(reference_points, target_points)
        per_frame[str(frame_index)] = rms
        rms_values.append(rms)
    return {
        "rigid_rmse_m": sum(rms_values) / len(rms_values) if rms_values else None,
        "per_frame_rmse_m": per_frame,
        "inlier_ratio": (
            sum(1 for value in rms_values if value <= 0.03) / len(rms_values)
            if rms_values
            else None
        ),
    }


def _track_trajectory_by_frame(track: dict[str, Any]) -> dict[int, list[float]]:
    out = {}
    for sample in track.get("samples", []):
        if (
            isinstance(sample, dict)
            and bool(sample.get("visible", False))
            and bool(sample.get("depth_valid", True))
            and isinstance(sample.get("xyz_world"), list)
            and len(sample["xyz_world"]) == 3
        ):
            out[int(sample.get("frame_index", 0))] = [float(value) for value in sample["xyz_world"]]
    return out


def _fit_rigid_rms(source_points: list[list[float]], target_points: list[list[float]]) -> float:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return float("nan")
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape[0] < 3:
        return float("inf")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    covariance = (source - source_centroid).T @ (target - target_centroid)
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_centroid - rotation @ source_centroid
    predicted = (rotation @ source.T).T + translation
    return float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))


def _track_endpoint_motion(track: dict[str, Any]) -> float:
    samples = [
        sample
        for sample in track.get("samples", [])
        if isinstance(sample, dict)
        and bool(sample.get("visible", False))
        and bool(sample.get("depth_valid", True))
        and isinstance(sample.get("xyz_world"), list)
        and len(sample.get("xyz_world")) == 3
    ]
    if len(samples) < 2:
        return 0.0
    samples = sorted(samples, key=lambda item: int(item.get("frame_index", 0)))
    start = [float(value) for value in samples[0]["xyz_world"]]
    end = [float(value) for value in samples[-1]["xyz_world"]]
    return _norm(_sub(end, start))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _object_mask_summary(
    *,
    predicted_joints: list[dict[str, Any]],
    gt_priors: dict[int, dict[str, Any]],
    directed_matches: list[dict[str, Any]],
    undirected_matches: list[dict[str, Any]],
    reverse_matches: list[dict[str, Any]],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    directed_type_correct = [
        bool(item.get("joint_type_correct", False))
        for item in directed_matches
        if "joint_type_correct" in item
    ]
    geometry_matches = [
        item for item in directed_matches if bool(item.get("joint_type_correct", False))
    ]
    directed_axis_errors = _finite_values(
        item.get("axis_angle_error_deg") for item in geometry_matches
    )
    directed_pivot_errors = _finite_values(item.get("pivot_error_m") for item in geometry_matches)
    return {
        "predicted_joint_count": len(predicted_joints),
        "ground_truth_joint_count": len(gt_priors),
        "directed_matched_joint_count": len(directed_matches),
        "undirected_matched_joint_count": len(undirected_matches),
        "reverse_directed_matched_joint_count": len(reverse_matches),
        "directed_joint_coverage": len(directed_matches) / max(1, len(gt_priors)),
        "undirected_joint_coverage": len(undirected_matches) / max(1, len(gt_priors)),
        "reverse_directed_joint_coverage": len(reverse_matches) / max(1, len(gt_priors)),
        "joint_type_accuracy": (
            sum(1 for value in directed_type_correct if value) / len(directed_type_correct)
            if directed_type_correct
            else None
        ),
        "axis_angle_error_deg_mean": _mean_or_none(directed_axis_errors),
        "axis_angle_error_deg_median": _median_or_none(directed_axis_errors),
        "axis_angle_error_deg_max": max(directed_axis_errors) if directed_axis_errors else None,
        "axis_evaluable_joint_count": len(directed_axis_errors),
        "pivot_error_m_mean": _mean_or_none(directed_pivot_errors),
        "pivot_error_m_median": _median_or_none(directed_pivot_errors),
        "pivot_error_m_max": max(directed_pivot_errors) if directed_pivot_errors else None,
        "by_ground_truth_joint_type": _summary_by_joint_type(directed_matches),
        "mean_cluster_purity": overlap.get("mean_purity"),
        "mean_gt_coverage": overlap.get("mean_gt_coverage"),
        "largest_cluster_ratio": overlap.get("largest_cluster_ratio"),
        "failure_reason_counts": _failure_reason_counts(directed_matches),
    }


def _joint_failure_reason_guess(joint: dict[str, Any]) -> str | None:
    if not bool(joint.get("directed_matched", False)):
        if bool(joint.get("reverse_directed_matched", False)):
            return "parent_child_direction_reversed"
        if bool(joint.get("undirected_matched", False)):
            return "joint_pair_found_but_direction_or_gt_mapping_mismatch"
        return "joint_pair_not_matched_to_gt"

    child_debug = joint.get("child_cluster_debug")
    child_purity = _debug_float(child_debug, "gt_purity")
    child_rigid_rmse = _debug_float(child_debug, "rigid_rmse_m")
    child_inlier_ratio = _debug_float(child_debug, "inlier_ratio")
    axis_error = _maybe_float(joint.get("axis_angle_error_deg"))
    pivot_error = _maybe_float(joint.get("pivot_error_m"))

    mixed = child_purity is not None and child_purity < 0.7
    nonrigid = child_rigid_rmse is not None and child_rigid_rmse > 0.03
    low_inlier = child_inlier_ratio is not None and child_inlier_ratio < 0.5
    high_pivot = pivot_error is not None and pivot_error > 0.1
    low_axis = axis_error is not None and axis_error < 15.0

    if mixed and (nonrigid or low_inlier):
        return "child_cluster_mixed_or_nonrigid"
    if high_pivot and low_axis and (mixed or nonrigid or low_inlier):
        return "pivot_sensitive_to_child_cluster_outliers"
    if high_pivot and low_axis:
        return "pivot_offset_with_good_axis"
    if nonrigid:
        return "noisy_or_nonrigid_child_tracks"
    if mixed:
        return "semantically_mixed_child_cluster"
    return None


def _joint_cleanup_recommendation(joint: dict[str, Any]) -> str | None:
    reason = joint.get("failure_reason_guess")
    if reason in {"child_cluster_mixed_or_nonrigid", "pivot_sensitive_to_child_cluster_outliers"}:
        return "local_split_child_cluster_then_refit_part_pose"
    if reason == "noisy_or_nonrigid_child_tracks":
        return "robust_track_outlier_filter_then_refit_part_pose"
    if reason == "semantically_mixed_child_cluster":
        return "inspect_cluster_gt_composition_and_consider_local_split"
    if reason == "pivot_offset_with_good_axis":
        return "refit_pivot_with_robust_child_track_subset"
    return None


def _failure_reason_counts(joints: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for joint in joints:
        reason = joint.get("failure_reason_guess")
        if not isinstance(reason, str) or not reason:
            reason = "none"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _debug_float(debug: Any, key: str) -> float | None:
    if not isinstance(debug, dict):
        return None
    return _maybe_float(debug.get(key))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _write_object_mask_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary", {})
    fields = [
        "joint_inference_path",
        "matching_metric",
        "predicted_joint_count",
        "ground_truth_joint_count",
        "directed_joint_coverage",
        "undirected_joint_coverage",
        "reverse_directed_joint_coverage",
        "joint_type_accuracy",
        "axis_angle_error_deg_mean",
        "pivot_error_m_mean",
        "mean_cluster_purity",
        "mean_gt_coverage",
        "largest_cluster_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "joint_inference_path": payload.get("joint_inference_path"),
                "matching_metric": payload.get("matching_metric"),
                **{field: summary.get(field) for field in fields if field not in {"joint_inference_path", "matching_metric"}},
            }
        )


def _best_item(values: dict[int, int]) -> tuple[int | None, int]:
    if not values:
        return None, 0
    key = max(sorted(values), key=lambda item: values[item])
    return key, values[key]


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _parse_vec3(raw: Any, fallback: list[float]) -> list[float]:
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    return list(fallback)


def _parse_limits(raw: Any) -> list[float] | None:
    if isinstance(raw, list) and len(raw) == 2:
        return [float(raw[0]), float(raw[1])]
    return None


def _axis_angle_error_deg(axis_a: list[float], axis_b: list[float]) -> float:
    a = _normalize(axis_a, fallback=[0.0, 0.0, 1.0])
    b = _normalize(axis_b, fallback=[0.0, 0.0, 1.0])
    cosine = max(-1.0, min(1.0, abs(_dot(a, b))))
    return math.degrees(math.acos(cosine))


def _line_distance(point_a: list[float], axis_a: list[float], point_b: list[float], axis_b: list[float]) -> float:
    a = _normalize(axis_a, fallback=[0.0, 0.0, 1.0])
    b = _normalize(axis_b, fallback=[0.0, 0.0, 1.0])
    delta = _sub(point_b, point_a)
    normal = _cross(a, b)
    normal_norm = _norm(normal)
    if normal_norm < 1e-9:
        return _norm(_cross(delta, a))
    return abs(_dot(delta, normal)) / normal_norm


def _joint_position_error(
    pred_pivot: list[float],
    pred_axis: list[float],
    predicted_type: str,
    gt_pivot: list[float],
    gt_axis: list[float],
    gt_type: str,
) -> float | None:
    pred_kind = _canonical_joint_type(predicted_type)
    gt_kind = _canonical_joint_type(gt_type)
    if pred_kind == "revolute" or gt_kind == "revolute":
        return _line_distance(pred_pivot, pred_axis, gt_pivot, gt_axis)
    if pred_kind == "prismatic" and gt_kind == "prismatic":
        return None
    return None


def _canonical_joint_type(joint_type: str) -> str:
    if joint_type in {"hinge", "revolute", "continuous"}:
        return "revolute"
    if joint_type in {"slide", "slider", "prismatic"}:
        return "prismatic"
    return joint_type


def _limit_error(predicted: list[float] | None, ground_truth: list[float] | None) -> dict[str, float] | None:
    if predicted is None or ground_truth is None:
        return None
    return {
        "lower_abs_error": abs(predicted[0] - ground_truth[0]),
        "upper_abs_error": abs(predicted[1] - ground_truth[1]),
        "span_abs_error": abs((predicted[1] - predicted[0]) - (ground_truth[1] - ground_truth[0])),
    }


def _q_error_metrics(samples: Any, q_by_frame: Any) -> dict[str, Any] | None:
    if not isinstance(samples, list) or not isinstance(q_by_frame, dict):
        return None
    pairs: list[tuple[float, float]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        frame_index = int(sample.get("frame_index", -1))
        if frame_index not in q_by_frame:
            continue
        pairs.append((float(sample.get("q", 0.0)), float(q_by_frame[frame_index])))
    if not pairs:
        return {"sample_count": 0}
    errors = [predicted - gt for predicted, gt in pairs]
    offset = sum(errors) / len(errors)
    aligned_errors = [error - offset for error in errors]
    return {
        "sample_count": len(pairs),
        "q_rmse": _rmse(errors),
        "q_mae": _mae(errors),
        "q_rmse_offset_aligned": _rmse(aligned_errors),
        "q_mae_offset_aligned": _mae(aligned_errors),
        "best_constant_offset": offset,
    }


def _finite_values(values: Any) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            out.append(numeric)
    return out


def _summary_by_joint_type(matched: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    joint_types = sorted({str(item.get("ground_truth_joint_type", "unknown")) for item in matched})
    for joint_type in joint_types:
        rows = [item for item in matched if str(item.get("ground_truth_joint_type", "unknown")) == joint_type]
        type_correct = [
            bool(item.get("joint_type_correct", False))
            for item in rows
            if "joint_type_correct" in item
        ]
        geometry_rows = [item for item in rows if bool(item.get("joint_type_correct", False))]
        axis_errors = _finite_values(item.get("axis_angle_error_deg") for item in geometry_rows)
        pivot_errors = _finite_values(item.get("pivot_error_m") for item in geometry_rows)
        q_rmse_aligned = []
        for item in rows:
            q_error = item.get("q_error")
            if isinstance(q_error, dict):
                q_rmse_aligned.extend(_finite_values([q_error.get("q_rmse_offset_aligned")]))
        out[joint_type] = {
            "joint_count": len(rows),
            "joint_type_accuracy": (
                sum(1 for value in type_correct if value) / len(type_correct)
                if type_correct
                else None
            ),
            "axis_angle_error_deg_mean": _mean_or_none(axis_errors),
            "axis_angle_error_deg_median": _median_or_none(axis_errors),
            "axis_angle_error_deg_max": max(axis_errors) if axis_errors else None,
            "axis_evaluable_joint_count": len(axis_errors),
            "pivot_error_m_mean": _mean_or_none(pivot_errors),
            "pivot_error_m_median": _median_or_none(pivot_errors),
            "pivot_error_m_max": max(pivot_errors) if pivot_errors else None,
            "q_rmse_offset_aligned_mean": _mean_or_none(q_rmse_aligned),
        }
    return out


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return _median(values) if values else None


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / max(1, len(errors)))


def _mae(errors: list[float]) -> float:
    return sum(abs(error) for error in errors) / max(1, len(errors))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
