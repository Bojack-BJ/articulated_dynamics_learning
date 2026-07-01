from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from .joint_inference import _dot, _load_mujoco_joint_priors, _norm, _normalize


@dataclass(slots=True)
class KinematicModelEvaluationConfig:
    joint_inference_path: str | Path
    part_pose_path: str | Path | None = None
    output_json: str | Path | None = None


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

        predicted_type = str(predicted.get("joint_type", "unknown"))
        gt_type = str(prior.get("joint_type", "unknown"))
        pred_axis = _parse_vec3(predicted.get("axis"), [0.0, 0.0, 1.0])
        gt_axis = _parse_vec3(prior.get("axis"), [0.0, 0.0, 1.0])
        pred_pivot = _parse_vec3(predicted.get("pivot"), [0.0, 0.0, 0.0])
        gt_pivot = _parse_vec3(prior.get("pivot"), [0.0, 0.0, 0.0])
        pred_limits = _parse_limits(predicted.get("limits"))
        gt_limits = _parse_limits(prior.get("limits"))

        axis_error_deg = _axis_angle_error_deg(pred_axis, gt_axis)
        pivot_error_m = (
            _line_distance(pred_pivot, pred_axis, gt_pivot, gt_axis)
            if predicted_type == "revolute" or gt_type == "revolute"
            else _norm(_sub(pred_pivot, gt_pivot))
        )
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
        axis_errors = _finite_values(item.get("axis_angle_error_deg") for item in matched)
        pivot_errors = _finite_values(item.get("pivot_error_m") for item in matched)
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
            "axis_angle_error_deg_max": max(axis_errors) if axis_errors else None,
            "pivot_error_m_mean": _mean_or_none(pivot_errors),
            "pivot_error_m_max": max(pivot_errors) if pivot_errors else None,
            "limit_abs_error_mean": _mean_or_none(limit_abs_errors),
            "q_rmse_mean": _mean_or_none(q_rmse),
            "q_rmse_offset_aligned_mean": _mean_or_none(q_rmse_aligned),
            "by_ground_truth_joint_type": _summary_by_joint_type(matched),
        }


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
        axis_errors = _finite_values(item.get("axis_angle_error_deg") for item in rows)
        pivot_errors = _finite_values(item.get("pivot_error_m") for item in rows)
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
            "axis_angle_error_deg_max": max(axis_errors) if axis_errors else None,
            "pivot_error_m_mean": _mean_or_none(pivot_errors),
            "pivot_error_m_max": max(pivot_errors) if pivot_errors else None,
            "q_rmse_offset_aligned_mean": _mean_or_none(q_rmse_aligned),
        }
    return out


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


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
