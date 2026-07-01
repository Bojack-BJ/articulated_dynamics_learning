from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from .evaluation import _axis_angle_error_deg, _line_distance, _mean_or_none


@dataclass(slots=True)
class FeedforwardArticulationEvaluationConfig:
    feedforward_root: str | Path
    reference_evaluation: str | Path | None = None
    output_json: str | Path | None = None
    yaw_degrees: tuple[int, ...] = (0, 90, 180, 270)
    angle_score_weight: float = 1.0 / 180.0
    position_score_weight: float = 1.0


class FeedforwardArticulationEvaluator:
    """Re-evaluate PARTICULATE URDF joints against an existing GT reference.

    PARTICULATE may export multiple candidate joints for one generated mesh.
    Earlier ad-hoc evaluation used a single exported joint, which penalizes
    objects where the useful door hinge is present but not the first/selected
    URDF joint. This evaluator searches all exported URDF joints and all
    configured yaw alignments, then reports the candidate with the lowest
    combined axis-line position and axis-angle score.
    """

    def evaluate(self, config: FeedforwardArticulationEvaluationConfig) -> Path:
        root = Path(config.feedforward_root).expanduser().resolve()
        reference_path = (
            Path(config.reference_evaluation).expanduser().resolve()
            if config.reference_evaluation is not None
            else root / "_evaluation" / "gt_axis_position_evaluation_unified_scale.json"
        )
        reference = load_json(reference_path)
        gt_by_object = self._gt_by_object(reference)
        rows = []
        for object_id in sorted(gt_by_object):
            object_dir = root / object_id
            urdf_path = self._latest_particulate_urdf(object_dir)
            rows.append(
                self._evaluate_object(
                    object_id=object_id,
                    urdf_path=urdf_path,
                    gt=gt_by_object[object_id],
                    yaw_degrees=config.yaw_degrees,
                    angle_score_weight=float(config.angle_score_weight),
                    position_score_weight=float(config.position_score_weight),
                )
            )

        summary = self._summary(rows, reference)
        output_path = (
            Path(config.output_json).expanduser().resolve()
            if config.output_json is not None
            else root / "_evaluation" / "gt_axis_position_evaluation_unified_scale_best_joint.json"
        )
        save_json(
            {
                "source": "feedforward-articulation-best-joint-evaluation",
                "feedforward_root": str(root),
                "reference_evaluation": str(reference_path),
                "summary": summary,
                "optimized_tracking": reference.get("optimized_tracking", []),
                "feedforward_particulate_best_joint": rows,
                "notes": [
                    "Each PARTICULATE URDF revolute/prismatic joint is treated as a candidate.",
                    "For each candidate, yaw rotations around +Z are tested and the lowest combined score is selected.",
                    "Score = position_score_weight * normalized_axis_position_error + angle_score_weight * axis_angle_error_deg.",
                    "The GT axis/pivot and object normalization are copied from the reference evaluation artifact.",
                ],
            },
            output_path,
        )
        self._write_tsv(output_path.with_suffix(".tsv"), rows)
        return output_path

    def _gt_by_object(self, reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in reference.get("feedforward_particulate_upY", []):
            if not isinstance(row, dict):
                continue
            object_id = row.get("object_id")
            if not isinstance(object_id, str):
                continue
            gt_axis = _vec3(row.get("gt_axis"), [0.0, 0.0, 1.0])
            gt_pivot = _vec3(row.get("gt_pivot_normalized"), _vec3(row.get("gt_pivot"), [0.0, 0.0, 0.0]))
            out[object_id] = {
                "gt_joint_type": row.get("gt_joint_type", "revolute"),
                "gt_axis": gt_axis,
                "gt_pivot_normalized": gt_pivot,
            }
        if out:
            return out

        for row in reference.get("optimized_tracking", []):
            if not isinstance(row, dict):
                continue
            object_id = row.get("object_id")
            if not isinstance(object_id, str):
                continue
            out[object_id] = {
                "gt_joint_type": row.get("gt_joint_type", "revolute"),
                "gt_axis": _vec3(row.get("gt_axis"), [0.0, 0.0, 1.0]),
                "gt_pivot_normalized": _vec3(row.get("gt_pivot_normalized"), [0.0, 0.0, 0.0]),
            }
        if not out:
            raise ValueError("Reference evaluation does not contain GT rows.")
        return out

    def _latest_particulate_urdf(self, object_dir: Path) -> Path | None:
        candidates = sorted((object_dir / "particulate").glob("urdf_*/model.urdf"))
        return candidates[-1] if candidates else None

    def _evaluate_object(
        self,
        object_id: str,
        urdf_path: Path | None,
        gt: dict[str, Any],
        yaw_degrees: tuple[int, ...],
        angle_score_weight: float,
        position_score_weight: float,
    ) -> dict[str, Any]:
        gt_type = str(gt.get("gt_joint_type", "revolute"))
        gt_axis = _vec3(gt.get("gt_axis"), [0.0, 0.0, 1.0])
        gt_pivot = _vec3(gt.get("gt_pivot_normalized"), [0.0, 0.0, 0.0])
        base: dict[str, Any] = {
            "object_id": object_id,
            "path": "feedforward_particulate_best_joint",
            "gt_joint_type": gt_type,
            "gt_axis": gt_axis,
            "gt_pivot_normalized": gt_pivot,
            "source_path": str(urdf_path) if urdf_path is not None else None,
        }
        if urdf_path is None or not urdf_path.exists():
            return {**base, "matched": False, "reason": "missing_particulate_urdf", "candidate_count": 0}

        candidates = _parse_urdf_joint_candidates(urdf_path)
        scored = []
        for index, candidate in enumerate(candidates):
            if candidate["joint_type"] != gt_type:
                continue
            for yaw_deg in yaw_degrees:
                axis = _rotate_z(candidate["axis"], yaw_deg)
                pivot = _rotate_z(candidate["pivot"], yaw_deg)
                angle_error = _axis_angle_error_deg(axis, gt_axis)
                position_error = _line_distance(pivot, axis, gt_pivot, gt_axis)
                score = position_score_weight * position_error + angle_score_weight * angle_error
                scored.append(
                    {
                        "candidate_index": index,
                        "joint_name": candidate["joint_name"],
                        "parent_link": candidate["parent_link"],
                        "child_link": candidate["child_link"],
                        "joint_type": candidate["joint_type"],
                        "yaw_deg": yaw_deg,
                        "score": score,
                        "axis_angle_error_deg": angle_error,
                        "axis_position_error_normalized": position_error,
                        "axis_position_mse_normalized": position_error * position_error,
                        "predicted_axis": axis,
                        "predicted_pivot_normalized": pivot,
                        "raw_axis": candidate["axis"],
                        "raw_pivot": candidate["pivot"],
                        "limits": candidate.get("limits"),
                    }
                )
        if not scored:
            return {
                **base,
                "matched": False,
                "reason": "no_joint_candidate_with_matching_type",
                "candidate_count": len(candidates),
                "candidates": candidates,
            }

        best = min(scored, key=lambda item: (float(item["score"]), float(item["axis_position_error_normalized"])))
        return {
            **base,
            "matched": True,
            "predicted_joint_type": best["joint_type"],
            "joint_type_correct": best["joint_type"] == gt_type,
            "candidate_count": len(candidates),
            "matching_type_candidate_count": len(scored) // max(1, len(yaw_degrees)),
            "selected_candidate_index": best["candidate_index"],
            "selected_joint_name": best["joint_name"],
            "selected_parent_link": best["parent_link"],
            "selected_child_link": best["child_link"],
            "best_yaw_deg": best["yaw_deg"],
            "match_score": best["score"],
            "axis_angle_error_deg": best["axis_angle_error_deg"],
            "axis_position_error_normalized": best["axis_position_error_normalized"],
            "axis_position_mse_normalized": best["axis_position_mse_normalized"],
            "predicted_axis": best["predicted_axis"],
            "predicted_pivot_normalized": best["predicted_pivot_normalized"],
            "raw_predicted_axis": best["raw_axis"],
            "raw_predicted_pivot": best["raw_pivot"],
            "selected_limits": best.get("limits"),
            "candidate_errors": scored,
        }

    def _summary(self, rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
        matched = [row for row in rows if row.get("matched")]
        type_correct = [row for row in matched if row.get("joint_type_correct")]
        angle_errors = _finite(row.get("axis_angle_error_deg") for row in matched)
        position_errors = _finite(row.get("axis_position_error_normalized") for row in matched)
        position_mse = _finite(row.get("axis_position_mse_normalized") for row in matched)
        return {
            "reference_summary": reference.get("summary", {}),
            "feedforward_particulate_best_joint_normalized": {
                "object_count": len(rows),
                "matched_object_count": len(matched),
                "joint_type_accuracy": len(type_correct) / len(matched) if matched else None,
                "correct_type_count": len(type_correct),
                "axis_position_mse_mean": _mean_or_none(position_mse),
                "axis_position_rmse": math.sqrt(sum(position_mse) / len(position_mse)) if position_mse else None,
                "axis_position_error_mean": _mean_or_none(position_errors),
                "axis_angle_error_deg_mean": _mean_or_none(angle_errors),
                "axis_angle_error_deg_median": _median_or_none(angle_errors),
                "axis_position_unit": "bbox_normalized^2",
                "matching_policy": "best over all same-type URDF joints and configured yaw rotations",
            },
        }

    def _write_tsv(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        columns = [
            "object_id",
            "matched",
            "joint_type_correct",
            "candidate_count",
            "matching_type_candidate_count",
            "selected_candidate_index",
            "selected_joint_name",
            "selected_parent_link",
            "selected_child_link",
            "best_yaw_deg",
            "axis_angle_error_deg",
            "axis_position_error_normalized",
            "match_score",
            "source_path",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["\t".join(columns)]
        for row in rows:
            lines.append("\t".join(_format_tsv(row.get(column)) for column in columns))
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_urdf_joint_candidates(urdf_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    joints = [joint for joint in root.findall("joint")]
    parent_names = {
        parent.get("link")
        for joint in joints
        for parent in [joint.find("parent")]
        if parent is not None and parent.get("link")
    }
    child_names = {
        child.get("link")
        for joint in joints
        for child in [joint.find("child")]
        if child is not None and child.get("link")
    }
    roots = sorted(parent_names - child_names)
    link_transforms: dict[str, tuple[list[list[float]], list[float]]] = {
        root_name: (_identity3(), [0.0, 0.0, 0.0]) for root_name in roots
    }
    pending = list(joints)
    while pending:
        progressed = False
        next_pending = []
        for joint in pending:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_link = parent.get("link")
            child_link = child.get("link")
            if not parent_link or not child_link or parent_link not in link_transforms:
                next_pending.append(joint)
                continue
            parent_rotation, parent_translation = link_transforms[parent_link]
            origin = joint.find("origin")
            local_translation = _parse_xyz(origin.get("xyz") if origin is not None else None)
            local_rotation = _rpy_to_matrix(origin.get("rpy") if origin is not None else None)
            world_rotation = _matmul3(parent_rotation, local_rotation)
            world_translation = _add(parent_translation, _matvec3(parent_rotation, local_translation))
            link_transforms[child_link] = (world_rotation, world_translation)
            progressed = True
        if not progressed:
            for joint in next_pending:
                child = joint.find("child")
                if child is not None and child.get("link"):
                    link_transforms.setdefault(child.get("link", ""), (_identity3(), [0.0, 0.0, 0.0]))
            break
        pending = next_pending

    candidates = []
    for joint in joints:
        joint_type = joint.get("type", "fixed")
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        parent_link = parent.get("link") if parent is not None else ""
        child_link = child.get("link") if child is not None else ""
        parent_rotation, parent_translation = link_transforms.get(parent_link, (_identity3(), [0.0, 0.0, 0.0]))
        origin = joint.find("origin")
        local_translation = _parse_xyz(origin.get("xyz") if origin is not None else None)
        local_rotation = _rpy_to_matrix(origin.get("rpy") if origin is not None else None)
        axis_el = joint.find("axis")
        local_axis = _parse_xyz(axis_el.get("xyz") if axis_el is not None else None, fallback=[0.0, 0.0, 1.0])
        origin_rotation = _matmul3(parent_rotation, local_rotation)
        axis_world = _normalize(_matvec3(origin_rotation, local_axis))
        pivot_world = _add(parent_translation, _matvec3(parent_rotation, local_translation))
        limit = joint.find("limit")
        candidates.append(
            {
                "joint_name": joint.get("name", ""),
                "parent_link": parent_link,
                "child_link": child_link,
                "joint_type": "revolute" if joint_type == "continuous" else joint_type,
                "axis": axis_world,
                "pivot": pivot_world,
                "limits": _parse_limit(limit),
            }
        )
    return candidates


def _parse_limit(limit: ET.Element | None) -> list[float] | None:
    if limit is None or limit.get("lower") is None or limit.get("upper") is None:
        return None
    return [float(limit.get("lower", "0")), float(limit.get("upper", "0"))]


def _parse_xyz(raw: str | None, fallback: list[float] | None = None) -> list[float]:
    if fallback is None:
        fallback = [0.0, 0.0, 0.0]
    if raw is None:
        return list(fallback)
    values = [float(item) for item in raw.split()]
    return values if len(values) == 3 else list(fallback)


def _rpy_to_matrix(raw: str | None) -> list[list[float]]:
    if raw is None:
        return _identity3()
    values = [float(item) for item in raw.split()]
    if len(values) != 3:
        return _identity3()
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotate_z(vec: list[float], yaw_deg: int | float) -> list[float]:
    rad = math.radians(float(yaw_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return [c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]]


def _vec3(raw: Any, fallback: list[float]) -> list[float]:
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    return list(fallback)


def _finite(values: Any) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            out.append(numeric)
    return out


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


def _identity3() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _matvec3(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vec[col] for col in range(3)) for row in range(3)]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _normalize(vec: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vec))
    if length < 1e-9:
        return [0.0, 0.0, 1.0]
    return [value / length for value in vec]


def _format_tsv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
