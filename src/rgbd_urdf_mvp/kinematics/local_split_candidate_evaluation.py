from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from ..perception.part_tracking import TrackPartPoseEstimationConfig, TrackPartPoseEstimator
from .evaluation import ObjectMaskKinematicEvaluationConfig, ObjectMaskKinematicEvaluator
from .joint_inference import JointInferenceConfig, JointInferencer


@dataclass(slots=True)
class LocalSplitCandidateEvaluationConfig:
    local_split_summary: str | Path
    output_dir: str | Path | None = None
    output_json: str | Path | None = None
    output_csv: str | Path | None = None
    min_tracks_per_part: int = 4
    mujoco_prior: str = "off"
    robust_track_model_trim_ratio: float = 0.0
    matching_metric: str = "iou"


class LocalSplitCandidateEvaluator:
    """Run downstream pose/joint diagnostics for local split proposals."""

    def evaluate(self, config: LocalSplitCandidateEvaluationConfig) -> Path:
        summary_path = Path(config.local_split_summary).expanduser().resolve()
        summary = load_json(summary_path)
        output_dir = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir is not None
            else summary_path.parent / "candidate_joint_evaluation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        artifacts = []
        for index, candidate in enumerate(summary.get("candidates", [])):
            if not isinstance(candidate, dict) or candidate.get("status") != "written":
                continue
            candidate_track_path = Path(str(candidate.get("output_json", ""))).expanduser()
            if not candidate_track_path.is_absolute():
                candidate_track_path = summary_path.parent / candidate_track_path
            if not candidate_track_path.exists():
                rows.append(self._missing_candidate_row(index, candidate, candidate_track_path))
                continue
            stem = candidate_track_path.stem
            part_pose_path = output_dir / f"{stem}.part_poses.json"
            joint_path = output_dir / f"{stem}.joint_inference.json"
            eval_path = output_dir / f"{stem}.object_mask_eval.json"
            eval_csv_path = output_dir / f"{stem}.object_mask_eval.csv"

            TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(
                    input_path=candidate_track_path,
                    output_json=part_pose_path,
                    min_tracks_per_part=max(3, int(config.min_tracks_per_part)),
                )
            ).estimate()
            JointInferencer(
                JointInferenceConfig(
                    input_path=part_pose_path,
                    output_json=joint_path,
                    mujoco_prior=config.mujoco_prior,
                    robust_track_model_trim_ratio=max(0.0, min(0.8, float(config.robust_track_model_trim_ratio))),
                )
            ).infer()

            object_eval = None
            try:
                ObjectMaskKinematicEvaluator().evaluate(
                    ObjectMaskKinematicEvaluationConfig(
                        joint_inference_path=joint_path,
                        part_pose_path=part_pose_path,
                        output_json=eval_path,
                        output_csv=eval_csv_path,
                        matching_metric=config.matching_metric,
                    )
                )
                object_eval = load_json(eval_path)
            except Exception as exc:
                object_eval = {"error": str(exc)}

            joint_artifact = load_json(joint_path)
            candidate_rows = self._candidate_rows(
                index=index,
                candidate=candidate,
                candidate_track_path=candidate_track_path,
                part_pose_path=part_pose_path,
                joint_path=joint_path,
                object_eval_path=eval_path if object_eval and "error" not in object_eval else None,
                joint_artifact=joint_artifact,
                object_eval=object_eval,
            )
            rows.extend(candidate_rows)
            artifacts.append(
                {
                    "candidate_index": index,
                    "track_path": str(candidate_track_path),
                    "part_pose_path": str(part_pose_path),
                    "joint_inference_path": str(joint_path),
                    "object_mask_eval_path": str(eval_path) if object_eval and "error" not in object_eval else None,
                    "object_mask_eval_error": object_eval.get("error") if isinstance(object_eval, dict) else None,
                }
            )

        output_json = (
            Path(config.output_json).expanduser().resolve()
            if config.output_json is not None
            else output_dir / "local_split_candidate_evaluation.json"
        )
        payload = {
            "source": "local-split-candidate-downstream-evaluation",
            "local_split_summary": str(summary_path),
            "output_dir": str(output_dir),
            "row_count": len(rows),
            "rows": rows,
            "artifacts": artifacts,
            "notes": [
                "selection_score_no_gt is a soft proposal score and does not use GT labels.",
                "joint_selection_score_no_gt combines proposal quality, saturated replay score, and anti-degeneracy terms.",
                "joint_replay_error_m alone can prefer near-static degenerate candidates; use it with motion and coverage terms.",
                "diagnostic_axis_error_gt and diagnostic_pivot_error_gt are only available in simulation diagnostics.",
                "axis_stability_bootstrap and pivot_stability_bootstrap are reserved for a later bootstrap pass.",
            ],
        }
        save_json(payload, output_json)
        output_csv = (
            Path(config.output_csv).expanduser().resolve()
            if config.output_csv is not None
            else output_json.with_suffix(".csv")
        )
        _write_candidate_csv(output_csv, rows)
        return output_json

    def _candidate_rows(
        self,
        *,
        index: int,
        candidate: dict[str, Any],
        candidate_track_path: Path,
        part_pose_path: Path,
        joint_path: Path,
        object_eval_path: Path | None,
        joint_artifact: dict[str, Any],
        object_eval: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        joints_by_child = {
            int(joint.get("child_part_id", -1)): joint
            for joint in joint_artifact.get("joints", [])
            if isinstance(joint, dict)
        }
        eval_by_child = {}
        if isinstance(object_eval, dict):
            for row in object_eval.get("per_joint", []):
                if isinstance(row, dict):
                    try:
                        eval_by_child[int(row.get("child_part_id"))] = row
                    except (TypeError, ValueError):
                        continue

        rows = []
        for cluster in candidate.get("after", {}).get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            part_id = int(cluster.get("part_id", -1))
            joint = joints_by_child.get(part_id, {})
            eval_row = eval_by_child.get(part_id, {})
            comparison = joint.get("metrics", {}).get("track_model_comparison")
            selected_type = _selected_replay_type(comparison)
            replay_error = _selected_replay_rmse(comparison, selected_type)
            selection = cluster.get("candidate_selection", {})
            joint_selection = _joint_selection_scores(
                selection=selection,
                replay_error=replay_error,
                mean_motion_m=cluster.get("mean_motion_m"),
            )
            rows.append(
                {
                    "candidate_id": f"{candidate.get('split_cluster_id')}_{candidate.get('ablation')}_K{candidate.get('local_k')}_part{part_id}",
                    "candidate_index": index,
                    "split_cluster_id": candidate.get("split_cluster_id"),
                    "local_k": candidate.get("local_k"),
                    "ablation": candidate.get("ablation"),
                    "part_id": part_id,
                    "track_count": cluster.get("track_count"),
                    "bbox_diag_m": cluster.get("bbox_diag_m"),
                    "mean_motion_m": cluster.get("mean_motion_m"),
                    "median_motion_m": cluster.get("median_motion_m"),
                    "rigid_rmse_m": cluster.get("rigid_rmse_m"),
                    "inlier_ratio": cluster.get("inlier_ratio"),
                    "visible_frame_ratio": cluster.get("visible_frame_ratio"),
                    "gt_purity": cluster.get("gt_purity"),
                    "selection_score_no_gt": selection.get("selection_score_no_gt"),
                    "diagnostic_score_with_gt": selection.get("diagnostic_score_with_gt"),
                    "base_like_low_motion": selection.get("base_like_low_motion"),
                    "joint_type": joint.get("joint_type"),
                    "joint_confidence": joint.get("confidence"),
                    "joint_replay_error_m": replay_error,
                    **joint_selection,
                    "joint_replay_selected_type": selected_type,
                    "joint_replay_decision_ratio": _nested(comparison, "decision_ratio"),
                    "axis_stability_bootstrap": None,
                    "pivot_stability_bootstrap": None,
                    "diagnostic_axis_error_gt": eval_row.get("axis_angle_error_deg"),
                    "diagnostic_pivot_error_gt": eval_row.get("pivot_error_m"),
                    "diagnostic_directed_matched": eval_row.get("directed_matched"),
                    "diagnostic_undirected_matched": eval_row.get("undirected_matched"),
                    "failure_reason_guess": eval_row.get("failure_reason_guess"),
                    "track_path": str(candidate_track_path),
                    "part_pose_path": str(part_pose_path),
                    "joint_inference_path": str(joint_path),
                    "object_mask_eval_path": str(object_eval_path) if object_eval_path is not None else None,
                }
            )
        return rows

    def _missing_candidate_row(self, index: int, candidate: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "candidate_id": f"{candidate.get('split_cluster_id')}_{candidate.get('ablation')}_K{candidate.get('local_k')}_missing",
            "candidate_index": index,
            "split_cluster_id": candidate.get("split_cluster_id"),
            "local_k": candidate.get("local_k"),
            "ablation": candidate.get("ablation"),
            "error": f"candidate track artifact not found: {path}",
        }


def _selected_replay_type(comparison: Any) -> str | None:
    if not isinstance(comparison, dict):
        return None
    selected = comparison.get("selected_type")
    return str(selected) if isinstance(selected, str) else None


def _selected_replay_rmse(comparison: Any, selected_type: str | None) -> float | None:
    if not isinstance(comparison, dict) or selected_type not in {"revolute", "prismatic"}:
        return None
    payload = comparison.get(selected_type)
    if not isinstance(payload, dict):
        return None
    value = payload.get("rmse_m")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(payload: Any, key: str) -> Any:
    return payload.get(key) if isinstance(payload, dict) else None


def _joint_selection_scores(*, selection: Any, replay_error: float | None, mean_motion_m: Any) -> dict[str, Any]:
    payload = selection if isinstance(selection, dict) else {}
    cluster_quality = _as_float(payload.get("selection_score_no_gt"), default=0.0)
    track_score = _as_float(payload.get("track_count_score"), default=0.0)
    bbox_score = _as_float(payload.get("bbox_score"), default=0.0)
    motion_score = _as_float(payload.get("motion_score"), default=0.0)
    inlier_score = _as_float(payload.get("inlier_score"), default=0.0)
    visibility_score = _as_float(payload.get("visibility_score"), default=0.0)
    base_like = bool(payload.get("base_like_low_motion", False))
    replay_score = math.exp(-max(0.0, float(replay_error)) / 0.05) if replay_error is not None else 0.0
    mean_motion = _as_unbounded_float(mean_motion_m, default=0.0)
    motion_normalized_replay = (
        float(replay_error) / max(1e-6, mean_motion)
        if replay_error is not None
        else None
    )

    degeneracy_penalty = 0.0
    degeneracy_penalty += 0.35 * (1.0 - track_score)
    degeneracy_penalty += 0.25 * (1.0 - bbox_score)
    degeneracy_penalty += 0.30 * (1.0 - motion_score)
    if base_like:
        degeneracy_penalty += 0.35
    degeneracy_penalty = max(0.0, min(1.0, degeneracy_penalty))

    joint_score = (
        0.25 * cluster_quality
        + 0.20 * replay_score
        + 0.15 * motion_score
        + 0.15 * bbox_score
        + 0.10 * track_score
        + 0.10 * inlier_score
        + 0.05 * visibility_score
        - 0.20 * degeneracy_penalty
        - (0.25 if base_like else 0.0)
    )
    return {
        "joint_selection_score_no_gt": max(0.0, min(1.0, float(joint_score))),
        "joint_replay_score": float(replay_score),
        "motion_normalized_replay_error": motion_normalized_replay,
        "degeneracy_penalty": float(degeneracy_penalty),
        "parent_child_motion_contrast": None,
    }


def _as_unbounded_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _as_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return max(0.0, min(1.0, numeric))


def _write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "split_cluster_id",
        "local_k",
        "ablation",
        "part_id",
        "selection_score_no_gt",
        "joint_selection_score_no_gt",
        "track_count",
        "bbox_diag_m",
        "mean_motion_m",
        "rigid_rmse_m",
        "inlier_ratio",
        "visible_frame_ratio",
        "joint_type",
        "joint_replay_error_m",
        "joint_replay_score",
        "motion_normalized_replay_error",
        "degeneracy_penalty",
        "parent_child_motion_contrast",
        "axis_stability_bootstrap",
        "pivot_stability_bootstrap",
        "diagnostic_axis_error_gt",
        "diagnostic_pivot_error_gt",
        "failure_reason_guess",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
