#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.kinematics.evaluation import KinematicModelEvaluationConfig, KinematicModelEvaluator
from rgbd_urdf_mvp.kinematics.joint_inference import JointInferenceConfig, JointInferencer


VARIANTS = (
    ("pose_gated", True),
    ("track_model_first", False),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pose-gated and track-model-first joint type selection on existing part tracks."
    )
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--optimized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rotation-threshold-rad", type=float, default=0.90)
    parser.add_argument("--translation-threshold-m", type=float, default=0.02)
    parser.add_argument("--track-residual-decision-ratio", type=float, default=0.85)
    parser.add_argument("--min-track-residual-samples", type=int, default=20)
    parser.add_argument("--min-track-residual-tracks", type=int, default=12)
    args = parser.parse_args()

    batch_config = args.batch_config.expanduser().resolve()
    optimized_root = args.optimized_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for object_id in _object_ids(batch_config):
        pointcloud_dir = optimized_root / object_id / "pointcloud_4d_partseg"
        part_poses = pointcloud_dir / "part_poses.json"
        if not part_poses.exists():
            rows.append({"object_id": object_id, "status": "missing_part_poses"})
            continue

        for variant, requires_pose_candidate in VARIANTS:
            variant_dir = output_dir / variant / object_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            joint_path = variant_dir / "joint_inference.json"
            eval_path = variant_dir / "kinematic_evaluation.json"

            infer_t0 = time.perf_counter()
            JointInferencer(
                JointInferenceConfig(
                    input_path=part_poses,
                    output_json=joint_path,
                    rotation_threshold_rad=float(args.rotation_threshold_rad),
                    translation_threshold_m=float(args.translation_threshold_m),
                    use_track_translation_axis=True,
                    use_track_residual_type=True,
                    track_residual_requires_pose_candidate=requires_pose_candidate,
                    track_residual_decision_ratio=float(args.track_residual_decision_ratio),
                    min_track_residual_samples=max(1, int(args.min_track_residual_samples)),
                    min_track_residual_tracks=max(1, int(args.min_track_residual_tracks)),
                    mujoco_prior="off",
                )
            ).infer()
            infer_s = time.perf_counter() - infer_t0

            eval_t0 = time.perf_counter()
            KinematicModelEvaluator().evaluate(
                KinematicModelEvaluationConfig(
                    joint_inference_path=joint_path,
                    part_pose_path=part_poses,
                    output_json=eval_path,
                )
            )
            eval_s = time.perf_counter() - eval_t0

            joint_artifact = _read_json(joint_path)
            eval_artifact = _read_json(eval_path)
            summary = eval_artifact.get("summary", {}) if isinstance(eval_artifact, dict) else {}
            decision_counts = _decision_counts(joint_artifact)
            rows.append(
                {
                    "object_id": object_id,
                    "variant": variant,
                    "status": "ok",
                    "track_residual_requires_pose_candidate": requires_pose_candidate,
                    "infer_s": infer_s,
                    "eval_s": eval_s,
                    "total_s": infer_s + eval_s,
                    "joint_type_accuracy": summary.get("joint_type_accuracy"),
                    "matched_joint_count": summary.get("matched_joint_count"),
                    "ground_truth_joint_count": summary.get("ground_truth_joint_count"),
                    "axis_angle_error_deg_mean": summary.get("axis_angle_error_deg_mean"),
                    "q_rmse_offset_aligned_mean": summary.get("q_rmse_offset_aligned_mean"),
                    "decision_counts": decision_counts,
                    "joint_inference_path": str(joint_path),
                    "kinematic_evaluation_path": str(eval_path),
                }
            )
            joint_rows.extend(_per_joint_rows(object_id, variant, eval_artifact, joint_artifact))

    payload = {
        "source": "joint-type-selection-ablation",
        "batch_config": str(batch_config),
        "optimized_root": str(optimized_root),
        "parameters": {
            "rotation_threshold_rad": float(args.rotation_threshold_rad),
            "translation_threshold_m": float(args.translation_threshold_m),
            "track_residual_decision_ratio": float(args.track_residual_decision_ratio),
            "min_track_residual_samples": int(args.min_track_residual_samples),
            "min_track_residual_tracks": int(args.min_track_residual_tracks),
        },
        "summary": {
            variant: _summarize([row for row in rows if row.get("variant") == variant])
            for variant, _ in VARIANTS
        },
        "delta_track_model_first_minus_pose_gated": _delta_summary(rows),
        "rows": rows,
        "per_joint": joint_rows,
        "notes": [
            "pose_gated reproduces the older behavior: track residual can decide type only when the pose threshold produced the same candidate.",
            "track_model_first lets decisive 3D track replay residual choose revolute/prismatic without requiring the pose-derived candidate gate.",
            "Both variants reuse the same part_poses.json and part_tracks.json; timings cover only joint inference and kinematic evaluation.",
        ],
    }
    (output_dir / "joint_type_ablation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_rows_tsv(output_dir / "joint_type_ablation_per_object.tsv", rows)
    _write_joint_tsv(output_dir / "joint_type_ablation_per_joint.tsv", joint_rows)
    _write_markdown(output_dir / "joint_type_ablation_summary.md", payload)
    print(json.dumps({"joint_type_ablation_summary": str((output_dir / "joint_type_ablation_summary.json").resolve())}, indent=2))
    return 0


def _object_ids(batch_config: Path) -> list[str]:
    out = []
    for line in batch_config.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) >= 3:
            out.append(fields[2])
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_counts(joint_artifact: dict[str, Any]) -> dict[str, int]:
    out = {
        "joint_count": 0,
        "track_model_comparison_count": 0,
        "residual_decisive_count": 0,
        "decision_applied_count": 0,
        "pose_candidate_blocked_count": 0,
    }
    for joint in joint_artifact.get("joints", []):
        if not isinstance(joint, dict):
            continue
        out["joint_count"] += 1
        metrics = joint.get("metrics")
        comparison = metrics.get("track_model_comparison") if isinstance(metrics, dict) else None
        if not isinstance(comparison, dict):
            continue
        out["track_model_comparison_count"] += 1
        if bool(comparison.get("residual_decisive", False)):
            out["residual_decisive_count"] += 1
        if bool(comparison.get("decision_applied", False)):
            out["decision_applied_count"] += 1
        if bool(comparison.get("residual_decisive", False)) and not bool(comparison.get("pose_candidate_allows_selected_type", True)):
            out["pose_candidate_blocked_count"] += 1
    return out


def _per_joint_rows(
    object_id: str,
    variant: str,
    eval_artifact: dict[str, Any],
    joint_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    joints_by_child = {
        int(joint.get("child_part_id", -1)): joint
        for joint in joint_artifact.get("joints", [])
        if isinstance(joint, dict)
    }
    out = []
    for item in eval_artifact.get("per_joint", []):
        if not isinstance(item, dict):
            continue
        child_part_id = int(item.get("child_part_id", -1))
        joint = joints_by_child.get(child_part_id, {})
        comparison = {}
        metrics = joint.get("metrics") if isinstance(joint, dict) else None
        if isinstance(metrics, dict) and isinstance(metrics.get("track_model_comparison"), dict):
            comparison = metrics["track_model_comparison"]
        out.append(
            {
                "object_id": object_id,
                "variant": variant,
                "child_part_id": child_part_id,
                "gt_joint_name": item.get("ground_truth_joint_name"),
                "gt_joint_type": item.get("ground_truth_joint_type"),
                "predicted_joint_type": item.get("predicted_joint_type"),
                "joint_type_correct": item.get("joint_type_correct"),
                "axis_angle_error_deg": item.get("axis_angle_error_deg"),
                "pivot_error_m": item.get("pivot_error_m"),
                "residual_selected_type": comparison.get("selected_type"),
                "residual_decisive": comparison.get("residual_decisive"),
                "decision_applied": comparison.get("decision_applied"),
                "pose_candidate_allows_selected_type": comparison.get("pose_candidate_allows_selected_type"),
                "decision_ratio": comparison.get("decision_ratio"),
                "prismatic_rmse_m": _nested(comparison, "prismatic", "rmse_m"),
                "revolute_rmse_m": _nested(comparison, "revolute", "rmse_m"),
            }
        )
    return out


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    decision_counts = {
        key: sum(int(row.get("decision_counts", {}).get(key, 0)) for row in ok)
        for key in (
            "joint_count",
            "track_model_comparison_count",
            "residual_decisive_count",
            "decision_applied_count",
            "pose_candidate_blocked_count",
        )
    }
    return {
        "object_count": len(ok),
        "mean_joint_type_accuracy": _mean(row.get("joint_type_accuracy") for row in ok),
        "mean_axis_angle_error_deg": _mean(row.get("axis_angle_error_deg_mean") for row in ok),
        "sum_infer_s": sum(float(row.get("infer_s", 0.0)) for row in ok),
        "mean_infer_s": _mean(row.get("infer_s") for row in ok),
        "sum_eval_s": sum(float(row.get("eval_s", 0.0)) for row in ok),
        "mean_eval_s": _mean(row.get("eval_s") for row in ok),
        "sum_total_s": sum(float(row.get("total_s", 0.0)) for row in ok),
        "mean_total_s": _mean(row.get("total_s") for row in ok),
        "decision_counts": decision_counts,
    }


def _delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_object_variant = {
        (str(row.get("object_id")), str(row.get("variant"))): row
        for row in rows
        if row.get("status") == "ok"
    }
    deltas = []
    for object_id in sorted({object_id for object_id, _variant in by_object_variant}):
        old = by_object_variant.get((object_id, "pose_gated"))
        new = by_object_variant.get((object_id, "track_model_first"))
        if old is None or new is None:
            continue
        deltas.append(
            {
                "object_id": object_id,
                "joint_type_accuracy_delta": _sub(new.get("joint_type_accuracy"), old.get("joint_type_accuracy")),
                "axis_angle_error_deg_mean_delta": _sub(new.get("axis_angle_error_deg_mean"), old.get("axis_angle_error_deg_mean")),
                "infer_s_delta": _sub(new.get("infer_s"), old.get("infer_s")),
                "total_s_delta": _sub(new.get("total_s"), old.get("total_s")),
            }
        )
    return {
        "object_count": len(deltas),
        "mean_joint_type_accuracy_delta": _mean(row.get("joint_type_accuracy_delta") for row in deltas),
        "mean_axis_angle_error_deg_delta": _mean(row.get("axis_angle_error_deg_mean_delta") for row in deltas),
        "sum_infer_s_delta": sum(float(row.get("infer_s_delta", 0.0)) for row in deltas if row.get("infer_s_delta") is not None),
        "mean_infer_s_delta": _mean(row.get("infer_s_delta") for row in deltas),
        "sum_total_s_delta": sum(float(row.get("total_s_delta", 0.0)) for row in deltas if row.get("total_s_delta") is not None),
        "mean_total_s_delta": _mean(row.get("total_s_delta") for row in deltas),
        "per_object": deltas,
    }


def _write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "object_id",
        "variant",
        "status",
        "joint_type_accuracy",
        "axis_angle_error_deg_mean",
        "infer_s",
        "eval_s",
        "total_s",
        "joint_inference_path",
        "kinematic_evaluation_path",
    ]
    _write_tsv(path, columns, rows)


def _write_joint_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "object_id",
        "variant",
        "child_part_id",
        "gt_joint_name",
        "gt_joint_type",
        "predicted_joint_type",
        "joint_type_correct",
        "axis_angle_error_deg",
        "pivot_error_m",
        "residual_selected_type",
        "residual_decisive",
        "decision_applied",
        "pose_candidate_allows_selected_type",
        "decision_ratio",
        "prismatic_rmse_m",
        "revolute_rmse_m",
    ]
    _write_tsv(path, columns, rows)


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Joint Type Selection Ablation",
        "",
        f"- Optimized root: `{payload['optimized_root']}`",
        f"- Batch config: `{payload['batch_config']}`",
        "",
        "## Summary",
        "",
        "| Variant | Objects | Mean type acc | Mean axis angle deg | Sum infer s | Mean infer s | Sum total s | Decisions applied | Pose-gate blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, _requires in VARIANTS:
        item = payload["summary"][variant]
        decisions = item.get("decision_counts", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(item.get("object_count", 0)),
                    _pct(item.get("mean_joint_type_accuracy")),
                    _fmt(item.get("mean_axis_angle_error_deg")),
                    _fmt(item.get("sum_infer_s")),
                    _fmt(item.get("mean_infer_s")),
                    _fmt(item.get("sum_total_s")),
                    str(decisions.get("decision_applied_count", 0)),
                    str(decisions.get("pose_candidate_blocked_count", 0)),
                ]
            )
            + " |"
        )
    delta = payload["delta_track_model_first_minus_pose_gated"]
    lines.extend(
        [
            "",
            "## Delta: Track-model-first Minus Pose-gated",
            "",
            f"- Mean type accuracy delta: `{_fmt(delta.get('mean_joint_type_accuracy_delta'))}`",
            f"- Mean axis-angle delta: `{_fmt(delta.get('mean_axis_angle_error_deg_delta'))}` deg",
            f"- Sum inference-time delta: `{_fmt(delta.get('sum_infer_s_delta'))}` s",
            f"- Mean inference-time delta: `{_fmt(delta.get('mean_infer_s_delta'))}` s/object",
            f"- Sum total-time delta: `{_fmt(delta.get('sum_total_s_delta'))}` s",
            "",
            "## Per-object Accuracy Changes",
            "",
            "| Object | Type acc delta | Axis angle delta deg | Infer s delta | Total s delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in delta.get("per_object", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("object_id", "")),
                    _fmt(row.get("joint_type_accuracy_delta")),
                    _fmt(row.get("axis_angle_error_deg_mean_delta")),
                    _fmt(row.get("infer_s_delta")),
                    _fmt(row.get("total_s_delta")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: Any) -> float | None:
    nums = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            nums.append(numeric)
    return sum(nums) / len(nums) if nums else None


def _sub(a: Any, b: Any) -> float | None:
    try:
        a_num = float(a)
        b_num = float(b)
    except (TypeError, ValueError):
        return None
    return a_num - b_num if math.isfinite(a_num) and math.isfinite(b_num) else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6g}"


def _pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{100.0 * numeric:.1f}%" if math.isfinite(numeric) else ""


if __name__ == "__main__":
    raise SystemExit(main())
