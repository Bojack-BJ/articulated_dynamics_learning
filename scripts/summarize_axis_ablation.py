#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SE(3)-pose and track-displacement prismatic axis inference.")
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--track-axis-root", type=Path, required=True)
    parser.add_argument("--se3-axis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for object_id in _object_ids(args.batch_config.expanduser().resolve()):
        track_eval = _read_json(args.track_axis_root / object_id / "pointcloud_4d_partseg" / "kinematic_evaluation.json")
        se3_eval = _read_json(args.se3_axis_root / object_id / "kinematic_evaluation.json")
        rows.extend(_compare_object(object_id, se3_eval, track_eval))

    payload = {
        "source": "axis-ablation-comparison",
        "track_axis_root": str(args.track_axis_root.expanduser().resolve()),
        "se3_axis_root": str(args.se3_axis_root.expanduser().resolve()),
        "summary": {
            "se3_pose_axis": _summarize(rows, "se3"),
            "track_displacement_axis": _summarize(rows, "track"),
            "delta_track_minus_se3": _delta_summary(rows),
        },
        "rows": rows,
        "notes": [
            "Both variants use the same dense tracks and part_poses; only prismatic axis estimation changes.",
            "SE(3)-pose axis estimates prismatic direction from per-frame rigid pose translations.",
            "Track-displacement axis estimates prismatic direction from endpoint centroid displacement over 3D tracks.",
            "Prismatic axis position is weakly observable from pure sliding motion; angle and q replay are usually more diagnostic.",
        ],
    }
    (output_dir / "axis_ablation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(output_dir / "axis_ablation_per_joint.tsv", rows)
    _write_markdown(output_dir / "axis_ablation_summary.md", payload)
    print(json.dumps({"axis_ablation_summary": str((output_dir / "axis_ablation_summary.json").resolve())}, indent=2))
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


def _compare_object(object_id: str, se3_eval: dict[str, Any], track_eval: dict[str, Any]) -> list[dict[str, Any]]:
    track_by_key = {
        (item.get("child_part_id"), item.get("ground_truth_joint_name")): item
        for item in track_eval.get("per_joint", [])
        if isinstance(item, dict) and item.get("matched")
    }
    rows = []
    for se3 in se3_eval.get("per_joint", []):
        if not isinstance(se3, dict) or not se3.get("matched"):
            continue
        key = (se3.get("child_part_id"), se3.get("ground_truth_joint_name"))
        track = track_by_key.get(key)
        if track is None:
            continue
        row = {
            "object_id": object_id,
            "gt_joint_name": se3.get("ground_truth_joint_name"),
            "gt_joint_type": se3.get("ground_truth_joint_type"),
            "se3_predicted_type": se3.get("predicted_joint_type"),
            "track_predicted_type": track.get("predicted_joint_type"),
            "se3_type_correct": se3.get("joint_type_correct"),
            "track_type_correct": track.get("joint_type_correct"),
            "se3_axis_angle_deg": _num(se3.get("axis_angle_error_deg")),
            "track_axis_angle_deg": _num(track.get("axis_angle_error_deg")),
            "se3_position_error_m": _num(se3.get("pivot_error_m")),
            "track_position_error_m": _num(track.get("pivot_error_m")),
            "se3_q_rmse_offset_aligned": _q(se3),
            "track_q_rmse_offset_aligned": _q(track),
        }
        row["axis_angle_delta_deg"] = _sub(row["track_axis_angle_deg"], row["se3_axis_angle_deg"])
        row["position_error_delta_m"] = _sub(row["track_position_error_m"], row["se3_position_error_m"])
        row["q_rmse_delta"] = _sub(row["track_q_rmse_offset_aligned"], row["se3_q_rmse_offset_aligned"])
        rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for joint_type in ("all", "prismatic", "revolute"):
        typed = rows if joint_type == "all" else [row for row in rows if row.get("gt_joint_type") == joint_type]
        out[joint_type] = {
            "joint_count": len(typed),
            "joint_type_accuracy": _mean(1.0 if row.get(f"{prefix}_type_correct") else 0.0 for row in typed),
            "axis_angle_deg_mean": _mean(row.get(f"{prefix}_axis_angle_deg") for row in typed),
            "axis_angle_deg_median": _median(row.get(f"{prefix}_axis_angle_deg") for row in typed),
            "position_error_m_mean": _mean(row.get(f"{prefix}_position_error_m") for row in typed),
            "q_rmse_offset_aligned_mean": _mean(row.get(f"{prefix}_q_rmse_offset_aligned") for row in typed),
        }
    return out


def _delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for joint_type in ("all", "prismatic", "revolute"):
        typed = rows if joint_type == "all" else [row for row in rows if row.get("gt_joint_type") == joint_type]
        out[joint_type] = {
            "axis_angle_delta_deg_mean": _mean(row.get("axis_angle_delta_deg") for row in typed),
            "position_error_delta_m_mean": _mean(row.get("position_error_delta_m") for row in typed),
            "q_rmse_delta_mean": _mean(row.get("q_rmse_delta") for row in typed),
        }
    return out


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "object_id",
        "gt_joint_name",
        "gt_joint_type",
        "se3_predicted_type",
        "track_predicted_type",
        "se3_type_correct",
        "track_type_correct",
        "se3_axis_angle_deg",
        "track_axis_angle_deg",
        "axis_angle_delta_deg",
        "se3_position_error_m",
        "track_position_error_m",
        "position_error_delta_m",
        "se3_q_rmse_offset_aligned",
        "track_q_rmse_offset_aligned",
        "q_rmse_delta",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Axis Inference Ablation",
        "",
        f"- SE(3)-axis root: `{payload['se3_axis_root']}`",
        f"- Track-axis root: `{payload['track_axis_root']}`",
        "- Negative delta means the track-displacement variant is better.",
        "",
        "## Summary",
        "",
        "| Variant | GT type | Joints | Type acc | Mean angle deg | Median angle deg | Mean pos err m | Mean q RMSE aligned |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, key in [("SE(3) pose axis", "se3_pose_axis"), ("Track displacement axis", "track_displacement_axis")]:
        for joint_type, item in summary[key].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        joint_type,
                        str(item["joint_count"]),
                        _pct(item.get("joint_type_accuracy")),
                        _fmt(item.get("axis_angle_deg_mean")),
                        _fmt(item.get("axis_angle_deg_median")),
                        _fmt(item.get("position_error_m_mean")),
                        _fmt(item.get("q_rmse_offset_aligned_mean")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Mean Delta: Track Minus SE(3)",
            "",
            "| GT type | Axis angle delta deg | Position error delta m | q RMSE delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for joint_type, item in summary["delta_track_minus_se3"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    joint_type,
                    _fmt(item.get("axis_angle_delta_deg_mean")),
                    _fmt(item.get("position_error_delta_m_mean")),
                    _fmt(item.get("q_rmse_delta_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Largest SE(3)-axis Errors",
            "",
            "| Object | GT joint | GT type | SE(3) angle | Track angle | Delta | SE(3) pos m | Track pos m |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    rows = sorted(payload["rows"], key=lambda row: row.get("se3_axis_angle_deg") or 0.0, reverse=True)[:20]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("object_id", "")),
                    str(row.get("gt_joint_name", "")),
                    str(row.get("gt_joint_type", "")),
                    _fmt(row.get("se3_axis_angle_deg")),
                    _fmt(row.get("track_axis_angle_deg")),
                    _fmt(row.get("axis_angle_delta_deg")),
                    _fmt(row.get("se3_position_error_m")),
                    _fmt(row.get("track_position_error_m")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in payload.get("notes", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _q(row: dict[str, Any]) -> float | None:
    q = row.get("q_error")
    return _num(q.get("q_rmse_offset_aligned")) if isinstance(q, dict) else None


def _sub(a: Any, b: Any) -> float | None:
    a_num = _num(a)
    b_num = _num(b)
    return None if a_num is None or b_num is None else a_num - b_num


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite(values: Any) -> list[float]:
    out = []
    for value in values:
        numeric = _num(value)
        if numeric is not None:
            out.append(numeric)
    return out


def _mean(values: Any) -> float | None:
    values = _finite(values)
    return sum(values) / len(values) if values else None


def _median(values: Any) -> float | None:
    values = _finite(values)
    return median(values) if values else None


def _fmt(value: Any) -> str:
    numeric = _num(value)
    return "" if numeric is None else f"{numeric:.4g}"


def _pct(value: Any) -> str:
    numeric = _num(value)
    return "" if numeric is None else f"{100.0 * numeric:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
