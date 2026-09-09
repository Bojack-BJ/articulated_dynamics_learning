#!/usr/bin/env python3
"""Evaluate child-only group-Kabsch and per-track voting axis proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.analytic_joint_axis import (
    axis_angle_error_deg,
    axis_line_distance,
)
from rgbd_urdf_mvp.kinematics.group_motion_axis import (
    build_group_motion_proposals,
    consensus_group_axis,
)
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    _load_relation_samples,
    _load_slot_model,
    _relation_targets,
)
from rgbd_urdf_mvp.kinematics.track_motion_voting import (
    build_track_motion_proposals,
    consensus_track_axes,
)
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-joints", type=int, default=32)
    parser.add_argument("--row-margin", type=int, default=2)
    parser.add_argument("--min-tracks", type=int, default=3,
                        help="Minimum common tracks for each group-Kabsch proposal.")
    parser.add_argument("--track-min-segment-frames", type=int, default=5)
    parser.add_argument("--track-max-gap", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _quality(sample: dict[str, Any]) -> np.ndarray:
    visibility = np.asarray(sample["visibility"], dtype=bool)
    observation = np.asarray(
        sample.get("observation_quality", np.ones(visibility.shape)), dtype=float
    )
    track = np.asarray(
        sample.get("track_quality", np.ones(visibility.shape[0])), dtype=float
    )
    if observation.shape != visibility.shape:
        observation = np.ones(visibility.shape, dtype=float)
    if track.shape != (visibility.shape[0],):
        track = np.ones(visibility.shape[0], dtype=float)
    return np.clip(observation, 0.0, 1.0) * np.clip(track[:, None], 0.0, 1.0)


def _estimate(
    method: str,
    points: np.ndarray,
    visibility: np.ndarray,
    quality: np.ndarray,
    membership: np.ndarray,
    joint_type: str,
    min_tracks: int,
    track_min_segment_frames: int,
    track_max_gap: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if method == "group_kabsch":
        if len(points) < min_tracks:
            return None, None, 0
        proposals = build_group_motion_proposals(
            points,
            visibility,
            quality=quality,
            membership=membership,
            strides=(1, 2, 4, 8),
            min_tracks=min_tracks,
            huber_delta=0.02,
            irls_iterations=4,
        )
        axis, point = consensus_group_axis(proposals, joint_type)
    elif method == "track_voting":
        proposals = build_track_motion_proposals(
            points,
            visibility,
            quality=quality,
            membership=membership,
            min_segment_frames=track_min_segment_frames,
            max_gap=track_max_gap,
        )
        axis, point = consensus_track_axes(proposals, joint_type)
    else:
        raise ValueError(method)
    return axis, point, len(proposals)


def _metrics(rows: list[dict[str, Any]], *, include_by_type: bool = True) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    errors = np.asarray([row["axis_error_deg"] for row in valid], dtype=float)
    line = np.asarray(
        [row["line_error"] for row in valid if row["line_error"] is not None],
        dtype=float,
    )
    penalized = [row["axis_error_deg"] if row["valid"] else 90.0 for row in rows]
    result = {
        "joint_count": len(rows),
        "valid_count": len(valid),
        "coverage": len(valid) / len(rows) if rows else 0.0,
        "axis_mean_deg": float(errors.mean()) if len(errors) else None,
        "axis_median_deg": float(np.median(errors)) if len(errors) else None,
        "axis_p90_deg": float(np.percentile(errors, 90)) if len(errors) else None,
        "axis_gt_80_count": int(np.sum(errors > 80.0)) if len(errors) else 0,
        "penalized_axis_mean_deg": float(np.mean(penalized)) if rows else None,
        "line_mean": float(line.mean()) if len(line) else None,
    }
    if include_by_type:
        result["by_type"] = {
            joint_type: _metrics(
                [row for row in rows if row["joint_type"] == joint_type],
                include_by_type=False,
            )
            for joint_type in ("prismatic", "revolute")
            if any(row["joint_type"] == joint_type for row in rows)
        }
    return result


def main() -> None:
    args = parse_args()
    import torch

    checkpoint = torch.load(args.slot_model, map_location="cpu", weights_only=False)
    manifest = load_pairwise_manifest(args.manifest)
    source_rows = [row for row in manifest if row["split"] == args.split]
    source_rows = source_rows[: args.max_joints * args.row_margin]
    samples = _load_relation_samples(source_rows, args.split, checkpoint)
    slot_model = _load_slot_model(checkpoint, torch, args.device)
    mean = checkpoint["feature_mean"].to(args.device)
    std = checkpoint["feature_std"].to(args.device)
    methods = ("group_kabsch", "track_voting")
    protocols = ("gt_child", "predicted_child_slot")
    rows: list[dict[str, Any]] = []
    joint_count = 0
    with torch.no_grad():
        for sample in samples:
            slot_features = torch.from_numpy(
                sample.get("slot_features", sample["features"])
            ).to(args.device)
            logits, _, slots = slot_model((slot_features - mean) / std, return_slots=True)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            targets = _relation_targets(
                sample, logits, int(slots.shape[0]), torch, args.device
            )
            relation_slots = {
                str(row["joint_name"]): int(row["child_slot"])
                for row in targets["relations"]
            }
            center = np.asarray(sample["canonical_center_m"], dtype=float)
            scale = max(float(sample["canonical_scale_m"]), 1e-8)
            points = (np.asarray(sample["points"], dtype=float) - center) / scale
            visibility = np.asarray(sample["visibility"], dtype=bool)
            quality = _quality(sample)
            labels = np.asarray(sample["labels"], dtype=int)
            for relation in sample["gt_relations"]:
                joint_type = str(relation["joint_type"])
                if joint_type not in {"prismatic", "revolute"}:
                    continue
                if joint_count >= args.max_joints:
                    break
                joint_count += 1
                child_label = int(relation["child_label"])
                child_slot = relation_slots[str(relation["joint_name"])]
                masks = {
                    "gt_child": labels == child_label,
                    "predicted_child_slot": probabilities.argmax(axis=1) == child_slot,
                }
                for protocol in protocols:
                    mask = masks[protocol]
                    membership = (
                        np.ones(int(mask.sum()), dtype=float)
                        if protocol == "gt_child"
                        else probabilities[mask, child_slot]
                    )
                    estimates: dict[str, dict[str, Any]] = {}
                    for method in methods:
                        axis, point, proposal_count = _estimate(
                            method,
                            points[mask],
                            visibility[mask],
                            quality[mask],
                            membership,
                            joint_type,
                            args.min_tracks,
                            args.track_min_segment_frames,
                            args.track_max_gap,
                        )
                        valid = axis is not None and np.all(np.isfinite(axis))
                        line_error = None
                        if valid and joint_type == "revolute" and point is not None:
                            line_error = axis_line_distance(
                                point, axis, relation["pivot"], relation["axis"]
                            )
                        row = {
                            "object_id": sample["object_id"],
                            "joint_name": relation["joint_name"],
                            "joint_type": joint_type,
                            "protocol": protocol,
                            "method": method,
                            "track_count": int(mask.sum()),
                            "proposal_count": proposal_count,
                            "valid": bool(valid),
                            "axis": axis.tolist() if valid else None,
                            "line_point": point.tolist() if point is not None else None,
                            "axis_error_deg": axis_angle_error_deg(axis, relation["axis"])
                            if valid else None,
                            "line_error": line_error,
                        }
                        rows.append(row)
                        estimates[method] = row
                    routed_method = (
                        "track_voting" if joint_type == "prismatic" else "group_kabsch"
                    )
                    routed = estimates[routed_method]
                    rows.append({
                        **routed,
                        "method": "type_routed_hybrid",
                        "source_method": routed_method,
                    })
            if joint_count >= args.max_joints:
                break
    summary = {
        protocol: {
            method: _metrics([
                row for row in rows
                if row["protocol"] == protocol and row["method"] == method
            ])
            for method in (*methods, "type_routed_hybrid")
        }
        for protocol in protocols
    }
    payload = {
        "evaluation_contract": {
            "split": args.split,
            "max_joints": args.max_joints,
            "manifest": str(args.manifest),
            "slot_model": str(args.slot_model),
            "quality": "observation_quality * track_quality",
            "group_min_tracks": args.min_tracks,
            "track_min_segment_frames": args.track_min_segment_frames,
            "track_max_gap": args.track_max_gap,
            "predicted_slot_mapping": "Hungarian label-to-slot mapping; hard predicted support with soft membership weights",
        },
        "summary": summary,
        "joints": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "motion_axis_proposal_report.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
