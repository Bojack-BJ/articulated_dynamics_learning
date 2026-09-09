#!/usr/bin/env python3
"""Measure GT-part rigidity and separability in AiM Gaussian trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from audit_aim_protocol import _read_ply_part_ids, _read_ply_xyz_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aim_run", type=Path)
    parser.add_argument("reference_ply", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=(0.0, 0.5, 1.0))
    parser.add_argument("--mapping-distance-ratio", type=float, default=0.02)
    return parser.parse_args()


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _replay_rmse(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    predicted = source @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    trajectories = [
        _read_ply_xyz_rgb(
            args.aim_run / f"motion_traj_t={time:.1f}" / "point_cloud_seq_0.ply"
        )[0]
        for time in args.times
    ]
    point_count = min(len(points) for points in trajectories)
    trajectory = np.stack([points[:point_count] for points in trajectories], axis=1)
    reference_xyz, reference_part = _read_ply_part_ids(args.reference_ply)
    bbox_diagonal = float(np.linalg.norm(np.ptp(reference_xyz, axis=0)))
    mapping_threshold = max(args.mapping_distance_ratio * bbox_diagonal, 1e-9)

    distance, gaussian_index = cKDTree(trajectory[:, -1]).query(reference_xyz, k=1)
    mapped = distance <= mapping_threshold
    part_indices: dict[int, np.ndarray] = {}
    for part_id in sorted(int(value) for value in np.unique(reference_part)):
        selected = mapped & (reference_part == part_id)
        part_indices[part_id] = np.unique(gaussian_index[selected]).astype(np.int64)

    models: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    for part_id, indices in part_indices.items():
        if len(indices) < 3:
            continue
        source = trajectory[indices, 0]
        models[part_id] = []
        residuals = []
        for time_index in range(len(args.times)):
            rotation, translation = _fit_rigid(source, trajectory[indices, time_index])
            models[part_id].append((rotation, translation))
            residuals.append(
                _replay_rmse(
                    source,
                    trajectory[indices, time_index],
                    rotation,
                    translation,
                )
            )
        displacement = np.linalg.norm(trajectory[indices, -1] - source, axis=1)
        rows.append(
            {
                "gt_part_id": part_id,
                "mapped_gaussian_count": len(indices),
                "reference_point_count": int(np.sum(reference_part == part_id)),
                "reference_mapping_coverage": float(
                    np.mean(mapped[reference_part == part_id])
                ),
                "mapping_distance_median_m": float(
                    np.median(distance[(reference_part == part_id) & mapped])
                ),
                "motion_displacement_mean_m": float(np.mean(displacement)),
                "motion_displacement_median_m": float(np.median(displacement)),
                "motion_displacement_std_m": float(np.std(displacement)),
                "rigid_rmse_mean_m": float(np.mean(residuals)),
                "rigid_rmse_max_m": float(np.max(residuals)),
                "per_time_rigid_rmse_m": json.dumps(residuals),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for model_part, transforms in models.items():
        for target_part, indices in part_indices.items():
            if model_part == target_part or len(indices) < 3:
                continue
            source = trajectory[indices, 0]
            cross_residuals = [
                _replay_rmse(
                    source,
                    trajectory[indices, time_index],
                    *transforms[time_index],
                )
                for time_index in range(len(args.times))
            ]
            target_row = next(row for row in rows if row["gt_part_id"] == target_part)
            own_rmse = float(target_row["rigid_rmse_mean_m"])
            cross_rmse = float(np.mean(cross_residuals))
            pair_rows.append(
                {
                    "model_gt_part_id": model_part,
                    "target_gt_part_id": target_part,
                    "cross_replay_rmse_mean_m": cross_rmse,
                    "target_own_rigid_rmse_mean_m": own_rmse,
                    "separability_ratio_cross_over_own": cross_rmse
                    / max(own_rmse, 1e-9),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_gt_part_trajectory.csv", rows)
    _write_csv(args.output_dir / "cross_part_separability.csv", pair_rows)
    result = {
        "aim_run": str(args.aim_run.resolve()),
        "reference_ply": str(args.reference_ply.resolve()),
        "times": list(args.times),
        "trajectory_point_count": point_count,
        "mapping_distance_ratio_bbox": args.mapping_distance_ratio,
        "mapping_distance_threshold_m": mapping_threshold,
        "parts": rows,
        "cross_part_separability": pair_rows,
    }
    (args.output_dir / "trajectory_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
