#!/usr/bin/env python3
"""Run sequential-RANSAC sensitivity tests on saved AiM Gaussian trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


GAUSSIAN_PATTERN = re.compile(
    r"(?P<static>\d+) Static Gaussians and (?P<moving>\d+) Moving Gaussians"
)
SEGMENTATION_COUNT_PATTERN = re.compile(
    r"^(?P<static>\d+)\s+(?P<moving>\d+)(?:\s+\[|$)", re.MULTILINE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aim_run", type=Path)
    parser.add_argument("reference_ply", type=Path)
    parser.add_argument("--aim-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-inliers-ratios",
        nargs="+",
        type=float,
        default=(0.10, 0.075, 0.05, 0.025),
    )
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_xyz(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        line = stream.readline()
        if line.strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header: {path}")
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        xyz = np.loadtxt(stream, dtype=np.float32, max_rows=count, usecols=(0, 1, 2))
    return np.atleast_2d(xyz)


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    properties: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        count = None
        while True:
            line = stream.readline()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line.strip() == "end_header":
                break
        data = np.loadtxt(stream, dtype=np.float64, max_rows=count)
    part_name = next(
        (name for name in ("original_part_id", "part_id", "label") if name in properties),
        None,
    )
    if part_name is None:
        raise ValueError(f"Reference PLY has no part-id property: {path}")
    return data[:, :3].astype(np.float32), data[:, properties.index(part_name)].astype(np.int64)


def _gaussian_counts(run: Path) -> tuple[int, int]:
    segmentation_log = run / "segmentation.log"
    if segmentation_log.is_file():
        matches = list(
            SEGMENTATION_COUNT_PATTERN.finditer(
                segmentation_log.read_text(encoding="utf-8", errors="replace")
            )
        )
        if matches:
            return int(matches[0]["static"]), int(matches[0]["moving"])
    matches = list(
        GAUSSIAN_PATTERN.finditer(
            (run / "train.log").read_text(encoding="utf-8", errors="replace")
        )
    )
    if not matches:
        raise RuntimeError(f"No Gaussian count in {run / 'train.log'}")
    return int(matches[-1]["static"]), int(matches[-1]["moving"])


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _oracle_models(
    trajectory: np.ndarray,
    gt_labels: np.ndarray,
    strict_valid: np.ndarray,
) -> dict[str, Any]:
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for part_id in sorted(int(value) for value in np.unique(gt_labels[strict_valid])):
        selected = strict_valid & (gt_labels == part_id)
        if selected.sum() < 3:
            continue
        rotation, translation = _fit_rigid(trajectory[selected, 0], trajectory[selected, 2])
        prediction = trajectory[selected, 0] @ rotation.T + translation
        residual = np.linalg.norm(prediction - trajectory[selected, 2], axis=1)
        models[part_id] = (rotation, translation)
        rows.append(
            {
                "gt_part_id": part_id,
                "gaussian_count": int(selected.sum()),
                "rigid_rmse_m": float(np.sqrt(np.mean(residual**2))),
                "rigid_median_m": float(np.median(residual)),
                "motion_median_m": float(
                    np.median(
                        np.linalg.norm(
                            trajectory[selected, 2] - trajectory[selected, 0], axis=1
                        )
                    )
                ),
            }
        )
    confusion = []
    for source_part, selected_model in models.items():
        source_mask = strict_valid & (gt_labels == source_part)
        for model_part, (rotation, translation) in models.items():
            prediction = trajectory[source_mask, 0] @ rotation.T + translation
            residual = np.linalg.norm(prediction - trajectory[source_mask, 2], axis=1)
            confusion.append(
                {
                    "gt_part_id": source_part,
                    "model_part_id": model_part,
                    "rmse_m": float(np.sqrt(np.mean(residual**2))),
                }
            )
    return {"per_gt_part": rows, "cross_model_residual": confusion}


def _composition(mask: torch.Tensor, labels: np.ndarray) -> dict[str, Any]:
    indices = torch.where(mask)[0].detach().cpu().numpy()
    values = labels[indices]
    values = values[values >= 0]
    counts = Counter(int(value) for value in values)
    total = sum(counts.values())
    dominant = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "size": int(mask.sum().item()),
        "strict_gt_mapped": int(total),
        "dominant_gt_part": dominant[0],
        "gt_purity": float(dominant[1] / total) if total else None,
        "gt_composition": {str(key): value for key, value in sorted(counts.items())},
    }


class _OfflineAiM:
    pass


def _run_variant(
    *,
    helper: Any,
    trajectory: torch.Tensor,
    gt_labels: np.ndarray,
    min_inliers: int,
    ransac_iterations: int,
    keep_largest_cc: bool,
    merge_components: bool,
    seed: int,
) -> dict[str, Any]:
    from utils.seg_tools import (  # type: ignore[import-not-found]
        em_inlier_joint_likelihood,
        fit_rigid_transform_weighted,
        screw_from_Rt,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x0, x05, x1 = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    labels = torch.full((len(trajectory),), -1, dtype=torch.long, device=trajectory.device)
    unassigned = torch.arange(len(trajectory), device=trajectory.device)
    rotations = [torch.eye(3, device=trajectory.device)]
    translations = [torch.zeros(3, device=trajectory.device)]
    proposals: list[dict[str, Any]] = []
    accepted = []
    stagnant = 0
    previous_size = -1
    while unassigned.numel() >= min_inliers and stagnant < 10:
        stagnant = stagnant + 1 if unassigned.numel() == previous_size else 0
        previous_size = unassigned.numel()
        current = trajectory[unassigned]
        raw_mask, errors, _, _, rotation, translation, _ = helper.ransac_propose_two_stage(
            current[:, 0],
            current[:, 1],
            current[:, 2],
            num_iter=ransac_iterations,
            sample_points=3,
            inlier_thresh=min(
                float(torch.linalg.norm(trajectory[:, 2] - trajectory[:, 0], dim=1).mean()),
                0.25,
            ),
        )
        proposal: dict[str, Any] = {
            "remaining_before": int(unassigned.numel()),
            "raw": _composition(raw_mask, gt_labels[unassigned.cpu().numpy()]),
        }
        em_local, weights, _, _, _ = em_inlier_joint_likelihood(
            errors[raw_mask], iters=50, odds_log_tau=0.0, w_thresh=0.5
        )
        em_mask = torch.zeros_like(raw_mask)
        em_mask[raw_mask] = em_local
        proposal["after_em"] = _composition(
            em_mask, gt_labels[unassigned.cpu().numpy()]
        )
        if em_mask.sum() == 0:
            proposal["decision"] = "reject_empty_em"
            proposals.append(proposal)
            continue
        if keep_largest_cc:
            cc_keep = helper.keep_largest_cc(current[em_mask, 2])
            final_mask = em_mask.clone()
            final_mask[em_mask] = cc_keep
        else:
            final_mask = em_mask
        proposal["after_cc"] = _composition(
            final_mask, gt_labels[unassigned.cpu().numpy()]
        )
        if final_mask.sum().item() < min_inliers:
            proposal["decision"] = "reject_min_inliers"
            proposals.append(proposal)
            continue
        selected_weights = weights[em_local][
            cc_keep if keep_largest_cc else torch.ones_like(em_local, dtype=torch.bool)
        ].clamp(1e-4, 1.0)
        refit_rotation, refit_translation = fit_rigid_transform_weighted(
            current[final_mask, 0], current[final_mask, 2], selected_weights
        )
        component_id = len(rotations)
        labels[unassigned[final_mask]] = component_id
        rotations.append(refit_rotation)
        translations.append(refit_translation)
        proposal["decision"] = "accept"
        proposal["component_id"] = component_id
        proposals.append(proposal)
        accepted.append(proposal["after_cc"])
        unassigned = unassigned[~final_mask]

    premerge_labels = labels.clone()
    premerge_count = len(rotations) - 1
    merge_map = {index: index for index in range(premerge_count)}
    if merge_components and premerge_count > 1:
        helper.motion_R_set = rotations
        helper.motion_t_set = translations
        models, merge_map = helper.search_and_merge(
            premerge_count + 1, x0, x1, labels
        )
        for old_id, new_id in merge_map.items():
            labels[premerge_labels == old_id + 1] = new_id + 1
        rotations = [torch.eye(3, device=trajectory.device)] + [
            model["R"] for model in models
        ]
        translations = [torch.zeros(3, device=trajectory.device)] + [
            model["t"] for model in models
        ]
    final_components = []
    for component_id in sorted(
        int(value) for value in torch.unique(labels).tolist() if value > 0
    ):
        selected = labels == component_id
        row = _composition(selected, gt_labels)
        rotation = rotations[component_id]
        translation = translations[component_id]
        _, axis, theta, phi, motion_type = screw_from_Rt(
            rotation, translation, eps_theta=0.2
        )
        row.update(
            {
                "component_id": component_id,
                "motion_type": str(motion_type),
                "axis": axis.detach().cpu().numpy().tolist(),
                "theta": float(theta),
                "phi": float(phi),
            }
        )
        final_components.append(row)
    return {
        "min_inliers": min_inliers,
        "keep_largest_cc": keep_largest_cc,
        "merge_components": merge_components,
        "premerge_component_count": premerge_count,
        "final_component_count": len(final_components),
        "unassigned_count": int((labels < 0).sum().item()),
        "accepted_components": accepted,
        "final_components": final_components,
        "merge_map": {str(key + 1): value + 1 for key, value in merge_map.items()},
        "proposal_count": len(proposals),
        "rejected_min_inliers_count": sum(
            row["decision"] == "reject_min_inliers" for row in proposals
        ),
        "proposals": proposals,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.aim_root.resolve()))
    from seg_main import AiM  # type: ignore[import-not-found]

    static_count, moving_count = _gaussian_counts(args.aim_run)
    sequence = np.stack(
        [
            _read_xyz(args.aim_run / f"motion_traj_t={time}/point_cloud_seq_0.ply")
            for time in ("0.0", "0.5", "1.0")
        ],
        axis=1,
    )
    if len(sequence) != static_count + moving_count:
        raise RuntimeError(
            f"Trajectory count {len(sequence)} != {static_count}+{moving_count}"
        )
    dynamic = sequence[static_count:]
    reference_xyz, reference_part = _read_reference(args.reference_ply)
    bbox_diagonal = float(np.linalg.norm(np.ptp(reference_xyz, axis=0)))
    distance, nearest = cKDTree(reference_xyz).query(dynamic[:, 2], k=1)
    strict_valid = distance <= 0.02 * bbox_diagonal
    gt_labels = np.full(moving_count, -1, dtype=np.int64)
    gt_labels[strict_valid] = reference_part[nearest[strict_valid]]
    helper = _OfflineAiM()
    helper.ransac_propose_two_stage = AiM.ransac_propose_two_stage.__get__(helper)
    helper.keep_largest_cc = AiM.keep_largest_cc.__get__(helper)
    helper.search_and_merge = AiM.search_and_merge.__get__(helper)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectory = torch.from_numpy(dynamic).to(device)
    variants = []
    started = time.perf_counter()
    for ratio in args.min_inliers_ratios:
        # Merge sensitivity is available from the premerge/final counts of the
        # same extraction, so it does not require another stochastic RANSAC run.
        for keep_cc, merge in ((True, True), (False, True)):
            variant_started = time.perf_counter()
            result = _run_variant(
                helper=helper,
                trajectory=trajectory,
                gt_labels=gt_labels,
                min_inliers=max(3, math.ceil(ratio * moving_count)),
                ransac_iterations=args.ransac_iterations,
                keep_largest_cc=keep_cc,
                merge_components=merge,
                seed=args.seed,
            )
            result.update(
                {
                    "min_inliers_ratio": ratio,
                    "runtime_s": time.perf_counter() - variant_started,
                }
            )
            variants.append(result)
    payload = {
        "aim_run": str(args.aim_run),
        "reference_ply": str(args.reference_ply),
        "static_gaussian_count": static_count,
        "dynamic_gaussian_count": moving_count,
        "strict_gt_mapping_ratio": float(strict_valid.mean()),
        "gt_dynamic_support": {
            str(part_id): int(np.sum(strict_valid & (gt_labels == part_id)))
            for part_id in sorted(int(value) for value in np.unique(gt_labels[strict_valid]))
        },
        "oracle_representation": _oracle_models(dynamic, gt_labels, strict_valid),
        "variants": variants,
        "runtime_s": time.perf_counter() - started,
    }
    output_json = args.output_dir / "ransac_sensitivity.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = [
        {
            "min_inliers_ratio": row["min_inliers_ratio"],
            "min_inliers": row["min_inliers"],
            "keep_largest_cc": row["keep_largest_cc"],
            "merge_components": row["merge_components"],
            "premerge_component_count": row["premerge_component_count"],
            "final_component_count": row["final_component_count"],
            "unassigned_count": row["unassigned_count"],
            "rejected_min_inliers_count": row["rejected_min_inliers_count"],
            "runtime_s": row["runtime_s"],
        }
        for row in variants
    ]
    with (args.output_dir / "variant_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output_json), "runtime_s": payload["runtime_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
