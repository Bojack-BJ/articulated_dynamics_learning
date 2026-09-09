#!/usr/bin/env python3
"""Diagnose world-frame alignment between AiM TSDF meshes and simulation GT.

This is deliberately *not* an evaluation metric. It exports raw and rigid-ICP
overlays and reports whether a low voxel IoU is explained by a global rigid
frame error, scale mismatch, or genuinely different geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment

from evaluate_aim_mesh_voxel_iou import _gt_part_meshes, _load_mesh, _voxel_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("aim_run", type=Path)
    parser.add_argument("--state", choices=("start", "end"), default="end")
    parser.add_argument("--voxel-size-m", type=float, default=0.01, help="Coarse diagnostic pitch; the final metric remains 4 mm.")
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _component_paths(run: Path, state: str) -> list[Path]:
    return sorted([
        *run.glob(f"sub_*_point_cloud_{state}/fuse_post.ply"),
        *run.glob(f"sub_*_point_cloud_{state}fuse_post.ply"),
    ])


def _centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)


def _bbox_diag(mesh: trimesh.Trimesh) -> float:
    return float(np.linalg.norm(np.asarray(mesh.bounds)[1] - np.asarray(mesh.bounds)[0]))


def _sample_vertices(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= count:
        return vertices
    return vertices[rng.choice(len(vertices), size=count, replace=False)]


def _initial_centroid_transform(source: trimesh.Trimesh, target: trimesh.Trimesh) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _centroid(target) - _centroid(source)
    return transform


def _voxel_iou(source: trimesh.Trimesh, target: trimesh.Trimesh, pitch: float) -> float:
    origin = np.floor(np.minimum(source.bounds[0], target.bounds[0]) / pitch) * pitch
    source_keys = _voxel_keys(source, pitch, origin)
    target_keys = _voxel_keys(target, pitch, origin)
    union = source_keys | target_keys
    return len(source_keys & target_keys) / len(union) if union else 0.0


def _colored(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    copied = mesh.copy()
    copied.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(copied.vertices), 1))
    return copied


def _export_overlay(path: Path, predicted: trimesh.Trimesh, target: trimesh.Trimesh, title: str) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(_colored(target, (48, 180, 90, 120)), node_name="gt_mesh")
    scene.add_geometry(_colored(predicted, (225, 65, 65, 120)), node_name="aim_tsdf_mesh")
    scene.metadata["title"] = title
    scene.export(path)


def main() -> int:
    args = parse_args()
    if args.voxel_size_m <= 0.0:
        raise ValueError("voxel-size-m must be positive")
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    gt_meshes = _gt_part_meshes(episode, args.state)
    paths = _component_paths(args.aim_run, args.state)
    if not paths:
        raise FileNotFoundError("No AiM TSDF meshes found. Run render_main.py first.")
    predicted = [_load_mesh(path) for path in paths]
    gt_ids = sorted(gt_meshes)
    raw_iou = np.asarray(
        [[_voxel_iou(pred, gt_meshes[part_id], args.voxel_size_m) for part_id in gt_ids] for pred in predicted],
        dtype=np.float64,
    )
    matched_rows, matched_cols = linear_sum_assignment(-raw_iou)
    rng = np.random.default_rng(0)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for pred_index, gt_index in zip(matched_rows, matched_cols, strict=True):
        source = predicted[int(pred_index)]
        target = gt_meshes[gt_ids[int(gt_index)]]
        source_points = _sample_vertices(source, args.sample_count, rng)
        target_points = _sample_vertices(target, args.sample_count, rng)
        initial = _initial_centroid_transform(source, target)
        matrix, _, cost = trimesh.registration.icp(
            source_points,
            target_points,
            initial=initial,
            max_iterations=80,
            threshold=1e-8,
            scale=False,
            reflection=False,
        )
        aligned = source.copy()
        aligned.apply_transform(matrix)
        component = paths[int(pred_index)].stem.replace("fuse_post", "").rstrip("_")
        _export_overlay(output / f"{component}_raw_overlay.glb", source, target, "raw world-frame overlay")
        _export_overlay(output / f"{component}_rigid_icp_overlay.glb", aligned, target, "rigid ICP diagnostic overlay")
        source_centroid = _centroid(source)
        target_centroid = _centroid(target)
        records.append(
            {
                "predicted_mesh": paths[int(pred_index)].name,
                "gt_part_id": gt_ids[int(gt_index)],
                "raw_centroid_m": source_centroid.tolist(),
                "gt_centroid_m": target_centroid.tolist(),
                "centroid_distance_m": float(np.linalg.norm(source_centroid - target_centroid)),
                "predicted_bbox_diagonal_m": _bbox_diag(source),
                "gt_bbox_diagonal_m": _bbox_diag(target),
                "predicted_over_gt_bbox_diagonal": _bbox_diag(source) / max(_bbox_diag(target), 1e-12),
                "raw_voxel_iou_diagnostic": float(raw_iou[int(pred_index), int(gt_index)]),
                "rigid_icp_voxel_iou_diagnostic": _voxel_iou(aligned, target, args.voxel_size_m),
                "icp_surface_cost_m": float(cost),
                "rigid_icp_matrix": matrix.tolist(),
            }
        )
    payload = {
        "purpose": "diagnostic only; raw evaluation coordinates are not modified",
        "state": args.state,
        "voxel_size_m": args.voxel_size_m,
        "matching": records,
        "interpretation": (
            "A large raw-to-rigid-ICP IoU increase indicates a likely global pose/frame mismatch. "
            "A small increase indicates that low raw IoU is not explained by rigid alignment alone."
        ),
    }
    (output / "mesh_alignment_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
