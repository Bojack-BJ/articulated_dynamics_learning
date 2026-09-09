#!/usr/bin/env python3
"""Evaluate official AiM TSDF component meshes against simulation GT part meshes.

The predicted meshes must be produced by AiM's unmodified ``render_main.py``.
This is a mesh-voxel metric, unlike this project's observed-point IoU.  The
evaluation uses a shared world-frame grid and Hungarian matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("aim_run", type=Path)
    parser.add_argument("--state", choices=("start", "end"), default="end")
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"Could not load triangle mesh: {path}")
    return loaded


def _set_frame_qpos(model: Any, data: Any, episode: dict[str, Any], state: str) -> None:
    import mujoco

    frame = episode["frames"][0 if state == "start" else -1]
    data.qpos[:] = model.qpos0
    positions = frame.get("action_log", {}).get("joint_positions", {})
    for name, value in positions.items():
        if value is None:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id >= 0:
            data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
    mujoco.mj_forward(model, data)


def _gt_part_meshes(episode: dict[str, Any], state: str) -> dict[int, trimesh.Trimesh]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(Path(episode["metadata"]["model_path"])))
    data = mujoco.MjData(model)
    _set_frame_qpos(model, data, episode, state)
    parts = episode["metadata"]["part_segmentation"]["parts"]
    result: dict[int, trimesh.Trimesh] = {}
    for part in parts:
        meshes: list[trimesh.Trimesh] = []
        for geom_id in part.get("visible_geom_ids", []):
            geom_id = int(geom_id)
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int64)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            meshes.append(trimesh.Trimesh(vertices=vertices @ rotation.T + position, faces=faces, process=False))
        if meshes:
            result[int(part["part_id"])] = trimesh.util.concatenate(meshes)
    if not result:
        raise ValueError("No visual GT meshes were found in episode metadata")
    return result


def _voxel_keys(mesh: trimesh.Trimesh, pitch: float, origin: np.ndarray) -> set[tuple[int, int, int]]:
    # AiM's TSDF output can be open. Fill when possible but retain surface voxels.
    voxels = mesh.voxelized(pitch)
    try:
        voxels = voxels.fill()
    except BaseException:
        pass
    indices = np.rint((np.asarray(voxels.points) - origin) / pitch).astype(np.int64)
    return {tuple(int(value) for value in row) for row in indices}


def _full_object_mesh_path(run: Path, state: str) -> Path | None:
    """Return AiM's complete (pre-component-decomposition) TSDF mesh."""
    time = "0.0" if state == "start" else "1.0"
    path = run / f"motion_traj_t={time}" / "fuse_post.ply"
    return path if path.exists() else None


def main() -> int:
    args = parse_args()
    if args.voxel_size_m <= 0.0:
        raise ValueError("voxel-size-m must be positive")
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    gt_meshes = _gt_part_meshes(episode, args.state)
    # AiM's released renderer uses two output spellings because one code path
    # concatenates ``model_path`` without inserting a slash. Support both
    # forms without altering the baseline renderer.
    prediction_paths = sorted(
        [
            *args.aim_run.glob(f"sub_*_point_cloud_{args.state}/fuse_post.ply"),
            *args.aim_run.glob(f"sub_*_point_cloud_{args.state}fuse_post.ply"),
        ]
    )
    if not prediction_paths:
        raise FileNotFoundError("No official AiM TSDF meshes found; run AiM render_main.py first")
    predicted_meshes = [_load_mesh(path) for path in prediction_paths]
    all_bounds = np.concatenate([mesh.bounds for mesh in [*gt_meshes.values(), *predicted_meshes]])
    origin = np.floor(all_bounds.min(axis=0) / args.voxel_size_m) * args.voxel_size_m
    gt_voxels = {part_id: _voxel_keys(mesh, args.voxel_size_m, origin) for part_id, mesh in gt_meshes.items()}
    pred_voxels = [_voxel_keys(mesh, args.voxel_size_m, origin) for mesh in predicted_meshes]
    full_prediction_path = _full_object_mesh_path(args.aim_run, args.state)
    full_object: dict[str, Any] | None = None
    if full_prediction_path is not None:
        full_predicted_mesh = _load_mesh(full_prediction_path)
        # Recompute the common origin with the complete predicted mesh present.
        # This avoids a one-voxel origin shift between full-object and parts.
        full_bounds = np.concatenate([mesh.bounds for mesh in [*gt_meshes.values(), full_predicted_mesh]])
        full_origin = np.floor(full_bounds.min(axis=0) / args.voxel_size_m) * args.voxel_size_m
        full_predicted_voxels = _voxel_keys(full_predicted_mesh, args.voxel_size_m, full_origin)
        full_gt_voxels = set().union(*[_voxel_keys(mesh, args.voxel_size_m, full_origin) for mesh in gt_meshes.values()])
        full_union = full_predicted_voxels | full_gt_voxels
        full_object = {
            "predicted_mesh": str(full_prediction_path.name),
            "voxel_iou": len(full_predicted_voxels & full_gt_voxels) / len(full_union) if full_union else 0.0,
            "predicted_voxels": len(full_predicted_voxels),
            "gt_voxels": len(full_gt_voxels),
        }
    gt_ids = sorted(gt_voxels)
    iou = np.zeros((len(pred_voxels), len(gt_ids)), dtype=np.float64)
    for row, pred in enumerate(pred_voxels):
        for column, part_id in enumerate(gt_ids):
            union = pred | gt_voxels[part_id]
            iou[row, column] = len(pred & gt_voxels[part_id]) / len(union) if union else 0.0
    rows, columns = linear_sum_assignment(-iou)
    matching = [
        {
            "predicted_mesh": prediction_paths[int(row)].name,
            "gt_part_id": gt_ids[int(column)],
            "voxel_iou": float(iou[row, column]),
            "predicted_voxels": len(pred_voxels[int(row)]),
            "gt_voxels": len(gt_voxels[gt_ids[int(column)]]),
        }
        for row, column in zip(rows, columns, strict=True)
    ]
    payload = {
        "metric": "AiM-official-TSDF-mesh-to-simulation-visual-mesh-voxel-IoU",
        "state": args.state,
        "voxel_size_m": args.voxel_size_m,
        "predicted_mesh_count": len(predicted_meshes),
        "gt_part_count": len(gt_meshes),
        "mean_matched_voxel_iou": float(np.mean([row["voxel_iou"] for row in matching])) if matching else None,
        "full_object": full_object,
        "matching": matching,
        "note": (
            "Predictions are rendered and TSDF-fused by unmodified AiM render_main.py. "
            "GT is the simulation visual mesh at the same articulation state."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
