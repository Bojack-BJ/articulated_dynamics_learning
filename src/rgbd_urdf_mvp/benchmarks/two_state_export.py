"""Export fixed start/end RGB-D scans for two-state external baselines."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_CAMERA_CONVENTIONS = {
    "mujoco-gl-forward",
    "camera-to-world-forward",
}


def export_two_state_package(
    episode_path: Path,
    output_dir: Path,
    *,
    object_id: str | None = None,
    category: str | None = None,
    gt_part_count: int | None = None,
    required_views_per_state: int = 100,
    copy_mode: str = "hardlink",
    overwrite: bool = False,
) -> Path:
    """Export one episode to canonical layouts for released two-state baselines."""
    episode_path = episode_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if required_views_per_state < 1:
        raise ValueError("required_views_per_state must be positive")
    if copy_mode not in {"copy", "hardlink"}:
        raise ValueError("copy_mode must be 'copy' or 'hardlink'")

    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    metadata = dict(episode.get("metadata", {}))
    convention = str(metadata.get("camera_pose_convention", ""))
    if convention not in SUPPORTED_CAMERA_CONVENTIONS:
        raise ValueError(f"Unsupported camera convention: {convention!r}")
    protocol = dict(metadata.get("aim_protocol") or {})
    states = {
        "start": _load_scan(
            episode_path, protocol.get("static_scan_manifest"), "start"
        ),
        "end": _load_scan(episode_path, protocol.get("end_scan_manifest"), "end"),
    }
    for state, scan in states.items():
        if len(scan["views"]) != required_views_per_state:
            raise ValueError(
                f"{state} scan has {len(scan['views'])} views; "
                f"expected {required_views_per_state}"
            )

    resolved_object_id = object_id or str(
        metadata.get("object_id") or episode_path.parent.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    package_manifest = {
        "schema": "articulated-two-state-acquisition-v1",
        "object_id": resolved_object_id,
        "category": category or metadata.get("category"),
        "source_episode": str(episode_path),
        "source_camera_pose_convention": convention,
        "observation_uses_gt_part_labels": False,
        "states": {
            state: {
                "view_count": len(scan["views"]),
                "joint_positions": scan["joint_positions"],
                "camera_intrinsics": scan["camera_intrinsics"],
                "source_manifest": str(scan["manifest_path"]),
            }
            for state, scan in states.items()
        },
        "adapters": {
            "dta": {
                "path": "dta",
                "input": "two-state RGB-D, object masks, calibrated cameras",
            },
            "artgs": {
                "path": "artgs",
                "input": "two-state multi-view RGBA/depth and OpenGL cameras",
            },
            "paris": {
                "path": "paris",
                "input": "two-state multi-view RGBA and OpenGL cameras",
                "scope": "two-part, one-joint objects only",
            },
            "ditto": {
                "path": "ditto",
                "input": "normalized fused start/end object point clouds",
                "scope": "two-part, one-joint objects only",
            },
            "gaussianart": {
                "path": "gaussianart",
                "input": "two-state RGB-D and calibrated cameras",
                "status": "requires external part-semantic initialization",
            },
        },
    }
    _write_json(output_dir / "package_manifest.json", package_manifest)
    _write_json(
        output_dir / "oracle_requirements.json",
        {
            "schema": "external-baseline-oracle-requirements-v1",
            "object_id": resolved_object_id,
            "gt_part_count": gt_part_count,
            "dta": {
                "requires_exact_num_parts": True,
                "cli_argument": "--num_parts",
                "value_if_oracle_run": gt_part_count,
            },
            "artgs": {
                "requires_num_slots": True,
                "value_if_oracle_run": gt_part_count,
                "joint_type_oracle_required": False,
                "note": (
                    "The released coarse-to-predict path can predict joint types, "
                    "but the slot count is configured per scene."
                ),
            },
            "paris": {
                "requires_exact_num_parts": False,
                "native_capacity": 2,
                "joint_type_oracle_required": False,
                "note": "Use the official SE(3) config before specialized fitting.",
            },
            "ditto": {
                "requires_exact_num_parts": False,
                "native_capacity": 2,
                "checkpoint_prior": "official synthetic checkpoint",
            },
            "gaussianart": {
                "requires_exact_num_parts": True,
                "value_if_oracle_run": gt_part_count,
                "requires_part_semantic_initialization": True,
                "requires_gt_motion_metadata_in_released_run_py": True,
            },
        },
    )
    _export_dta(states, output_dir / "dta", episode_path.parent, copy_mode)
    _export_artgs(
        states,
        output_dir / "artgs",
        episode_path.parent,
        copy_mode,
        resolved_object_id,
    )
    _export_paris(states, output_dir / "paris", episode_path.parent, copy_mode)
    _export_ditto(states, output_dir / "ditto", episode_path.parent)
    _export_gaussianart_scaffold(
        output_dir / "artgs",
        output_dir / "gaussianart",
        resolved_object_id,
    )
    return output_dir / "package_manifest.json"


def camera_to_opengl(camera_to_world_forward: Any) -> list[list[float]]:
    """Convert recorder +Z-forward camera coordinates to OpenGL -Z-forward."""
    matrix = np.asarray(camera_to_world_forward, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 camera pose, got {matrix.shape}")
    return (matrix @ np.diag([1.0, 1.0, -1.0, 1.0])).tolist()


def _load_scan(episode_path: Path, relative_path: Any, state: str) -> dict[str, Any]:
    if not relative_path:
        raise ValueError(
            f"Episode does not contain a fixed {state} scan. "
            "Use recording_protocol=aim_style_fixed_end."
        )
    manifest_path = episode_path.parent / str(relative_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = list(payload.get("views", []))
    if not views:
        raise ValueError(f"{state} scan has no views: {manifest_path}")
    fixed_joint_positions = payload.get("joint_positions", {})
    for view in views:
        if view.get("joint_positions", {}) != fixed_joint_positions:
            raise ValueError(f"{state} joint state changed during the fixed scan")
        for key in ("rgb_path", "depth_path", "mask_path", "camera_pose"):
            if view.get(key) is None:
                raise ValueError(f"{state} view is missing {key}: {view}")
    intrinsics = dict(payload.get("camera_intrinsics") or {})
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise ValueError(f"{state} scan camera intrinsics are missing {key}")
    return {
        "manifest_path": manifest_path,
        "views": views,
        "joint_positions": fixed_joint_positions,
        "camera_intrinsics": intrinsics,
    }


def _export_dta(
    states: dict[str, dict[str, Any]],
    output_dir: Path,
    episode_root: Path,
    copy_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = states["start"]["camera_intrinsics"]
    np.savetxt(
        output_dir / "cam_K.txt",
        np.asarray(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        fmt="%.12g",
    )
    keyframes: dict[str, Any] = {}
    frame_index = 0
    for state_index, state in enumerate(("start", "end")):
        for view in states[state]["views"]:
            frame_id = f"{state_index}_{int(view.get('view_index', frame_index)):03d}"
            rgb_source = episode_root / view["rgb_path"]
            depth_source = episode_root / view["depth_path"]
            mask_source = episode_root / view["mask_path"]
            _write_segmented_rgb(
                rgb_source,
                mask_source,
                output_dir / "color_segmented" / f"{frame_id}.png",
            )
            _copy_or_link(
                depth_source,
                output_dir / "depth_filtered" / f"{frame_id}.png",
                copy_mode,
            )
            _write_binary_mask(
                mask_source, output_dir / "mask" / f"{frame_id}.png"
            )
            keyframes[f"frame_{frame_id}"] = {
                "cam_in_ob": np.asarray(
                    camera_to_opengl(view["camera_pose"]), dtype=np.float64
                ).reshape(-1).tolist(),
                "time": float(state_index),
                "state": state,
                "source_view_index": int(view.get("view_index", frame_index)),
            }
            frame_index += 1
    # JSON is valid YAML and avoids a runtime PyYAML dependency in the exporter.
    _write_json(output_dir / "init_keyframes.yml", keyframes)
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "digital-twin-art-input-v1",
            "upstream_commit": "1a48b402e4bf4bb7731296e8e230f0db3d86fe4f",
            "camera_pose_convention": "opengl-camera-to-world-minus-z-forward",
            "depth_unit": "millimeter_uint16",
            "object_masks_only": True,
            "part_masks_exported": False,
            "uses_gt_part_count": False,
            "note": "Running DTA still requires an explicit --num_parts oracle value.",
        },
    )


def _export_artgs(
    states: dict[str, dict[str, Any]],
    output_dir: Path,
    episode_root: Path,
    copy_mode: str,
    object_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_payloads: dict[str, dict[str, Any]] = {}
    for state in ("start", "end"):
        scan = states[state]
        intrinsics = scan["camera_intrinsics"]
        frames: list[dict[str, Any]] = []
        first_rgb = episode_root / scan["views"][0]["rgb_path"]
        width, height = Image.open(first_rgb).size
        for output_index, view in enumerate(scan["views"]):
            stem = f"{output_index:04d}"
            rgba_path = output_dir / state / "train" / "rgba" / f"{stem}.png"
            depth_path = output_dir / state / "train" / "depth" / f"{stem}.png"
            _write_rgba(
                episode_root / view["rgb_path"],
                episode_root / view["mask_path"],
                rgba_path,
            )
            _copy_or_link(
                episode_root / view["depth_path"], depth_path, copy_mode
            )
            frames.append(
                {
                    "file_path": f"./{state}/train/rgba/{stem}",
                    "depth_path": f"./{state}/train/depth/{stem}.png",
                    "time": 0.0 if state == "start" else 1.0,
                    "transform_matrix": camera_to_opengl(view["camera_pose"]),
                    "source_view_index": int(view.get("view_index", output_index)),
                }
            )
        payload = {
            "camera_angle_x": 2.0
            * math.atan(width / (2.0 * float(intrinsics["fx"]))),
            "camera_angle_y": 2.0
            * math.atan(height / (2.0 * float(intrinsics["fy"]))),
            "frames": frames,
        }
        state_payloads[state] = payload
        _write_json(output_dir / f"transforms_train_{state}.json", payload)
    # ArtGS detects Blender-style scenes through this conventional filename,
    # while its two-state loader reads the start/end manifests above.
    _write_json(output_dir / "transforms_train.json", state_payloads["start"])
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "artgs-two-state-input-v1",
            "object_id": object_id,
            "upstream_commit": "7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a",
            "camera_pose_convention": "blender-opengl-minus-z-forward",
            "object_masks_encoded_as_alpha": True,
            "part_masks_exported": False,
            "uses_gt_part_count": False,
            "released_path_note": (
                "The released implementation reads depth for initialization/losses "
                "even though the method is commonly summarized as two-state RGB."
            ),
            "oracle_note": (
                "The official scripts require a per-scene num_slots value. "
                "Use oracle_requirements.json only in explicitly labeled oracle runs."
            ),
        },
    )


def _export_paris(
    states: dict[str, dict[str, Any]],
    output_dir: Path,
    episode_root: Path,
    copy_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for state in ("start", "end"):
        scan = states[state]
        intrinsics = scan["camera_intrinsics"]
        camera_payload: dict[str, Any] = {
            "K": [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ]
        }
        for index, view in enumerate(scan["views"]):
            stem = f"{index:04d}"
            for split in ("train", "val", "test"):
                _write_rgba(
                    episode_root / view["rgb_path"],
                    episode_root / view["mask_path"],
                    output_dir / state / split / f"{stem}.png",
                )
            camera_payload[stem] = camera_to_opengl(view["camera_pose"])
        for split in ("train", "val", "test"):
            _write_json(
                output_dir / state / f"camera_{split}.json", camera_payload
            )
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "paris-two-state-input-v1",
            "camera_pose_convention": "blender-opengl-minus-z-forward",
            "object_masks_encoded_as_alpha": True,
            "part_masks_exported": False,
            "native_model_capacity": "one static part, one moving part, one joint",
            "recommended_first_config": "configs/se3.yaml",
        },
    )


def _export_ditto(
    states: dict[str, dict[str, Any]],
    output_dir: Path,
    episode_root: Path,
    *,
    point_count: int = 8192,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clouds = {
        state: _fuse_masked_scan_points(states[state], episode_root)
        for state in ("start", "end")
    }
    combined = np.concatenate([clouds["start"], clouds["end"]], axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    center = (lower + upper) * 0.5
    scale = float(np.max(upper - lower) * 1.1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Ditto point cloud has invalid normalization scale")
    rng = np.random.default_rng(0)
    sampled: dict[str, np.ndarray] = {}
    for state, points in clouds.items():
        indices = rng.choice(
            len(points), size=point_count, replace=len(points) < point_count
        )
        sampled[state] = ((points[indices] - center) / scale).astype(np.float32)
    np.savez_compressed(
        output_dir / "two_state_points.npz",
        pc_start=sampled["start"],
        pc_end=sampled["end"],
        normalization_center=center.astype(np.float32),
        normalization_scale=np.float32(scale),
    )
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "ditto-two-state-pointcloud-v1",
            "point_count_per_state": point_count,
            "normalization": "(x - joint_bbox_center) / (1.1 * max_bbox_extent)",
            "native_model_capacity": "one static part, one moving part, one joint",
            "uses_gt_part_labels": False,
        },
    )


def _fuse_masked_scan_points(
    scan: dict[str, Any], episode_root: Path
) -> np.ndarray:
    intrinsics = scan["camera_intrinsics"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    clouds: list[np.ndarray] = []
    for view in scan["views"]:
        depth = np.asarray(Image.open(episode_root / view["depth_path"]))
        if np.issubdtype(depth.dtype, np.integer):
            depth = depth.astype(np.float64) / 1000.0
        else:
            depth = depth.astype(np.float64)
        mask = np.asarray(Image.open(episode_root / view["mask_path"])) > 0
        valid = mask & np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        z = depth[ys, xs]
        camera_points = np.column_stack(
            ((xs - cx) * z / fx, -(ys - cy) * z / fy, z)
        )
        pose = np.asarray(view["camera_pose"], dtype=np.float64)
        world = camera_points @ pose[:3, :3].T + pose[:3, 3]
        clouds.append(world)
    if not clouds:
        raise ValueError("No valid masked depth points for Ditto export")
    return np.concatenate(clouds, axis=0)


def _export_gaussianart_scaffold(
    artgs_dir: Path, output_dir: Path, object_id: str
) -> None:
    shutil.copytree(artgs_dir, output_dir, dirs_exist_ok=True)
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "gaussianart-two-state-input-v1",
            "object_id": object_id,
            "upstream_commit": "e296670c3864554451005142be10638ec4d4d2be",
            "rgbd_and_camera_data_ready": True,
            "part_semantic_initialization_ready": False,
            "gt_motion_metadata_ready": False,
            "blocked_files": [
                "per-view semantic/*.npy or equivalent Art-SAM predictions",
                "gt/trans.json required by the released run.py",
            ],
            "note": (
                "The released code derives num_parts and prismatic indices from "
                "gt/trans.json. This scaffold does not fabricate those oracle inputs."
            ),
        },
    )


def _write_segmented_rgb(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path)) > 0
    segmented = np.where(mask[..., None], rgb, 0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(segmented, mode="RGB").save(output_path)


def _write_binary_mask(source: Path, destination: Path) -> None:
    mask = (np.asarray(Image.open(source)) > 0).astype(np.uint8) * 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(destination)


def _write_rgba(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    mask = (np.asarray(Image.open(mask_path)) > 0).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack((rgb, mask)), mode="RGBA").save(output_path)


def _copy_or_link(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
