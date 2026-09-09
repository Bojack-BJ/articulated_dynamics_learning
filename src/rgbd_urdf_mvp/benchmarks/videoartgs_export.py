"""Export recorded monocular RGB-D interactions to VideoArtGS inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def export_videoartgs_native_package(
    static_cameras_path: Path,
    interaction_episode_path: Path,
    joint_infos_path: Path,
    output_dir: Path,
    *,
    joint_inventory_source: str,
    overwrite: bool = False,
) -> Path:
    """Export VideoArtGS' canonical-scan plus monocular-interaction protocol.

    The released VideoArtGS preprocessing assumes the first 100 observations
    describe one canonical articulation state. Track extraction is deliberately
    left to the official TAPIP3D preprocessing script.
    """
    if joint_inventory_source not in {"vlm", "manual", "gt_oracle"}:
        raise ValueError("joint_inventory_source must be vlm, manual, or gt_oracle")
    static_cameras_path = static_cameras_path.expanduser().resolve()
    interaction_episode_path = interaction_episode_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    static = json.loads(static_cameras_path.read_text(encoding="utf-8"))
    interaction = json.loads(interaction_episode_path.read_text(encoding="utf-8"))
    static_views = list(static.get("views", []))
    if len(static_views) != 100:
        raise ValueError(
            "VideoArtGS native self-captured protocol requires exactly "
            f"100 canonical views, got {len(static_views)}"
        )
    interaction_frames = list(interaction.get("frames", []))
    if not interaction_frames:
        raise ValueError("Interaction episode contains no frames")

    static_root = _resolve_static_scan_root(static_cameras_path)
    interaction_root = interaction_episode_path.parent
    observations: list[dict[str, Any]] = []
    for view in static_views:
        observations.append(
            {
                "rgb": static_root / view["rgb_path"],
                "depth": static_root / view["depth_path"],
                "mask": static_root / view["mask_path"],
                "camera_pose": view["camera_pose"],
                "stage": "canonical",
            }
        )
    for frame in interaction_frames:
        rgb_paths = frame.get("rgb_paths_by_view") or [frame["rgb_path"]]
        depth_paths = frame.get("depth_paths_by_view") or [frame["depth_path"]]
        mask_paths = frame.get("mask_paths_by_view") or [frame["mask_path"]]
        camera_poses = frame.get("camera_poses_by_view") or [frame["camera_pose"]]
        if len(rgb_paths) != 1:
            raise ValueError("VideoArtGS interaction must be monocular per timestep")
        observations.append(
            {
                "rgb": interaction_root / rgb_paths[0],
                "depth": interaction_root / depth_paths[0],
                "mask": interaction_root / mask_paths[0],
                "camera_pose": camera_poses[0],
                "stage": "interaction",
            }
        )

    intrinsics = dict(interaction["camera_intrinsics"])
    static_intrinsics = dict(static["camera_intrinsics"])
    if any(
        not np.isclose(float(intrinsics[key]), float(static_intrinsics[key]))
        for key in ("fx", "fy", "cx", "cy")
    ):
        raise ValueError("Static and interaction camera intrinsics do not match")

    video, depths, masks, poses = _load_observations(observations)
    k = _intrinsics_matrix(intrinsics)
    c2w, extrinsics = _convert_camera_poses(poses)
    video_array = np.stack(video).astype(np.uint8)
    depth_array = np.stack(depths).astype(np.float32)
    mask_array = np.stack(masks).astype(np.uint8)
    frame_times = np.concatenate(
        (
            np.zeros(len(static_views), dtype=np.float32),
            np.linspace(
                0.0,
                1.0,
                len(interaction_frames),
                dtype=np.float32,
            ),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (rgb, mask) in enumerate(zip(video_array, mask_array)):
        image_path = output_dir / "images" / f"{index:06d}.png"
        mask_path = output_dir / "masks" / f"{index:06d}.npy"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(image_path)
        np.save(mask_path, np.stack((mask, np.zeros_like(mask))))
    _write_initial_point_cloud(
        output_dir / "point_cloud.ply",
        video_array[:100],
        depth_array[:100],
        mask_array[:100],
        c2w[:100],
        k,
        stride=8,
    )
    np.savez_compressed(
        output_dir / "data.npz",
        video=video_array.transpose(0, 3, 1, 2),
        depths=depth_array,
        intrinsics=np.repeat(k[None], len(video_array), axis=0),
        poses=c2w,
        extrinsics=extrinsics,
        masks=mask_array[:, None],
        frame_times=frame_times,
        canonical_frame_count=np.asarray(100, dtype=np.int32),
    )
    _write_videoartgs_joint_inventory(
        joint_infos_path,
        output_dir / "joint_infos_vlm.json",
    )
    width, height = video_array.shape[2], video_array.shape[1]
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "videoartgs-native-self-captured-input-v1",
            "static_cameras": str(static_cameras_path),
            "interaction_episode": str(interaction_episode_path),
            "canonical_frame_count": 100,
            "interaction_frame_count": len(interaction_frames),
            "frame_count": len(video_array),
            "image_size": [width, height],
            "uses_gt_part_labels": False,
            "camera_depth_source": "recorded_gt_controlled_variant",
            "track_source": "official_tapip3d_required",
            "filtered_tracks_present": False,
            "joint_inventory_source": joint_inventory_source,
            "joint_inventory_is_oracle": joint_inventory_source == "gt_oracle",
        },
    )
    return output_dir / "adapter_manifest.json"


def export_videoartgs_package(
    episode_path: Path,
    tracks_path: Path,
    joint_infos_path: Path,
    output_dir: Path,
    *,
    joint_inventory_source: str,
    view_index: int = 0,
    overwrite: bool = False,
) -> Path:
    """Create VideoArtGS ``data.npz`` and ``filtered.npz`` without GT labels."""
    if joint_inventory_source not in {"vlm", "manual", "gt_oracle"}:
        raise ValueError("joint_inventory_source must be vlm, manual, or gt_oracle")
    episode_path = episode_path.expanduser().resolve()
    tracks_path = tracks_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    frames = list(episode.get("frames", []))
    sampled = list(tracks.get("sampled_frame_indices", range(len(frames))))
    if len(sampled) != int(tracks.get("frame_count", len(sampled))):
        raise ValueError("Track frame metadata is inconsistent")

    intrinsics = dict(episode["camera_intrinsics"])
    video, depths, masks, poses = [], [], [], []
    for source_index in sampled:
        frame = frames[int(source_index)]
        rgb_paths = frame.get("rgb_paths_by_view") or [frame["rgb_path"]]
        depth_paths = frame.get("depth_paths_by_view") or [frame["depth_path"]]
        mask_paths = frame.get("mask_paths_by_view") or [frame["mask_path"]]
        camera_poses = frame.get("camera_poses_by_view") or [frame["camera_pose"]]
        if view_index >= len(rgb_paths):
            raise ValueError(f"Frame {source_index} has no view {view_index}")
        video.append(
            np.asarray(Image.open(episode_path.parent / rgb_paths[view_index]).convert("RGB"))
        )
        depth = np.asarray(Image.open(episode_path.parent / depth_paths[view_index]))
        depths.append(depth.astype(np.float32) / 1000.0)
        masks.append(
            (np.asarray(Image.open(episode_path.parent / mask_paths[view_index])) > 0)
            .astype(np.uint8)
        )
        poses.append(np.asarray(camera_poses[view_index], dtype=np.float32))

    height, width = video[0].shape[:2]
    k = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # VideoArtGS stores OpenCV world-to-camera extrinsics in data.npz.
    c2w = np.stack(poses)
    c2w[:, :3, :3] = c2w[:, :3, :3] @ np.diag([1.0, -1.0, -1.0])
    extrinsics = np.linalg.inv(c2w).astype(np.float32)
    video_array = np.stack(video).astype(np.uint8)
    depth_array = np.stack(depths).astype(np.float32)
    mask_array = np.stack(masks).astype(np.uint8)
    data = {
        "video": video_array.transpose(0, 3, 1, 2),
        "depths": depth_array,
        "intrinsics": np.repeat(k[None], len(video), axis=0),
        "poses": c2w.astype(np.float32),
        "extrinsics": extrinsics,
        "masks": mask_array[:, None],
    }

    selected_tracks = [
        track for track in tracks.get("tracks", [])
        if int(track.get("view_index", 0)) == view_index
    ]
    coords = np.full((len(sampled), len(selected_tracks), 3), np.nan, np.float32)
    visibs = np.zeros((len(sampled), len(selected_tracks)), bool)
    for column, track in enumerate(selected_tracks):
        samples = track.get("samples", [])
        if len(samples) != len(sampled):
            raise ValueError(f"Track {track.get('track_id')} has wrong sample count")
        for row, sample in enumerate(samples):
            xyz = sample.get("xyz_world")
            visible = bool(sample.get("visible")) and xyz is not None
            if xyz is not None:
                coords[row, column] = np.asarray(xyz, dtype=np.float32)
            visibs[row, column] = visible
    coords, visibs = _fill_track_gaps(coords, visibs)

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (rgb, mask) in enumerate(zip(video_array, mask_array)):
        image_path = output_dir / "images" / f"{index:06d}.png"
        mask_path = output_dir / "masks" / f"{index:06d}.npy"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(image_path)
        np.save(mask_path, np.stack((mask, np.zeros_like(mask))))
    _write_initial_point_cloud(
        output_dir / "point_cloud.ply",
        video_array,
        depth_array,
        mask_array,
        c2w,
        k,
        stride=8,
    )
    np.savez_compressed(output_dir / "data.npz", **data)
    np.savez_compressed(output_dir / "filtered.npz", coords=coords, visibs=visibs)
    shutil.copy2(joint_infos_path, output_dir / "joint_infos.json")
    _write_json(
        output_dir / "adapter_manifest.json",
        {
            "schema": "videoartgs-known-rgbd-input-v1",
            "source_episode": str(episode_path),
            "source_tracks": str(tracks_path),
            "frame_count": len(video),
            "image_size": [width, height],
            "view_index": view_index,
            "track_count": int(coords.shape[1]),
            "uses_gt_part_labels": False,
            "camera_depth_source": "recorded_gt_controlled_variant",
            "joint_inventory_source": joint_inventory_source,
            "joint_inventory_is_oracle": joint_inventory_source == "gt_oracle",
        },
    )
    return output_dir / "adapter_manifest.json"


def _resolve_static_scan_root(cameras_path: Path) -> Path:
    """Resolve paths stored relative to the recording episode root."""
    for candidate in (cameras_path.parent, *cameras_path.parents):
        probe = candidate / "aim_protocol/static_scan"
        if probe.resolve() == cameras_path.parent.resolve():
            return candidate
    return cameras_path.parent


def _write_videoartgs_joint_inventory(source: Path, destination: Path) -> None:
    """Convert an internal joint inventory to VideoArtGS' VLM schema."""
    entries = json.loads(source.read_text(encoding="utf-8"))
    inventory = []
    for index, entry in enumerate(entries):
        raw_type = str(entry.get("joint_type", entry.get("joint", ""))).lower()
        if raw_type in {"r", "revolute", "hinge"}:
            joint = "hinge"
        elif raw_type in {"p", "prismatic", "slider"}:
            joint = "slider"
        else:
            continue
        inventory.append(
            {
                "id": len(inventory) + 1,
                "name": str(entry.get("name", f"moving_part_{index}")),
                "joint": joint,
                "parent": int(entry.get("parent", 0)),
            }
        )
    if not inventory:
        raise ValueError("Joint inventory contains no revolute or prismatic joints")
    _write_json(destination, inventory)


def _load_observations(
    observations: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    video: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for observation in observations:
        video.append(np.asarray(Image.open(observation["rgb"]).convert("RGB")))
        depth = np.asarray(Image.open(observation["depth"]))
        depths.append(depth.astype(np.float32) / 1000.0)
        masks.append(
            (np.asarray(Image.open(observation["mask"])) > 0).astype(np.uint8)
        )
        poses.append(np.asarray(observation["camera_pose"], dtype=np.float32))
    return video, depths, masks, np.stack(poses)


def _intrinsics_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _convert_camera_poses(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert recorder poses to forward C2W and OpenCV W2C matrices.

    MuJoCo recordings already use a positive-Z forward camera frame with
    positive Y pointing up. TAPIP3D expects OpenCV world-to-camera matrices,
    so only the camera Y axis changes sign.
    """
    c2w = np.asarray(poses, dtype=np.float32).copy()
    recorder_to_opencv = np.diag([1.0, -1.0, 1.0, 1.0])
    extrinsics = recorder_to_opencv @ np.linalg.inv(c2w)
    return c2w, extrinsics.astype(np.float32)


def _fill_track_gaps(
    coords: np.ndarray,
    visibs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill coordinate gaps without changing the track visibility signal."""
    keep = np.sum(np.all(np.isfinite(coords), axis=2), axis=0) >= 2
    coords = coords[:, keep].copy()
    visibs = visibs[:, keep].copy()
    frame_indices = np.arange(coords.shape[0], dtype=np.float32)
    for track_index in range(coords.shape[1]):
        finite = np.all(np.isfinite(coords[:, track_index]), axis=1)
        source_frames = frame_indices[finite]
        for axis in range(3):
            coords[:, track_index, axis] = np.interp(
                frame_indices,
                source_frames,
                coords[finite, track_index, axis],
            )
    return coords, visibs


def _write_initial_point_cloud(
    path: Path,
    images: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int,
) -> None:
    """Write a deterministic masked RGB-D fusion for VideoArtGS initialization."""
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    height, width = depths.shape[1:]
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    for image, depth, mask, pose in zip(images, depths, masks, poses):
        z = depth[vv, uu]
        valid = (z > 0.0) & (mask[vv, uu] > 0)
        if not np.any(valid):
            continue
        z = z[valid]
        x = (uu[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = -(vv[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera = np.stack((x, y, z), axis=1)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        points.append(world.astype(np.float32))
        colors.append(image[vv[valid], uu[valid]].astype(np.uint8))
    if not points:
        raise ValueError("No masked depth points available for VideoArtGS initialization")
    cloud = np.concatenate(points, axis=0)
    rgb = np.concatenate(colors, axis=0)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(cloud)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                *(
                    " ".join(
                        (
                            *(str(float(value)) for value in point),
                            *(str(int(value)) for value in color),
                        )
                    )
                    for point, color in zip(cloud, rgb)
                ),
            ]
        )
        + "\n",
        encoding="ascii",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
