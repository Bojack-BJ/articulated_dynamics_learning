from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.benchmarks.videoartgs_export import (
    _fill_track_gaps,
    export_videoartgs_native_package,
    export_videoartgs_package,
)


def test_videoartgs_export_keeps_time_and_marks_inventory_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episode"
    frames = []
    for index in range(2):
        asset = root / "assets" / f"{index}"
        asset.mkdir(parents=True)
        Image.new("RGB", (4, 3), (index, 2, 3)).save(asset / "rgb.png")
        Image.fromarray(np.full((3, 4), 1000, np.uint16)).save(asset / "depth.png")
        Image.fromarray(np.full((3, 4), 255, np.uint8)).save(asset / "mask.png")
        frames.append(
            {
                "rgb_path": f"assets/{index}/rgb.png",
                "depth_path": f"assets/{index}/depth.png",
                "mask_path": f"assets/{index}/mask.png",
                "camera_pose": np.eye(4).tolist(),
            }
        )
    episode = {
        "camera_intrinsics": {"fx": 4, "fy": 4, "cx": 1.5, "cy": 1},
        "frames": frames,
    }
    episode_path = root / "episode.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    tracks = {
        "frame_count": 2,
        "sampled_frame_indices": [0, 1],
        "tracks": [
            {
                "track_id": 1,
                "view_index": 0,
                "samples": [
                    {"visible": True, "xyz_world": [0, 0, 1]},
                    {"visible": True, "xyz_world": [0.1, 0, 1]},
                ],
            }
        ],
    }
    tracks_path = root / "tracks.json"
    tracks_path.write_text(json.dumps(tracks), encoding="utf-8")
    joints_path = root / "joint_infos.json"
    joints_path.write_text(
        json.dumps([{"joint_type": "r", "parent": 0}]), encoding="utf-8"
    )

    output = tmp_path / "videoartgs"
    manifest_path = export_videoartgs_package(
        episode_path,
        tracks_path,
        joints_path,
        output,
        joint_inventory_source="vlm",
    )
    data = np.load(output / "data.npz")
    filtered = np.load(output / "filtered.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["video"].shape == (2, 3, 3, 4)
    assert data["poses"].shape == (2, 4, 4)
    assert data["masks"].shape == (2, 1, 3, 4)
    assert filtered["coords"].shape == (2, 1, 3)
    assert (output / "images/000000.png").exists()
    assert np.load(output / "masks/000000.npy").shape == (2, 3, 4)
    assert (output / "point_cloud.ply").exists()
    ply_text = (output / "point_cloud.ply").read_text(encoding="ascii")
    assert "property uchar red" in ply_text
    assert "property uchar green" in ply_text
    assert "property uchar blue" in ply_text
    assert manifest["joint_inventory_source"] == "vlm"
    assert manifest["joint_inventory_is_oracle"] is False


def test_fill_track_gaps_interpolates_coordinates_but_preserves_visibility() -> None:
    coords = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan]],
            [[3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    visibility = np.asarray([[True], [False], [True]])

    filled, filled_visibility = _fill_track_gaps(coords, visibility)

    np.testing.assert_allclose(filled[:, 0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(filled_visibility[:, 0], visibility[:, 0])


def test_native_export_combines_100_static_views_and_monocular_interaction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recording"
    scan = root / "aim_protocol/static_scan"
    scan.mkdir(parents=True)
    views = []
    for index in range(100):
        view = scan / f"view_{index:03d}"
        view.mkdir()
        Image.new("RGB", (4, 3), (index, 2, 3)).save(view / "rgb.png")
        Image.fromarray(np.full((3, 4), 1000, np.uint16)).save(view / "depth.png")
        Image.fromarray(np.full((3, 4), 255, np.uint8)).save(view / "mask.png")
        views.append(
            {
                "rgb_path": f"aim_protocol/static_scan/view_{index:03d}/rgb.png",
                "depth_path": f"aim_protocol/static_scan/view_{index:03d}/depth.png",
                "mask_path": f"aim_protocol/static_scan/view_{index:03d}/mask.png",
                "camera_pose": np.eye(4).tolist(),
            }
        )
    intrinsics = {"fx": 4, "fy": 4, "cx": 1.5, "cy": 1}
    cameras_path = scan / "cameras.json"
    cameras_path.write_text(
        json.dumps({"camera_intrinsics": intrinsics, "views": views}),
        encoding="utf-8",
    )

    interaction_root = tmp_path / "interaction"
    frames = []
    for index in range(2):
        assets = interaction_root / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), (100 + index, 2, 3)).save(
            assets / f"{index}_rgb.png"
        )
        Image.fromarray(np.full((3, 4), 1000, np.uint16)).save(
            assets / f"{index}_depth.png"
        )
        Image.fromarray(np.full((3, 4), 255, np.uint8)).save(
            assets / f"{index}_mask.png"
        )
        pose = np.eye(4)
        pose[0, 3] = index
        frames.append(
            {
                "rgb_path": f"assets/{index}_rgb.png",
                "depth_path": f"assets/{index}_depth.png",
                "mask_path": f"assets/{index}_mask.png",
                "camera_pose": pose.tolist(),
            }
        )
    episode_path = interaction_root / "episode.json"
    episode_path.write_text(
        json.dumps({"camera_intrinsics": intrinsics, "frames": frames}),
        encoding="utf-8",
    )
    joints_path = tmp_path / "joint_infos.json"
    joints_path.write_text(
        json.dumps(
            [
                {"joint_type": "s"},
                {"joint_type": "r", "name": "door", "parent": 0},
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "native"
    manifest_path = export_videoartgs_native_package(
        cameras_path,
        episode_path,
        joints_path,
        output,
        joint_inventory_source="manual",
    )

    data = np.load(output / "data.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["video"].shape == (102, 3, 3, 4)
    assert data["frame_times"].tolist() == [0.0] * 101 + [1.0]
    assert int(data["canonical_frame_count"]) == 100
    expected_w2c = np.diag([1.0, -1.0, 1.0, 1.0]) @ np.linalg.inv(
        np.asarray(frames[0]["camera_pose"])
    )
    np.testing.assert_allclose(data["extrinsics"][100], expected_w2c)
    np.testing.assert_allclose(data["poses"][100], frames[0]["camera_pose"])
    assert not (output / "filtered.npz").exists()
    assert not (output / "joint_infos.json").exists()
    assert json.loads((output / "joint_infos_vlm.json").read_text()) == [
        {"id": 1, "name": "door", "joint": "hinge", "parent": 0}
    ]
    assert manifest["track_source"] == "official_tapip3d_required"
    assert manifest["filtered_tracks_present"] is False
