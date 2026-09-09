from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.benchmarks.two_state_export import (
    camera_to_opengl,
    export_two_state_package,
)


def test_camera_to_opengl_flips_forward_axis() -> None:
    converted = np.asarray(camera_to_opengl(np.eye(4)))
    np.testing.assert_allclose(converted, np.diag([1.0, 1.0, -1.0, 1.0]))


def test_exports_two_state_baselines_without_part_masks(tmp_path: Path) -> None:
    episode_dir = tmp_path / "recording"
    views_by_state = {}
    for state, joint_value in (("static_scan", 0.0), ("end_scan", 1.0)):
        views = []
        for index in range(2):
            view_dir = episode_dir / "aim_protocol" / state / f"view_{index:03d}"
            view_dir.mkdir(parents=True)
            Image.new("RGB", (8, 6), (20 + index, 30, 40)).save(
                view_dir / "rgb.png"
            )
            Image.fromarray(
                np.full((6, 8), 1000 + index, dtype=np.uint16)
            ).save(view_dir / "depth.png")
            mask = np.zeros((6, 8), dtype=np.uint16)
            mask[1:5, 2:7] = 65535
            Image.fromarray(mask).save(view_dir / "mask.png")
            Image.fromarray(np.where(mask > 0, 7, 0).astype(np.uint16)).save(
                view_dir / "part_mask.png"
            )
            views.append(
                {
                    "view_index": index,
                    "rgb_path": str(
                        (view_dir / "rgb.png").relative_to(episode_dir)
                    ),
                    "depth_path": str(
                        (view_dir / "depth.png").relative_to(episode_dir)
                    ),
                    "mask_path": str(
                        (view_dir / "mask.png").relative_to(episode_dir)
                    ),
                    "part_mask_path": str(
                        (view_dir / "part_mask.png").relative_to(episode_dir)
                    ),
                    "camera_pose": np.eye(4).tolist(),
                    "joint_positions": {"joint_0": joint_value},
                }
            )
        payload = {
            "camera_intrinsics": {"fx": 10, "fy": 10, "cx": 4, "cy": 3},
            "joint_positions": {"joint_0": joint_value},
            "views": views,
        }
        manifest = episode_dir / "aim_protocol" / state / "cameras.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        views_by_state[state] = manifest

    episode = {
        "metadata": {
            "camera_pose_convention": "mujoco-gl-forward",
            "recording_protocol": "aim_style_fixed_end",
            "aim_protocol": {
                "static_scan_manifest": str(
                    views_by_state["static_scan"].relative_to(episode_dir)
                ),
                "end_scan_manifest": str(
                    views_by_state["end_scan"].relative_to(episode_dir)
                ),
            },
        }
    }
    episode_path = episode_dir / "episode.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    output = tmp_path / "export"

    manifest_path = export_two_state_package(
        episode_path,
        output,
        object_id="partnet_test",
        category="drawer",
        gt_part_count=2,
        required_views_per_state=2,
    )

    package = json.loads(manifest_path.read_text(encoding="utf-8"))
    oracle = json.loads(
        (output / "oracle_requirements.json").read_text(encoding="utf-8")
    )
    dta_frames = json.loads(
        (output / "dta" / "init_keyframes.yml").read_text(encoding="utf-8")
    )
    artgs_start = json.loads(
        (output / "artgs" / "transforms_train_start.json").read_text(
            encoding="utf-8"
        )
    )
    artgs_compatibility_manifest = json.loads(
        (output / "artgs" / "transforms_train.json").read_text(encoding="utf-8")
    )
    assert package["observation_uses_gt_part_labels"] is False
    assert oracle["dta"]["requires_exact_num_parts"] is True
    assert len(dta_frames) == 4
    assert len(artgs_start["frames"]) == 2
    assert artgs_compatibility_manifest == artgs_start
    assert not list(output.rglob("*part_mask*"))
    assert Image.open(output / "artgs/start/train/rgba/0000.png").mode == "RGBA"
    assert np.asarray(Image.open(output / "dta/mask/0_000.png")).max() == 255
    assert Image.open(output / "paris/start/train/0000.png").mode == "RGBA"
    paris_camera = json.loads(
        (output / "paris/start/camera_train.json").read_text(encoding="utf-8")
    )
    assert set(paris_camera) == {"K", "0000", "0001"}
    ditto = np.load(output / "ditto/two_state_points.npz")
    assert ditto["pc_start"].shape == (8192, 3)
    assert ditto["pc_end"].shape == (8192, 3)
    gaussianart = json.loads(
        (output / "gaussianart/adapter_manifest.json").read_text(encoding="utf-8")
    )
    assert gaussianart["part_semantic_initialization_ready"] is False
