import json
import csv

import numpy as np
from PIL import Image

from scripts.export_episode_to_artipoint import (
    artipoint_camera_to_world,
    camera_info,
    export_episode,
)


def test_camera_info_contains_expected_intrinsics() -> None:
    text = camera_info(640, 480, {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0})
    assert "width: 640" in text
    assert "height: 480" in text
    assert "K: (500.0, 0.0, 320.0, 0.0, 501.0, 240.0" in text


def test_mujoco_camera_pose_is_converted_to_proper_rotation() -> None:
    converted = artipoint_camera_to_world(
        np.diag([1.0, -1.0, 1.0, 1.0]), convention="mujoco-gl-forward"
    )
    np.testing.assert_allclose(converted, np.eye(4))
    assert np.isclose(np.linalg.det(converted[:3, :3]), 1.0)


def test_export_records_camera_convention(tmp_path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    frames = []
    for index in range(2):
        rgb = assets / f"rgb_{index}.png"
        depth = assets / f"depth_{index}.png"
        mask = assets / f"mask_{index}.png"
        Image.new("RGB", (4, 3), (10, 20, 30)).save(rgb)
        Image.new("I;16", (4, 3), 1000).save(depth)
        Image.new("L", (4, 3), 255).save(mask)
        frames.append(
            {
                "rgb_path": str(rgb.relative_to(tmp_path)),
                "depth_path": str(depth.relative_to(tmp_path)),
                "mask_path": str(mask.relative_to(tmp_path)),
                "camera_pose": np.diag([1.0, -1.0, 1.0, 1.0]).tolist(),
            }
        )
    episode = tmp_path / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "object_instance_id": "partnet_test",
                "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.0},
                "metadata": {"camera_pose_convention": "mujoco-gl-forward"},
                "frames": frames,
            }
        )
    )
    manifest = export_episode(episode, tmp_path / "export")
    assert manifest["source_camera_pose_convention"] == "mujoco-gl-forward"
    assert manifest["camera_pose_convention"] == "right-handed-camera-to-world"
    with (tmp_path / "export" / "matched_cues.csv").open(newline="") as handle:
        cue = next(csv.DictReader(handle))
    assert cue["CUE_START"] == "0"
    assert cue["CUE_END"] == "1"
    assert cue["VERIFICATION"] == "VERIFIED"
