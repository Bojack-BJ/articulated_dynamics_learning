from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.cli_parser import build_parser
from rgbd_urdf_mvp.core.serialization import load_episode, validate_episode
from rgbd_urdf_mvp.perception.rbo_adapter import RBORecordingImportConfig, RBORecordingImporter


class RBORecordingImporterTest(unittest.TestCase):
    def test_imports_ros_export_as_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "microwave01_o"
            (source / "camera_rgb").mkdir(parents=True)
            (source / "camera_depth_registered").mkdir(parents=True)
            _write_camera_info(source / "camera_depth_registered_camera_info.csv")
            _write_joint_states(source / "microwave_joint_states.csv")
            _write_rgb(source / "camera_rgb" / "000000-100.000.png")
            _write_rgb(source / "camera_rgb" / "000001-100.100.png")
            np.savetxt(source / "camera_depth_registered" / "000000-100.004.txt", np.full((4, 4), 1.2))
            np.savetxt(source / "camera_depth_registered" / "000001-100.104.txt", np.full((4, 4), 1.4))

            episode_path = RBORecordingImporter().import_recording(
                RBORecordingImportConfig(
                    input_dir=source,
                    output_dir=root / "out",
                    max_sync_delta_s=0.02,
                    mask_mode="depth-near",
                )
            )

            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["object_instance_id"], "microwave01_o")
            self.assertEqual(payload["category"], "microwave")
            self.assertEqual(payload["camera_intrinsics"]["fx"], 532.0)
            self.assertEqual(len(payload["frames"]), 2)
            self.assertEqual(payload["frames"][0]["timestamp_s"], 0.0)
            self.assertEqual(payload["frames"][0]["rgb_path"], "assets/view_0/frame_0000_rgb.png")
            self.assertEqual(payload["frames"][0]["mask_path"], "assets/view_0/frame_0000_mask.png")
            self.assertEqual(payload["frames"][0]["joint_position_hint"], 0.5)
            self.assertEqual(validate_episode(load_episode(episode_path)), [])
            self.assertTrue((root / "out" / "assets" / "view_0" / "frame_0000_depth.png").exists())

    def test_cli_parser_accepts_import_rbo_recording(self) -> None:
        args = build_parser().parse_args(
            [
                "import-rbo-recording",
                "real_data/microwave01_o",
                "--output-dir",
                "outputs/real/microwave01_o",
                "--mask-mode",
                "depth-near",
                "--max-frames",
                "10",
            ]
        )
        self.assertEqual(args.command, "import-rbo-recording")
        self.assertEqual(args.mask_mode, "depth-near")
        self.assertEqual(args.max_frames, 10)


def _write_camera_info(path: Path) -> None:
    fieldnames = ["field.P0", "field.P2", "field.P5", "field.P6", "field.K0", "field.K2", "field.K4", "field.K5"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "field.P0": "532.0",
                "field.P2": "313.0",
                "field.P5": "534.0",
                "field.P6": "243.0",
                "field.K0": "531.0",
                "field.K2": "312.0",
                "field.K4": "533.0",
                "field.K5": "242.0",
            }
        )


def _write_joint_states(path: Path) -> None:
    fieldnames = ["%time", "field.header.stamp", "field.name0", "field.position0"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "%time": "100000000000",
                "field.header.stamp": "100000000000",
                "field.name0": "j_0_1",
                "field.position0": "0.5",
            }
        )


def _write_rgb(path: Path) -> None:
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8), mode="RGB").save(path)


if __name__ == "__main__":
    unittest.main()
