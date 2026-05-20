from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.generation_preprocess import (
    GenerationImagePreparationConfig,
    GenerationImagePreparer,
    load_generation_image_manifest,
)


def _write_png_rgb(path: Path, width: int, height: int) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        raise unittest.SkipTest("Pillow is not installed")
    image = Image.new("RGB", (width, height), (20, 30, 40))
    for x_coord in range(1, 3):
        for y_coord in range(1, 3):
            image.putpixel((x_coord, y_coord), (200, 120, 30))
    image.save(path)


def _write_png_mask(path: Path, width: int, height: int) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError:
        raise unittest.SkipTest("Pillow/NumPy is not installed")
    mask = np.zeros((height, width), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    Image.fromarray(mask, mode="I;16").save(path)


class GenerationPreprocessTests(unittest.TestCase):
    def test_parser_accepts_prepare_generation_images(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "prepare-generation-images",
                "episode.json",
                "--frame-index",
                "2",
                "--view-indices",
                "0",
                "1",
                "--image-views",
                "front",
                "left",
                "--background",
                "white",
            ]
        )
        self.assertEqual(args.command, "prepare-generation-images")
        self.assertEqual(args.view_indices, [0, 1])
        self.assertEqual(args.image_views, ["front", "left"])

    def test_remote_articulate_parser_accepts_episode_generation_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "remote-articulate-generate",
                "--server-url",
                "http://localhost:8090",
                "--output-dir",
                "outputs/remote",
                "--episode",
                "episode.json",
                "--generation-image-output-dir",
                "prepared",
                "--frame-index",
                "3",
                "--view-indices",
                "0",
                "2",
                "--image-views",
                "front",
                "right",
                "--mask-source",
                "part",
                "--background",
                "white",
            ]
        )
        self.assertEqual(args.command, "remote-articulate-generate")
        self.assertEqual(args.episode, Path("episode.json"))
        self.assertEqual(args.generation_image_output_dir, Path("prepared"))
        self.assertEqual(args.frame_index, 3)
        self.assertEqual(args.view_indices, [0, 2])
        self.assertEqual(args.image_views, ["front", "right"])
        self.assertEqual(args.mask_source, "part")
        self.assertEqual(args.background, "white")

    def test_prepare_images_uses_prior_mask_and_manifest(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode_dir = root / "episode"
            view_dir = episode_dir / "assets" / "view_0"
            view_dir.mkdir(parents=True)
            rgb_path = view_dir / "frame_0000_rgb.png"
            mask_path = view_dir / "frame_0000_mask.png"
            _write_png_rgb(rgb_path, 4, 4)
            _write_png_mask(mask_path, 4, 4)

            episode_path = episode_dir / "episode.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "object_instance_id": "masked-object",
                        "category": "microwave",
                        "camera_intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.5},
                        "frames": [
                            {
                                "timestamp_s": 0.0,
                                "rgb_path": "assets/view_0/frame_0000_rgb.png",
                                "depth_path": "assets/view_0/frame_0000_mask.png",
                                "rgb_paths_by_view": ["assets/view_0/frame_0000_rgb.png"],
                                "mask_paths_by_view": ["assets/view_0/frame_0000_mask.png"],
                                "camera_pose": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest_path = GenerationImagePreparer().prepare(
                GenerationImagePreparationConfig(
                    episode_path=episode_path,
                    output_dir=root / "prepared",
                    image_views=["front"],
                    padding_ratio=0.0,
                    min_mask_pixels=1,
                )
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mask_provider"], "recorded-prior-mask")
            self.assertEqual(payload["image_views"], ["front"])
            image_paths, image_views = load_generation_image_manifest(manifest_path)
            self.assertEqual(image_views, ["front"])
            image = Image.open(image_paths[0])
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (2, 2))
            alpha_values = list(image.getchannel("A").getdata())
            self.assertTrue(all(value == 255 for value in alpha_values))


if __name__ == "__main__":
    unittest.main()
