from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.hunyuan3d_client import (
    Hunyuan3DClient,
    Hunyuan3DClientError,
    Hunyuan3DGenerationConfig,
    build_generation_payload,
)


class _FakeHunyuanClient(Hunyuan3DClient):
    def __init__(self) -> None:
        super().__init__("http://fake-server")
        self.sent_payload: dict[str, object] | None = None

    def send(self, payload: dict[str, object]) -> str:
        self.sent_payload = payload
        return "job-001"

    def status(self, uid: str) -> dict[str, object]:
        return {
            "status": "completed",
            "model_base64": base64.b64encode(b"glb-async").decode("ascii"),
        }


class Hunyuan3DClientTests(unittest.TestCase):
    def test_hunyuan3d_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "hunyuan3d-generate",
                "--server-url",
                "https://example.com",
                "--image",
                "input.png",
                "--image-manifest",
                "generation_images.json",
                "--output",
                "model.glb",
                "--mode",
                "async",
                "--api-token",
                "secret",
            ]
        )

        self.assertEqual(args.command, "hunyuan3d-generate")
        self.assertEqual(args.server_url, "https://example.com")
        self.assertEqual(args.image, [Path("input.png")])
        self.assertEqual(args.image_manifest, Path("generation_images.json"))
        self.assertEqual(args.output, Path("model.glb"))
        self.assertEqual(args.api_token, "secret")

    def test_build_generation_payload_base64_encodes_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            image_path.write_bytes(b"fake-png")

            payload = build_generation_payload(
                Hunyuan3DGenerationConfig(
                    server_url="http://localhost",
                    image_path=image_path,
                    output_path=Path(temp_dir) / "out.glb",
                    seed=99,
                    octree_resolution=256,
                )
            )

            self.assertEqual(base64.b64decode(payload["image"]), b"fake-png")
            self.assertEqual(payload["seed"], 99)
            self.assertEqual(payload["octree_resolution"], 256)

    def test_build_generation_payload_supports_multiview_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            front_path = Path(temp_dir) / "front.png"
            left_path = Path(temp_dir) / "left.png"
            right_path = Path(temp_dir) / "right.png"
            front_path.write_bytes(b"front")
            left_path.write_bytes(b"left")
            right_path.write_bytes(b"right")

            payload = build_generation_payload(
                Hunyuan3DGenerationConfig(
                    server_url="http://localhost",
                    image_path=[front_path, left_path, right_path],
                    image_views=["front", "left", "right"],
                    output_path=Path(temp_dir) / "out.glb",
                )
            )

            image_payload = payload["image"]
            self.assertIsInstance(image_payload, dict)
            assert isinstance(image_payload, dict)
            self.assertEqual(base64.b64decode(image_payload["front"]), b"front")
            self.assertEqual(base64.b64decode(image_payload["left"]), b"left")
            self.assertEqual(base64.b64decode(image_payload["right"]), b"right")

    def test_build_generation_payload_sends_single_named_view_as_plain_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            front_path = Path(temp_dir) / "front.png"
            front_path.write_bytes(b"front")

            payload = build_generation_payload(
                Hunyuan3DGenerationConfig(
                    server_url="http://localhost",
                    image_path=[front_path],
                    image_views=["front"],
                    output_path=Path(temp_dir) / "out.glb",
                )
            )

            self.assertIsInstance(payload["image"], str)
            self.assertEqual(base64.b64decode(payload["image"]), b"front")

    def test_build_generation_payload_rejects_view_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "front.png"
            image_path.write_bytes(b"front")

            with self.assertRaisesRegex(Hunyuan3DClientError, "image_views length"):
                build_generation_payload(
                    Hunyuan3DGenerationConfig(
                        server_url="http://localhost",
                        image_path=[image_path],
                        image_views=["front", "left"],
                        output_path=Path(temp_dir) / "out.glb",
                    )
                )

    def test_async_generation_saves_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            image_path.write_bytes(b"fake-png")
            output_path = Path(temp_dir) / "model.glb"
            client = _FakeHunyuanClient()

            result = client.generate(
                Hunyuan3DGenerationConfig(
                    server_url="http://fake-server",
                    image_path=image_path,
                    output_path=output_path,
                    timeout_s=5.0,
                    poll_interval_s=0.1,
                )
            )

            self.assertEqual(result, output_path.resolve())
            self.assertEqual(output_path.read_bytes(), b"glb-async")
            self.assertIsNotNone(client.sent_payload)


if __name__ == "__main__":
    unittest.main()
