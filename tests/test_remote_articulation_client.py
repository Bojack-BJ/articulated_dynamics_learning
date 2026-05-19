from __future__ import annotations

import base64
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from rgbd_urdf_mvp.perception.remote_articulation_client import RemoteArticulationClient, RemoteArticulationConfig


class RemoteArticulationClientTests(unittest.TestCase):
    def test_generate_unpacks_completed_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "front.png"
            image_path.write_bytes(b"png")
            zip_path = root / "result.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("particulate/particulate_result.json", json.dumps({"ok": True}))
            status = {
                "uid": "job1",
                "status": "completed",
                "result_zip_base64": base64.b64encode(zip_path.read_bytes()).decode("utf-8"),
            }

            client = RemoteArticulationClient("https://example.test")
            with mock.patch.object(client, "_send", return_value="job1") as send_mock:
                with mock.patch.object(client, "_wait", return_value=status):
                    output_dir = client.generate(
                        RemoteArticulationConfig(
                            server_url="https://example.test",
                            image_path=image_path,
                            output_dir=root / "out",
                        )
                    )

            self.assertTrue((output_dir / "particulate" / "particulate_result.json").exists())
            sent_payload = send_mock.call_args.args[0]
            self.assertIn("hunyuan", sent_payload)
            self.assertIn("particulate", sent_payload)
            self.assertEqual(sent_payload["hunyuan"]["type"], "glb")


if __name__ == "__main__":
    unittest.main()
