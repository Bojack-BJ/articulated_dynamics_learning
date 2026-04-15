from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.core.cli_config import config_to_argv, expand_config_argv


class CLIConfigTests(unittest.TestCase):
    def test_expand_yaml_config_into_run_command(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "command: run",
                        "episode: examples/episodes/door/episode.json",
                        "output-dir: outputs/from_yaml",
                        "path: feedforward",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = expand_config_argv([str(config_path)], parser)

            self.assertEqual(argv[0], "run")
            self.assertIn("examples/episodes/door/episode.json", argv)
            self.assertIn("--output-dir", argv)
            self.assertIn("outputs/from_yaml", argv)
            self.assertIn("--path", argv)
            self.assertIn("feedforward", argv)

    def test_config_to_argv_handles_lists_and_flags(self) -> None:
        parser = build_parser()
        argv = config_to_argv(
            {
                "command": "record-mujoco",
                "args": {
                    "model": "model.xml",
                    "category": "door",
                    "object-id": "door-yaml",
                    "lookat": [0.0, 0.0, 0.3],
                    "video": True,
                    "duration-s": 4,
                    "fps": 20,
                },
            },
            parser,
        )

        self.assertEqual(argv[:4], ["record-mujoco", "model.xml", "--category", "door"])
        self.assertIn("--object-id", argv)
        self.assertIn("door-yaml", argv)
        self.assertIn("--lookat", argv)
        lookat_index = argv.index("--lookat")
        self.assertEqual(argv[lookat_index + 1 : lookat_index + 4], ["0.0", "0.0", "0.3"])
        self.assertIn("--video", argv)

    def test_expand_config_keeps_cli_overrides_last(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "command: run",
                        "episode: examples/episodes/door/episode.json",
                        "output-dir: outputs/from_yaml",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = expand_config_argv(
                [str(config_path), "--output-dir", "outputs/override"],
                parser,
            )

            self.assertEqual(argv[-2:], ["--output-dir", "outputs/override"])


if __name__ == "__main__":
    unittest.main()
