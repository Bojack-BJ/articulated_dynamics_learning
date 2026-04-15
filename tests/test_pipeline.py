from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp import RGBDToURDFPipeline, load_episode
from rgbd_urdf_mvp.core.serialization import validate_episode
from rgbd_urdf_mvp.kinematics.refit import differentiate_joint_signal, smooth_sequence
from rgbd_urdf_mvp.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_episode_load_and_validate(self) -> None:
        episode = load_episode(ROOT / "examples" / "episodes" / "door" / "episode.json")
        self.assertEqual(episode.category, "door")
        self.assertFalse(validate_episode(episode))
        self.assertEqual(len(episode.frames), 4)

    def test_door_pipeline_outputs_revolute_joint(self) -> None:
        episode = load_episode(ROOT / "examples" / "episodes" / "door" / "episode.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = RGBDToURDFPipeline().run(episode, temp_dir)
            self.assertTrue(Path(result.urdf_package.urdf_path).exists())
            self.assertTrue(Path(result.urdf_package.mjcf_path).exists())
            articulation = Path(result.articulation_artifact_path).read_text(encoding="utf-8")
            self.assertIn('"joint_type": "revolute"', articulation)
            self.assertIn('"qdot"', articulation)

    def test_drawer_pipeline_outputs_prismatic_joint(self) -> None:
        episode = load_episode(ROOT / "examples" / "episodes" / "drawer" / "episode.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = RGBDToURDFPipeline().run(episode, temp_dir)
            self.assertTrue(Path(result.urdf_package.urdf_path).exists())
            articulation = Path(result.articulation_artifact_path).read_text(encoding="utf-8")
            self.assertIn('"joint_type": "prismatic"', articulation)
            self.assertIn('"moving_link"', articulation)

    def test_feedforward_path_skips_temporal_refit(self) -> None:
        episode = load_episode(ROOT / "examples" / "episodes" / "door" / "episode.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = RGBDToURDFPipeline().run(episode, temp_dir, path_mode="feedforward")
            articulation = Path(result.articulation_artifact_path).read_text(encoding="utf-8")
            self.assertIn('"pipeline_path": "feedforward"', articulation)
            self.assertNotIn('"qdot"', articulation)

    def test_run_parser_accepts_path_switch(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "examples/episodes/door/episode.json",
                "--path",
                "feedforward",
            ]
        )
        self.assertEqual(args.path, "feedforward")

    def test_signal_smoothing_and_qdot(self) -> None:
        smoothed = smooth_sequence([0.0, 1.0, 0.0, 1.0, 0.0], window_size=3)
        self.assertLess(smoothed[1], 1.0)
        qdot = differentiate_joint_signal([0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.3, 0.6])
        self.assertEqual(len(qdot), 4)
        self.assertGreater(qdot[-1], qdot[0])


if __name__ == "__main__":
    unittest.main()
