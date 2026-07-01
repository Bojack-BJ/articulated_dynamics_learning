from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.kinematics.feedforward_evaluation import (
    FeedforwardArticulationEvaluationConfig,
    FeedforwardArticulationEvaluator,
)


class FeedforwardEvaluationTests(unittest.TestCase):
    def test_parser_accepts_feedforward_eval_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate-feedforward-articulation",
                "outputs/feedforward",
                "--reference-evaluation",
                "reference.json",
                "--output-json",
                "best.json",
            ]
        )
        self.assertEqual(args.command, "evaluate-feedforward-articulation")
        self.assertEqual(args.feedforward_root, Path("outputs/feedforward"))

    def test_selects_best_joint_candidate_from_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feedforward_root = root / "feedforward"
            object_dir = feedforward_root / "microwave001" / "particulate" / "urdf_001"
            eval_dir = feedforward_root / "_evaluation"
            object_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)
            (object_dir / "model.urdf").write_text(
                """
<robot name="test">
  <link name="base"/>
  <link name="wrong"/>
  <link name="door"/>
  <joint name="wrong_joint" type="revolute">
    <parent link="base"/>
    <child link="wrong"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="door_joint" type="revolute">
    <parent link="base"/>
    <child link="door"/>
    <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
  </joint>
</robot>
""".strip()
                + "\n",
                encoding="utf-8",
            )
            reference_path = eval_dir / "gt_axis_position_evaluation_unified_scale.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "summary": {},
                        "optimized_tracking": [],
                        "feedforward_particulate_upY": [
                            {
                                "object_id": "microwave001",
                                "gt_joint_type": "revolute",
                                "gt_axis": [0.0, 0.0, 1.0],
                                "gt_pivot_normalized": [0.1, 0.2, 0.3],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = FeedforwardArticulationEvaluator().evaluate(
                FeedforwardArticulationEvaluationConfig(feedforward_root=feedforward_root)
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            row = payload["feedforward_particulate_best_joint"][0]
            self.assertEqual(row["candidate_count"], 2)
            self.assertEqual(row["selected_joint_name"], "door_joint")
            self.assertAlmostEqual(row["axis_angle_error_deg"], 0.0)
            self.assertAlmostEqual(row["axis_position_error_normalized"], 0.0)


if __name__ == "__main__":
    unittest.main()
