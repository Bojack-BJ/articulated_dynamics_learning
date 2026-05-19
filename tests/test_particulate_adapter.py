from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.perception.particulate_adapter import (
    ArticulationBackendComparisonConfig,
    ArticulationBackendComparator,
    ParticulateInferenceConfig,
    ParticulateInferenceRunner,
)


class ParticulateAdapterTests(unittest.TestCase):
    def test_dry_run_writes_manifest_without_launching_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            particulate_root = root / "Particulate"
            particulate_root.mkdir()
            mesh_path = root / "object.glb"
            mesh_path.write_bytes(b"glb")
            output_dir = root / "out"

            manifest_path = ParticulateInferenceRunner().run(
                ParticulateInferenceConfig(
                    mesh_path=mesh_path,
                    output_dir=output_dir,
                    particulate_root=particulate_root,
                    python_bin="python",
                    dry_run=True,
                )
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["input_mesh_path"], str(mesh_path.resolve()))
            self.assertIn("--export_urdf", payload["command"])
            self.assertIn("--export_mjcf", payload["command"])
            self.assertIn("--eval", payload["command"])

    def test_compare_summarizes_tracking_and_particulate_npz(self) -> None:
        try:
            np = __import__("numpy")
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed in this test environment")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            joint_path = root / "joint_inference.json"
            joint_path.write_text(
                json.dumps(
                    {
                        "estimator": "relative-se3-geometry",
                        "anchor_part_id": 1,
                        "joints": [
                            {
                                "name": "door_joint",
                                "joint_type": "revolute",
                                "parent_name": "base",
                                "child_name": "door",
                                "axis": [0, 0, 1],
                                "pivot": [0, 0, 0],
                                "limits": [-1.0, 0.0],
                                "confidence": 0.8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            eval_dir = root / "particulate" / "eval"
            eval_dir.mkdir(parents=True)
            npz_path = eval_dir / "pred.npz"
            np.savez(
                npz_path,
                face_part_ids=np.array([0, 0, 1, 1]),
                motion_hierarchy=np.array([[0, 1]]),
                is_part_revolute=np.array([False, True]),
                is_part_prismatic=np.array([False, False]),
            )
            particulate_path = root / "particulate" / "particulate_result.json"
            particulate_path.write_text(
                json.dumps(
                    {
                        "input_mesh_path": str(root / "object.glb"),
                        "output_dir": str(root / "particulate"),
                        "artifacts": {"eval_npz": str(npz_path)},
                    }
                ),
                encoding="utf-8",
            )

            output_path = ArticulationBackendComparator().compare(
                ArticulationBackendComparisonConfig(
                    tracking_joint_inference_path=joint_path,
                    particulate_result_path=particulate_path,
                )
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["tracking"]["joint_count"], 1)
            self.assertEqual(payload["tracking"]["joint_type_counts"]["revolute"], 1)
            self.assertEqual(payload["particulate"]["part_count"], 2)
            self.assertEqual(payload["particulate"]["hierarchy_edge_count"], 1)
            self.assertEqual(payload["particulate"]["revolute_part_count"], 1)


if __name__ == "__main__":
    unittest.main()
