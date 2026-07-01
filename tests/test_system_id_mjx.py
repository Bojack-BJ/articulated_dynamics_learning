from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import mujoco

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.dynamics.system_id_mjx import MJXDynamicsIdentificationConfig, MJXDynamicsIdentifier


MODEL_XML = """
<mujoco model="system_id_mjx_test">
  <option timestep="0.01" gravity="0 0 0" />
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="door" pos="0 0 0">
        <inertial pos="0 0 0" quat="1 0 0 0" mass="1.5" diaginertia="0.12 0.12 0.12" />
        <joint name="door_joint" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.35" frictionloss="0.03" />
        <geom type="box" pos="0.25 0 0" size="0.25 0.03 0.35" />
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _simulate_track(xml_path: Path, times: list[float]) -> list[dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_joint")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr] = 0.0
    data.qvel[dof_adr] = 3.0
    mujoco.mj_forward(model, data)
    samples: list[dict[str, float]] = []
    for target in times:
        while data.time + model.opt.timestep < target - 1e-12:
            mujoco.mj_step(model, data)
        if data.time < target - 1e-12:
            mujoco.mj_step(model, data)
        samples.append(
            {
                "timestamp_s": target,
                "q": float(data.qpos[qpos_adr]),
                "qdot": float(data.qvel[dof_adr]),
            }
        )
    return samples


class MJXDynamicsIdentificationTests(unittest.TestCase):
    def test_identify_dynamics_mjx_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "identify-dynamics-mjx",
                "episode.json",
                "articulation_artifact.json",
                "model.mjcf.xml",
                "--max-iterations",
                "6",
                "--learning-rate",
                "0.1",
                "--jax-platform",
                "metal",
                "--enable-pjrt-compatibility",
                "--enable-contact",
                "--no-jit",
            ]
        )
        self.assertEqual(args.command, "identify-dynamics-mjx")
        self.assertEqual(args.episode, Path("episode.json"))
        self.assertEqual(args.articulation_artifact, Path("articulation_artifact.json"))
        self.assertEqual(args.mjcf, Path("model.mjcf.xml"))
        self.assertEqual(args.max_iterations, 6)
        self.assertAlmostEqual(args.learning_rate, 0.1)
        self.assertEqual(args.jax_platform, "metal")
        self.assertTrue(args.enable_pjrt_compatibility)
        self.assertTrue(args.enable_contact)
        self.assertTrue(args.no_jit)

    def test_identify_dynamics_mjx_writes_outputs_when_jax_is_available(self) -> None:
        try:
            os.environ["JAX_PLATFORMS"] = "cpu"
            import jax  # noqa: F401
            from mujoco import mjx  # noqa: F401
        except Exception:
            self.skipTest("JAX + MuJoCo MJX not installed in this environment")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.mjcf.xml"
            model_path.write_text(MODEL_XML + "\n", encoding="utf-8")

            times = [index * 0.02 for index in range(20)]
            observed = _simulate_track(model_path, times)

            episode_path = root / "episode.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "object_instance_id": "dyn-test",
                        "category": "microwave",
                        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 160.0, "cy": 120.0},
                        "frames": [
                            {
                                "timestamp_s": sample["timestamp_s"],
                                "rgb_path": "rgb.png",
                                "depth_path": "depth.png",
                                "camera_pose": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                            }
                            for sample in observed
                        ],
                        "metadata": {
                            "source": "mujoco-recorder",
                            "control_mode": "free",
                            "sim_dt": 0.01,
                            "frame_dt": 0.02,
                            "model_path": str(model_path),
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            articulation_path = root / "articulation_artifact.json"
            articulation_path.write_text(
                json.dumps(
                    {
                        "parts": [
                            {"name": "base", "role": "static", "primitive_boxes": [], "canonical_pose": {}, "confidence": 1.0},
                            {"name": "door", "role": "moving", "primitive_boxes": [], "canonical_pose": {}, "confidence": 1.0},
                        ],
                        "joints": [
                            {
                                "name": "door_joint",
                                "joint_type": "revolute",
                                "parent": "base",
                                "child": "door",
                                "axis": [0.0, 0.0, 1.0],
                                "origin": [0.0, 0.0, 0.0],
                                "limits": [-3.14, 3.14],
                                "confidence": 1.0,
                            }
                        ],
                        "state": [],
                        "fit_metrics": {
                            "joint_state_tracks": {
                                "door_joint": [
                                    {
                                        "frame_index": index,
                                        "timestamp_s": sample["timestamp_s"],
                                        "q": sample["q"],
                                    }
                                    for index, sample in enumerate(observed)
                                ]
                            }
                        },
                        "low_confidence_components": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = MJXDynamicsIdentifier().run(
                MJXDynamicsIdentificationConfig(
                    episode_path=episode_path,
                    articulation_artifact_path=articulation_path,
                    mjcf_path=model_path,
                    output_dir=root / "dynamics_identification_mjx",
                    jax_platform="cpu",
                    max_iterations=3,
                    learning_rate=0.05,
                    jit=False,
                )
            )

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            history = artifact["optimizer"]["history"]
            self.assertTrue(output_path.exists())
            self.assertTrue(Path(artifact["mjcf_optimized_path"]).exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertEqual(artifact["source"], "mjx-autodiff-system-id")
            self.assertEqual(artifact["optimizer"]["method"], "jax-adam-autodiff-forward-mode")
            self.assertFalse(artifact["simulation_options"]["enable_contact"])
            optimized_text = Path(artifact["mjcf_optimized_path"]).read_text(encoding="utf-8")
            self.assertIn('contype="0"', optimized_text)
            self.assertIn('conaffinity="0"', optimized_text)
            self.assertIn("trajectory_error", artifact["fit_metrics"])
            self.assertIn("ground_truth_comparison", artifact)
            self.assertIn("render_artifacts", artifact)
            self.assertTrue(artifact["ground_truth_comparison"]["available"])
            self.assertIn("overflow_penalty", artifact["fit_metrics"]["loss_breakdown"])
            self.assertIn("invalid_sample_count", artifact["fit_metrics"]["loss_breakdown"])


if __name__ == "__main__":
    unittest.main()
