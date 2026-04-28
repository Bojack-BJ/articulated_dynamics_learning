from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.dynamics.plotting import DynamicsOptimizationPlotConfig, DynamicsOptimizationPlotter
from rgbd_urdf_mvp.dynamics.system_id import DynamicsIdentificationConfig, DynamicsIdentifier


MODEL_XML = """
<mujoco model="system_id_test">
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


def _force_at_time(force_schedule: list[tuple[float, float]] | None, time_s: float) -> float:
    if not force_schedule:
        return 0.0
    force = 0.0
    for timestamp_s, value in force_schedule:
        if timestamp_s <= time_s + 1e-12:
            force = value
        else:
            break
    return force


def _simulate_track(
    xml_path: Path,
    times: list[float],
    initial_qvel: float = 3.0,
    force_schedule: list[tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_joint")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr] = 0.0
    data.qvel[dof_adr] = initial_qvel
    mujoco.mj_forward(model, data)
    samples: list[dict[str, float]] = []
    for target in times:
        while data.time + model.opt.timestep < target - 1e-12:
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = _force_at_time(force_schedule, float(data.time))
            mujoco.mj_step(model, data)
        if data.time < target - 1e-12:
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = _force_at_time(force_schedule, float(data.time))
            mujoco.mj_step(model, data)
        samples.append(
            {
                "timestamp_s": target,
                "q": float(data.qpos[qpos_adr]),
                "qdot": float(data.qvel[dof_adr]),
            }
        )
    return samples


class DynamicsIdentificationTests(unittest.TestCase):
    def test_identify_dynamics_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "identify-dynamics",
                "episode.json",
                "articulation_artifact.json",
                "model.mjcf.xml",
                "--max-iterations",
                "6",
                "--learning-rate",
                "0.1",
                "--enable-contact",
                "--gravity-mode",
                "zero",
            ]
        )
        self.assertEqual(args.command, "identify-dynamics")
        self.assertEqual(args.episode, Path("episode.json"))
        self.assertEqual(args.articulation_artifact, Path("articulation_artifact.json"))
        self.assertEqual(args.mjcf, Path("model.mjcf.xml"))
        self.assertEqual(args.max_iterations, 6)
        self.assertAlmostEqual(args.learning_rate, 0.1)
        self.assertTrue(args.enable_contact)
        self.assertEqual(args.gravity_mode, "zero")

    def test_plot_dynamics_identification_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "plot-dynamics-identification",
                "dynamics_identification.json",
                "--output-svg",
                "history.svg",
                "--linear-loss",
            ]
        )
        self.assertEqual(args.command, "plot-dynamics-identification")
        self.assertEqual(args.input, Path("dynamics_identification.json"))
        self.assertEqual(args.output_svg, Path("history.svg"))
        self.assertTrue(args.linear_loss)

    def test_plot_dynamics_identification_writes_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path = root / "dynamics_identification.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "optimizer": {
                            "history": [
                                {
                                    "iteration": 0,
                                    "loss": 1.0,
                                    "loss_breakdown": {"q_mse": 0.8, "qdot_mse": 4.0},
                                    "parameter_values": {
                                        "body::door::mass": 0.5,
                                        "joint::door_joint::damping": 0.1,
                                    },
                                },
                                {
                                    "iteration": 1,
                                    "loss": 0.1,
                                    "loss_breakdown": {"q_mse": 0.08, "qdot_mse": 0.4},
                                    "parameter_values": {
                                        "body::door::mass": 1.0,
                                        "joint::door_joint::damping": 1.0,
                                    },
                                },
                            ]
                        },
                        "fit_metrics": {
                            "trajectory_error": {"q_rmse": 0.01, "qdot_rmse": 0.02}
                        },
                        "ground_truth_comparison": {
                            "summary": {"effective_inertia_rel_error_mean": 0.05}
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output_svg = DynamicsOptimizationPlotter().plot(
                DynamicsOptimizationPlotConfig(input_path=artifact_path)
            )
            svg = output_svg.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Dynamics Optimization History", svg)
            self.assertIn("door/mass", svg)
            self.assertIn("effective_inertia_rel_error", svg)

    def test_identify_dynamics_reduces_rollout_loss_and_writes_outputs(self) -> None:
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
                                "rgb_path": "rgb.ppm",
                                "depth_path": "depth.pgm",
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

            output_path = DynamicsIdentifier().run(
                DynamicsIdentificationConfig(
                    episode_path=episode_path,
                    articulation_artifact_path=articulation_path,
                    mjcf_path=model_path,
                    output_dir=root / "dynamics_identification",
                    max_iterations=4,
                    learning_rate=0.2,
                )
            )

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            history = artifact["optimizer"]["history"]
            self.assertTrue(output_path.exists())
            self.assertTrue(Path(artifact["mjcf_optimized_path"]).exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertLessEqual(
                float(artifact["fit_metrics"]["best_loss"]),
                float(history[0]["loss"]) + 1e-9,
            )
            optimized_text = Path(artifact["mjcf_optimized_path"]).read_text(encoding="utf-8")
            self.assertIn("\n  <worldbody>", optimized_text)
            self.assertIn("damping=", optimized_text)
            self.assertIn("frictionloss=", optimized_text)
            self.assertIn('contype="0"', optimized_text)
            self.assertIn('conaffinity="0"', optimized_text)
            self.assertIn("<inertial ", optimized_text)
            part_names = {part["name"] for part in artifact["identified_parameters"]["parts"]}
            self.assertIn("door", part_names)
            joint_names = {joint["name"] for joint in artifact["identified_parameters"]["joints"]}
            self.assertIn("door_joint", joint_names)
            self.assertIn("trajectory_error", artifact["fit_metrics"])
            self.assertFalse(artifact["simulation_options"]["enable_contact"])
            self.assertIn("ground_truth_comparison", artifact)
            self.assertIn("render_artifacts", artifact)
            self.assertTrue(artifact["ground_truth_comparison"]["available"])
            self.assertIn("parts", artifact["ground_truth_comparison"])
            self.assertIn("joints", artifact["ground_truth_comparison"])
            self.assertGreaterEqual(artifact["fit_metrics"]["trajectory_error"]["sample_count"], 1)

    def test_identify_dynamics_replays_recorded_generalized_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.mjcf.xml"
            model_path.write_text(MODEL_XML + "\n", encoding="utf-8")

            times = [index * 0.02 for index in range(30)]
            force_schedule = [
                (index * 0.01, 1.2 if 0.05 <= index * 0.01 < 0.20 else 0.0)
                for index in range(80)
            ]
            observed = _simulate_track(
                model_path,
                times,
                initial_qvel=0.0,
                force_schedule=force_schedule,
            )

            dynamics_log_path = root / "dynamics_log.jsonl"
            dynamics_log_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp_s": round(timestamp_s, 9),
                            "joints": {
                                "source_door_joint": {
                                    "applied_generalized_force": force,
                                    "excitation_force": force,
                                }
                            },
                        }
                    )
                    for timestamp_s, force in force_schedule
                )
                + "\n",
                encoding="utf-8",
            )

            episode_path = root / "episode.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "object_instance_id": "dyn-force-test",
                        "category": "microwave",
                        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 160.0, "cy": 120.0},
                        "frames": [
                            {
                                "timestamp_s": sample["timestamp_s"],
                                "rgb_path": "rgb.ppm",
                                "depth_path": "depth.pgm",
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
                            "recording_variant": "forced-excitation",
                            "sim_dt": 0.01,
                            "frame_dt": 0.02,
                            "model_path": str(model_path),
                            "joint_name": "source_door_joint",
                            "excitation": {
                                "mode": "pulse",
                                "dynamics_log_path": "dynamics_log.jsonl",
                            },
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
                                "metadata": {"prior": {"joint_name": "source_door_joint"}},
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

            output_path = DynamicsIdentifier().run(
                DynamicsIdentificationConfig(
                    episode_path=episode_path,
                    articulation_artifact_path=articulation_path,
                    mjcf_path=model_path,
                    output_dir=root / "dynamics_identification",
                    max_iterations=1,
                    render_gl_backend="none",
                    prior_weight=0.0,
                )
            )

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            force_replay = artifact["simulation_options"]["force_replay"]
            self.assertTrue(force_replay["available"])
            self.assertEqual(force_replay["mapped_joints"][0]["source_joint_name"], "source_door_joint")
            self.assertGreater(force_replay["mapped_joints"][0]["nonzero_sample_count"], 0)
            self.assertLess(artifact["fit_metrics"]["trajectory_error"]["q_rmse"], 0.02)


if __name__ == "__main__":
    unittest.main()
