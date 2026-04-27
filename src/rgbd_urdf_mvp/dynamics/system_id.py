from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.serialization import load_episode, load_json, save_json
from ..kinematics.refit import differentiate_joint_signal


@dataclass(slots=True)
class DynamicsIdentificationConfig:
    episode_path: str | Path
    articulation_artifact_path: str | Path
    mjcf_path: str | Path
    output_dir: str | Path | None = None
    max_iterations: int = 10
    learning_rate: float = 0.25
    q_weight: float = 1.0
    qdot_weight: float = 0.05
    prior_weight: float = 0.02
    optimize_static_parts: bool = False
    optimize_mass: bool = True
    optimize_damping: bool = True
    optimize_friction: bool = True


@dataclass(slots=True)
class _Trajectory:
    joint_name: str
    timestamps_s: list[float]
    q_observed: list[float]
    qdot_observed: list[float]
    qpos_adr: int
    dof_adr: int


@dataclass(slots=True)
class _ParameterSpec:
    name: str
    kind: str
    target_name: str
    index: int
    initial_value: float
    lower: float
    upper: float
    reference_inertia: np.ndarray | None = None
    fd_step: float = 0.05

    def encode(self, value: float) -> float:
        safe = max(self.lower, min(self.upper, value))
        if self.kind == "body_mass":
            return math.log(max(safe, 1e-6))
        return math.log(max(safe + 1e-6, 1e-6))

    def decode(self, theta: float) -> float:
        if self.kind == "body_mass":
            value = math.exp(theta)
        else:
            value = max(0.0, math.exp(theta) - 1e-6)
        return max(self.lower, min(self.upper, value))

    def prior_scale(self) -> float:
        if self.kind == "body_mass":
            return max(1e-3, 0.25 * max(self.initial_value, 1e-3))
        return max(1e-3, 0.1 * max(self.upper - self.lower, 1e-3), self.initial_value)


class DynamicsIdentifier:
    def run(self, config: DynamicsIdentificationConfig) -> Path:
        episode = load_episode(config.episode_path)
        if str(episode.metadata.get("control_mode", "free")) != "free":
            raise ValueError("identify-dynamics currently supports only recordings made with control_mode='free'.")

        articulation = load_json(config.articulation_artifact_path)
        mjcf_path = Path(config.mjcf_path).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))

        trajectories = self._load_trajectories(articulation, model)
        if not trajectories:
            raise ValueError("No joint state tracks found in articulation_artifact.fit_metrics['joint_state_tracks'].")

        specs = self._build_parameter_specs(model, articulation, config)
        if not specs:
            raise ValueError("No dynamic parameters selected for optimization.")

        sim_dt = float(episode.metadata.get("sim_dt", model.opt.timestep))
        if sim_dt > 0.0:
            model.opt.timestep = sim_dt

        theta = np.array([spec.encode(spec.initial_value) for spec in specs], dtype=float)
        evaluation_cache: dict[tuple[float, ...], tuple[float, dict[str, float], dict[str, list[dict[str, float]]], list[float]]] = {}

        def evaluate(theta_value: np.ndarray) -> tuple[float, dict[str, float], dict[str, list[dict[str, float]]], list[float]]:
            key = tuple(round(float(item), 10) for item in theta_value)
            cached = evaluation_cache.get(key)
            if cached is not None:
                return cached
            actual_values = [spec.decode(float(item)) for spec, item in zip(specs, theta_value)]
            self._apply_parameters(model, specs, actual_values)
            trajectories_sim = self._rollout(model, trajectories)
            loss, breakdown = self._loss(
                specs=specs,
                actual_values=actual_values,
                simulated=trajectories_sim,
                observed=trajectories,
                q_weight=float(config.q_weight),
                qdot_weight=float(config.qdot_weight),
                prior_weight=float(config.prior_weight),
            )
            cached = (loss, breakdown, trajectories_sim, actual_values)
            evaluation_cache[key] = cached
            return cached

        history: list[dict[str, Any]] = []
        best_theta = theta.copy()
        best_loss, best_breakdown, best_simulated, best_values = evaluate(theta)
        history.append(
            {
                "iteration": 0,
                "loss": best_loss,
                "loss_breakdown": best_breakdown,
                "parameter_values": _parameter_value_map(specs, best_values),
            }
        )

        learning_rate = max(1e-4, float(config.learning_rate))
        for iteration in range(1, max(1, int(config.max_iterations)) + 1):
            grads = np.zeros_like(theta)
            base_loss = float(best_loss if np.allclose(theta, best_theta) else evaluate(theta)[0])
            for index, spec in enumerate(specs):
                theta_eps = theta.copy()
                theta_eps[index] += spec.fd_step
                loss_eps = float(evaluate(theta_eps)[0])
                grads[index] = (loss_eps - base_loss) / spec.fd_step

            candidate_found = False
            for step_scale in (1.0, 0.5, 0.25, 0.1):
                theta_candidate = theta - learning_rate * step_scale * grads
                candidate_loss, candidate_breakdown, candidate_simulated, candidate_values = evaluate(theta_candidate)
                if candidate_loss + 1e-9 < base_loss:
                    theta = theta_candidate
                    base_loss = candidate_loss
                    candidate_found = True
                    if candidate_loss + 1e-9 < best_loss:
                        best_loss = candidate_loss
                        best_theta = theta_candidate.copy()
                        best_breakdown = candidate_breakdown
                        best_simulated = candidate_simulated
                        best_values = candidate_values
                    history.append(
                        {
                            "iteration": iteration,
                            "loss": candidate_loss,
                            "loss_breakdown": candidate_breakdown,
                            "parameter_values": _parameter_value_map(specs, candidate_values),
                            "accepted_step_scale": step_scale,
                            "gradient_norm": float(np.linalg.norm(grads)),
                        }
                    )
                    break
            if not candidate_found:
                learning_rate *= 0.5
                history.append(
                    {
                        "iteration": iteration,
                        "loss": base_loss,
                        "loss_breakdown": evaluate(theta)[1],
                        "parameter_values": _parameter_value_map(specs, [spec.decode(float(item)) for spec, item in zip(specs, theta)]),
                        "accepted_step_scale": 0.0,
                        "gradient_norm": float(np.linalg.norm(grads)),
                    }
                )
                if learning_rate < 1e-4:
                    break

        output_dir = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir is not None
            else mjcf_path.parent / "dynamics_identification"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        optimized_xml_path = output_dir / f"{mjcf_path.stem}.identified.xml"
        self._write_optimized_mjcf(
            mjcf_path=mjcf_path,
            output_path=optimized_xml_path,
            specs=specs,
            values=best_values,
        )
        mujoco.MjModel.from_xml_path(str(optimized_xml_path))

        artifact_path = output_dir / "dynamics_identification.json"
        save_json(
            {
                "source": "mujoco-rollout-system-id",
                "episode_path": str(Path(config.episode_path).resolve()),
                "articulation_artifact_path": str(Path(config.articulation_artifact_path).resolve()),
                "mjcf_input_path": str(mjcf_path),
                "mjcf_optimized_path": str(optimized_xml_path.resolve()),
                "optimizer": {
                    "method": "finite-difference-gradient-descent",
                    "max_iterations": int(config.max_iterations),
                    "learning_rate": float(config.learning_rate),
                    "q_weight": float(config.q_weight),
                    "qdot_weight": float(config.qdot_weight),
                    "prior_weight": float(config.prior_weight),
                    "history": history,
                },
                "identified_parameters": {
                    "parts": self._part_parameter_summary(articulation, model, specs, best_values),
                    "joints": self._joint_parameter_summary(articulation, specs, best_values),
                },
                "fit_metrics": {
                    "best_loss": float(best_loss),
                    "loss_breakdown": best_breakdown,
                },
                "trajectories": {
                    "observed": {traj.joint_name: _trajectory_to_samples(traj.timestamps_s, traj.q_observed, traj.qdot_observed) for traj in trajectories},
                    "simulated": best_simulated,
                },
                "limitations": [
                    "The current optimizer uses finite-difference rollout fitting inside MuJoCo, not exact end-to-end autodiff.",
                    "Absolute mass can be weakly identifiable from passive single-DOF trajectories; damping/friction are usually better constrained.",
                    "control_mode='track' and contact-rich episodes are not yet replayed in this MVP.",
                ],
            },
            artifact_path,
        )
        return artifact_path

    def _load_trajectories(self, articulation: dict[str, Any], model: mujoco.MjModel) -> list[_Trajectory]:
        joint_tracks = articulation.get("fit_metrics", {}).get("joint_state_tracks", {})
        trajectories: list[_Trajectory] = []
        if isinstance(joint_tracks, dict) and joint_tracks:
            for joint_name, raw_samples in joint_tracks.items():
                if not isinstance(raw_samples, list) or not raw_samples:
                    continue
                samples = [sample for sample in raw_samples if isinstance(sample, dict)]
                samples.sort(key=lambda item: float(item.get("timestamp_s", 0.0)))
                times = [float(sample.get("timestamp_s", 0.0)) for sample in samples]
                qs = [float(sample.get("q", 0.0)) for sample in samples]
                qdots = differentiate_joint_signal(times, qs)
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
                if joint_id < 0:
                    continue
                trajectories.append(
                    _Trajectory(
                        joint_name=str(joint_name),
                        timestamps_s=times,
                        q_observed=qs,
                        qdot_observed=qdots,
                        qpos_adr=int(model.jnt_qposadr[joint_id]),
                        dof_adr=int(model.jnt_dofadr[joint_id]),
                    )
                )
        if trajectories:
            return trajectories

        joints = articulation.get("joints", [])
        state = articulation.get("state", [])
        if len(joints) == 1 and isinstance(state, list) and state:
            joint_name = str(joints[0].get("name", "joint"))
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                times = [float(sample.get("timestamp_s", 0.0)) for sample in state if isinstance(sample, dict)]
                qs = [float(sample.get("q", 0.0)) for sample in state if isinstance(sample, dict)]
                qdots = [float(sample.get("qdot", 0.0)) for sample in state if isinstance(sample, dict)]
                return [
                    _Trajectory(
                        joint_name=joint_name,
                        timestamps_s=times,
                        q_observed=qs,
                        qdot_observed=qdots,
                        qpos_adr=int(model.jnt_qposadr[joint_id]),
                        dof_adr=int(model.jnt_dofadr[joint_id]),
                    )
                ]
        return []

    def _build_parameter_specs(
        self,
        model: mujoco.MjModel,
        articulation: dict[str, Any],
        config: DynamicsIdentificationConfig,
    ) -> list[_ParameterSpec]:
        specs: list[_ParameterSpec] = []
        parts = articulation.get("parts", [])
        if bool(config.optimize_mass):
            for raw_part in parts:
                if not isinstance(raw_part, dict):
                    continue
                role = str(raw_part.get("role", "moving"))
                if role == "static" and not bool(config.optimize_static_parts):
                    continue
                body_name = str(raw_part.get("name", "")).strip()
                if not body_name:
                    continue
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id <= 0:
                    continue
                mass = float(model.body_mass[body_id])
                inertia = np.array(model.body_inertia[body_id], dtype=float)
                specs.append(
                    _ParameterSpec(
                        name=f"body::{body_name}::mass",
                        kind="body_mass",
                        target_name=body_name,
                        index=body_id,
                        initial_value=mass,
                        lower=max(1e-3, 0.2 * mass),
                        upper=max(1e-2, 5.0 * mass),
                        reference_inertia=inertia,
                        fd_step=0.03,
                    )
                )

        if bool(config.optimize_damping) or bool(config.optimize_friction):
            for raw_joint in articulation.get("joints", []):
                if not isinstance(raw_joint, dict):
                    continue
                joint_name = str(raw_joint.get("name", "")).strip()
                if not joint_name:
                    continue
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    continue
                dof_adr = int(model.jnt_dofadr[joint_id])
                if bool(config.optimize_damping):
                    initial = float(model.dof_damping[dof_adr])
                    specs.append(
                        _ParameterSpec(
                            name=f"joint::{joint_name}::damping",
                            kind="joint_damping",
                            target_name=joint_name,
                            index=dof_adr,
                            initial_value=initial,
                            lower=0.0,
                            upper=max(1.0, 10.0 * max(initial, 0.1)),
                            fd_step=0.05,
                        )
                    )
                if bool(config.optimize_friction):
                    initial = float(model.dof_frictionloss[dof_adr])
                    specs.append(
                        _ParameterSpec(
                            name=f"joint::{joint_name}::frictionloss",
                            kind="joint_frictionloss",
                            target_name=joint_name,
                            index=dof_adr,
                            initial_value=initial,
                            lower=0.0,
                            upper=max(0.2, 10.0 * max(initial, 0.02)),
                            fd_step=0.05,
                        )
                    )
        return specs

    def _apply_parameters(self, model: mujoco.MjModel, specs: list[_ParameterSpec], values: list[float]) -> None:
        data = mujoco.MjData(model)
        for spec, value in zip(specs, values):
            if spec.kind == "body_mass":
                model.body_mass[spec.index] = value
                if spec.reference_inertia is not None:
                    scale = value / max(spec.initial_value, 1e-9)
                    model.body_inertia[spec.index] = np.maximum(1e-9, spec.reference_inertia * scale)
            elif spec.kind == "joint_damping":
                model.dof_damping[spec.index] = value
            elif spec.kind == "joint_frictionloss":
                model.dof_frictionloss[spec.index] = value
        mujoco.mj_setConst(model, data)

    def _rollout(self, model: mujoco.MjModel, trajectories: list[_Trajectory]) -> dict[str, list[dict[str, float]]]:
        schedule = sorted({round(t, 9) for traj in trajectories for t in traj.timestamps_s})
        if not schedule:
            return {}
        data = mujoco.MjData(model)
        for traj in trajectories:
            data.qpos[traj.qpos_adr] = traj.q_observed[0]
            data.qvel[traj.dof_adr] = traj.qdot_observed[0]
        mujoco.mj_forward(model, data)

        results = {traj.joint_name: [] for traj in trajectories}
        schedule_index = 0
        target = schedule[schedule_index]
        while schedule_index < len(schedule):
            while data.time + model.opt.timestep < target - 1e-12:
                mujoco.mj_step(model, data)
            if data.time < target - 1e-12:
                mujoco.mj_step(model, data)
            for traj in trajectories:
                results[traj.joint_name].append(
                    {
                        "timestamp_s": target,
                        "q": float(data.qpos[traj.qpos_adr]),
                        "qdot": float(data.qvel[traj.dof_adr]),
                    }
                )
            schedule_index += 1
            if schedule_index >= len(schedule):
                break
            target = schedule[schedule_index]
        return results

    def _loss(
        self,
        specs: list[_ParameterSpec],
        actual_values: list[float],
        simulated: dict[str, list[dict[str, float]]],
        observed: list[_Trajectory],
        q_weight: float,
        qdot_weight: float,
        prior_weight: float,
    ) -> tuple[float, dict[str, float]]:
        q_err = 0.0
        qdot_err = 0.0
        sample_count = 0
        for traj in observed:
            simulated_samples = simulated.get(traj.joint_name, [])
            simulated_by_time = {round(sample["timestamp_s"], 9): sample for sample in simulated_samples}
            for timestamp_s, q_obs, qdot_obs in zip(traj.timestamps_s, traj.q_observed, traj.qdot_observed):
                sample = simulated_by_time.get(round(timestamp_s, 9))
                if sample is None:
                    continue
                q_err += (float(sample["q"]) - q_obs) ** 2
                qdot_err += (float(sample["qdot"]) - qdot_obs) ** 2
                sample_count += 1
        if sample_count == 0:
            return 1e12, {"q_mse": 1e12, "qdot_mse": 1e12, "prior_penalty": 0.0}
        q_mse = q_err / sample_count
        qdot_mse = qdot_err / sample_count
        prior_penalty = 0.0
        for spec, value in zip(specs, actual_values):
            scale = spec.prior_scale()
            prior_penalty += ((value - spec.initial_value) / scale) ** 2
        total = q_weight * q_mse + qdot_weight * qdot_mse + prior_weight * prior_penalty
        return total, {
            "q_mse": q_mse,
            "qdot_mse": qdot_mse,
            "prior_penalty": prior_penalty,
        }

    def _write_optimized_mjcf(
        self,
        mjcf_path: Path,
        output_path: Path,
        specs: list[_ParameterSpec],
        values: list[float],
    ) -> None:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        by_name = {spec.name: value for spec, value in zip(specs, values)}

        body_updates: dict[str, tuple[float, np.ndarray]] = {}
        for spec, value in zip(specs, values):
            if spec.kind != "body_mass":
                continue
            reference_inertia = spec.reference_inertia if spec.reference_inertia is not None else np.array([1e-6, 1e-6, 1e-6])
            scale = value / max(spec.initial_value, 1e-9)
            body_updates[spec.target_name] = (value, np.maximum(1e-9, reference_inertia * scale))

        for body in root.findall(".//body"):
            body_name = (body.get("name") or "").strip()
            if body_name in body_updates:
                mass_value, inertia_diag = body_updates[body_name]
                inertial = body.find("inertial")
                if inertial is None:
                    inertial = ET.SubElement(body, "inertial", pos="0 0 0", quat="1 0 0 0")
                inertial.set("mass", f"{mass_value:.6f}")
                inertial.set("diaginertia", " ".join(f"{value:.9f}" for value in inertia_diag.tolist()))

        for joint in root.findall(".//joint"):
            joint_name = (joint.get("name") or "").strip()
            damping_key = f"joint::{joint_name}::damping"
            friction_key = f"joint::{joint_name}::frictionloss"
            if damping_key in by_name:
                joint.set("damping", f"{float(by_name[damping_key]):.6f}")
            if friction_key in by_name:
                joint.set("frictionloss", f"{float(by_name[friction_key]):.6f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _part_parameter_summary(
        self,
        articulation: dict[str, Any],
        model: mujoco.MjModel,
        specs: list[_ParameterSpec],
        values: list[float],
    ) -> list[dict[str, Any]]:
        mass_map = {
            spec.target_name: (spec, value)
            for spec, value in zip(specs, values)
            if spec.kind == "body_mass"
        }
        parts_summary: list[dict[str, Any]] = []
        for raw_part in articulation.get("parts", []):
            if not isinstance(raw_part, dict):
                continue
            part_name = str(raw_part.get("name", "")).strip()
            if not part_name:
                continue
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part_name)
            if body_id < 0:
                continue
            spec_value = mass_map.get(part_name)
            if spec_value is not None:
                spec, value = spec_value
                inertia_diag = (spec.reference_inertia * (value / max(spec.initial_value, 1e-9))).tolist() if spec.reference_inertia is not None else [0.0, 0.0, 0.0]
                optimized = True
                initial_mass = spec.initial_value
            else:
                value = float(model.body_mass[body_id])
                inertia_diag = [float(item) for item in model.body_inertia[body_id].tolist()]
                optimized = False
                initial_mass = value
            parts_summary.append(
                {
                    "name": part_name,
                    "role": str(raw_part.get("role", "moving")),
                    "mass_kg": float(value),
                    "initial_mass_kg": float(initial_mass),
                    "inertia_diag": [float(item) for item in inertia_diag],
                    "optimized": optimized,
                }
            )
        return parts_summary

    def _joint_parameter_summary(
        self,
        articulation: dict[str, Any],
        specs: list[_ParameterSpec],
        values: list[float],
    ) -> list[dict[str, Any]]:
        by_name = {
            spec.name: (spec, value)
            for spec, value in zip(specs, values)
            if spec.kind in {"joint_damping", "joint_frictionloss"}
        }
        joints_summary: list[dict[str, Any]] = []
        for raw_joint in articulation.get("joints", []):
            if not isinstance(raw_joint, dict):
                continue
            joint_name = str(raw_joint.get("name", "")).strip()
            if not joint_name:
                continue
            damping_item = by_name.get(f"joint::{joint_name}::damping")
            friction_item = by_name.get(f"joint::{joint_name}::frictionloss")
            joints_summary.append(
                {
                    "name": joint_name,
                    "joint_type": str(raw_joint.get("joint_type", "unknown")),
                    "damping": float(damping_item[1]) if damping_item is not None else None,
                    "initial_damping": float(damping_item[0].initial_value) if damping_item is not None else None,
                    "frictionloss": float(friction_item[1]) if friction_item is not None else None,
                    "initial_frictionloss": float(friction_item[0].initial_value) if friction_item is not None else None,
                }
            )
        return joints_summary


def _parameter_value_map(specs: list[_ParameterSpec], values: list[float]) -> dict[str, float]:
    return {spec.name: float(value) for spec, value in zip(specs, values)}


def _trajectory_to_samples(times: list[float], q_values: list[float], qdot_values: list[float]) -> list[dict[str, float]]:
    return [
        {
            "timestamp_s": float(timestamp_s),
            "q": float(q_value),
            "qdot": float(qdot_value),
        }
        for timestamp_s, q_value, qdot_value in zip(times, q_values, qdot_values)
    ]
