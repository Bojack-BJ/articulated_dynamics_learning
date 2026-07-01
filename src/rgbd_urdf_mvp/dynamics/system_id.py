from __future__ import annotations

import contextlib
import gc
import io
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.serialization import load_episode, load_json, save_json
from ..core.xml_utils import write_xml_tree
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
    enable_contact: bool = False
    render_gl_backend: str | None = None
    gravity_mode: str = "model"


@dataclass(slots=True)
class _Trajectory:
    joint_name: str
    timestamps_s: list[float]
    q_observed: list[float]
    qdot_observed: list[float]
    qpos_adr: int
    dof_adr: int


@dataclass(slots=True)
class _ForceSchedule:
    joint_name: str
    source_joint_name: str
    timestamps_s: np.ndarray
    forces: np.ndarray


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
        if config.render_gl_backend and config.render_gl_backend not in {"auto", "none"}:
            import os

            os.environ["MUJOCO_GL"] = str(config.render_gl_backend)
        if str(episode.metadata.get("control_mode", "free")) != "free":
            raise ValueError("identify-dynamics currently supports only recordings made with control_mode='free'.")

        articulation = load_json(config.articulation_artifact_path)
        mjcf_path = Path(config.mjcf_path).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        if not bool(config.enable_contact):
            self._disable_geom_contacts(model)
        self._apply_gravity_mode(model, config.gravity_mode)

        trajectories = self._load_trajectories(articulation, model)
        if not trajectories:
            raise ValueError("No joint state tracks found in articulation_artifact.fit_metrics['joint_state_tracks'].")
        force_schedules, force_replay_summary = self._load_force_schedules(
            episode_path=Path(config.episode_path).expanduser().resolve(),
            episode=episode,
            articulation=articulation,
            trajectories=trajectories,
        )

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
            trajectories_sim = self._rollout(model, trajectories, force_schedules)
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
                "phase": "initial",
                "loss": best_loss,
                "loss_breakdown": best_breakdown,
                "parameter_values": _parameter_value_map(specs, best_values),
            }
        )

        theta, best_theta, best_loss, best_breakdown, best_simulated, best_values = self._coordinate_search_warm_start(
            theta=theta,
            best_theta=best_theta,
            best_loss=best_loss,
            best_breakdown=best_breakdown,
            best_simulated=best_simulated,
            best_values=best_values,
            specs=specs,
            evaluate=evaluate,
            history=history,
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
                            "phase": "gradient_descent",
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
                        "phase": "gradient_descent",
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
            enable_contact=bool(config.enable_contact),
            gravity_mode=str(config.gravity_mode),
        )
        optimized_model = mujoco.MjModel.from_xml_path(str(optimized_xml_path))
        if not bool(config.enable_contact):
            self._disable_geom_contacts(optimized_model)
        self._apply_gravity_mode(optimized_model, config.gravity_mode)
        trajectory_errors = self._trajectory_error_summary(trajectories, best_simulated)
        gt_comparison = self._ground_truth_parameter_comparison(
            episode=episode,
            articulation=articulation,
            estimated_model=optimized_model,
        )
        render_summary = self._render_rollout_comparison(
            episode=episode,
            episode_path=Path(config.episode_path).expanduser().resolve(),
            model=optimized_model,
            trajectories=trajectories,
            force_schedules=force_schedules,
            output_dir=output_dir,
            gl_backend=config.render_gl_backend,
        )

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
                "simulation_options": {
                    "enable_contact": bool(config.enable_contact),
                    "contact_policy": (
                        "MJCF geom contacts enabled"
                        if bool(config.enable_contact)
                        else "MJCF geom contacts disabled for dynamics rollout"
                    ),
                    "force_replay": force_replay_summary,
                    "gravity_mode": str(config.gravity_mode),
                    "gravity": [float(value) for value in model.opt.gravity.tolist()],
                },
                "identified_parameters": {
                    "parts": self._part_parameter_summary(articulation, model, specs, best_values),
                    "joints": self._joint_parameter_summary(articulation, specs, best_values),
                },
                "fit_metrics": {
                    "best_loss": float(best_loss),
                    "loss_breakdown": best_breakdown,
                    "trajectory_error": trajectory_errors,
                },
                "ground_truth_comparison": gt_comparison,
                "render_artifacts": render_summary,
                "trajectories": {
                    "observed": {traj.joint_name: _trajectory_to_samples(traj.timestamps_s, traj.q_observed, traj.qdot_observed) for traj in trajectories},
                    "simulated": best_simulated,
                },
                "limitations": [
                    "The current optimizer uses finite-difference rollout fitting inside MuJoCo, not exact end-to-end autodiff.",
                    "Absolute mass can be weakly identifiable from passive single-DOF trajectories; damping/friction are usually better constrained.",
                    "control_mode='track' contact-rich episodes are not yet a target for this MVP.",
                ],
            },
            artifact_path,
        )
        return artifact_path

    def _disable_geom_contacts(self, model: mujoco.MjModel) -> None:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0

    def _apply_gravity_mode(self, model: mujoco.MjModel, gravity_mode: str) -> None:
        mode = str(gravity_mode).strip().lower()
        if mode == "model":
            return
        if mode == "zero":
            model.opt.gravity[:] = 0.0
            return
        raise ValueError(f"Unsupported gravity_mode: {gravity_mode!r}")

    def _load_force_schedules(
        self,
        episode_path: Path,
        episode: Any,
        articulation: dict[str, Any],
        trajectories: list[_Trajectory],
    ) -> tuple[dict[str, _ForceSchedule], dict[str, Any]]:
        log_path = self._resolve_dynamics_log_path(episode_path, episode)
        if log_path is None:
            return {}, {"available": False, "reason": "episode has no dynamics log path"}
        if not log_path.is_file():
            return {}, {"available": False, "reason": f"dynamics log does not exist: {log_path}"}

        joint_log_candidates = self._force_log_name_candidates(episode, articulation, trajectories)
        samples: dict[str, list[tuple[float, float, str]]] = {traj.joint_name: [] for traj in trajectories}
        line_count = 0
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                line_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp_s = float(row.get("timestamp_s", 0.0))
                raw_joints = row.get("joints", {})
                if not isinstance(raw_joints, dict) or not raw_joints:
                    continue
                single_logged_joint = next(iter(raw_joints)) if len(raw_joints) == 1 else None
                for traj in trajectories:
                    source_name = None
                    source_payload = None
                    for candidate in joint_log_candidates.get(traj.joint_name, []):
                        if candidate in raw_joints and isinstance(raw_joints[candidate], dict):
                            source_name = candidate
                            source_payload = raw_joints[candidate]
                            break
                    if source_payload is None and single_logged_joint is not None and len(trajectories) == 1:
                        maybe_payload = raw_joints.get(single_logged_joint)
                        if isinstance(maybe_payload, dict):
                            source_name = str(single_logged_joint)
                            source_payload = maybe_payload
                    if source_payload is None:
                        continue
                    force = float(
                        source_payload.get(
                            "applied_generalized_force",
                            source_payload.get("excitation_force", 0.0),
                        )
                    )
                    samples[traj.joint_name].append((timestamp_s, force, str(source_name)))

        schedules: dict[str, _ForceSchedule] = {}
        mapped: list[dict[str, Any]] = []
        for traj in trajectories:
            raw_samples = samples.get(traj.joint_name, [])
            if not raw_samples:
                continue
            source_names = [item[2] for item in raw_samples]
            source_name = max(set(source_names), key=source_names.count)
            schedules[traj.joint_name] = _ForceSchedule(
                joint_name=traj.joint_name,
                source_joint_name=source_name,
                timestamps_s=np.array([item[0] for item in raw_samples], dtype=float),
                forces=np.array([item[1] for item in raw_samples], dtype=float),
            )
            nonzero_count = sum(1 for _, force, _ in raw_samples if abs(force) > 1e-12)
            mapped.append(
                {
                    "joint_name": traj.joint_name,
                    "source_joint_name": source_name,
                    "sample_count": len(raw_samples),
                    "nonzero_sample_count": nonzero_count,
                    "force_min": float(min(force for _, force, _ in raw_samples)),
                    "force_max": float(max(force for _, force, _ in raw_samples)),
                }
            )

        if not schedules:
            return {}, {
                "available": False,
                "reason": "dynamics log did not contain any mappable joint force samples",
                "dynamics_log_path": str(log_path),
                "line_count": line_count,
            }

        return schedules, {
            "available": True,
            "dynamics_log_path": str(log_path),
            "line_count": line_count,
            "mapped_joints": mapped,
        }

    def _resolve_dynamics_log_path(self, episode_path: Path, episode: Any) -> Path | None:
        excitation = episode.metadata.get("excitation", {}) if hasattr(episode, "metadata") else {}
        raw_log_path = excitation.get("dynamics_log_path") if isinstance(excitation, dict) else None
        if not raw_log_path:
            candidate = episode_path.parent / "dynamics_log.jsonl"
            return candidate if candidate.exists() else None
        log_path = Path(str(raw_log_path)).expanduser()
        if not log_path.is_absolute():
            log_path = episode_path.parent / log_path
        return log_path.resolve()

    def _force_log_name_candidates(
        self,
        episode: Any,
        articulation: dict[str, Any],
        trajectories: list[_Trajectory],
    ) -> dict[str, list[str]]:
        joint_metadata = {
            str(joint.get("name", "")): dict(joint.get("metadata", {}))
            for joint in articulation.get("joints", [])
            if isinstance(joint, dict)
        }
        episode_joint_names = [
            str(name)
            for name in (episode.metadata.get("joint_names", []) if hasattr(episode, "metadata") else [])
            if isinstance(name, str)
        ]
        primary_episode_joint_name = (
            str(episode.metadata.get("joint_name", ""))
            if hasattr(episode, "metadata") and episode.metadata.get("joint_name")
            else ""
        )

        candidates: dict[str, list[str]] = {}
        for traj in trajectories:
            names = [traj.joint_name]
            metadata = joint_metadata.get(traj.joint_name, {})
            prior = metadata.get("prior")
            if isinstance(prior, dict) and prior.get("joint_name"):
                names.append(str(prior["joint_name"]))
            if primary_episode_joint_name:
                names.append(primary_episode_joint_name)
            names.extend(episode_joint_names)
            deduped: list[str] = []
            for name in names:
                if name and name not in deduped:
                    deduped.append(name)
            candidates[traj.joint_name] = deduped
        return candidates

    def _coordinate_search_warm_start(
        self,
        theta: np.ndarray,
        best_theta: np.ndarray,
        best_loss: float,
        best_breakdown: dict[str, float],
        best_simulated: dict[str, list[dict[str, float]]],
        best_values: list[float],
        specs: list[_ParameterSpec],
        evaluate: Any,
        history: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, float], dict[str, list[dict[str, float]]], list[float]]:
        # Local finite differences are brittle when joint limits or dry
        # friction introduce discontinuities. A tiny coordinate scan gives the
        # optimizer a better basin without making the MVP dependent on scipy.
        current_theta = theta.copy()
        current_loss = best_loss
        current_breakdown = best_breakdown
        current_simulated = best_simulated
        current_values = best_values

        for pass_index in range(2):
            improved_this_pass = False
            for index, spec in enumerate(specs):
                for candidate_value in self._warm_start_candidate_values(spec):
                    theta_candidate = current_theta.copy()
                    theta_candidate[index] = spec.encode(candidate_value)
                    candidate_loss, candidate_breakdown, candidate_simulated, candidate_values = evaluate(theta_candidate)
                    if candidate_loss + 1e-9 >= current_loss:
                        continue
                    current_theta = theta_candidate
                    current_loss = candidate_loss
                    current_breakdown = candidate_breakdown
                    current_simulated = candidate_simulated
                    current_values = candidate_values
                    improved_this_pass = True
                    if candidate_loss + 1e-9 < best_loss:
                        best_loss = candidate_loss
                        best_theta = theta_candidate.copy()
                        best_breakdown = candidate_breakdown
                        best_simulated = candidate_simulated
                        best_values = candidate_values
                    history.append(
                        {
                            "iteration": 0,
                            "phase": "coordinate_search",
                            "coordinate_search_pass": pass_index,
                            "parameter": spec.name,
                            "loss": candidate_loss,
                            "loss_breakdown": candidate_breakdown,
                            "parameter_values": _parameter_value_map(specs, candidate_values),
                        }
                    )
            if not improved_this_pass:
                break
        return current_theta, best_theta, best_loss, best_breakdown, best_simulated, best_values

    def _warm_start_candidate_values(self, spec: _ParameterSpec) -> list[float]:
        initial = max(spec.lower, min(spec.upper, spec.initial_value))
        if spec.kind == "body_mass":
            raw_values = [
                spec.lower,
                0.25 * initial,
                0.5 * initial,
                initial,
                2.0 * initial,
                5.0 * initial,
                10.0 * initial,
                20.0 * initial,
                spec.upper,
            ]
        else:
            span = max(spec.upper - spec.lower, 1e-9)
            raw_values = [
                spec.lower,
                initial,
                0.25 * spec.upper,
                0.5 * spec.upper,
                spec.upper,
                spec.lower + 0.05 * span,
            ]
        values: list[float] = []
        for value in raw_values:
            clamped = max(spec.lower, min(spec.upper, float(value)))
            if all(abs(clamped - existing) > 1e-12 for existing in values):
                values.append(clamped)
        return values

    def _trajectory_error_summary(
        self,
        observed: list[_Trajectory],
        simulated: dict[str, list[dict[str, float]]],
    ) -> dict[str, Any]:
        per_joint: dict[str, dict[str, float]] = {}
        q_errors_all: list[float] = []
        qdot_errors_all: list[float] = []
        for traj in observed:
            simulated_samples = simulated.get(traj.joint_name, [])
            simulated_by_time = {round(sample["timestamp_s"], 9): sample for sample in simulated_samples}
            q_errors: list[float] = []
            qdot_errors: list[float] = []
            for timestamp_s, q_obs, qdot_obs in zip(traj.timestamps_s, traj.q_observed, traj.qdot_observed):
                sample = simulated_by_time.get(round(timestamp_s, 9))
                if sample is None:
                    continue
                q_error = float(sample["q"]) - q_obs
                qdot_error = float(sample["qdot"]) - qdot_obs
                if not (math.isfinite(q_error) and math.isfinite(qdot_error)):
                    continue
                q_errors.append(q_error)
                qdot_errors.append(qdot_error)
            per_joint[traj.joint_name] = {
                "sample_count": len(q_errors),
                "q_mse": _mse(q_errors),
                "q_rmse": _rmse(q_errors),
                "q_mae": _mae(q_errors),
                "q_max_abs": _max_abs(q_errors),
                "qdot_mse": _mse(qdot_errors),
                "qdot_rmse": _rmse(qdot_errors),
                "qdot_mae": _mae(qdot_errors),
                "qdot_max_abs": _max_abs(qdot_errors),
            }
            q_errors_all.extend(q_errors)
            qdot_errors_all.extend(qdot_errors)
        return {
            "sample_count": len(q_errors_all),
            "q_mse": _mse(q_errors_all),
            "q_rmse": _rmse(q_errors_all),
            "q_mae": _mae(q_errors_all),
            "q_max_abs": _max_abs(q_errors_all),
            "qdot_mse": _mse(qdot_errors_all),
            "qdot_rmse": _rmse(qdot_errors_all),
            "qdot_mae": _mae(qdot_errors_all),
            "qdot_max_abs": _max_abs(qdot_errors_all),
            "per_joint": per_joint,
        }

    def _ground_truth_parameter_comparison(
        self,
        episode: Any,
        articulation: dict[str, Any],
        estimated_model: mujoco.MjModel,
    ) -> dict[str, Any]:
        gt_path = _resolve_ground_truth_mjcf_path(episode)
        if gt_path is None or not gt_path.exists():
            return {
                "available": False,
                "reason": "episode.metadata.model_path missing or does not exist",
            }
        gt_model = mujoco.MjModel.from_xml_path(str(gt_path))

        parts: list[dict[str, Any]] = []
        joints: list[dict[str, Any]] = []
        mass_rel_errors: list[float] = []
        inertia_rel_errors: list[float] = []
        damping_rel_errors: list[float] = []
        friction_rel_errors: list[float] = []
        effective_inertia_rel_errors: list[float] = []

        for raw_part in articulation.get("parts", []):
            if not isinstance(raw_part, dict):
                continue
            body_name = str(raw_part.get("name", "")).strip()
            if not body_name:
                continue
            estimated_body_id = mujoco.mj_name2id(estimated_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            gt_body_id = mujoco.mj_name2id(gt_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if estimated_body_id < 0 or gt_body_id < 0:
                continue
            est_mass = float(estimated_model.body_mass[estimated_body_id])
            gt_mass = float(gt_model.body_mass[gt_body_id])
            est_inertia = [float(value) for value in estimated_model.body_inertia[estimated_body_id].tolist()]
            gt_inertia = [float(value) for value in gt_model.body_inertia[gt_body_id].tolist()]
            mass_rel = _relative_error(est_mass, gt_mass)
            inertia_rel = [_relative_error(est, gt) for est, gt in zip(est_inertia, gt_inertia)]
            mass_rel_errors.append(mass_rel)
            inertia_rel_errors.extend(inertia_rel)
            parts.append(
                {
                    "name": body_name,
                    "estimated_mass_kg": est_mass,
                    "ground_truth_mass_kg": gt_mass,
                    "mass_abs_error": abs(est_mass - gt_mass),
                    "mass_rel_error": mass_rel,
                    "estimated_inertia_diag": est_inertia,
                    "ground_truth_inertia_diag": gt_inertia,
                    "inertia_abs_error_diag": [abs(est - gt) for est, gt in zip(est_inertia, gt_inertia)],
                    "inertia_rel_error_diag": inertia_rel,
                    "inertia_rel_error_mean": float(sum(inertia_rel) / len(inertia_rel)) if inertia_rel else 0.0,
                }
            )

        for raw_joint in articulation.get("joints", []):
            if not isinstance(raw_joint, dict):
                continue
            joint_name = str(raw_joint.get("name", "")).strip()
            if not joint_name:
                continue
            estimated_joint_id = mujoco.mj_name2id(estimated_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            gt_joint_id = self._match_ground_truth_joint_id(gt_model, raw_joint)
            if estimated_joint_id < 0 or gt_joint_id < 0:
                continue
            estimated_dof_adr = int(estimated_model.jnt_dofadr[estimated_joint_id])
            gt_dof_adr = int(gt_model.jnt_dofadr[gt_joint_id])
            est_damping = float(estimated_model.dof_damping[estimated_dof_adr])
            gt_damping = float(gt_model.dof_damping[gt_dof_adr])
            est_friction = float(estimated_model.dof_frictionloss[estimated_dof_adr])
            gt_friction = float(gt_model.dof_frictionloss[gt_dof_adr])
            est_effective_inertia = self._joint_effective_inertia(estimated_model, estimated_joint_id)
            gt_effective_inertia = self._joint_effective_inertia(gt_model, gt_joint_id)
            damping_rel = _relative_error(est_damping, gt_damping)
            friction_rel = _relative_error(est_friction, gt_friction)
            effective_inertia_rel = _relative_error(est_effective_inertia, gt_effective_inertia)
            damping_rel_errors.append(damping_rel)
            friction_rel_errors.append(friction_rel)
            effective_inertia_rel_errors.append(effective_inertia_rel)
            joints.append(
                {
                    "name": joint_name,
                    "estimated_effective_inertia": est_effective_inertia,
                    "ground_truth_effective_inertia": gt_effective_inertia,
                    "effective_inertia_abs_error": abs(est_effective_inertia - gt_effective_inertia),
                    "effective_inertia_rel_error": effective_inertia_rel,
                    "estimated_damping": est_damping,
                    "ground_truth_damping": gt_damping,
                    "damping_abs_error": abs(est_damping - gt_damping),
                    "damping_rel_error": damping_rel,
                    "estimated_frictionloss": est_friction,
                    "ground_truth_frictionloss": gt_friction,
                    "frictionloss_abs_error": abs(est_friction - gt_friction),
                    "frictionloss_rel_error": friction_rel,
                }
            )

        return {
            "available": True,
            "ground_truth_mjcf_path": str(gt_path),
            "summary": {
                "part_count": len(parts),
                "joint_count": len(joints),
                "mass_rel_error_mean": _mean(mass_rel_errors),
                "inertia_rel_error_mean": _mean(inertia_rel_errors),
                "effective_inertia_rel_error_mean": _mean(effective_inertia_rel_errors),
                "damping_rel_error_mean": _mean(damping_rel_errors),
                "frictionloss_rel_error_mean": _mean(friction_rel_errors),
            },
            "parts": parts,
            "joints": joints,
        }

    def _render_rollout_comparison(
        self,
        episode: Any,
        episode_path: Path,
        model: mujoco.MjModel,
        trajectories: list[_Trajectory],
        force_schedules: dict[str, _ForceSchedule],
        output_dir: Path,
        gl_backend: str | None = None,
    ) -> dict[str, Any]:
        if gl_backend == "none":
            return {"available": False, "reason": "rendering disabled by render_gl_backend=none"}
        try:
            import imageio.v2 as imageio
        except Exception:
            return {"available": False, "reason": "imageio not installed"}

        if not episode.frames:
            return {"available": False, "reason": "episode has no frames"}

        frame_dt = float(episode.metadata.get("frame_dt", model.opt.timestep))
        if frame_dt <= 0.0:
            frame_dt = float(model.opt.timestep)

        view_index = self._render_view_index(episode)
        first_rgb_path = self._frame_rgb_path(episode_path, episode.frames[0], view_index)
        if first_rgb_path is None or not first_rgb_path.exists():
            return {"available": False, "reason": "original RGB frame missing"}

        first_rgb = imageio.imread(first_rgb_path)
        height = int(first_rgb.shape[0])
        width = int(first_rgb.shape[1])
        renderer_stderr = io.StringIO()
        original_renderer_del = getattr(mujoco.Renderer, "__del__", None)

        def safe_renderer_del(renderer: Any) -> None:
            if original_renderer_del is None:
                return
            if not hasattr(renderer, "_mjr_context"):
                return
            try:
                original_renderer_del(renderer)
            except AttributeError:
                return

        if original_renderer_del is not None:
            mujoco.Renderer.__del__ = safe_renderer_del  # type: ignore[method-assign]
        try:
            with contextlib.redirect_stderr(renderer_stderr):
                renderer = mujoco.Renderer(model, width=width, height=height)
        except Exception as exc:
            with contextlib.redirect_stderr(renderer_stderr):
                gc.collect()
            extra = renderer_stderr.getvalue().strip()
            reason = f"renderer initialization failed: {exc}"
            if extra:
                reason = f"{reason}; stderr: {extra}"
            return {"available": False, "backend": gl_backend or "auto", "reason": reason}

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._configure_render_camera(camera, episode, view_index)
        camera_fovy_deg = episode.metadata.get("camera_fovy_deg")
        if camera_fovy_deg is not None:
            model.vis.global_.fovy = float(camera_fovy_deg)

        data = mujoco.MjData(model)
        for traj in trajectories:
            data.qpos[traj.qpos_adr] = traj.q_observed[0]
            data.qvel[traj.dof_adr] = traj.qdot_observed[0]
        mujoco.mj_forward(model, data)

        sim_video_path = output_dir / f"simulated_view_{view_index}.mp4"
        compare_video_path = output_dir / f"comparison_view_{view_index}.mp4"
        sim_writer = None
        compare_writer = None
        rendered_frames = 0
        try:
            sim_writer = imageio.get_writer(sim_video_path, fps=float(1.0 / max(frame_dt, 1e-6)), codec="libx264")
            compare_writer = imageio.get_writer(compare_video_path, fps=float(1.0 / max(frame_dt, 1e-6)), codec="libx264")
            for frame in episode.frames:
                target_time = float(frame.timestamp_s)
                while data.time + model.opt.timestep < target_time - 1e-12:
                    self._apply_force_replay(data, trajectories, force_schedules)
                    mujoco.mj_step(model, data)
                if data.time < target_time - 1e-12:
                    self._apply_force_replay(data, trajectories, force_schedules)
                    mujoco.mj_step(model, data)

                renderer.update_scene(data, camera=camera)
                sim_rgb = renderer.render()
                original_rgb_path = self._frame_rgb_path(episode_path, frame, view_index)
                if original_rgb_path is None or not original_rgb_path.exists():
                    continue
                original_rgb = imageio.imread(original_rgb_path)
                if original_rgb.shape[0] != sim_rgb.shape[0]:
                    continue
                sim_writer.append_data(sim_rgb)
                compare_writer.append_data(np.concatenate([original_rgb[:, :, :3], sim_rgb[:, :, :3]], axis=1))
                rendered_frames += 1
        finally:
            if sim_writer is not None:
                sim_writer.close()
            if compare_writer is not None:
                compare_writer.close()
            try:
                renderer.close()
            finally:
                if original_renderer_del is not None:
                    mujoco.Renderer.__del__ = original_renderer_del  # type: ignore[method-assign]

        return {
            "available": rendered_frames > 0,
            "backend": gl_backend or "auto",
            "view_index": view_index,
            "rendered_frame_count": rendered_frames,
            "simulated_video_path": str(sim_video_path.resolve()),
            "comparison_video_path": str(compare_video_path.resolve()),
        }

    def _render_view_index(self, episode: Any) -> int:
        if episode.metadata.get("camera_mode") == "triview":
            return 1
        return 0

    def _frame_rgb_path(self, episode_path: Path, frame: Any, view_index: int) -> Path | None:
        episode_dir = episode_path.parent
        paths_by_view = list(getattr(frame, "rgb_paths_by_view", []) or [])
        if view_index < len(paths_by_view):
            return (episode_dir / paths_by_view[view_index]).resolve()
        rgb_path = getattr(frame, "rgb_path", None)
        if rgb_path is None:
            return None
        return (episode_dir / str(rgb_path)).resolve()

    def _configure_render_camera(self, camera: Any, episode: Any, view_index: int) -> None:
        metadata = dict(episode.metadata)
        camera.lookat[:] = [float(value) for value in metadata.get("lookat", [0.0, 0.0, 0.0])]
        camera.distance = float(metadata.get("camera_distance", 1.5))
        camera.elevation = float(metadata.get("camera_elevation_deg", 0.0))
        azimuths = metadata.get("camera_azimuths_deg") or []
        if metadata.get("camera_mode") == "triview" and view_index < len(azimuths):
            camera.azimuth = float(azimuths[view_index])
            return
        if metadata.get("camera_mode") == "orbit" and len(azimuths) >= 1:
            camera.azimuth = float(azimuths[0])
            return
        camera.azimuth = float(azimuths[min(view_index, len(azimuths) - 1)]) if azimuths else 0.0

    def _match_ground_truth_joint_id(self, gt_model: mujoco.MjModel, inferred_joint: dict[str, Any]) -> int:
        joint_name = str(inferred_joint.get("name", "")).strip()
        if joint_name:
            direct = mujoco.mj_name2id(gt_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if direct >= 0:
                return direct

        child_body_name = str(inferred_joint.get("child", "")).strip()
        if child_body_name:
            child_body_id = mujoco.mj_name2id(gt_model, mujoco.mjtObj.mjOBJ_BODY, child_body_name)
            if child_body_id >= 0:
                body_jntadr = int(gt_model.body_jntadr[child_body_id])
                body_jntnum = int(gt_model.body_jntnum[child_body_id])
                if body_jntnum == 1:
                    return body_jntadr
        return -1

    def _joint_effective_inertia(self, model: mujoco.MjModel, joint_id: int) -> float:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dense_mass = np.zeros((model.nv, model.nv), dtype=float)
        mujoco.mj_fullM(model, dense_mass, data.qM)
        dof_adr = int(model.jnt_dofadr[joint_id])
        return float(dense_mass[dof_adr, dof_adr])

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
                        upper=max(1e-2, 100.0 * mass),
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

    def _rollout(
        self,
        model: mujoco.MjModel,
        trajectories: list[_Trajectory],
        force_schedules: dict[str, _ForceSchedule],
    ) -> dict[str, list[dict[str, float]]]:
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
                self._apply_force_replay(data, trajectories, force_schedules)
                mujoco.mj_step(model, data)
            if data.time < target - 1e-12:
                self._apply_force_replay(data, trajectories, force_schedules)
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

    def _apply_force_replay(
        self,
        data: mujoco.MjData,
        trajectories: list[_Trajectory],
        force_schedules: dict[str, _ForceSchedule],
    ) -> None:
        data.qfrc_applied[:] = 0.0
        if not force_schedules:
            return
        time_s = float(data.time)
        for traj in trajectories:
            schedule = force_schedules.get(traj.joint_name)
            if schedule is None or schedule.timestamps_s.size == 0:
                continue
            force = self._sample_force_schedule(schedule, time_s)
            data.qfrc_applied[traj.dof_adr] = force

    def _sample_force_schedule(self, schedule: _ForceSchedule, time_s: float) -> float:
        index = int(np.searchsorted(schedule.timestamps_s, time_s + 1e-12, side="right")) - 1
        if index < 0:
            return 0.0
        if index >= int(schedule.forces.size):
            index = int(schedule.forces.size) - 1
        return float(schedule.forces[index])

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
        enable_contact: bool = False,
        gravity_mode: str = "model",
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

        if not bool(enable_contact):
            for geom in root.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        if str(gravity_mode).strip().lower() == "zero":
            option = root.find("option")
            if option is None:
                option = ET.Element("option")
                root.insert(0, option)
            option.set("gravity", "0 0 0")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_xml_tree(tree, output_path)

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


def _resolve_ground_truth_mjcf_path(episode: Any) -> Path | None:
    raw_path = episode.metadata.get("model_path") if hasattr(episode, "metadata") else None
    if raw_path is None:
        return None
    try:
        return Path(str(raw_path)).expanduser().resolve()
    except Exception:
        return None




def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mse(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return float(sum(error * error for error in errors) / len(errors))


def _rmse(errors: list[float]) -> float:
    return math.sqrt(_mse(errors))


def _mae(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return float(sum(abs(error) for error in errors) / len(errors))


def _max_abs(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return float(max(abs(error) for error in errors))


def _relative_error(estimated: float, ground_truth: float) -> float:
    return abs(float(estimated) - float(ground_truth)) / max(abs(float(ground_truth)), 1e-9)
