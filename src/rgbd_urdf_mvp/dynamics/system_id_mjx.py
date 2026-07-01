from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.jax_runtime import configure_jax_runtime
from ..core.serialization import load_episode, load_json, save_json
from .system_id import (
    DynamicsIdentifier,
    _ParameterSpec,
    _Trajectory,
    _parameter_value_map,
    _trajectory_to_samples,
)


@dataclass(slots=True)
class MJXDynamicsIdentificationConfig:
    episode_path: str | Path
    articulation_artifact_path: str | Path
    mjcf_path: str | Path
    output_dir: str | Path | None = None
    jax_platform: str = "cpu"
    enable_pjrt_compatibility: bool | None = None
    max_iterations: int = 25
    learning_rate: float = 0.05
    q_weight: float = 1.0
    qdot_weight: float = 0.05
    prior_weight: float = 0.02
    optimize_static_parts: bool = False
    optimize_mass: bool = True
    optimize_damping: bool = True
    optimize_friction: bool = True
    enable_contact: bool = False
    jit: bool = True
    render_gl_backend: str | None = None


class MJXDynamicsIdentifier:
    def run(self, config: MJXDynamicsIdentificationConfig) -> Path:
        episode = load_episode(config.episode_path)
        if str(episode.metadata.get("control_mode", "free")) != "free":
            raise ValueError("identify-dynamics-mjx currently supports only recordings made with control_mode='free'.")

        configure_jax_runtime(
            platform=str(config.jax_platform),
            enable_pjrt_compatibility=config.enable_pjrt_compatibility,
        )
        jax, jnp, mjx = self._import_jax_mjx()
        helper = DynamicsIdentifier()

        articulation = load_json(config.articulation_artifact_path)
        mjcf_path = Path(config.mjcf_path).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        if not bool(config.enable_contact):
            helper._disable_geom_contacts(model)
        trajectories = helper._load_trajectories(articulation, model)
        if not trajectories:
            raise ValueError("No joint state tracks found in articulation_artifact.fit_metrics['joint_state_tracks'].")
        shared_times = self._shared_schedule(trajectories)
        specs = helper._build_parameter_specs(model, articulation, config)
        if not specs:
            raise ValueError("No dynamic parameters selected for optimization.")

        sim_dt = float(episode.metadata.get("sim_dt", model.opt.timestep))
        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive for MJX rollout.")
        model.opt.timestep = sim_dt

        observed_q = jnp.asarray([traj.q_observed for traj in trajectories], dtype=jnp.float32).T
        observed_qdot = jnp.asarray([traj.qdot_observed for traj in trajectories], dtype=jnp.float32).T
        timestamps = jnp.asarray(shared_times, dtype=jnp.float32)
        qpos_indices = jnp.asarray([traj.qpos_adr for traj in trajectories], dtype=jnp.int32)
        dof_indices = jnp.asarray([traj.dof_adr for traj in trajectories], dtype=jnp.int32)
        step_count_list = self._step_counts(shared_times, sim_dt)
        step_counts = jnp.asarray(step_count_list, dtype=jnp.int32)
        max_step_count = int(max(step_count_list)) if step_count_list else 0

        theta0 = jnp.asarray([spec.encode(spec.initial_value) for spec in specs], dtype=jnp.float32)
        initial_values = jnp.asarray([spec.initial_value for spec in specs], dtype=jnp.float32)
        lower_bounds = jnp.asarray([spec.lower for spec in specs], dtype=jnp.float32)
        upper_bounds = jnp.asarray([spec.upper for spec in specs], dtype=jnp.float32)
        prior_scales = jnp.asarray([spec.prior_scale() for spec in specs], dtype=jnp.float32)
        body_mass_indices = jnp.asarray(
            [spec.index for spec in specs if spec.kind == "body_mass"],
            dtype=jnp.int32,
        )
        body_mass_initials = jnp.asarray(
            [spec.initial_value for spec in specs if spec.kind == "body_mass"],
            dtype=jnp.float32,
        )
        body_mass_reference_inertias = jnp.asarray(
            [
                spec.reference_inertia.tolist() if spec.reference_inertia is not None else [1e-6, 1e-6, 1e-6]
                for spec in specs
                if spec.kind == "body_mass"
            ],
            dtype=jnp.float32,
        )
        damping_indices = jnp.asarray(
            [spec.index for spec in specs if spec.kind == "joint_damping"],
            dtype=jnp.int32,
        )
        friction_indices = jnp.asarray(
            [spec.index for spec in specs if spec.kind == "joint_frictionloss"],
            dtype=jnp.int32,
        )
        kind_codes = tuple(self._kind_code(spec.kind) for spec in specs)

        initial_data = mujoco.MjData(model)
        for traj in trajectories:
            initial_data.qpos[traj.qpos_adr] = traj.q_observed[0]
            initial_data.qvel[traj.dof_adr] = traj.qdot_observed[0]
        mujoco.mj_forward(model, initial_data)

        jax_device = jax.devices()[0]
        try:
            # MuJoCo MJX does not yet auto-resolve the Apple METAL backend in
            # put_model/put_data, so we pass the JAX implementation/device pair
            # explicitly. This still keeps the rollout and gradients on the
            # active JAX backend selected by the CLI runtime.
            mx_template = mjx.put_model(model, impl="jax", device=jax_device)
            dx0 = mjx.put_data(model, initial_data, impl="jax", device=jax_device)
        except Exception as exc:
            if str(config.jax_platform).lower() == "metal" and "default_memory_space" in str(exc):
                raise RuntimeError(
                    "JAX Metal backend initialized, but MJX model upload failed with "
                    "'default_memory_space is not supported'. Re-run with --jax-platform cpu "
                    "as the safe fallback, or use a dedicated Metal environment with a "
                    "known-good pinned jax/jaxlib/jax-metal stack."
                ) from exc
            raise

        def decode_theta(theta):
            values = []
            for index, kind_code in enumerate(kind_codes):
                raw = theta[index]
                if kind_code == 0:
                    value = jnp.exp(raw)
                else:
                    value = jnp.maximum(0.0, jnp.exp(raw) - 1e-6)
                value = jnp.clip(value, lower_bounds[index], upper_bounds[index])
                values.append(value)
            return jnp.stack(values) if values else jnp.zeros((0,), dtype=jnp.float32)

        def apply_values(mx_model, values):
            body_mass = mx_model.body_mass
            body_inertia = mx_model.body_inertia
            dof_damping = mx_model.dof_damping
            dof_frictionloss = mx_model.dof_frictionloss

            mass_offset = 0
            damping_offset = 0
            friction_offset = 0
            for index, kind_code in enumerate(kind_codes):
                value = values[index]
                if kind_code == 0:
                    body_index = body_mass_indices[mass_offset]
                    initial_mass = body_mass_initials[mass_offset]
                    scale = value / jnp.maximum(initial_mass, 1e-9)
                    body_mass = body_mass.at[body_index].set(value)
                    body_inertia = body_inertia.at[body_index].set(
                        jnp.maximum(1e-9, body_mass_reference_inertias[mass_offset] * scale)
                    )
                    mass_offset += 1
                elif kind_code == 1:
                    dof_damping = dof_damping.at[damping_indices[damping_offset]].set(value)
                    damping_offset += 1
                else:
                    dof_frictionloss = dof_frictionloss.at[friction_indices[friction_offset]].set(value)
                    friction_offset += 1
            return self._replace_pytree(
                mx_model,
                body_mass=body_mass,
                body_inertia=body_inertia,
                dof_damping=dof_damping,
                dof_frictionloss=dof_frictionloss,
            )

        def rollout(theta):
            values = decode_theta(theta)
            mx_model = apply_values(mx_template, values)
            step_indices = jnp.arange(max_step_count, dtype=jnp.int32)

            def advance(data, steps):
                def masked_step(carry, step_index):
                    stepped = mjx.step(mx_model, carry)
                    next_carry = jax.lax.cond(
                        step_index < steps,
                        lambda _: stepped,
                        lambda _: carry,
                        operand=None,
                    )
                    return next_carry, None

                data, _ = jax.lax.scan(masked_step, data, step_indices)
                sample_q = data.qpos[qpos_indices]
                sample_qdot = data.qvel[dof_indices]
                return data, (sample_q, sample_qdot)

            _, samples = jax.lax.scan(advance, dx0, step_counts)
            sim_q, sim_qdot = samples
            return values, sim_q, sim_qdot

        def metrics_fn(theta):
            values, sim_q, sim_qdot = rollout(theta)
            finite_mask = jnp.isfinite(sim_q) & jnp.isfinite(sim_qdot)
            invalid_sample_count = jnp.sum(~finite_mask)

            q_sanitized = jnp.nan_to_num(sim_q, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
            qdot_sanitized = jnp.nan_to_num(sim_qdot, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

            q_clip_limit = jnp.float32(5.0)
            qdot_clip_limit = jnp.float32(50.0)
            q_clipped = jnp.clip(q_sanitized, -q_clip_limit, q_clip_limit)
            qdot_clipped = jnp.clip(qdot_sanitized, -qdot_clip_limit, qdot_clip_limit)

            q_mse = jnp.mean((q_clipped - observed_q) ** 2)
            qdot_mse = jnp.mean((qdot_clipped - observed_qdot) ** 2)
            prior_penalty = jnp.sum(((values - initial_values) / prior_scales) ** 2)
            overflow_penalty = (
                jnp.mean(jnp.square(jnp.maximum(0.0, jnp.abs(q_sanitized) - q_clip_limit)))
                + 0.1 * jnp.mean(jnp.square(jnp.maximum(0.0, jnp.abs(qdot_sanitized) - qdot_clip_limit)))
                + 100.0 * invalid_sample_count.astype(jnp.float32)
            )
            total = (
                float(config.q_weight) * q_mse
                + float(config.qdot_weight) * qdot_mse
                + float(config.prior_weight) * prior_penalty
                + overflow_penalty
            )
            return (
                total,
                q_mse,
                qdot_mse,
                prior_penalty,
                overflow_penalty,
                invalid_sample_count.astype(jnp.float32),
                values,
                sim_q,
                sim_qdot,
            )

        def loss_fn(theta):
            total, *_ = metrics_fn(theta)
            return total

        if bool(config.jit):
            metrics_eval = jax.jit(metrics_fn)
            loss_eval = jax.jit(loss_fn)
            loss_grad = jax.jit(jax.jacfwd(loss_fn))
        else:
            metrics_eval = metrics_fn
            loss_eval = loss_fn
            loss_grad = jax.jacfwd(loss_fn)

        theta = theta0
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        history: list[dict[str, Any]] = []
        best_payload = metrics_eval(theta)
        best_loss = float(best_payload[0])
        best_breakdown = self._breakdown_dict(best_payload)
        best_theta = np.asarray(theta, dtype=float)
        best_values = np.asarray(best_payload[6], dtype=float)
        best_sim_q = np.asarray(best_payload[7], dtype=float)
        best_sim_qdot = np.asarray(best_payload[8], dtype=float)
        history.append(
            {
                "iteration": 0,
                "loss": best_loss,
                "loss_breakdown": best_breakdown,
                "parameter_values": _parameter_value_map(specs, best_values.tolist()),
            }
        )

        learning_rate = max(1e-5, float(config.learning_rate))
        for iteration in range(1, max(1, int(config.max_iterations)) + 1):
            loss_value = loss_eval(theta)
            grads = loss_grad(theta)
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            m = beta1 * m + (1.0 - beta1) * grads
            v = beta2 * v + (1.0 - beta2) * (grads * grads)
            m_hat = m / (1.0 - beta1**iteration)
            v_hat = v / (1.0 - beta2**iteration)
            theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

            payload = metrics_eval(theta)
            current_loss = float(payload[0])
            current_values = np.asarray(payload[6], dtype=float)
            history.append(
                {
                    "iteration": iteration,
                    "loss": current_loss,
                    "loss_breakdown": self._breakdown_dict(payload),
                    "parameter_values": _parameter_value_map(specs, current_values.tolist()),
                    "gradient_norm": float(np.linalg.norm(np.asarray(grads, dtype=float))),
                }
            )
            if current_loss + 1e-9 < best_loss:
                best_loss = current_loss
                best_breakdown = self._breakdown_dict(payload)
                best_theta = np.asarray(theta, dtype=float)
                best_values = current_values
                best_sim_q = np.asarray(payload[7], dtype=float)
                best_sim_qdot = np.asarray(payload[8], dtype=float)

        helper._apply_parameters(model, specs, best_values.tolist())
        output_dir = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir is not None
            else mjcf_path.parent / "dynamics_identification_mjx"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        optimized_xml_path = output_dir / f"{mjcf_path.stem}.identified.xml"
        helper._write_optimized_mjcf(
            mjcf_path=mjcf_path,
            output_path=optimized_xml_path,
            specs=specs,
            values=best_values.tolist(),
            enable_contact=bool(config.enable_contact),
        )
        optimized_model = mujoco.MjModel.from_xml_path(str(optimized_xml_path))
        if not bool(config.enable_contact):
            helper._disable_geom_contacts(optimized_model)
        simulated_payload = self._simulated_payload(trajectories, timestamps.tolist(), best_sim_q, best_sim_qdot)
        trajectory_errors = helper._trajectory_error_summary(trajectories, simulated_payload)
        gt_comparison = helper._ground_truth_parameter_comparison(
            episode=episode,
            articulation=articulation,
            estimated_model=optimized_model,
        )
        render_summary = helper._render_rollout_comparison(
            episode=episode,
            episode_path=Path(config.episode_path).expanduser().resolve(),
            model=optimized_model,
            trajectories=trajectories,
            force_schedules={},
            output_dir=output_dir,
            gl_backend=config.render_gl_backend,
        )

        artifact_path = output_dir / "dynamics_identification.json"
        save_json(
            {
                "source": "mjx-autodiff-system-id",
                "episode_path": str(Path(config.episode_path).resolve()),
                "articulation_artifact_path": str(Path(config.articulation_artifact_path).resolve()),
                "mjcf_input_path": str(mjcf_path),
                "mjcf_optimized_path": str(optimized_xml_path.resolve()),
                "runtime": {
                    "jax_platform": str(config.jax_platform),
                    "enable_pjrt_compatibility": config.enable_pjrt_compatibility,
                },
                "optimizer": {
                    "method": "jax-adam-autodiff-forward-mode",
                    "jit_enabled": bool(config.jit),
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
                },
                "identified_parameters": {
                    "parts": helper._part_parameter_summary(articulation, model, specs, best_values.tolist()),
                    "joints": helper._joint_parameter_summary(articulation, specs, best_values.tolist()),
                },
                "fit_metrics": {
                    "best_loss": float(best_loss),
                    "loss_breakdown": best_breakdown,
                    "trajectory_error": trajectory_errors,
                },
                "ground_truth_comparison": gt_comparison,
                "render_artifacts": render_summary,
                "trajectories": {
                    "observed": {
                        traj.joint_name: _trajectory_to_samples(traj.timestamps_s, traj.q_observed, traj.qdot_observed)
                        for traj in trajectories
                    },
                    "simulated": simulated_payload,
                },
                "limitations": [
                    "The current MJX path assumes all observed joints share a single timestamp schedule.",
                    "This MVP still targets free-motion single-scene episodes; contact-rich replay and control replay are not implemented.",
                    "Absolute mass can be weakly identifiable from passive single-DOF trajectories; damping/friction are usually better constrained.",
                ],
                "debug": {
                    "best_theta": [float(value) for value in best_theta.tolist()],
                },
            },
            artifact_path,
        )
        return artifact_path

    def _import_jax_mjx(self):
        try:
            import jax
            import jax.numpy as jnp
            from mujoco import mjx
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "identify-dynamics-mjx requires JAX + MuJoCo MJX. Install with: pip install -e '.[simulation,mjx]'"
            ) from exc
        return jax, jnp, mjx

    def _shared_schedule(self, trajectories: list[_Trajectory]) -> list[float]:
        reference = [round(float(timestamp_s), 9) for timestamp_s in trajectories[0].timestamps_s]
        for traj in trajectories[1:]:
            candidate = [round(float(timestamp_s), 9) for timestamp_s in traj.timestamps_s]
            if candidate != reference:
                raise ValueError(
                    "identify-dynamics-mjx currently requires all joint_state_tracks to share the same timestamps."
                )
        return [float(timestamp_s) for timestamp_s in reference]

    def _step_counts(self, timestamps_s: list[float], sim_dt: float) -> list[int]:
        if not timestamps_s:
            return []
        counts = [0]
        for previous, current in zip(timestamps_s[:-1], timestamps_s[1:]):
            delta = max(0.0, float(current) - float(previous))
            counts.append(max(0, int(round(delta / sim_dt))))
        return counts

    def _kind_code(self, kind: str) -> int:
        if kind == "body_mass":
            return 0
        if kind == "joint_damping":
            return 1
        if kind == "joint_frictionloss":
            return 2
        raise ValueError(f"Unsupported parameter kind for MJX system ID: {kind}")

    def _replace_pytree(self, obj: Any, **updates: Any) -> Any:
        if hasattr(obj, "replace"):
            return obj.replace(**updates)
        return dataclasses.replace(obj, **updates)

    def _breakdown_dict(self, payload: tuple[Any, ...]) -> dict[str, float]:
        return {
            "q_mse": float(payload[1]),
            "qdot_mse": float(payload[2]),
            "prior_penalty": float(payload[3]),
            "overflow_penalty": float(payload[4]),
            "invalid_sample_count": float(payload[5]),
        }

    def _simulated_payload(
        self,
        trajectories: list[_Trajectory],
        timestamps_s: list[float],
        sim_q: np.ndarray,
        sim_qdot: np.ndarray,
    ) -> dict[str, list[dict[str, float]]]:
        result: dict[str, list[dict[str, float]]] = {}
        for joint_index, traj in enumerate(trajectories):
            result[traj.joint_name] = _trajectory_to_samples(
                timestamps_s,
                sim_q[:, joint_index].tolist(),
                sim_qdot[:, joint_index].tolist(),
            )
        return result
