from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class MuJoCoPushPlanConfig:
    mjcf_path: str | Path
    target_q: float
    output_dir: str | Path | None = None
    dynamics_identification_path: str | Path | None = None
    joint_name: str | None = None
    joint_id: int | None = None
    mode: str = "initial_velocity"
    initial_q: float | None = None
    duration_s: float = 1.5
    sim_dt: float | None = None
    num_candidates: int = 81
    max_initial_qvel: float = 8.0
    max_pulse_force: float = 5.0
    pulse_duration_s: float = 0.15
    tolerance: float = 0.03
    enable_contact: bool = False
    gravity_mode: str = "model"


@dataclass(slots=True)
class _JointSpec:
    joint_id: int
    joint_name: str
    joint_type: str
    body_id: int
    body_name: str
    qpos_adr: int
    dof_adr: int
    lower: float
    upper: float


@dataclass(slots=True)
class _RolloutResult:
    command_value: float
    final_q: float
    final_qdot: float
    objective: float
    reached: bool
    samples: list[dict[str, float]]


class MuJoCoPushPlanner:
    """Plan a one-shot joint-space push for an articulated object in MuJoCo."""

    def run(self, config: MuJoCoPushPlanConfig) -> Path:
        mode = str(config.mode).strip().lower()
        if mode not in {"initial_velocity", "pulse_force"}:
            raise ValueError("mode must be 'initial_velocity' or 'pulse_force'")

        mjcf_path = self._resolve_model_path(config)
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        if config.sim_dt is not None:
            model.opt.timestep = float(config.sim_dt)
        if not bool(config.enable_contact):
            model.geom_contype[:] = 0
            model.geom_conaffinity[:] = 0
        self._apply_gravity_mode(model, str(config.gravity_mode))

        joint = self._resolve_joint(model, config)
        initial_q = self._initial_q(model, joint, config)
        target_q = self._clamp_to_joint_range(float(config.target_q), joint)
        candidates = self._candidate_values(mode, initial_q, target_q, config)
        rollouts = [
            self._rollout(
                model=model,
                joint=joint,
                mode=mode,
                command_value=value,
                initial_q=initial_q,
                target_q=target_q,
                duration_s=max(float(config.duration_s), model.opt.timestep),
                pulse_duration_s=max(0.0, float(config.pulse_duration_s)),
                tolerance=max(0.0, float(config.tolerance)),
            )
            for value in candidates
        ]
        best = min(rollouts, key=lambda item: item.objective)

        output_dir = (
            Path(config.output_dir).expanduser().resolve()
            if config.output_dir is not None
            else mjcf_path.parent / "downstream_manipulation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "manipulation_plan.json"

        dynamics_artifact_path = self._resolve_dynamics_artifact_path(config)
        dynamics_artifact = self._load_optional_json(dynamics_artifact_path)
        save_json(
            {
                "source": "mujoco-downstream-push-planner",
                "mjcf_path": str(mjcf_path),
                "dynamics_identification_path": (
                    str(dynamics_artifact_path)
                    if dynamics_artifact_path is not None
                    else None
                ),
                "task": {
                    "type": "single_joint_reach",
                    "joint_name": joint.joint_name,
                    "joint_type": joint.joint_type,
                    "body_name": joint.body_name,
                    "initial_q": float(initial_q),
                    "target_q": float(target_q),
                    "tolerance": float(config.tolerance),
                    "duration_s": float(config.duration_s),
                },
                "command": self._command_payload(mode, best, config),
                "policy_conditioning": self._policy_conditioning(
                    model=model,
                    joint=joint,
                    dynamics_artifact=dynamics_artifact,
                ),
                "simulation_options": {
                    "mode": mode,
                    "sim_dt": float(model.opt.timestep),
                    "enable_contact": bool(config.enable_contact),
                    "gravity_mode": str(config.gravity_mode),
                    "gravity": [float(value) for value in model.opt.gravity.tolist()],
                    "candidate_count": len(candidates),
                },
                "result": {
                    "success": bool(best.reached),
                    "final_q": float(best.final_q),
                    "final_qdot": float(best.final_qdot),
                    "abs_error": abs(float(best.final_q) - float(target_q)),
                    "objective": float(best.objective),
                },
                "rollout": {
                    "best": best.samples,
                    "candidates": [
                        {
                            "command_value": float(item.command_value),
                            "final_q": float(item.final_q),
                            "final_qdot": float(item.final_qdot),
                            "objective": float(item.objective),
                            "reached": bool(item.reached),
                        }
                        for item in rollouts
                    ],
                },
                "limitations": [
                    "This is a joint-space MuJoCo push planner, not a full robot-arm contact planner.",
                    "The command can supervise or condition a visual policy; executing it on hardware still needs a controller that maps end-effector motion to the articulated part.",
                    "Mass and damping are most useful when the downstream policy sees enough variation across objects or domains.",
                ],
            },
            artifact_path,
        )
        return artifact_path

    def _resolve_model_path(self, config: MuJoCoPushPlanConfig) -> Path:
        mjcf_path = Path(config.mjcf_path).expanduser()
        if mjcf_path.suffix.lower() == ".json":
            artifact = load_json(mjcf_path)
            optimized = artifact.get("mjcf_optimized_path")
            if not optimized:
                raise ValueError(f"{mjcf_path} is JSON but does not contain mjcf_optimized_path")
            mjcf_path = Path(str(optimized)).expanduser()
        return mjcf_path.resolve()

    def _resolve_dynamics_artifact_path(self, config: MuJoCoPushPlanConfig) -> Path | None:
        if config.dynamics_identification_path is not None:
            return Path(config.dynamics_identification_path).expanduser().resolve()
        candidate = Path(config.mjcf_path).expanduser()
        if candidate.suffix.lower() == ".json":
            return candidate.resolve()
        return None

    def _resolve_joint(self, model: mujoco.MjModel, config: MuJoCoPushPlanConfig) -> _JointSpec:
        allowed = {
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        }
        if config.joint_name is not None and config.joint_id is not None:
            raise ValueError("Provide only one of joint_name or joint_id")
        if config.joint_id is not None:
            joint_id = int(config.joint_id)
            if joint_id < 0 or joint_id >= model.njnt:
                raise ValueError(f"joint_id out of range: {joint_id}")
        elif config.joint_name is not None:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(config.joint_name)))
            if joint_id < 0:
                raise ValueError(f"Unknown joint_name: {config.joint_name}")
        else:
            candidates = [joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) in allowed]
            if not candidates:
                raise ValueError("MJCF has no hinge or slide joint to plan for")
            joint_id = candidates[0]

        joint_type_code = int(model.jnt_type[joint_id])
        if joint_type_code not in allowed:
            raise ValueError("plan-manipulation supports hinge and slide joints only")
        lower, upper = self._joint_limits(model, joint_id, allowed[joint_type_code])
        body_id = int(model.jnt_bodyid[joint_id])
        return _JointSpec(
            joint_id=joint_id,
            joint_name=self._name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, f"joint_{joint_id}"),
            joint_type=allowed[joint_type_code],
            body_id=body_id,
            body_name=self._name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
            qpos_adr=int(model.jnt_qposadr[joint_id]),
            dof_adr=int(model.jnt_dofadr[joint_id]),
            lower=lower,
            upper=upper,
        )

    def _joint_limits(self, model: mujoco.MjModel, joint_id: int, joint_type: str) -> tuple[float, float]:
        if bool(model.jnt_limited[joint_id]):
            return float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])
        if joint_type == "hinge":
            return -math.pi, math.pi
        return -1.0, 1.0

    def _initial_q(self, model: mujoco.MjModel, joint: _JointSpec, config: MuJoCoPushPlanConfig) -> float:
        if config.initial_q is not None:
            return self._clamp_to_joint_range(float(config.initial_q), joint)
        return self._clamp_to_joint_range(float(model.qpos0[joint.qpos_adr]), joint)

    def _clamp_to_joint_range(self, value: float, joint: _JointSpec) -> float:
        return max(float(joint.lower), min(float(joint.upper), float(value)))

    def _candidate_values(
        self,
        mode: str,
        initial_q: float,
        target_q: float,
        config: MuJoCoPushPlanConfig,
    ) -> np.ndarray:
        count = max(3, int(config.num_candidates))
        direction = 1.0 if target_q >= initial_q else -1.0
        if mode == "initial_velocity":
            vmax = abs(float(config.max_initial_qvel))
            return np.linspace(0.0, direction * vmax, count, dtype=float)
        fmax = abs(float(config.max_pulse_force))
        return np.linspace(0.0, direction * fmax, count, dtype=float)

    def _rollout(
        self,
        model: mujoco.MjModel,
        joint: _JointSpec,
        mode: str,
        command_value: float,
        initial_q: float,
        target_q: float,
        duration_s: float,
        pulse_duration_s: float,
        tolerance: float,
    ) -> _RolloutResult:
        data = mujoco.MjData(model)
        data.qpos[joint.qpos_adr] = float(initial_q)
        if mode == "initial_velocity":
            data.qvel[joint.dof_adr] = float(command_value)
        mujoco.mj_forward(model, data)

        samples: list[dict[str, float]] = []
        best_abs_error = abs(float(data.qpos[joint.qpos_adr]) - target_q)
        max_overshoot = 0.0
        direction = 1.0 if target_q >= initial_q else -1.0
        while data.time < duration_s - 1e-12:
            data.qfrc_applied[:] = 0.0
            if mode == "pulse_force" and data.time <= pulse_duration_s + 1e-12:
                data.qfrc_applied[joint.dof_adr] = float(command_value)
            mujoco.mj_step(model, data)
            q = float(data.qpos[joint.qpos_adr])
            qdot = float(data.qvel[joint.dof_adr])
            error = abs(q - target_q)
            best_abs_error = min(best_abs_error, error)
            signed_progress = direction * (q - target_q)
            if signed_progress > 0.0:
                max_overshoot = max(max_overshoot, signed_progress)
            samples.append({"timestamp_s": float(data.time), "q": q, "qdot": qdot})

        final_q = float(data.qpos[joint.qpos_adr])
        final_qdot = float(data.qvel[joint.dof_adr])
        final_error = abs(final_q - target_q)
        objective = final_error + 0.25 * best_abs_error + 0.05 * abs(final_qdot) + 0.5 * max_overshoot
        return _RolloutResult(
            command_value=float(command_value),
            final_q=final_q,
            final_qdot=final_qdot,
            objective=float(objective),
            reached=final_error <= tolerance or best_abs_error <= tolerance,
            samples=samples,
        )

    def _command_payload(
        self,
        mode: str,
        best: _RolloutResult,
        config: MuJoCoPushPlanConfig,
    ) -> dict[str, Any]:
        if mode == "initial_velocity":
            return {
                "type": "initial_joint_velocity",
                "initial_qvel": float(best.command_value),
                "controller_hint": "Use a short end-effector velocity push along the part motion direction, then release.",
            }
        return {
            "type": "generalized_force_pulse",
            "force": float(best.command_value),
            "duration_s": float(config.pulse_duration_s),
            "controller_hint": "Approximate this as a short contact impulse on the moving part.",
        }

    def _policy_conditioning(
        self,
        model: mujoco.MjModel,
        joint: _JointSpec,
        dynamics_artifact: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body_mass = float(model.body_mass[joint.body_id])
        effective_inertia = float(model.dof_M0[joint.dof_adr]) if model.dof_M0.size else 0.0
        damping = float(model.dof_damping[joint.dof_adr])
        frictionloss = float(model.dof_frictionloss[joint.dof_adr])
        armature = float(model.dof_armature[joint.dof_adr])
        lower = float(joint.lower)
        upper = float(joint.upper)
        features = [
            "body_mass",
            "effective_joint_inertia",
            "joint_damping",
            "joint_frictionloss",
            "joint_armature",
            "joint_range_lower",
            "joint_range_upper",
        ]
        values = [body_mass, effective_inertia, damping, frictionloss, armature, lower, upper]
        payload: dict[str, Any] = {
            "intended_use": "Concatenate these dynamics values with visual/state features, or use them as FiLM/adaptor conditioning for a visual policy.",
            "joint_name": joint.joint_name,
            "features": features,
            "values": values,
            "by_name": dict(zip(features, values, strict=True)),
        }
        if dynamics_artifact is not None:
            payload["source_identified_parameters"] = dynamics_artifact.get("identified_parameters", {})
            payload["source_fit_metrics"] = dynamics_artifact.get("fit_metrics", {})
        return payload

    def _load_optional_json(self, path: str | Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        return load_json(path)

    def _apply_gravity_mode(self, model: mujoco.MjModel, gravity_mode: str) -> None:
        mode = gravity_mode.strip().lower()
        if mode == "model":
            return
        if mode == "zero":
            model.opt.gravity[:] = 0.0
            return
        raise ValueError(f"Unsupported gravity_mode: {gravity_mode!r}")

    def _name(self, model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int, fallback: str) -> str:
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        return str(name) if name else fallback
