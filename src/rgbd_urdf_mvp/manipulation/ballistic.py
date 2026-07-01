from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.serialization import load_json


CONDITION_FEATURES = [
    "log_effective_joint_inertia",
    "log_joint_damping",
    "log_joint_frictionloss",
    "damping_over_inertia",
    "frictionloss_over_inertia",
    "joint_range_lower",
    "joint_range_upper",
]

OBS_FEATURES = ["q0", "qdot0", "target_q", "release_duration_s"]
ACTION_FEATURES = ["initial_qvel"]


@dataclass(slots=True)
class BallisticJointSpec:
    joint_id: int
    joint_name: str
    joint_type: str
    body_id: int
    body_name: str
    qpos_adr: int
    dof_adr: int
    lower: float
    upper: float


def resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path).expanduser()
    if model_path.suffix.lower() == ".json":
        artifact = load_json(model_path)
        optimized = artifact.get("mjcf_optimized_path")
        if not optimized:
            raise ValueError(f"{model_path} is JSON but does not contain mjcf_optimized_path")
        model_path = Path(str(optimized)).expanduser()
    return model_path.resolve()


def load_model(path: str | Path, sim_dt: float | None = None, enable_contact: bool = False, gravity_mode: str = "zero") -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(path)))
    if sim_dt is not None:
        model.opt.timestep = float(sim_dt)
    if not enable_contact:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    if gravity_mode == "zero":
        model.opt.gravity[:] = 0.0
    elif gravity_mode != "model":
        raise ValueError(f"Unsupported gravity_mode: {gravity_mode!r}")
    return model


def resolve_joint(model: mujoco.MjModel, joint_name: str | None = None, joint_id: int | None = None) -> BallisticJointSpec:
    allowed = {
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    }
    if joint_name is not None and joint_id is not None:
        raise ValueError("Provide only one of joint_name or joint_id")
    if joint_id is not None:
        resolved_id = int(joint_id)
        if resolved_id < 0 or resolved_id >= model.njnt:
            raise ValueError(f"joint_id out of range: {resolved_id}")
    elif joint_name is not None:
        resolved_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)))
        if resolved_id < 0:
            raise ValueError(f"Unknown joint_name: {joint_name}")
    else:
        candidates = [jid for jid in range(model.njnt) if int(model.jnt_type[jid]) in allowed]
        if not candidates:
            raise ValueError("MJCF has no hinge or slide joint")
        resolved_id = candidates[0]

    joint_type_code = int(model.jnt_type[resolved_id])
    if joint_type_code not in allowed:
        raise ValueError("Ballistic IL supports hinge and slide joints only")
    if bool(model.jnt_limited[resolved_id]):
        lower = float(model.jnt_range[resolved_id][0])
        upper = float(model.jnt_range[resolved_id][1])
    elif allowed[joint_type_code] == "hinge":
        lower, upper = -math.pi, math.pi
    else:
        lower, upper = -1.0, 1.0
    body_id = int(model.jnt_bodyid[resolved_id])
    return BallisticJointSpec(
        joint_id=resolved_id,
        joint_name=_name(model, mujoco.mjtObj.mjOBJ_JOINT, resolved_id, f"joint_{resolved_id}"),
        joint_type=allowed[joint_type_code],
        body_id=body_id,
        body_name=_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
        qpos_adr=int(model.jnt_qposadr[resolved_id]),
        dof_adr=int(model.jnt_dofadr[resolved_id]),
        lower=lower,
        upper=upper,
    )


def apply_dynamics_scale(
    model: mujoco.MjModel,
    joint: BallisticJointSpec,
    mass_scale: float,
    damping_scale: float,
    friction_scale: float,
) -> dict[str, float]:
    body = joint.body_id
    dof = joint.dof_adr
    model.body_mass[body] = max(1e-6, float(model.body_mass[body]) * float(mass_scale))
    model.body_inertia[body] = np.maximum(1e-9, model.body_inertia[body] * float(mass_scale))
    model.dof_damping[dof] = max(0.0, float(model.dof_damping[dof]) * float(damping_scale))
    model.dof_frictionloss[dof] = max(0.0, float(model.dof_frictionloss[dof]) * float(friction_scale))
    try:
        mujoco.mj_setConst(model, mujoco.MjData(model))
    except Exception:
        pass
    return dynamics_values(model, joint)


def dynamics_values(model: mujoco.MjModel, joint: BallisticJointSpec) -> dict[str, float]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inertia = float(model.dof_M0[joint.dof_adr]) if model.dof_M0.size else float(model.body_mass[joint.body_id])
    inertia = max(1e-9, inertia)
    damping = max(0.0, float(model.dof_damping[joint.dof_adr]))
    friction = max(0.0, float(model.dof_frictionloss[joint.dof_adr]))
    return {
        "body_mass": float(model.body_mass[joint.body_id]),
        "effective_joint_inertia": inertia,
        "joint_damping": damping,
        "joint_frictionloss": friction,
        "joint_range_lower": float(joint.lower),
        "joint_range_upper": float(joint.upper),
    }


def condition_vector(values: dict[str, float], joint: BallisticJointSpec) -> np.ndarray:
    inertia = max(1e-9, float(values["effective_joint_inertia"]))
    damping = max(0.0, float(values["joint_damping"]))
    friction = max(0.0, float(values["joint_frictionloss"]))
    return np.asarray(
        [
            math.log(inertia),
            math.log(damping + 1e-6),
            math.log(friction + 1e-6),
            damping / inertia,
            friction / inertia,
            float(joint.lower),
            float(joint.upper),
        ],
        dtype=np.float32,
    )


def rollout_initial_qvel(
    model: mujoco.MjModel,
    joint: BallisticJointSpec,
    initial_q: float,
    initial_qvel: float,
    duration_s: float,
    sample_count: int,
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[joint.qpos_adr] = float(np.clip(initial_q, joint.lower, joint.upper))
    data.qvel[joint.dof_adr] = float(initial_qvel)
    mujoco.mj_forward(model, data)
    duration = max(float(duration_s), float(model.opt.timestep))
    sample_count = max(2, int(sample_count))
    sample_times = np.linspace(float(model.opt.timestep), duration, sample_count)
    samples = np.zeros((sample_count, 3), dtype=np.float32)
    for index, target_time in enumerate(sample_times):
        while data.time + 1e-12 < float(target_time):
            data.qfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
        samples[index] = [float(data.time), float(data.qpos[joint.qpos_adr]), float(data.qvel[joint.dof_adr])]
    return samples


def rollout_metrics(response: np.ndarray, target_q: float, initial_q: float, tolerance: float, qdot_tolerance: float) -> dict[str, float | bool]:
    final_q = float(response[-1, 1])
    final_qdot = float(response[-1, 2])
    final_error = abs(final_q - float(target_q))
    direction = 1.0 if target_q >= initial_q else -1.0
    overshoot = float(max(0.0, np.max(direction * (response[:, 1] - float(target_q)))))
    return {
        "success": bool(final_error <= tolerance and abs(final_qdot) <= qdot_tolerance),
        "final_q": final_q,
        "final_qdot": final_qdot,
        "final_error": final_error,
        "overshoot": overshoot,
        "settling_qdot": abs(final_qdot),
    }


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int, fallback: str) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return str(name) if name else fallback

