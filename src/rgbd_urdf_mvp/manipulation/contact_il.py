from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from ..core.serialization import save_json
from .ballistic import (
    CONDITION_FEATURES,
    OBS_FEATURES,
    apply_dynamics_scale,
    condition_vector,
    dynamics_values,
    load_model,
    resolve_joint,
    resolve_model_path,
    rollout_metrics,
)
from .il_policy import load_checkpoint, policy_input, _import_torch


CONTACT_ACTION_FEATURES = [
    "contact_point_x",
    "contact_point_y",
    "contact_point_z",
    "push_dir_x",
    "push_dir_y",
    "push_dir_z",
    "force",
    "contact_duration_s",
]


@dataclass(slots=True)
class ContactILDatasetConfig:
    mjcf_path: str | Path
    output_dir: str | Path
    robot_model_path: str | Path | None = None
    robot_end_effector_body: str = "link6"
    task: Literal["impulse-to-target", "timing-gate", "energy-budget"] = "impulse-to-target"
    num_episodes: int = 256
    release_duration_s: float = 1.0
    contact_duration_s: float = 0.12
    condition_source: Literal["oracle", "estimated"] = "estimated"
    joint_name: str | None = None
    joint_id: int | None = None
    sim_dt: float | None = None
    max_force: float = 20.0
    num_teacher_candidates: int = 101
    response_samples: int = 64
    tolerance: float = 0.05
    qdot_tolerance: float = 0.25
    seed: int = 0
    enable_contact: bool = False
    gravity_mode: str = "zero"
    mass_scale_range: tuple[float, float] = (0.5, 2.0)
    damping_scale_range: tuple[float, float] = (0.25, 4.0)
    friction_scale_range: tuple[float, float] = (0.5, 3.0)


class ContactILDatasetGenerator:
    def generate(self, config: ContactILDatasetConfig) -> Path:
        rng = np.random.default_rng(int(config.seed))
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolve_model_path(config.mjcf_path)
        robot_model_path = str(Path(config.robot_model_path).expanduser().resolve()) if config.robot_model_path else None

        obs_rows: list[np.ndarray] = []
        cond_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        response_rows: list[np.ndarray] = []
        metric_rows: list[dict[str, float | bool]] = []

        for _ in range(max(1, int(config.num_episodes))):
            model = load_model(model_path, sim_dt=config.sim_dt, enable_contact=config.enable_contact, gravity_mode=config.gravity_mode)
            joint = resolve_joint(model, joint_name=config.joint_name, joint_id=config.joint_id)
            params = apply_dynamics_scale(
                model,
                joint,
                mass_scale=_uniform_log(rng, config.mass_scale_range),
                damping_scale=_uniform_log(rng, config.damping_scale_range),
                friction_scale=_uniform_log(rng, config.friction_scale_range),
            )
            span = max(1e-6, joint.upper - joint.lower)
            q0 = float(rng.uniform(joint.lower + 0.15 * span, joint.upper - 0.15 * span))
            delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.2 * span, 0.55 * span))
            target_q = float(np.clip(q0 + delta, joint.lower + 0.05 * span, joint.upper - 0.05 * span))
            force, action_geometry, response, metrics = _teacher_search_contact(
                model=model,
                joint=joint,
                q0=q0,
                target_q=target_q,
                duration_s=float(config.release_duration_s),
                contact_duration_s=float(config.contact_duration_s),
                max_force=float(config.max_force),
                num_candidates=max(3, int(config.num_teacher_candidates)),
                response_samples=max(2, int(config.response_samples)),
                tolerance=float(config.tolerance),
                qdot_tolerance=float(config.qdot_tolerance),
            )
            obs_rows.append(np.asarray([q0, 0.0, target_q, float(config.release_duration_s)], dtype=np.float32))
            cond_rows.append(condition_vector(params, joint))
            action_rows.append(np.asarray([*action_geometry, force, float(config.contact_duration_s)], dtype=np.float32))
            response_rows.append(response)
            metric_rows.append(metrics)

        dataset_path = output_dir / "contact_il_dataset.npz"
        np.savez_compressed(
            dataset_path,
            obs=np.stack(obs_rows, axis=0).astype(np.float32),
            condition=np.stack(cond_rows, axis=0).astype(np.float32),
            action=np.stack(action_rows, axis=0).astype(np.float32),
            free_response=np.stack(response_rows, axis=0).astype(np.float32),
        )
        success_rate = float(np.mean([1.0 if row["success"] else 0.0 for row in metric_rows]))
        save_json(
            {
                "source": "contact-impulse-il-demo-generator",
                "dataset_path": str(dataset_path),
                "mjcf_path": str(model_path),
                "robot_model_path": robot_model_path,
                "robot_end_effector_body": config.robot_end_effector_body,
                "task": config.task,
                "condition_source": config.condition_source,
                "num_episodes": len(obs_rows),
                "obs_features": OBS_FEATURES,
                "condition_features": CONDITION_FEATURES,
                "action_features": CONTACT_ACTION_FEATURES,
                "teacher": {
                    "type": "grid_search_contact_force",
                    "max_force": float(config.max_force),
                    "num_candidates": int(config.num_teacher_candidates),
                    "contact_duration_s": float(config.contact_duration_s),
                    "success_rate": success_rate,
                },
                "metrics": {
                    "success_rate": success_rate,
                    "final_error_mean": float(np.mean([float(row["final_error"]) for row in metric_rows])),
                    "settling_qdot_mean": float(np.mean([float(row["settling_qdot"]) for row in metric_rows])),
                    "overshoot_mean": float(np.mean([float(row["overshoot"]) for row in metric_rows])),
                },
            },
            output_dir / "contact_il_dataset_manifest.json",
        )
        return dataset_path


@dataclass(slots=True)
class ContactILEvalConfig:
    policy_path: str | Path
    mjcf_path: str | Path
    output_dir: str | Path
    num_episodes: int = 64
    release_duration_s: float = 1.0
    joint_name: str | None = None
    joint_id: int | None = None
    sim_dt: float | None = None
    response_samples: int = 64
    tolerance: float = 0.05
    qdot_tolerance: float = 0.25
    seed: int = 1
    max_force: float = 40.0
    max_contact_duration_s: float = 0.25
    enable_contact: bool = False
    gravity_mode: str = "zero"


class ContactILEvaluator:
    def evaluate(self, config: ContactILEvalConfig) -> Path:
        torch = _import_torch()
        policy, policy_config, metadata = load_checkpoint(config.policy_path)
        rng = np.random.default_rng(int(config.seed))
        model_path = resolve_model_path(config.mjcf_path)
        rows: list[dict[str, float | bool]] = []
        for _ in range(max(1, int(config.num_episodes))):
            model = load_model(model_path, sim_dt=config.sim_dt, enable_contact=config.enable_contact, gravity_mode=config.gravity_mode)
            joint = resolve_joint(model, joint_name=config.joint_name, joint_id=config.joint_id)
            params = apply_dynamics_scale(
                model,
                joint,
                mass_scale=_uniform_log(rng, (0.5, 2.0)),
                damping_scale=_uniform_log(rng, (0.25, 4.0)),
                friction_scale=_uniform_log(rng, (0.5, 3.0)),
            )
            span = max(1e-6, joint.upper - joint.lower)
            q0 = float(rng.uniform(joint.lower + 0.15 * span, joint.upper - 0.15 * span))
            target_q = float(np.clip(q0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.2 * span, 0.55 * span), joint.lower + 0.05 * span, joint.upper - 0.05 * span))
            obs = np.asarray([[q0, 0.0, target_q, float(config.release_duration_s)]], dtype=np.float32)
            cond = condition_vector(params, joint)[None, :]
            x = policy_input(obs, cond, policy_config.condition_mode)
            with torch.no_grad():
                action = policy(torch.from_numpy(x)).detach().cpu().numpy()[0]
            point = np.asarray(action[:3], dtype=float)
            direction = _normalize(np.asarray(action[3:6], dtype=float))
            force = float(np.clip(action[6], 0.0, abs(float(config.max_force))))
            contact_duration = float(np.clip(action[7], 0.0, float(config.max_contact_duration_s)))
            response = rollout_contact_force(
                model,
                joint,
                q0,
                force,
                (point, direction),
                contact_duration,
                float(config.release_duration_s),
                int(config.response_samples),
            )
            metrics = rollout_metrics(response, target_q, q0, float(config.tolerance), float(config.qdot_tolerance))
            rows.append(
                {
                    "force": force,
                    "contact_duration_s": contact_duration,
                    **metrics,
                }
            )

        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "contact_il_eval.json"
        save_json(
            {
                "source": "contact-il-evaluator",
                "policy_path": str(Path(config.policy_path).expanduser().resolve()),
                "mjcf_path": str(model_path),
                "policy_metadata": metadata,
                "num_episodes": len(rows),
                "success_rate": float(np.mean([1.0 if row["success"] else 0.0 for row in rows])),
                "final_error_mean": float(np.mean([float(row["final_error"]) for row in rows])),
                "settling_qdot_mean": float(np.mean([float(row["settling_qdot"]) for row in rows])),
                "overshoot_mean": float(np.mean([float(row["overshoot"]) for row in rows])),
                "episodes": rows,
            },
            output_path,
        )
        return output_path


def _teacher_search_contact(
    model,
    joint,
    q0: float,
    target_q: float,
    duration_s: float,
    contact_duration_s: float,
    max_force: float,
    num_candidates: int,
    response_samples: int,
    tolerance: float,
    qdot_tolerance: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float | bool]]:
    direction = 1.0 if target_q >= q0 else -1.0
    forces = np.linspace(0.0, abs(max_force), num_candidates, dtype=float)
    geometry = _contact_geometry(model, joint, q0, direction)
    best_force = float(forces[0])
    best_response = rollout_contact_force(model, joint, q0, best_force, geometry, contact_duration_s, duration_s, response_samples)
    best_metrics = rollout_metrics(best_response, target_q, q0, tolerance, qdot_tolerance)
    best_objective = _objective(best_metrics)
    for force in forces[1:]:
        response = rollout_contact_force(model, joint, q0, float(force), geometry, contact_duration_s, duration_s, response_samples)
        metrics = rollout_metrics(response, target_q, q0, tolerance, qdot_tolerance)
        objective = _objective(metrics)
        if objective < best_objective:
            best_force = float(force)
            best_response = response
            best_metrics = metrics
            best_objective = objective
    action_geometry = np.asarray([*geometry[0], *geometry[1]], dtype=np.float32)
    return best_force, action_geometry, best_response, best_metrics


def rollout_contact_force(
    model,
    joint,
    initial_q: float,
    force_magnitude: float,
    geometry: tuple[np.ndarray, np.ndarray],
    contact_duration_s: float,
    duration_s: float,
    sample_count: int,
) -> np.ndarray:
    point_world, direction_world = geometry
    data = mujoco.MjData(model)
    data.qpos[joint.qpos_adr] = float(np.clip(initial_q, joint.lower, joint.upper))
    data.qvel[joint.dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    sample_times = np.linspace(float(model.opt.timestep), max(float(duration_s), float(model.opt.timestep)), max(2, int(sample_count)))
    samples = np.zeros((len(sample_times), 3), dtype=np.float32)
    qfrc = np.zeros(model.nv, dtype=float)
    zero_torque = np.zeros(3, dtype=float)
    for index, target_time in enumerate(sample_times):
        while data.time + 1e-12 < float(target_time):
            data.qfrc_applied[:] = 0.0
            if data.time <= float(contact_duration_s) + 1e-12 and abs(force_magnitude) > 0.0:
                qfrc[:] = 0.0
                force = np.asarray(direction_world, dtype=float) * float(force_magnitude)
                mujoco.mj_applyFT(model, data, force, zero_torque, point_world, joint.body_id, qfrc)
                data.qfrc_applied[:] = qfrc
            mujoco.mj_step(model, data)
        samples[index] = [float(data.time), float(data.qpos[joint.qpos_adr]), float(data.qvel[joint.dof_adr])]
    return samples


def _contact_geometry(model, joint, q0: float, direction: float) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[joint.qpos_adr] = float(np.clip(q0, joint.lower, joint.upper))
    mujoco.mj_forward(model, data)
    anchor = np.asarray(data.xanchor[joint.joint_id], dtype=float)
    axis = _normalize(np.asarray(data.xaxis[joint.joint_id], dtype=float))
    point, radius = _best_contact_point(model, data, joint.body_id, anchor, axis)
    if point is None or np.linalg.norm(radius) < 1e-6:
        radius = _orthogonal(axis) * 0.05
        point = anchor + radius
    push_dir = _normalize(np.cross(axis, radius)) * float(direction)
    return point.astype(float), push_dir.astype(float)


def _best_contact_point(model, data, body_id: int, anchor: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    best_point: np.ndarray | None = None
    best_radius = np.zeros(3, dtype=float)
    best_norm = -1.0
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        for point in _geom_contact_candidates(model, data, geom_id):
            radius = point - anchor
            radius = radius - axis * float(np.dot(radius, axis))
            radius_norm = float(np.linalg.norm(radius))
            if radius_norm > best_norm:
                best_point = point
                best_radius = radius
                best_norm = radius_norm
    if best_point is not None:
        return best_point, best_radius
    point = np.asarray(data.xipos[body_id], dtype=float)
    radius = point - anchor
    radius = radius - axis * float(np.dot(radius, axis))
    return point, radius


def _geom_contact_candidates(model, data, geom_id: int) -> list[np.ndarray]:
    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        points = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = mat @ np.asarray([sx * size[0], sy * size[1], sz * size[2]], dtype=float)
                    points.append(center + offset)
        points.append(center)
        return points
    if geom_type in {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)}:
        mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        return [
            center,
            center + mat[:, 0] * float(size[0]),
            center - mat[:, 0] * float(size[0]),
            center + mat[:, 1] * float(size[0]),
            center - mat[:, 1] * float(size[0]),
        ]
    return [center]


def _objective(metrics: dict[str, float | bool]) -> float:
    return float(metrics["final_error"]) + 0.1 * float(metrics["settling_qdot"]) + 0.5 * float(metrics["overshoot"])


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def _orthogonal(axis: np.ndarray) -> np.ndarray:
    candidate = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, axis))) > 0.9:
        candidate = np.asarray([0.0, 1.0, 0.0], dtype=float)
    return _normalize(candidate - axis * float(np.dot(candidate, axis)))


def _uniform_log(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = max(1e-6, float(bounds[0])), max(1e-6, float(bounds[1]))
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
