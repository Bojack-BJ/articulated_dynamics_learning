from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.serialization import save_json
from .ballistic import (
    ACTION_FEATURES,
    CONDITION_FEATURES,
    OBS_FEATURES,
    apply_dynamics_scale,
    condition_vector,
    load_model,
    resolve_joint,
    resolve_model_path,
    rollout_initial_qvel,
    rollout_metrics,
)


TaskName = Literal["impulse-to-target", "timing-gate", "energy-budget"]
ConditionSource = Literal["oracle", "estimated"]


@dataclass(slots=True)
class BallisticILDatasetConfig:
    mjcf_path: str | Path
    output_dir: str | Path
    task: TaskName = "impulse-to-target"
    num_episodes: int = 256
    release_duration_s: float = 1.0
    condition_source: ConditionSource = "oracle"
    joint_name: str | None = None
    joint_id: int | None = None
    sim_dt: float | None = None
    max_initial_qvel: float = 8.0
    num_teacher_candidates: int = 121
    response_samples: int = 64
    tolerance: float = 0.05
    qdot_tolerance: float = 0.25
    seed: int = 0
    enable_contact: bool = False
    gravity_mode: str = "zero"
    mass_scale_range: tuple[float, float] = (0.5, 2.0)
    damping_scale_range: tuple[float, float] = (0.25, 4.0)
    friction_scale_range: tuple[float, float] = (0.5, 3.0)


class BallisticILDatasetGenerator:
    def generate(self, config: BallisticILDatasetConfig) -> Path:
        if config.task not in {"impulse-to-target", "timing-gate", "energy-budget"}:
            raise ValueError(f"Unsupported task: {config.task!r}")
        rng = np.random.default_rng(int(config.seed))
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolve_model_path(config.mjcf_path)

        obs_rows: list[np.ndarray] = []
        cond_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        response_rows: list[np.ndarray] = []
        metric_rows: list[dict[str, float | bool]] = []
        raw_params: list[dict[str, float]] = []

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
            qvel, response, metrics = _teacher_search(
                model=model,
                joint=joint,
                q0=q0,
                target_q=target_q,
                duration_s=float(config.release_duration_s),
                max_initial_qvel=float(config.max_initial_qvel),
                num_candidates=max(3, int(config.num_teacher_candidates)),
                response_samples=max(2, int(config.response_samples)),
                tolerance=float(config.tolerance),
                qdot_tolerance=float(config.qdot_tolerance),
            )
            obs_rows.append(np.asarray([q0, 0.0, target_q, float(config.release_duration_s)], dtype=np.float32))
            cond_rows.append(condition_vector(params, joint))
            action_rows.append(np.asarray([qvel], dtype=np.float32))
            response_rows.append(response)
            metric_rows.append(metrics)
            raw_params.append(params)

        dataset_path = output_dir / "il_dataset.npz"
        np.savez_compressed(
            dataset_path,
            obs=np.stack(obs_rows, axis=0).astype(np.float32),
            condition=np.stack(cond_rows, axis=0).astype(np.float32),
            action=np.stack(action_rows, axis=0).astype(np.float32),
            free_response=np.stack(response_rows, axis=0).astype(np.float32),
        )
        manifest_path = output_dir / "il_dataset_manifest.json"
        success_rate = float(np.mean([1.0 if row["success"] else 0.0 for row in metric_rows])) if metric_rows else 0.0
        save_json(
            {
                "source": "ballistic-il-demo-generator",
                "dataset_path": str(dataset_path),
                "mjcf_path": str(model_path),
                "task": config.task,
                "condition_source": config.condition_source,
                "num_episodes": len(obs_rows),
                "obs_features": OBS_FEATURES,
                "condition_features": CONDITION_FEATURES,
                "action_features": ACTION_FEATURES,
                "release_duration_s": float(config.release_duration_s),
                "teacher": {
                    "type": "grid_search_initial_qvel",
                    "max_initial_qvel": float(config.max_initial_qvel),
                    "num_candidates": int(config.num_teacher_candidates),
                    "success_rate": success_rate,
                },
                "metrics": {
                    "success_rate": success_rate,
                    "final_error_mean": float(np.mean([float(row["final_error"]) for row in metric_rows])),
                    "settling_qdot_mean": float(np.mean([float(row["settling_qdot"]) for row in metric_rows])),
                },
                "raw_parameter_examples": raw_params[: min(5, len(raw_params))],
                "manifest_path": str(manifest_path),
            },
            manifest_path,
        )
        return dataset_path


def _teacher_search(
    model,
    joint,
    q0: float,
    target_q: float,
    duration_s: float,
    max_initial_qvel: float,
    num_candidates: int,
    response_samples: int,
    tolerance: float,
    qdot_tolerance: float,
) -> tuple[float, np.ndarray, dict[str, float | bool]]:
    direction = 1.0 if target_q >= q0 else -1.0
    candidates = np.linspace(0.0, direction * abs(max_initial_qvel), num_candidates, dtype=float)
    best_qvel = float(candidates[0])
    best_response = rollout_initial_qvel(model, joint, q0, best_qvel, duration_s, response_samples)
    best_metrics = rollout_metrics(best_response, target_q, q0, tolerance, qdot_tolerance)
    best_objective = _objective(best_metrics)
    for qvel in candidates[1:]:
        response = rollout_initial_qvel(model, joint, q0, float(qvel), duration_s, response_samples)
        metrics = rollout_metrics(response, target_q, q0, tolerance, qdot_tolerance)
        objective = _objective(metrics)
        if objective < best_objective:
            best_qvel = float(qvel)
            best_response = response
            best_metrics = metrics
            best_objective = objective
    return best_qvel, best_response, best_metrics


def _objective(metrics: dict[str, float | bool]) -> float:
    return (
        float(metrics["final_error"])
        + 0.1 * float(metrics["settling_qdot"])
        + 0.5 * float(metrics["overshoot"])
    )


def _uniform_log(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = max(1e-6, float(bounds[0])), max(1e-6, float(bounds[1]))
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

