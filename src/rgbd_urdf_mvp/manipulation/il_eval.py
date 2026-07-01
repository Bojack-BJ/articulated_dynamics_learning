from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.serialization import save_json
from .ballistic import apply_dynamics_scale, condition_vector, load_model, resolve_joint, resolve_model_path, rollout_initial_qvel, rollout_metrics
from .il_policy import load_checkpoint, policy_input, _import_torch


@dataclass(slots=True)
class BallisticILEvalConfig:
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
    max_abs_initial_qvel: float = 8.0
    enable_contact: bool = False
    gravity_mode: str = "zero"


class BallisticILEvaluator:
    def evaluate(self, config: BallisticILEvalConfig) -> Path:
        torch = _import_torch()
        model_policy, policy_config, metadata = load_checkpoint(config.policy_path)
        rng = np.random.default_rng(int(config.seed))
        model_path = resolve_model_path(config.mjcf_path)
        rows: list[dict[str, float | bool]] = []
        for _ in range(max(1, int(config.num_episodes))):
            model = load_model(model_path, sim_dt=config.sim_dt, enable_contact=config.enable_contact, gravity_mode=config.gravity_mode)
            joint = resolve_joint(model, joint_name=config.joint_name, joint_id=config.joint_id)
            params = apply_dynamics_scale(
                model,
                joint,
                mass_scale=float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))),
                damping_scale=float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))),
                friction_scale=float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
            )
            span = max(1e-6, joint.upper - joint.lower)
            q0 = float(rng.uniform(joint.lower + 0.15 * span, joint.upper - 0.15 * span))
            target_q = float(np.clip(q0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.2 * span, 0.55 * span), joint.lower + 0.05 * span, joint.upper - 0.05 * span))
            obs = np.asarray([[q0, 0.0, target_q, float(config.release_duration_s)]], dtype=np.float32)
            cond = condition_vector(params, joint)[None, :]
            x = policy_input(obs, cond, policy_config.condition_mode)
            with torch.no_grad():
                pred = model_policy(torch.from_numpy(x)).detach().cpu().numpy()[0]
            qvel = float(np.clip(pred[0], -abs(float(config.max_abs_initial_qvel)), abs(float(config.max_abs_initial_qvel))))
            response = rollout_initial_qvel(model, joint, q0, qvel, float(config.release_duration_s), int(config.response_samples))
            metrics = rollout_metrics(response, target_q, q0, float(config.tolerance), float(config.qdot_tolerance))
            rows.append({"initial_qvel": qvel, **metrics})

        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "il_eval.json"
        save_json(
            {
                "source": "ballistic-il-evaluator",
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

