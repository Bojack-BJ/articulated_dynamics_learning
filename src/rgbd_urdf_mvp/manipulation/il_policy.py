from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


ConditionMode = Literal["none", "oracle", "estimated", "noisy-estimated"]


@dataclass(slots=True)
class BCPolicyConfig:
    obs_dim: int
    condition_dim: int
    action_dim: int
    condition_mode: ConditionMode = "oracle"
    hidden_dim: int = 128
    hidden_layers: int = 2


def build_policy(config: BCPolicyConfig):
    torch = _import_torch()
    input_dim = int(config.obs_dim) if config.condition_mode == "none" else int(config.obs_dim) + int(config.condition_dim)
    layers: list[Any] = []
    current = input_dim
    for _ in range(max(1, int(config.hidden_layers))):
        layers.append(torch.nn.Linear(current, int(config.hidden_dim)))
        layers.append(torch.nn.ReLU())
        current = int(config.hidden_dim)
    layers.append(torch.nn.Linear(current, int(config.action_dim)))
    return torch.nn.Sequential(*layers)


def policy_input(obs: np.ndarray, condition: np.ndarray, mode: ConditionMode, noise_std: float = 0.05) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    condition = np.asarray(condition, dtype=np.float32)
    if mode == "none":
        return obs
    if mode == "noisy-estimated":
        rng = np.random.default_rng(0)
        condition = condition + rng.normal(0.0, float(noise_std), size=condition.shape).astype(np.float32)
    return np.concatenate([obs, condition], axis=-1).astype(np.float32)


def save_checkpoint(path: str | Path, model: Any, config: BCPolicyConfig, metadata: dict[str, Any]) -> None:
    torch = _import_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "obs_dim": config.obs_dim,
                "condition_dim": config.condition_dim,
                "action_dim": config.action_dim,
                "condition_mode": config.condition_mode,
                "hidden_dim": config.hidden_dim,
                "hidden_layers": config.hidden_layers,
            },
            "metadata": metadata,
        },
        output,
    )


def load_checkpoint(path: str | Path):
    torch = _import_torch()
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = BCPolicyConfig(**checkpoint["config"])
    model = build_policy(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, checkpoint.get("metadata", {})


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Imitation learning requires torch. Install with: pip install -e '.[imitation]'") from exc
    return torch

