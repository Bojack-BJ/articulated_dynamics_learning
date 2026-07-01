from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.serialization import save_json
from .il_policy import BCPolicyConfig, ConditionMode, build_policy, policy_input, save_checkpoint, _import_torch


@dataclass(slots=True)
class BallisticILTrainConfig:
    dataset_path: str | Path
    output_dir: str | Path
    condition_mode: ConditionMode = "oracle"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    hidden_layers: int = 2
    val_fraction: float = 0.2
    seed: int = 0
    noisy_condition_std: float = 0.05


class BallisticILTrainer:
    def train(self, config: BallisticILTrainConfig) -> Path:
        torch = _import_torch()
        data = np.load(Path(config.dataset_path))
        obs = data["obs"].astype(np.float32)
        condition = data["condition"].astype(np.float32)
        action = data["action"].astype(np.float32)
        x = policy_input(obs, condition, config.condition_mode, noise_std=float(config.noisy_condition_std))
        y = action
        rng = np.random.default_rng(int(config.seed))
        indices = rng.permutation(len(x))
        val_count = min(len(x) - 1, max(1, int(round(len(x) * float(config.val_fraction))))) if len(x) > 1 else 0
        val_idx = indices[:val_count]
        train_idx = indices[val_count:] if val_count else indices

        policy_config = BCPolicyConfig(
            obs_dim=int(obs.shape[1]),
            condition_dim=int(condition.shape[1]),
            action_dim=int(action.shape[1]),
            condition_mode=config.condition_mode,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        )
        model = build_policy(policy_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
        loss_fn = torch.nn.MSELoss()
        history: list[dict[str, float]] = []
        batch_size = max(1, int(config.batch_size))
        for epoch in range(max(1, int(config.epochs))):
            model.train()
            shuffled = rng.permutation(train_idx)
            train_losses: list[float] = []
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                xb = torch.from_numpy(x[batch])
                yb = torch.from_numpy(y[batch])
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                if len(val_idx):
                    val_loss = float(loss_fn(model(torch.from_numpy(x[val_idx])), torch.from_numpy(y[val_idx])).detach().cpu())
            history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "policy.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            policy_config,
            {
                "dataset_path": str(Path(config.dataset_path).expanduser().resolve()),
                "condition_mode": config.condition_mode,
                "train_count": int(len(train_idx)),
                "val_count": int(len(val_idx)),
            },
        )
        save_json({"source": "ballistic-bc-trainer", "policy_path": str(checkpoint_path), "history": history}, output_dir / "training_metrics.json")
        return checkpoint_path

