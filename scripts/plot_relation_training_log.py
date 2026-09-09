#!/usr/bin/env python3
"""Recover relation-head training curves from JSON events in a text log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TRAIN_KEYS = (
    "train_total",
    "train_edge",
    "train_type",
    "train_axis",
    "train_axis_line",
    "train_joint_replay",
    "train_slot_assignment",
    "train_slot_existence",
    "train_equivariance",
    "train_equivariance_axis_line",
    "train_equivariance_edge",
    "train_equivariance_type",
    "train_slot_consistency",
)
VAL_KEYS = (
    "val_loss",
    "val_axis_error_deg",
    "val_axis_error_median_deg",
    "val_axis_error_p90_deg",
    "val_axis_line_error_normalized",
    "val_joint_type_accuracy",
    "val_edge_f1",
)


def load_epochs(path: Path) -> list[dict]:
    epochs = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "relation_training_epoch":
            epochs.append(event)
    if not epochs:
        raise ValueError(f"No relation_training_epoch events found in {path}")
    return epochs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    epochs = load_epochs(args.log)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    columns = ("epoch", "epoch_time_s", *TRAIN_KEYS, *VAL_KEYS)
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in epochs)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = [row["epoch"] for row in epochs]
    active_train_keys = [
        key for key in TRAIN_KEYS
        if any(abs(float(row.get(key, 0.0) or 0.0)) > 1e-12 for row in epochs)
    ]
    for key in active_train_keys:
        axes[0].plot(x, [row.get(key) for row in epochs], marker="o", label=key.removeprefix("train_"))
    axes[0].set_title("Training losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    for key in ("val_loss", "val_axis_line_error_normalized"):
        axes[1].plot(x, [row.get(key) for row in epochs], marker="o", label=key)
    axes[1].set_title("Validation losses")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    for key in ("val_axis_error_deg", "val_axis_error_median_deg", "val_axis_error_p90_deg"):
        axes[2].plot(x, [row.get(key) for row in epochs], marker="o", label=key.removeprefix("val_axis_error_"))
    axes[2].set_title("Validation axis error (deg)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), dpi=180)
    print(json.dumps({"csv": str(prefix.with_suffix('.csv')), "png": str(prefix.with_suffix('.png'))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
