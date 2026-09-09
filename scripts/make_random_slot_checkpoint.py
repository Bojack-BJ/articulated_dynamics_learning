#!/usr/bin/env python3
"""Reset a slot checkpoint's learned parameters while preserving its schema/statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import _load_slot_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    model = _load_slot_model(checkpoint, torch, "cpu")
    for module in model.modules():
        if module is model:
            continue
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    if hasattr(model, "queries"):
        torch.nn.init.normal_(model.queries, mean=0.0, std=0.02)
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["scratch_initialization"] = True
    checkpoint["scratch_seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
