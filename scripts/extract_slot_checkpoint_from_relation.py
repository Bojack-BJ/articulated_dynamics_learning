#!/usr/bin/env python3
"""Materialize the slot state embedded in a relation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_slot_checkpoint", type=Path)
    parser.add_argument("relation_checkpoint", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    args = parser.parse_args()

    import torch

    slot = torch.load(args.base_slot_checkpoint, map_location="cpu", weights_only=True)
    relation = torch.load(args.relation_checkpoint, map_location="cpu", weights_only=True)
    state = relation.get("slot_state_dict")
    if state is None:
        raise ValueError("Relation checkpoint does not contain a fine-tuned slot state")
    slot["state_dict"] = state
    slot["derived_from_relation_checkpoint"] = str(args.relation_checkpoint.resolve())
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slot, args.output_checkpoint)
    print(args.output_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
