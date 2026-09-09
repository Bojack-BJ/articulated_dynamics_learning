#!/usr/bin/env python3
"""Export per-joint GT and predicted axes for Relation Head diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    JOINT_TYPES,
    _build_relation_model,
    _load_relation_samples,
    _load_slot_model,
    _relation_targets,
    _slot_motion_summary,
)
from rgbd_urdf_mvp.perception.motion_part_slots import _require_torch, _resolve_device
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("relation_model", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch = _require_torch()
    device = _resolve_device(torch, args.device)
    slot_checkpoint = torch.load(args.slot_model.resolve(), map_location="cpu", weights_only=True)
    relation_checkpoint = torch.load(args.relation_model.resolve(), map_location="cpu", weights_only=True)
    slot_model = _load_slot_model(slot_checkpoint, torch, device)
    if relation_checkpoint.get("slot_state_dict") is not None:
        slot_model.load_state_dict(relation_checkpoint["slot_state_dict"])
    relation_model = _build_relation_model(
        torch,
        slot_dim=int(relation_checkpoint["slot_dim"]),
        hidden_dim=int(relation_checkpoint["hidden_dim"]),
        motion_summary_dim=int(relation_checkpoint.get("motion_summary_dim", 0)),
    ).to(device)
    relation_model.load_state_dict(relation_checkpoint["state_dict"])
    relation_model.eval()
    slot_model.eval()
    samples = _load_relation_samples(
        load_pairwise_manifest(args.manifest.resolve()), "test", slot_checkpoint
    )
    mean = slot_checkpoint["feature_mean"].to(device)
    std = slot_checkpoint["feature_std"].to(device)
    rows = []
    with torch.no_grad():
        for sample in samples:
            features = torch.from_numpy(sample["features"]).to(device)
            logits, _, slots = slot_model((features - mean) / std, return_slots=True)
            probabilities = torch.softmax(logits, dim=-1)
            motion_summary = _slot_motion_summary(
                features, probabilities, int(sample["embedding_dim"]), torch
            )
            prediction = relation_model(slots, motion_summary)
            target = _relation_targets(sample, logits, int(slots.shape[0]), torch, device)
            for relation in target["relations"]:
                parent, child = int(relation["parent_slot"]), int(relation["child_slot"])
                predicted_axis = prediction["axes"][parent, child]
                gt_axis = target["axis"][parent, child]
                dot = float(torch.sum(predicted_axis * gt_axis).abs().clamp(0.0, 1.0).cpu())
                predicted_type_index = int(prediction["type_logits"][parent, child].argmax().cpu())
                rows.append({
                    "object_id": str(sample["object_id"]),
                    "joint_name": str(relation["joint_name"]),
                    "parent_label": int(relation["parent_label"]),
                    "child_label": int(relation["child_label"]),
                    "gt_joint_type": str(relation["joint_type"]),
                    "predicted_joint_type": JOINT_TYPES[predicted_type_index],
                    "type_correct": JOINT_TYPES[predicted_type_index] == str(relation["joint_type"]),
                    "gt_axis": [float(value) for value in gt_axis.cpu()],
                    "predicted_axis": [float(value) for value in predicted_axis.cpu()],
                    "axis_error_deg": math.degrees(math.acos(dot)),
                    "edge_probability": float(torch.sigmoid(prediction["edge_logits"][parent, child]).cpu()),
                })
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"joints": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output), "joint_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
