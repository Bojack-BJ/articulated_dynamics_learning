#!/usr/bin/env python3
"""Measure relation-output dependence on slot ordering and decoded slot features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    _apply_slot_geometry_representation,
    _apply_slot_input_mode,
    _build_relation_model,
    _load_relation_samples,
    _load_slot_model,
    _relation_slot_probabilities,
    _slot_motion_summary,
    _trajectory_tensors,
)
from rgbd_urdf_mvp.perception.motion_part_slots import _require_numpy, _require_torch, _resolve_device
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def _model(checkpoint, torch, device):
    model = _build_relation_model(
        torch,
        slot_dim=int(checkpoint["slot_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        motion_summary_dim=int(checkpoint.get("motion_summary_dim", 0)),
        axis_geometry_branch=bool(checkpoint.get("axis_geometry_branch", False)),
        axis_head_type=str(checkpoint.get("axis_head_type", "direct")),
        vector_pivot_parameterization=str(checkpoint.get("vector_pivot_parameterization", "legacy_center_delta")),
        geometry_encoder_type=str(checkpoint.get("geometry_encoder_type", "track_gru_average")),
        trajectory_hidden_dim=int(checkpoint.get("trajectory_hidden_dim", 128)),
        geometry_max_tracks=int(checkpoint.get("geometry_max_tracks", 64)),
        geometry_attention_heads=int(checkpoint.get("geometry_attention_heads", 4)),
        geometry_transformer_layers=int(checkpoint.get("geometry_transformer_layers", 1)),
        joint_type_head_type=str(checkpoint.get("joint_type_head_type", "pair_context")),
        edge_head_type=str(checkpoint.get("edge_head_type", "pair_context")),
        relation_context_source=str(checkpoint.get("relation_context_source", "decoded_slots")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.trajectory_samples = int(checkpoint.get("trajectory_samples", 32))
    model.quality_weighted_trajectories = bool(checkpoint.get("quality_weighted_trajectories", False))
    model.robust_segment_weights = bool(checkpoint.get("robust_segment_weights", False))
    model.eval()
    return model


def _mean_abs(first, second, torch):
    return float(torch.mean(torch.abs(first - second)).cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("relation_model", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    np, torch = _require_numpy(), _require_torch()
    device = _resolve_device(torch, args.device)
    slot_checkpoint = torch.load(args.slot_model, map_location="cpu", weights_only=True)
    relation_checkpoint = torch.load(args.relation_model, map_location="cpu", weights_only=True)
    slot_model = _load_slot_model(slot_checkpoint, torch, device)
    if relation_checkpoint.get("slot_state_dict") is not None:
        slot_model.load_state_dict(relation_checkpoint["slot_state_dict"])
    slot_model.eval()
    relation_model = _model(relation_checkpoint, torch, device)
    samples = _load_relation_samples(load_pairwise_manifest(args.manifest), args.split, slot_checkpoint)
    samples = _apply_slot_input_mode(samples, str(relation_checkpoint.get("slot_input_mode", "full")), slot_checkpoint["feature_mean"].cpu().numpy())
    samples = _apply_slot_geometry_representation(samples, str(relation_checkpoint.get("slot_geometry_representation", "raw")), np)
    mean = relation_checkpoint.get("slot_feature_mean", slot_checkpoint["feature_mean"]).to(device)
    std = relation_checkpoint.get("slot_feature_std", slot_checkpoint["feature_std"]).to(device)
    rows = []
    with torch.no_grad():
        for sample in samples[: max(1, args.max_objects)]:
            features = torch.from_numpy(sample["features"]).to(device)
            slot_features = torch.from_numpy(sample.get("slot_features", sample["features"])).to(device)
            logits, _, slots = slot_model((slot_features - mean) / std, return_slots=True)
            probabilities = _relation_slot_probabilities(logits, sample, torch, "predicted")
            motion_summary = _slot_motion_summary(features, probabilities, int(sample["embedding_dim"]), torch)
            tokens, visibility = _trajectory_tensors(
                sample, torch, device,
                sample_count=relation_model.trajectory_samples,
                quality_weighted=relation_model.quality_weighted_trajectories,
                robust_segment_weights=relation_model.robust_segment_weights,
            )
            kwargs = dict(trajectory_tokens=tokens, trajectory_visibility=visibility)
            base = relation_model(slots, motion_summary, slot_probabilities=probabilities, **kwargs)
            permutation = torch.randperm(slots.shape[0], device=device)
            sync = relation_model(
                slots[permutation], motion_summary[permutation],
                slot_probabilities=probabilities[:, permutation], **kwargs,
            )
            inverse = torch.argsort(permutation)
            feature_permuted = relation_model(
                slots[permutation], motion_summary,
                slot_probabilities=probabilities, **kwargs,
            )
            feature_zeroed = relation_model(
                torch.zeros_like(slots), motion_summary,
                slot_probabilities=probabilities, **kwargs,
            )
            row = {"object_id": str(sample["object_id"])}
            for key in ("edge_logits", "type_logits", "axes", "pivots"):
                aligned = sync[key][inverse]
                aligned = aligned[:, inverse] if aligned.ndim >= 2 else aligned
                row[f"sync_permutation_{key}_mae"] = _mean_abs(base[key], aligned, torch)
                row[f"slot_feature_permutation_{key}_mae"] = _mean_abs(base[key], feature_permuted[key], torch)
                row[f"slot_feature_zero_{key}_mae"] = _mean_abs(base[key], feature_zeroed[key], torch)
            rows.append(row)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0] if key != "object_id"
    }
    payload = {
        "relation_context_source": relation_checkpoint.get("relation_context_source", "decoded_slots"),
        "objects": len(rows), "summary": summary, "per_object": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
