#!/usr/bin/env python3
"""Evaluate a trained slot Relation Head by object category without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import (
    JOINT_TYPES,
    _apply_slot_geometry_representation,
    _apply_slot_input_mode,
    _build_relation_model,
    _evaluate_relation_samples,
    _load_relation_samples,
    _load_slot_model,
)
from rgbd_urdf_mvp.perception.motion_part_slots import (
    _require_numpy,
    _require_torch,
    _resolve_device,
)
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("relation_model", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def load_categories(path: Path) -> dict[str, str]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    return {
        str(row["object_id"]): str(row.get("category", row.get("source_category", "unknown"))).lower()
        for row in rows
    }


def sample_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    relations = [relation for sample in samples for relation in sample.get("gt_relations", [])]
    return {
        "object_count": len(samples),
        "joint_count": len(relations),
        **{
            f"{joint_type}_joint_count": sum(
                str(relation.get("joint_type")) == joint_type for relation in relations
            )
            for joint_type in JOINT_TYPES
        },
    }


def main() -> int:
    args = parse_args()
    np = _require_numpy()
    torch = _require_torch()
    device = _resolve_device(torch, args.device)
    slot_checkpoint = torch.load(args.slot_model.expanduser().resolve(), map_location="cpu", weights_only=True)
    relation_checkpoint = torch.load(
        args.relation_model.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    slot_model = _load_slot_model(slot_checkpoint, torch, device)
    if relation_checkpoint.get("slot_state_dict") is not None:
        slot_model.load_state_dict(relation_checkpoint["slot_state_dict"])
    relation_model = _build_relation_model(
        torch,
        slot_dim=int(relation_checkpoint["slot_dim"]),
        hidden_dim=int(relation_checkpoint["hidden_dim"]),
        motion_summary_dim=int(relation_checkpoint.get("motion_summary_dim", 0)),
        axis_geometry_branch=bool(relation_checkpoint.get("axis_geometry_branch", False)),
        axis_head_type=str(relation_checkpoint.get("axis_head_type", "direct")),
        vector_pivot_parameterization=str(
            relation_checkpoint.get("vector_pivot_parameterization", "legacy_center_delta")
        ),
        geometry_encoder_type=str(
            relation_checkpoint.get("geometry_encoder_type", "track_gru_average")
        ),
        trajectory_hidden_dim=int(relation_checkpoint.get("trajectory_hidden_dim", 128)),
        geometry_max_tracks=int(relation_checkpoint.get("geometry_max_tracks", 64)),
        geometry_attention_heads=int(
            relation_checkpoint.get("geometry_attention_heads", 4)
        ),
        geometry_transformer_layers=int(
            relation_checkpoint.get("geometry_transformer_layers", 1)
        ),
        joint_type_head_type=str(
            relation_checkpoint.get("joint_type_head_type", "pair_context")
        ),
        edge_head_type=str(relation_checkpoint.get("edge_head_type", "pair_context")),
    ).to(device)
    relation_model.trajectory_samples = int(relation_checkpoint.get("trajectory_samples", 32))
    relation_model.quality_weighted_trajectories = bool(
        relation_checkpoint.get("quality_weighted_trajectories", False)
    )
    relation_model.load_state_dict(relation_checkpoint["state_dict"])
    samples = _load_relation_samples(
        load_pairwise_manifest(args.manifest.expanduser().resolve()), "test", slot_checkpoint
    )
    samples = _apply_slot_input_mode(
        samples,
        str(relation_checkpoint.get("slot_input_mode", "full")),
        slot_checkpoint["feature_mean"].cpu().numpy(),
    )
    samples = _apply_slot_geometry_representation(
        samples,
        str(relation_checkpoint.get("slot_geometry_representation", "raw")),
        np,
    )
    categories = load_categories(args.catalog)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[categories.get(str(sample["object_id"]), "unknown")].append(sample)

    mean = relation_checkpoint.get("slot_feature_mean", slot_checkpoint["feature_mean"]).to(device)
    std = relation_checkpoint.get("slot_feature_std", slot_checkpoint["feature_std"]).to(device)
    rows = []
    for category, category_samples in [("all", samples), *sorted(groups.items())]:
        metrics = _evaluate_relation_samples(
            category_samples, slot_model, relation_model, mean, std, torch, device
        )
        rows.append({"category": category, **sample_counts(category_samples), **metrics})

    per_object = []
    for sample in samples:
        metrics = _evaluate_relation_samples(
            [sample], slot_model, relation_model, mean, std, torch, device
        )
        per_object.append({
            "object_id": str(sample["object_id"]),
            "category": categories.get(str(sample["object_id"]), "unknown"),
            **sample_counts([sample]),
            **metrics,
        })
    per_object.sort(key=lambda row: (str(row["category"]), str(row["object_id"])))

    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"rows": rows, "per_object": per_object}, indent=2) + "\n",
        encoding="utf-8",
    )
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else output_json.with_suffix(".csv")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
