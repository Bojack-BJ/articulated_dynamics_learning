#!/usr/bin/env python3
"""Train and compare child-only analytic and equivariant feedforward axis heads."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.analytic_joint_axis import axis_angle_error_deg, axis_line_distance
from rgbd_urdf_mvp.kinematics.child_motion_axis import (
    build_child_motion_axis_model,
    estimate_child_motion_axis,
)
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import _load_relation_samples
from rgbd_urdf_mvp.perception.pairwise_affinity import load_pairwise_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("slot_model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def examples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for sample in samples:
        center = np.asarray(sample["canonical_center_m"], dtype=np.float32)
        scale = max(float(sample["canonical_scale_m"]), 1e-6)
        points = (np.asarray(sample["points"], dtype=np.float32) - center) / scale
        visibility = np.asarray(sample["visibility"], dtype=bool)
        labels = np.asarray(sample["labels"], dtype=int)
        for relation in sample["gt_relations"]:
            joint_type = str(relation["joint_type"])
            if joint_type not in {"revolute", "prismatic"}:
                continue
            child = labels == int(relation["child_label"])
            if int(child.sum()) < 6:
                continue
            result.append({
                "object_id": sample["object_id"],
                "joint_name": relation["joint_name"],
                "joint_type": joint_type,
                "points": points[child],
                "visibility": visibility[child],
                "axis": np.asarray(relation["axis"], dtype=np.float32),
                "pivot": np.asarray(relation["pivot"], dtype=np.float32),
            })
    return result


def tensor_example(row: dict[str, Any], torch: Any, device: str) -> tuple[Any, Any]:
    points = torch.as_tensor(row["points"], dtype=torch.float32, device=device)[None]
    visible = torch.as_tensor(row["visibility"], dtype=torch.bool, device=device)[None]
    return points, visible


def loss_for(model: Any, row: dict[str, Any], torch: Any, device: str) -> Any:
    points, visible = tensor_example(row, torch, device)
    output = model(points, visible)
    type_index = 0 if row["joint_type"] == "prismatic" else 1
    target_type = torch.tensor([type_index], device=device)
    type_loss = torch.nn.functional.cross_entropy(output["type_logits"], target_type)
    axis = output["prismatic_axis"] if type_index == 0 else output["revolute_axis"]
    target_axis = torch.as_tensor(row["axis"], device=device)[None]
    direction_loss = 1.0 - torch.sum(axis * target_axis, dim=-1).abs().mean()
    if type_index == 1:
        target_pivot = torch.as_tensor(row["pivot"], device=device)[None]
        line_loss = torch.linalg.vector_norm(
            torch.linalg.cross(output["line_point"] - target_pivot, target_axis, dim=-1), dim=-1
        ).mean()
    else:
        line_loss = direction_loss * 0.0
    return type_loss + 2.0 * direction_loss + line_loss


def evaluate(model: Any, rows: list[dict[str, Any]], torch: Any, device: str) -> dict[str, Any]:
    joint_rows = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            points, visible = tensor_example(row, torch, device)
            output = model(points, visible)
            predicted_type = "prismatic" if int(output["type_logits"].argmax(-1)) == 0 else "revolute"
            # Direction is evaluated under GT type for a head-isolation comparison;
            # type classification remains an independently reported metric.
            neural_axis = output[f"{row['joint_type']}_axis"][0].cpu().numpy()
            neural_line = output["line_point"][0].cpu().numpy()
            analytic = estimate_child_motion_axis(
                row["points"], row["visibility"], row["joint_type"]
            )
            neural_correct = predicted_type == row["joint_type"]
            joint_rows.append({
                "object_id": row["object_id"], "joint_name": row["joint_name"],
                "joint_type": row["joint_type"], "neural_type": predicted_type,
                "neural_axis_error_deg": axis_angle_error_deg(neural_axis, row["axis"]),
                "neural_line_error": axis_line_distance(neural_line, neural_axis, row["pivot"], row["axis"])
                if row["joint_type"] == "revolute" else None,
                "analytic_valid": analytic.valid,
                "analytic_axis_error_deg": axis_angle_error_deg(analytic.axis, row["axis"])
                if analytic.valid else None,
                "analytic_line_error": axis_line_distance(analytic.line_point, analytic.axis, row["pivot"], row["axis"])
                if analytic.valid and row["joint_type"] == "revolute" and analytic.line_point is not None else None,
            })
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in joint_rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "joint_count": len(joint_rows),
        "neural_type_accuracy": float(np.mean([
            row["neural_type"] == row["joint_type"] for row in joint_rows
        ])) if joint_rows else 0.0,
        "neural_axis_error_mean_deg": mean("neural_axis_error_deg"),
        "neural_line_error_mean": mean("neural_line_error"),
        "analytic_coverage": float(np.mean([row["analytic_valid"] for row in joint_rows])) if joint_rows else 0.0,
        "analytic_axis_error_mean_deg": mean("analytic_axis_error_deg"),
        "analytic_line_error_mean": mean("analytic_line_error"),
        "joints": joint_rows,
    }


def main() -> None:
    args = parse_args()
    import torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.slot_model, map_location="cpu", weights_only=False)
    manifest = load_pairwise_manifest(args.manifest)
    # Cap manifest rows before loading large CoTracker feature archives. A row
    # can contain multiple joints, so retain a small margin over the joint cap.
    train_rows = [row for row in manifest if row["split"] == "train"][: args.max_train * 2]
    val_rows = [row for row in manifest if row["split"] == "val"][: args.max_val * 2]
    train = examples(_load_relation_samples(train_rows, "train", checkpoint))[: args.max_train]
    validation = examples(_load_relation_samples(val_rows, "val", checkpoint))[: args.max_val]
    if not validation:
        test_rows = [row for row in manifest if row["split"] == "test"][: args.max_val * 2]
        validation = examples(_load_relation_samples(test_rows, "test", checkpoint))[: args.max_val]
    if not train or not validation:
        raise RuntimeError(f"insufficient examples: train={len(train)} val={len(validation)}")
    model = build_child_motion_axis_model(torch).to(device)
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train(); random.shuffle(train); losses = []
        for row in train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, row, torch, device)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation, torch, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **{
            key: value for key, value in metrics.items() if key != "joints"
        }})
        print(json.dumps(history[-1]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": 64}, args.output_dir / "child_motion_axis_head.pt")
    summary = {
        "protocol": "oracle_child_membership_head_isolation",
        "train_examples": len(train), "validation_examples": len(validation),
        "history": history, "validation": evaluate(model, validation, torch, device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
