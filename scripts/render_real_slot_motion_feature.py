#!/usr/bin/env python3
"""Render the real assignment-weighted slot motion features for Figure 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rgbd_urdf_mvp.kinematics.pairwise_relation_head import _slot_motion_summary
from rgbd_urdf_mvp.perception.cotracker_features import load_cotracker_feature_map
from rgbd_urdf_mvp.perception.motion_part_slots import _build_slot_model, _sample_from_artifact


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/learning_features_v1/refrigerator045"
TRACKS = CASE / "predicted_slots_balanced_v2.json"
FEATURES = CASE / "cotracker_features.npz"
OUTPUT = ROOT / "outputs/paper_assets/refrigerator045/slot_motion_feature_real.png"
DETAILS = ROOT / "outputs/paper_assets/refrigerator045/slot_motion_feature_real.json"

SEMANTIC_COLORS = {1: "#858B93", 2: "#397BC5", 3: "#F28E2B", 4: "#397BC5"}
SEMANTIC_NAMES = {1: "base", 2: "drawer", 3: "door", 4: "drawer"}


def main() -> None:
    import torch

    artifact = json.loads(TRACKS.read_text(encoding="utf-8"))
    checkpoint_path = Path(artifact["motion_segmentation"]["model_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sample = _sample_from_artifact(
        artifact,
        load_cotracker_feature_map(FEATURES),
        object_id="refrigerator045",
        require_labels=False,
        canonicalize_geometry=bool(checkpoint.get("canonicalize_geometry", False)),
    )
    model = _build_slot_model(
        torch,
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_slots=int(checkpoint["max_slots"]),
        encoder_layers=int(checkpoint["encoder_layers"]),
        decoder_layers=int(checkpoint["decoder_layers"]),
        attention_heads=int(checkpoint["attention_heads"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = torch.from_numpy(sample["features"])
    with torch.no_grad():
        logits, existence, _ = model(
            (features - checkpoint["feature_mean"]) / checkpoint["feature_std"],
            return_slots=True,
        )
        probabilities = torch.softmax(logits, dim=-1)
        summary = _slot_motion_summary(
            features, probabilities, int(sample["embedding_dim"]), torch
        ).cpu().numpy()
        existence_prob = torch.sigmoid(existence).cpu().numpy()

    active = [int(value) for value in artifact["motion_segmentation"]["active_slots"]]
    raw = summary[active]
    offset = int(sample["embedding_dim"])
    mean = checkpoint["feature_mean"].cpu().numpy()[offset:]
    std = checkpoint["feature_std"].cpu().numpy()[offset:]
    standardized = np.clip((raw - mean[None, :]) / np.maximum(std[None, :], 1e-6), -2.5, 2.5)

    labels = np.asarray(
        [int(track.get("original_part_id", track.get("part_id", 0))) for track in sample["tracks"]],
        dtype=np.int64,
    )
    probs = probabilities.cpu().numpy()
    roles = []
    for slot in active:
        scores = {
            int(label): float(probs[labels == label, slot].sum())
            for label in np.unique(labels)
        }
        gt_part = max(scores, key=scores.get)
        roles.append((gt_part, SEMANTIC_NAMES.get(gt_part, f"part {gt_part}")))

    points = np.asarray(sample["points"], dtype=np.float32)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    trajectories = []
    for slot in active:
        slot_weights = probs[:, slot]
        centroids = []
        for frame_index in range(points.shape[1]):
            valid = visibility[:, frame_index] & np.isfinite(points[:, frame_index]).all(axis=1)
            weights = slot_weights[valid]
            if weights.sum() <= 1e-8:
                centroids.append(centroids[-1] if centroids else np.zeros(3, dtype=np.float32))
                continue
            centroids.append(np.average(points[valid, frame_index], axis=0, weights=weights))
        trajectories.append(np.asarray(centroids, dtype=np.float32))
    trajectories = np.asarray(trajectories)
    centered = trajectories - trajectories[:, :1]
    flattened = centered.reshape(-1, 3)
    _, _, vt = np.linalg.svd(flattened - flattened.mean(axis=0), full_matrices=False)
    projected = centered @ vt[:2].T

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 4.5), dpi=220)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.50)
    heat = fig.add_subplot(grid[0])
    image = heat.imshow(standardized, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    heat.set_yticks(range(len(active)))
    heat.set_yticklabels([f"{name}  (slot {slot})" for slot, (_, name) in zip(active, roles)])
    heat.set_xticks([1, 4, 6, 7, 11, 17, 23, 29])
    heat.set_xticklabels(["ref", "end", "mag", "vis", "t1", "t3", "t5", "t7"], fontsize=8)
    for boundary in (2.5, 5.5, 6.5, 7.5):
        heat.axvline(boundary, color="#172033", linewidth=1.2)
    for row, (part_id, _) in enumerate(roles):
        heat.get_yticklabels()[row].set_color(SEMANTIC_COLORS.get(part_id, "#172033"))
        heat.get_yticklabels()[row].set_fontweight("bold")
    heat.set_title("Real assignment-weighted slot motion descriptors", fontsize=14, weight="bold", pad=10)
    heat.set_xlabel("32D canonical trajectory descriptor", fontsize=9)
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_color("#A7B8CC")
    colorbar = fig.colorbar(image, ax=heat, fraction=0.018, pad=0.015)
    colorbar.set_label("standardized value", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    glyph = fig.add_subplot(grid[1])
    temporal_indices = np.linspace(0, trajectories.shape[1] - 1, 8).round().astype(int)
    for row, ((part_id, name), trajectory) in enumerate(zip(roles, projected)):
        color = SEMANTIC_COLORS.get(part_id, "#64748B")
        x = trajectory[:, 0]
        y = trajectory[:, 1] + row * 0.22
        glyph.plot(x, y, color=color, linewidth=2.6, alpha=0.92)
        glyph.scatter(
            x[temporal_indices], y[temporal_indices], color=color,
            s=np.linspace(10, 25, len(temporal_indices)), alpha=np.linspace(0.35, 1.0, len(temporal_indices)),
            edgecolors="none", zorder=3,
        )
        glyph.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8})
        glyph.text(x[0] - 0.02, row * 0.22, name, color=color, ha="right", va="center", weight="bold", fontsize=9)
    glyph.set_title("Assignment-weighted 3D slot centroid trajectories", fontsize=10, pad=8)
    glyph.set_aspect("equal", adjustable="datalim")
    glyph.axis("off")
    fig.patch.set_alpha(0.0)
    fig.savefig(OUTPUT, dpi=650, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    DETAILS.write_text(
        json.dumps(
            {
                "tracks": str(TRACKS),
                "features": str(FEATURES),
                "checkpoint": str(checkpoint_path),
                "active_slots": active,
                "slot_existence": [float(existence_prob[slot]) for slot in active],
                "dominant_gt_parts": [int(part_id) for part_id, _ in roles],
                "raw_slot_motion_summary": raw.tolist(),
                "assignment_weighted_slot_centroid_trajectories": trajectories.tolist(),
                "descriptor_layout": {
                    "reference_xyz": [0, 3],
                    "endpoint_displacement_xyz": [3, 6],
                    "motion_magnitude": [6, 7],
                    "visibility_ratio": [7, 8],
                    "eight_sampled_xyz_displacements": [8, 32],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
