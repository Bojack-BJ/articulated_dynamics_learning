"""Permutation-invariant per-track part-slot prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from .cotracker_features import load_cotracker_feature_map
from .object_mask_flow_html import ObjectMaskFlowHtmlBuilder, ObjectMaskFlowHtmlConfig
from .pairwise_affinity import load_pairwise_manifest


@dataclass(slots=True)
class TrackSlotTrainingConfig:
    manifest_path: str | Path
    output_dir: str | Path
    max_slots: int = 8
    hidden_dim: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    pairwise_loss_weight: float = 0.2
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class TrackSlotPredictionConfig:
    tracks_path: str | Path
    features_npz: str | Path
    model_path: str | Path
    output_json: str | Path
    output_viewer: str | Path | None = None
    device: str = "auto"
    generate_viewer: bool = True


class TrackSlotTrainer:
    def __init__(self, config: TrackSlotTrainingConfig) -> None:
        self.config = config

    def train(self) -> Path:
        np = _require_numpy()
        torch = _require_torch()
        rows = load_pairwise_manifest(self.config.manifest_path)
        train_objects = _load_objects(rows, "train")
        val_objects = _load_objects(rows, "val") or train_objects
        if not train_objects:
            raise ValueError("Track-slot training manifest has no train objects.")
        feature_dim = int(train_objects[0]["features"].shape[1])
        if any(int(item["features"].shape[1]) != feature_dim for item in train_objects + val_objects):
            raise ValueError("All track-slot feature artifacts must use the same feature dimension.")
        max_gt_parts = max(len(set(item["labels"].tolist())) for item in train_objects + val_objects)
        if max_gt_parts > int(self.config.max_slots):
            raise ValueError(f"max_slots={self.config.max_slots} is smaller than GT part count {max_gt_parts}.")

        device = _resolve_device(torch, self.config.device)
        torch.manual_seed(int(self.config.seed))
        model = _TrackSlotModel(torch, feature_dim, int(self.config.hidden_dim), int(self.config.max_slots)).module.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay)
        )
        history: list[dict[str, Any]] = []
        best_score = -math.inf
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "track_slot_head.pt"
        rng = np.random.default_rng(self.config.seed)

        for epoch in range(max(1, int(self.config.epochs))):
            model.train()
            losses = []
            for index in rng.permutation(len(train_objects)):
                item = train_objects[int(index)]
                features = torch.from_numpy(item["features"]).float().to(device)
                labels = torch.from_numpy(item["labels"]).long().to(device)
                logits = model(features)
                target = _hungarian_slot_targets(logits.detach(), labels, int(self.config.max_slots), torch)
                counts = torch.bincount(target, minlength=int(self.config.max_slots)).float()
                class_weights = torch.where(counts > 0, labels.numel() / counts.clamp_min(1.0), 0.0)
                class_weights = class_weights / class_weights[class_weights > 0].mean().clamp_min(1e-6)
                ce_loss = torch.nn.functional.cross_entropy(logits, target, weight=class_weights)
                pair_loss = _pairwise_partition_loss(logits, labels, torch)
                loss = ce_loss + float(self.config.pairwise_loss_weight) * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = _evaluate_objects(model, val_objects, device, torch)
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
            score = float(
                metrics["mean_cluster_purity"] + metrics["mean_gt_coverage"]
                - metrics["oversegmentation_ratio"] - metrics["undersegmentation_ratio"]
            )
            if score >= best_score:
                best_score = score
                torch.save({
                    "state_dict": model.state_dict(), "feature_dim": feature_dim,
                    "hidden_dim": int(self.config.hidden_dim), "max_slots": int(self.config.max_slots),
                }, checkpoint_path)
        save_json({
            "model_path": str(checkpoint_path), "best_validation_score": best_score,
            "train_object_count": len(train_objects), "validation_object_count": len(val_objects),
            "history": history, "config": {
                field: str(value) if isinstance(value, Path) else value
                for field in self.config.__dataclass_fields__
                for value in [getattr(self.config, field)]
            },
        }, output_dir / "training_summary.json")
        return checkpoint_path


class TrackSlotPredictor:
    def __init__(self, config: TrackSlotPredictionConfig) -> None:
        self.config = config

    def predict(self) -> tuple[Path, Path | None]:
        np = _require_numpy()
        torch = _require_torch()
        artifact = load_json(self.config.tracks_path)
        feature_map = load_cotracker_feature_map(self.config.features_npz)
        tracks = [track for track in artifact.get("tracks", []) if int(track.get("track_id", -1)) in feature_map]
        if not tracks:
            raise ValueError("No tracks align with the track-slot feature artifact.")
        checkpoint = torch.load(Path(self.config.model_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        feature_dim = int(checkpoint["feature_dim"])
        features = np.stack([feature_map[int(track["track_id"])] for track in tracks]).astype(np.float32)
        if features.shape[1] != feature_dim:
            raise ValueError(f"Track-slot model expects {feature_dim} features, got {features.shape[1]}.")
        device = _resolve_device(torch, self.config.device)
        model = _TrackSlotModel(torch, feature_dim, int(checkpoint["hidden_dim"]), int(checkpoint["max_slots"])).module
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.from_numpy(features).to(device)), dim=-1).cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        output_tracks = []
        for track, slot, probability in zip(tracks, predicted.tolist(), probabilities.tolist()):
            item = dict(track)
            item.update({
                "part_id": int(slot) + 1, "part_name": f"learned_slot_{int(slot) + 1}",
                "predicted_slot_id": int(slot), "predicted_slot_confidence": float(max(probability)),
                "predicted_slot_probabilities": [float(value) for value in probability],
            })
            output_tracks.append(item)
        output = Path(self.config.output_json).expanduser().resolve()
        save_json({
            **{key: value for key, value in artifact.items() if key != "tracks"},
            "estimator": "permutation-invariant-track-slot-head", "slot_model": str(Path(self.config.model_path).resolve()),
            "parts": _part_summary(output_tracks), "tracks": output_tracks,
        }, output)
        viewer = None
        if self.config.generate_viewer:
            viewer = Path(self.config.output_viewer).expanduser().resolve() if self.config.output_viewer else output.with_suffix(".viewer.html")
            ObjectMaskFlowHtmlBuilder().build(ObjectMaskFlowHtmlConfig(
                motion_tracks=output, output_html=viewer, color_by="pred_cluster", frame_stride=1,
            ))
        return output, viewer


def _load_objects(rows: list[dict[str, str]], split: str) -> list[dict[str, Any]]:
    np = _require_numpy()
    objects = []
    for row in rows:
        if row["split"] != split:
            continue
        artifact = load_json(row["tracks_path"])
        feature_map = load_cotracker_feature_map(row["features_npz"])
        tracks = [track for track in artifact.get("tracks", []) if "original_part_id" in track and int(track.get("track_id", -1)) in feature_map]
        if not tracks:
            continue
        objects.append({
            "object_id": row["object_id"],
            "features": np.stack([feature_map[int(track["track_id"])] for track in tracks]).astype(np.float32),
            "labels": np.asarray([int(track["original_part_id"]) for track in tracks], dtype=np.int64),
        })
    return objects


def _hungarian_slot_targets(logits: Any, labels: Any, max_slots: int, torch: Any) -> Any:
    from scipy.optimize import linear_sum_assignment

    probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-8)
    unique = torch.unique(labels, sorted=True)
    cost = torch.stack([-torch.log(probabilities[labels == label]).sum(dim=0) for label in unique])
    gt_rows, slot_columns = linear_sum_assignment(cost.cpu().numpy())
    target = torch.zeros_like(labels)
    for gt_row, slot in zip(gt_rows.tolist(), slot_columns.tolist()):
        target[labels == unique[int(gt_row)]] = int(slot)
    return target


def _pairwise_partition_loss(logits: Any, labels: Any, torch: Any, max_pairs: int = 8192) -> Any:
    count = int(logits.shape[0])
    if count < 2:
        return logits.sum() * 0.0
    pairs = torch.combinations(torch.arange(count, device=logits.device), r=2)
    if int(pairs.shape[0]) > max_pairs:
        pairs = pairs[torch.randperm(pairs.shape[0], device=logits.device)[:max_pairs]]
    probabilities = torch.softmax(logits, dim=-1)
    same_probability = (probabilities[pairs[:, 0]] * probabilities[pairs[:, 1]]).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
    same_label = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).float()
    positive_count = same_label.sum().clamp_min(1.0)
    negative_count = (1.0 - same_label).sum().clamp_min(1.0)
    weights = torch.where(same_label > 0.5, 0.5 / positive_count, 0.5 / negative_count)
    losses = torch.nn.functional.binary_cross_entropy(same_probability, same_label, reduction="none")
    return (losses * weights).sum()


def _evaluate_objects(model: Any, objects: list[dict[str, Any]], device: str, torch: Any) -> dict[str, float]:
    purities, coverages, oversegments, undersegments = [], [], [], []
    model.eval()
    with torch.no_grad():
        for item in objects:
            labels = item["labels"]
            predicted = model(torch.from_numpy(item["features"]).float().to(device)).argmax(dim=-1).cpu().numpy()
            metrics = _partition_metrics(predicted, labels)
            purities.append(metrics["mean_cluster_purity"])
            coverages.append(metrics["mean_gt_coverage"])
            oversegments.append(metrics["oversegmentation_ratio"])
            undersegments.append(metrics["undersegmentation_ratio"])
    return {
        "mean_cluster_purity": float(sum(purities) / len(purities)),
        "mean_gt_coverage": float(sum(coverages) / len(coverages)),
        "oversegmentation_ratio": float(sum(oversegments) / len(oversegments)),
        "undersegmentation_ratio": float(sum(undersegments) / len(undersegments)),
    }


def _partition_metrics(predicted: Any, labels: Any) -> dict[str, float]:
    np = _require_numpy()
    clusters, gt_parts = np.unique(predicted), np.unique(labels)
    overlap = np.asarray([[np.sum((predicted == cluster) & (labels == gt)) for gt in gt_parts] for cluster in clusters])
    purity = float(np.mean(overlap.max(axis=1) / overlap.sum(axis=1)))
    coverage = float(np.mean(overlap.max(axis=0) / overlap.sum(axis=0)))
    denominator = max(1, len(gt_parts))
    return {
        "mean_cluster_purity": purity,
        "mean_gt_coverage": coverage,
        "oversegmentation_ratio": max(0.0, (len(clusters) - len(gt_parts)) / denominator),
        "undersegmentation_ratio": max(0.0, (len(gt_parts) - len(clusters)) / denominator),
    }


def _part_summary(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for track in tracks:
        key = str(track["part_id"])
        counts[key] = counts.get(key, 0) + 1
    return {key: {"name": f"learned_slot_{key}", "count": value} for key, value in sorted(counts.items())}


class _TrackSlotModel:
    def __init__(self, torch: Any, feature_dim: int, hidden_dim: int, max_slots: int) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU(),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * 2, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, max_slots)
                )

            def forward(self, features: Any) -> Any:
                tokens = self.encoder(features)
                context = tokens.mean(dim=0, keepdim=True).expand_as(tokens)
                return self.head(torch.cat([tokens, context], dim=-1))

        self.module = Model()


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_numpy() -> Any:
    import numpy as np
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Track-slot learning requires the tracking extra with PyTorch.") from exc
    return torch
