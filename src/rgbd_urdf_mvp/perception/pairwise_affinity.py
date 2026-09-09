from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json
from .cotracker_features import _binary_auc, load_cotracker_feature_map
from .motion_segmentation import (
    _distance,
    _pair_metrics,
    _reference_position,
    _track_motion_m,
    _trajectory_by_frame,
)


PAIR_SCALAR_FEATURES = (
    "feature_cosine",
    "reference_distance_m",
    "rigidity_rmse_m",
    "motion_disagreement_m",
    "displacement_curve_rmse_m",
    "velocity_cosine",
    "common_frame_ratio",
    "track_a_motion_m",
    "track_b_motion_m",
    "motion_difference_m",
)


@dataclass(slots=True)
class PairwiseAffinityTrainingConfig:
    manifest_path: str | Path
    output_dir: str | Path
    knn_k: int = 12
    min_common_frames: int = 3
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    hard_negative_weight: float = 2.0
    hard_mining_k: int = 8
    max_positive_negative_ratio: float = 2.0
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class PairwiseAffinityEvaluationConfig:
    manifest_path: str | Path
    model_path: str | Path
    output_json: str | Path
    split: str = "test"
    knn_k: int = 12
    min_common_frames: int = 3
    device: str = "auto"


def load_pairwise_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path).expanduser().resolve()
    delimiter = "\t" if manifest_path.suffix.lower() == ".tsv" else ","
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    required = {"object_id", "tracks_path", "features_npz", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Pairwise manifest requires columns: {sorted(required)}")
    for row in rows:
        for key in ("tracks_path", "features_npz", "relation_gt_path"):
            if not row.get(key):
                continue
            candidate = Path(row[key]).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            row[key] = str(candidate.resolve())
        row["split"] = str(row["split"]).strip().lower()
    return rows


def build_pair_dataset(
    rows: list[dict[str, str]],
    *,
    split: str,
    knn_k: int,
    min_common_frames: int,
    hard_negative_weight: float,
    hard_mining_k: int = 0,
    max_positive_negative_ratio: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    np = _require_numpy()
    vectors: list[Any] = []
    labels: list[float] = []
    sample_weights: list[float] = []
    object_ids: list[str] = []
    pair_rows: list[dict[str, Any]] = []
    feature_dim: int | None = None

    for row in rows:
        if row["split"] != split:
            continue
        artifact = load_json(row["tracks_path"])
        feature_map = load_cotracker_feature_map(row["features_npz"])
        tracks = [
            track
            for track in artifact.get("tracks", [])
            if isinstance(track, dict)
            and "original_part_id" in track
            and int(track.get("track_id", -1)) in feature_map
            and _trajectory_by_frame(track)
        ]
        if not tracks:
            continue
        trajectories = [_trajectory_by_frame(track) for track in tracks]
        references = [_reference_position(track) for track in tracks]
        seen: dict[tuple[int, int], set[str]] = {}
        labels_by_track = [int(track["original_part_id"]) for track in tracks]
        feature_matrix = np.stack([feature_map[int(track["track_id"])] for track in tracks]).astype(np.float64)
        endpoint_motion = np.asarray([_endpoint_displacement(track) for track in tracks], dtype=np.float64)

        def add_pair(i: int, j: int, source: str) -> None:
            pair = (min(i, j), max(i, j))
            seen.setdefault(pair, set()).add(source)

        for i, reference in enumerate(references):
            neighbors = sorted(
                (_distance(reference, references[j]), j)
                for j in range(len(tracks))
                if j != i
            )[: max(1, int(knn_k))]
            for reference_distance, j in neighbors:
                add_pair(i, j, "spatial_knn")

        mining_k = max(0, int(hard_mining_k))
        if mining_k > 0:
            for i in range(len(tracks)):
                cross_part = [j for j in range(len(tracks)) if j != i and labels_by_track[j] != labels_by_track[i]]
                same_part = [j for j in range(len(tracks)) if j != i and labels_by_track[j] == labels_by_track[i]]
                feature_hard = sorted(cross_part, key=lambda j: -float(feature_matrix[i] @ feature_matrix[j]))[:mining_k]
                motion_hard = sorted(
                    cross_part,
                    key=lambda j: float(np.linalg.norm(endpoint_motion[i] - endpoint_motion[j])),
                )[:mining_k]
                distant_positive = sorted(
                    same_part,
                    key=lambda j: -_distance(references[i], references[j]),
                )[: max(1, mining_k // 2)]
                for j in feature_hard:
                    add_pair(i, j, "feature_hard_negative")
                for j in motion_hard:
                    add_pair(i, j, "motion_hard_negative")
                for j in distant_positive:
                    add_pair(i, j, "distant_positive")

        object_vectors: list[Any] = []
        object_labels: list[float] = []
        object_weights: list[float] = []
        object_pair_rows: list[dict[str, Any]] = []
        for (i, j), sources in sorted(seen.items()):
            reference_distance = _distance(references[i], references[j])
            metrics = _pair_metrics(trajectories[i], trajectories[j], min_common_frames)
            if metrics is None:
                continue
            a_feature = feature_map[int(tracks[i]["track_id"])]
            b_feature = feature_map[int(tracks[j]["track_id"])]
            if feature_dim is None:
                feature_dim = int(a_feature.shape[0])
            elif int(a_feature.shape[0]) != feature_dim or int(b_feature.shape[0]) != feature_dim:
                raise ValueError("All CoTracker feature artifacts must use the same feature dimension.")
            vector = pair_feature_vector(
                a_feature,
                b_feature,
                tracks[i],
                tracks[j],
                metrics,
                reference_distance_m=float(metrics.get("median_distance_m", reference_distance)),
            )
            label = float(labels_by_track[i] == labels_by_track[j])
            geometric_affinity = _diagnostic_geometric_affinity(metrics, float(reference_distance))
            is_mined_negative = not label and any(source.endswith("hard_negative") for source in sources)
            weight = 1.0 + float(is_mined_negative) * max(0.0, float(hard_negative_weight)) * max(
                0.25, geometric_affinity
            )
            object_vectors.append(vector)
            object_labels.append(label)
            object_weights.append(weight)
            object_pair_rows.append(
                {
                    "object_id": row["object_id"],
                    "track_i": int(tracks[i]["track_id"]),
                    "track_j": int(tracks[j]["track_id"]),
                    "label": int(label),
                    "sources": sorted(sources),
                    "geometric_affinity": geometric_affinity,
                    "sample_weight": weight,
                }
            )

        keep = _balanced_pair_indices(
            object_labels,
            max_positive_negative_ratio=max_positive_negative_ratio,
            seed=seed + len(object_ids),
        )
        for index in keep:
            vectors.append(object_vectors[index])
            labels.append(object_labels[index])
            sample_weights.append(object_weights[index])
            object_ids.append(row["object_id"])
            pair_rows.append(object_pair_rows[index])
    if not vectors:
        raise ValueError(f"No pair samples found for split '{split}'.")
    return {
        "features": np.stack(vectors).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "object_ids": object_ids,
        "pair_rows": pair_rows,
        "cotracker_feature_dim": int(feature_dim or 0),
    }


def pair_feature_vector(
    a_feature: Any,
    b_feature: Any,
    a_track: dict[str, Any],
    b_track: dict[str, Any],
    metrics: dict[str, Any],
    *,
    reference_distance_m: float,
) -> Any:
    np = _require_numpy()
    a = np.asarray(a_feature, dtype=np.float64)
    b = np.asarray(b_feature, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Pairwise CoTracker features must be aligned one-dimensional vectors.")
    cosine = float(max(-1.0, min(1.0, float(a @ b))))
    a_motion = float(_track_motion_m(a_track))
    b_motion = float(_track_motion_m(b_track))
    common_frames = float(metrics.get("common_frames", 0.0))
    scalar = np.asarray(
        [
            cosine,
            float(reference_distance_m),
            float(metrics.get("rigidity_rmse_m", 0.0)),
            float(metrics.get("motion_disagreement_m", 0.0)),
            float(metrics.get("displacement_curve_rmse_m", 0.0)),
            float(metrics.get("velocity_cosine", 0.0)),
            min(1.0, common_frames / 60.0),
            min(a_motion, b_motion),
            max(a_motion, b_motion),
            abs(a_motion - b_motion),
        ],
        dtype=np.float64,
    )
    # abs-difference and product make the head invariant to pair ordering.
    return np.concatenate([np.abs(a - b), a * b, scalar]).astype(np.float32)


class PairwiseAffinityTrainer:
    def __init__(self, config: PairwiseAffinityTrainingConfig) -> None:
        self.config = config

    def train(self) -> Path:
        np = _require_numpy()
        torch = _require_torch()
        rows = load_pairwise_manifest(self.config.manifest_path)
        train_data = build_pair_dataset(
            rows,
            split="train",
            knn_k=self.config.knn_k,
            min_common_frames=self.config.min_common_frames,
            hard_negative_weight=self.config.hard_negative_weight,
            hard_mining_k=self.config.hard_mining_k,
            max_positive_negative_ratio=self.config.max_positive_negative_ratio,
            seed=self.config.seed,
        )
        validation_split = "val" if any(row["split"] == "val" for row in rows) else "train"
        val_data = build_pair_dataset(
            rows,
            split=validation_split,
            knn_k=self.config.knn_k,
            min_common_frames=self.config.min_common_frames,
            hard_negative_weight=self.config.hard_negative_weight,
            hard_mining_k=self.config.hard_mining_k,
            max_positive_negative_ratio=self.config.max_positive_negative_ratio,
            seed=self.config.seed + 1,
        )
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        device = _resolve_torch_device(torch, self.config.device)
        torch.manual_seed(int(self.config.seed))
        mean = np.mean(train_data["features"], axis=0).astype(np.float32)
        std = np.std(train_data["features"], axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)
        model = _build_model(torch, train_data["features"].shape[1], int(self.config.hidden_dim)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        train_x = torch.from_numpy((train_data["features"] - mean) / std)
        train_y = torch.from_numpy(train_data["labels"])
        train_w = torch.from_numpy(train_data["sample_weights"])
        negative_count = max(1.0, float((train_data["labels"] == 0).sum()))
        positive_count = max(1.0, float((train_data["labels"] == 1).sum()))
        positive_weight = min(10.0, negative_count / positive_count)
        rng = np.random.default_rng(self.config.seed)
        history: list[dict[str, Any]] = []
        best_auc = -math.inf
        checkpoint_path = output_dir / "pairwise_affinity.pt"

        for epoch in range(max(1, int(self.config.epochs))):
            model.train()
            order = rng.permutation(train_x.shape[0])
            losses = []
            for start in range(0, len(order), max(1, int(self.config.batch_size))):
                indices = torch.from_numpy(order[start : start + self.config.batch_size]).long()
                batch_x = train_x[indices].to(device)
                batch_y = train_y[indices].to(device)
                batch_w = train_w[indices].to(device)
                logits = model(batch_x).squeeze(-1)
                class_weight = torch.where(batch_y > 0.5, positive_weight, 1.0)
                loss_values = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_y,
                    reduction="none",
                )
                loss = (loss_values * batch_w * class_weight).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = _evaluate_model(model, val_data, mean, std, device, torch)
            row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics}
            history.append(row)
            if metrics["auc"] >= best_auc:
                best_auc = float(metrics["auc"])
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": int(train_data["features"].shape[1]),
                        "hidden_dim": int(self.config.hidden_dim),
                        "cotracker_feature_dim": int(train_data["cotracker_feature_dim"]),
                        "feature_mean": torch.from_numpy(mean),
                        "feature_std": torch.from_numpy(std),
                        "scalar_feature_names": list(PAIR_SCALAR_FEATURES),
                        "knn_k": int(self.config.knn_k),
                        "min_common_frames": int(self.config.min_common_frames),
                    },
                    checkpoint_path,
                )
        save_json(
            {
                "model_path": str(checkpoint_path),
                "device": device,
                "validation_split": validation_split,
                "train": _dataset_summary(train_data),
                "validation": _dataset_summary(val_data),
                "best_val_auc": best_auc,
                "history": history,
                "config": {
                    "knn_k": self.config.knn_k,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "hard_negative_weight": self.config.hard_negative_weight,
                    "hard_mining_k": self.config.hard_mining_k,
                    "max_positive_negative_ratio": self.config.max_positive_negative_ratio,
                },
            },
            output_dir / "training_summary.json",
        )
        return checkpoint_path


class PairwiseAffinityPredictor:
    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        torch = _require_torch()
        self.torch = torch
        self.device = _resolve_torch_device(torch, device)
        checkpoint = torch.load(Path(model_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        self.feature_dim = int(checkpoint["cotracker_feature_dim"])
        self.model = _build_model(torch, int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.mean = checkpoint["feature_mean"].float().to(self.device)
        self.std = checkpoint["feature_std"].float().to(self.device)

    def predict(
        self,
        a_feature: Any,
        b_feature: Any,
        a_track: dict[str, Any],
        b_track: dict[str, Any],
        metrics: dict[str, Any],
    ) -> float:
        vector = pair_feature_vector(
            a_feature,
            b_feature,
            a_track,
            b_track,
            metrics,
            reference_distance_m=float(metrics.get("median_distance_m", 0.0)),
        )
        tensor = self.torch.from_numpy(vector).float().to(self.device)
        with self.torch.no_grad():
            logit = self.model(((tensor - self.mean) / self.std).unsqueeze(0)).squeeze()
            return float(self.torch.sigmoid(logit).cpu())

    def predict_many(self, vectors: list[Any], *, batch_size: int = 2048) -> list[float]:
        """Score precomputed pair vectors without one device round-trip per edge."""
        if not vectors:
            return []
        np = _require_numpy()
        matrix = np.stack(vectors).astype(np.float32)
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, matrix.shape[0], max(1, int(batch_size))):
                tensor = self.torch.from_numpy(matrix[start : start + batch_size]).to(self.device)
                logits = self.model((tensor - self.mean) / self.std).squeeze(-1)
                scores.extend(float(value) for value in self.torch.sigmoid(logits).cpu().tolist())
        return scores


class PairwiseAffinityEvaluator:
    def __init__(self, config: PairwiseAffinityEvaluationConfig) -> None:
        self.config = config

    def evaluate(self) -> Path:
        rows = load_pairwise_manifest(self.config.manifest_path)
        data = build_pair_dataset(
            rows,
            split=self.config.split,
            knn_k=self.config.knn_k,
            min_common_frames=self.config.min_common_frames,
            hard_negative_weight=0.0,
        )
        predictor = PairwiseAffinityPredictor(self.config.model_path, device=self.config.device)
        metrics = _evaluate_checkpoint_arrays(predictor, data)
        output = Path(self.config.output_json).expanduser().resolve()
        save_json({"split": self.config.split, "dataset": _dataset_summary(data), "metrics": metrics}, output)
        return output


def _evaluate_checkpoint_arrays(predictor: PairwiseAffinityPredictor, data: dict[str, Any]) -> dict[str, float]:
    torch = predictor.torch
    tensor = torch.from_numpy(data["features"]).float().to(predictor.device)
    with torch.no_grad():
        scores = torch.sigmoid(predictor.model((tensor - predictor.mean) / predictor.std).squeeze(-1)).cpu().numpy()
    return _classification_metrics(scores, data["labels"])


def _evaluate_model(model: Any, data: dict[str, Any], mean: Any, std: Any, device: str, torch: Any) -> dict[str, float]:
    model.eval()
    tensor = torch.from_numpy((data["features"] - mean) / std).float().to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(tensor).squeeze(-1)).cpu().numpy()
    return _classification_metrics(scores, data["labels"])


def _classification_metrics(scores: Any, labels: Any) -> dict[str, float]:
    np = _require_numpy()
    predicted = scores >= 0.5
    positive = labels >= 0.5
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "auc": float(_binary_auc(scores, positive)),
        "accuracy": float(np.mean(predicted == positive)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
    }


def _dataset_summary(data: dict[str, Any]) -> dict[str, Any]:
    np = _require_numpy()
    labels = data["labels"]
    source_counts: dict[str, int] = {}
    for row in data.get("pair_rows", []):
        for source in row.get("sources", ["spatial_knn"]):
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    return {
        "pair_count": int(labels.shape[0]),
        "positive_count": int(np.sum(labels >= 0.5)),
        "negative_count": int(np.sum(labels < 0.5)),
        "object_ids": sorted(set(data["object_ids"])),
        "input_dim": int(data["features"].shape[1]),
        "source_counts": source_counts,
    }


def _endpoint_displacement(track: dict[str, Any]) -> list[float]:
    trajectory = _trajectory_by_frame(track)
    if len(trajectory) < 2:
        return [0.0, 0.0, 0.0]
    frames = sorted(trajectory)
    start = trajectory[frames[0]]
    end = trajectory[frames[-1]]
    return [float(end[index] - start[index]) for index in range(3)]


def _balanced_pair_indices(
    labels: list[float],
    *,
    max_positive_negative_ratio: float,
    seed: int,
) -> list[int]:
    if max_positive_negative_ratio <= 0.0:
        return list(range(len(labels)))
    np = _require_numpy()
    positives = [index for index, label in enumerate(labels) if label >= 0.5]
    negatives = [index for index, label in enumerate(labels) if label < 0.5]
    if not positives or not negatives:
        return list(range(len(labels)))
    max_positives = max(1, int(math.ceil(len(negatives) * max_positive_negative_ratio)))
    if len(positives) > max_positives:
        rng = np.random.default_rng(seed)
        positives = sorted(int(index) for index in rng.choice(positives, size=max_positives, replace=False))
    return sorted(positives + negatives)


def _diagnostic_geometric_affinity(metrics: dict[str, Any], reference_distance_m: float) -> float:
    rigidity = math.exp(-float(metrics.get("rigidity_rmse_m", 0.0)) / 0.015)
    endpoint = math.exp(-float(metrics.get("motion_disagreement_m", 0.0)) / 0.03)
    spatial = math.exp(-max(0.0, reference_distance_m) / 0.18)
    visibility = min(1.0, float(metrics.get("common_frames", 0.0)) / 6.0)
    return float(max(0.0, min(1.0, (0.55 * rigidity + 0.30 * endpoint + 0.15 * visibility) * spatial)))


def _build_model(torch: Any, input_dim: int, hidden_dim: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
        torch.nn.GELU(),
        torch.nn.Linear(max(16, hidden_dim // 2), 1),
    )


def _resolve_torch_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported torch device: {requested}")
    return requested


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pairwise affinity requires NumPy.") from exc
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pairwise affinity training requires the tracking extra with PyTorch.") from exc
    return torch
