from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class CoTrackerFeatureProbeConfig:
    tracks_path: str | Path
    features_npz: str | Path
    output_json: str | Path | None = None
    output_embedding_csv: str | Path | None = None
    label_field: str = "original_part_id"
    max_pairs: int = 200_000
    cluster_k: int | None = None
    seed: int = 0


def load_cotracker_feature_map(path: str | Path) -> dict[int, Any]:
    np = _require_numpy()
    with np.load(Path(path).expanduser().resolve()) as payload:
        track_ids = np.asarray(payload["track_ids"], dtype=np.int64)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    if embeddings.ndim != 2 or track_ids.ndim != 1 or embeddings.shape[0] != track_ids.shape[0]:
        raise ValueError("CoTracker feature NPZ must contain aligned track_ids [N] and embeddings [N,D].")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return {int(track_id): embeddings[index] for index, track_id in enumerate(track_ids.tolist())}


class CoTrackerFeatureProber:
    """Diagnostic-only evaluator for frozen CoTracker track representations."""

    def __init__(self, config: CoTrackerFeatureProbeConfig) -> None:
        self.config = config

    def probe(self) -> Path:
        np = _require_numpy()
        tracks_path = Path(self.config.tracks_path).expanduser().resolve()
        features_path = Path(self.config.features_npz).expanduser().resolve()
        tracks_artifact = load_json(tracks_path)
        feature_map = load_cotracker_feature_map(features_path)

        rows: list[tuple[int, int, Any]] = []
        for track in tracks_artifact.get("tracks", []):
            if not isinstance(track, dict) or self.config.label_field not in track:
                continue
            track_id = int(track.get("track_id", -1))
            if track_id in feature_map:
                rows.append((track_id, int(track[self.config.label_field]), feature_map[track_id]))
        if len(rows) < 2:
            raise ValueError(
                f"Need at least two aligned tracks containing label field '{self.config.label_field}'."
            )
        labels = np.asarray([row[1] for row in rows], dtype=np.int64)
        features = np.stack([row[2] for row in rows])
        unique_labels = sorted(int(value) for value in np.unique(labels).tolist())
        if len(unique_labels) < 2:
            raise ValueError(
                f"Feature probing requires at least two '{self.config.label_field}' values; found {unique_labels}."
            )

        pair_metrics = _pairwise_probe(features, labels, self.config.max_pairs, self.config.seed)
        cluster_k = int(self.config.cluster_k or len(unique_labels))
        assignments = _cosine_kmeans(features, cluster_k, self.config.seed)
        clustering = _clustering_metrics(assignments, labels)
        pca = _pca_2d(features)

        output_json = (
            Path(self.config.output_json).expanduser().resolve()
            if self.config.output_json is not None
            else features_path.with_name("cotracker_feature_probe.json")
        )
        payload = {
            "diagnostic_only": True,
            "tracks_path": str(tracks_path),
            "features_npz": str(features_path),
            "label_field": self.config.label_field,
            "track_count": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "gt_part_count": len(unique_labels),
            "cluster_k": cluster_k,
            "pairwise": pair_metrics,
            "hidden_only_clustering": clustering,
        }
        save_json(payload, output_json)

        if self.config.output_embedding_csv is not None:
            csv_path = Path(self.config.output_embedding_csv).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["track_id", "gt_part_id", "cluster_id", "pca_x", "pca_y"])
                writer.writeheader()
                for index, (track_id, label, _feature) in enumerate(rows):
                    writer.writerow(
                        {
                            "track_id": track_id,
                            "gt_part_id": label,
                            "cluster_id": int(assignments[index]),
                            "pca_x": float(pca[index, 0]),
                            "pca_y": float(pca[index, 1]),
                        }
                    )
        return output_json


def _pairwise_probe(features: Any, labels: Any, max_pairs: int, seed: int) -> dict[str, Any]:
    np = _require_numpy()
    row_indices, col_indices = np.triu_indices(features.shape[0], k=1)
    pair_count = int(row_indices.shape[0])
    if pair_count > max(1, int(max_pairs)):
        rng = np.random.default_rng(seed)
        selected = rng.choice(pair_count, size=max(1, int(max_pairs)), replace=False)
        row_indices = row_indices[selected]
        col_indices = col_indices[selected]
    scores = np.sum(features[row_indices] * features[col_indices], axis=1)
    same = labels[row_indices] == labels[col_indices]
    positive = scores[same]
    negative = scores[~same]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Sampled pairs must include both same-part and cross-part examples.")
    return {
        "sampled_pair_count": int(scores.size),
        "same_part_pair_count": int(positive.size),
        "cross_part_pair_count": int(negative.size),
        "same_part_cosine": _summary(positive),
        "cross_part_cosine": _summary(negative),
        "same_vs_cross_auc": _binary_auc(scores, same),
    }


def _summary(values: Any) -> dict[str, float]:
    np = _require_numpy()
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _binary_auc(scores: Any, positive_mask: Any) -> float:
    np = _require_numpy()
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=float)
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive_count = int(np.sum(positive_mask))
    negative_count = int(scores.shape[0] - positive_count)
    rank_sum = float(np.sum(ranks[positive_mask]))
    return float((rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count))


def _cosine_kmeans(features: Any, k: int, seed: int, max_iterations: int = 50) -> Any:
    np = _require_numpy()
    if k < 1 or k > features.shape[0]:
        raise ValueError(f"cluster_k must be in [1, {features.shape[0]}], got {k}.")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, features.shape[0]))
    centers = [features[first]]
    while len(centers) < k:
        similarities = features @ np.stack(centers).T
        distance = 1.0 - np.max(similarities, axis=1)
        centers.append(features[int(np.argmax(distance))])
    centers_array = np.stack(centers)
    assignments = np.full(features.shape[0], -1, dtype=np.int64)
    for _ in range(max_iterations):
        updated = np.argmax(features @ centers_array.T, axis=1)
        if np.array_equal(updated, assignments):
            assignments = updated
            break
        assignments = updated
        for cluster_id in range(k):
            members = features[assignments == cluster_id]
            if members.size == 0:
                continue
            center = np.mean(members, axis=0)
            centers_array[cluster_id] = center / max(float(np.linalg.norm(center)), 1e-12)
    return assignments


def _clustering_metrics(predicted: Any, labels: Any) -> dict[str, float]:
    np = _require_numpy()
    pred_values = sorted(int(value) for value in np.unique(predicted).tolist())
    gt_values = sorted(int(value) for value in np.unique(labels).tolist())
    overlap = np.zeros((len(pred_values), len(gt_values)), dtype=np.int64)
    for pred_index, pred_value in enumerate(pred_values):
        for gt_index, gt_value in enumerate(gt_values):
            overlap[pred_index, gt_index] = int(np.sum((predicted == pred_value) & (labels == gt_value)))
    cluster_sizes = np.sum(overlap, axis=1)
    gt_sizes = np.sum(overlap, axis=0)
    purity = float(np.sum(np.max(overlap, axis=1)) / max(1, int(np.sum(overlap))))
    mean_gt_coverage = float(np.mean(np.max(overlap, axis=0) / np.maximum(gt_sizes, 1)))
    return {
        "purity": purity,
        "mean_gt_coverage": mean_gt_coverage,
        "adjusted_rand_index": _adjusted_rand_index(overlap),
        "normalized_mutual_information": _normalized_mutual_information(overlap),
        "largest_cluster_ratio": float(np.max(cluster_sizes) / max(1, int(np.sum(cluster_sizes)))),
    }


def _adjusted_rand_index(overlap: Any) -> float:
    np = _require_numpy()
    combination = lambda values: np.sum(values * (values - 1) / 2.0)
    total = float(np.sum(overlap))
    if total < 2:
        return 1.0
    index = float(combination(overlap))
    row_pairs = float(combination(np.sum(overlap, axis=1)))
    col_pairs = float(combination(np.sum(overlap, axis=0)))
    total_pairs = total * (total - 1.0) / 2.0
    expected = row_pairs * col_pairs / total_pairs
    maximum = 0.5 * (row_pairs + col_pairs)
    return float((index - expected) / max(maximum - expected, 1e-12))


def _normalized_mutual_information(overlap: Any) -> float:
    np = _require_numpy()
    total = float(np.sum(overlap))
    probabilities = overlap / max(total, 1.0)
    row_probabilities = np.sum(probabilities, axis=1)
    col_probabilities = np.sum(probabilities, axis=0)
    mutual_information = 0.0
    for row_index, col_index in zip(*np.nonzero(probabilities)):
        value = float(probabilities[row_index, col_index])
        mutual_information += value * math.log(value / (row_probabilities[row_index] * col_probabilities[col_index]))
    row_entropy = -float(np.sum(row_probabilities[row_probabilities > 0] * np.log(row_probabilities[row_probabilities > 0])))
    col_entropy = -float(np.sum(col_probabilities[col_probabilities > 0] * np.log(col_probabilities[col_probabilities > 0])))
    return float(mutual_information / max(math.sqrt(row_entropy * col_entropy), 1e-12))


def _pca_2d(features: Any) -> Any:
    np = _require_numpy()
    centered = features - np.mean(features, axis=0, keepdims=True)
    left, singular_values, _right = np.linalg.svd(centered, full_matrices=False)
    coordinates = left[:, :2] * singular_values[:2]
    if coordinates.shape[1] == 1:
        coordinates = np.concatenate([coordinates, np.zeros((coordinates.shape[0], 1))], axis=1)
    return coordinates


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CoTracker feature probing requires NumPy.") from exc
    return np
