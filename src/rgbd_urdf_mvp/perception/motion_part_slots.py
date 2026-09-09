from __future__ import annotations

import json
import itertools
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from ..core.serialization import load_json, save_json
from .cotracker_features import load_cotracker_feature_map
from .motion_segmentation import (
    _best_model_assignment,
    _best_rigid_ransac_model,
    _model_track_residual,
    _reference_position,
    _track_motion_m,
    _trajectory_by_frame,
)
from .pairwise_affinity import load_pairwise_manifest
from .part_segmentation import collapse_fixed_connected_parts
from .quality_weights import observation_weight, track_quality_score


@dataclass(slots=True)
class MotionPartSlotTrainingConfig:
    manifest_path: str | Path
    output_dir: str | Path
    max_slots: int = 8
    hidden_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    rigid_loss_weight: float = 0.1
    dice_loss_weight: float = 0.5
    pairwise_loss_weight: float = 0.25
    existence_loss_weight: float = 0.25
    part_balanced_assignment: bool = True
    canonicalize_geometry: bool = True
    geometry_augmentation: bool = True
    geometry_noise_std: float = 0.003
    depth_bias_probability: float = 0.0
    depth_bias_std: float = 0.01
    sampled_depth_spike_probability: float = 0.0
    track_dropout_ratio: float = 0.1
    view_dropout_probability: float = 0.0
    max_dropped_views: int = 2
    pair_samples_per_object: int = 4096
    object_batch_size: int = 1
    data_loader_workers: int = 0
    topology_balanced_sampling: bool = True
    collapse_fixed_connected_labels: bool = False
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class MotionPartSlotInferenceConfig:
    tracks_path: str | Path
    features_npz: str | Path
    model_path: str | Path
    output_json: str | Path
    device: str = "auto"
    post_ransac_refine: bool = False
    ransac_iterations: int = 128
    ransac_inlier_threshold_m: float = 0.025
    ransac_min_inliers: int = 8
    seed: int = 0
    slot_existence_threshold: float = 0.5
    min_visible_frames: int = 0
    min_visible_ratio: float = 0.0
    max_trajectory_jump_m: float = 0.0


class MotionPartSlotTrainer:
    def __init__(self, config: MotionPartSlotTrainingConfig) -> None:
        self.config = config

    def train(self) -> Path:
        np = _require_numpy()
        torch = _require_torch()
        rows = load_pairwise_manifest(self.config.manifest_path)
        train_samples = _load_samples(
            rows, "train", canonicalize_geometry=self.config.canonicalize_geometry,
            collapse_fixed_connected_labels=self.config.collapse_fixed_connected_labels,
            workers=self.config.data_loader_workers,
        )
        validation_split = "val" if any(row["split"] == "val" for row in rows) else "train"
        val_samples = _load_samples(
            rows, validation_split, canonicalize_geometry=self.config.canonicalize_geometry,
            collapse_fixed_connected_labels=self.config.collapse_fixed_connected_labels,
            workers=self.config.data_loader_workers,
        )
        if not train_samples:
            raise ValueError("No motion-part slot training objects found.")
        max_gt_parts = max(len(set(sample["labels"].tolist())) for sample in train_samples + val_samples)
        if max_gt_parts > self.config.max_slots:
            raise ValueError(f"max_slots={self.config.max_slots} is smaller than GT part count {max_gt_parts}.")

        all_features = np.concatenate([sample["features"] for sample in train_samples], axis=0)
        mean = all_features.mean(axis=0).astype(np.float32)
        std = np.maximum(all_features.std(axis=0), 1e-6).astype(np.float32)
        device = _resolve_device(torch, self.config.device)
        torch.manual_seed(int(self.config.seed))
        model = _build_slot_model(
            torch,
            input_dim=all_features.shape[1],
            hidden_dim=self.config.hidden_dim,
            max_slots=self.config.max_slots,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            attention_heads=self.config.attention_heads,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        feature_mean = torch.from_numpy(mean).to(device)
        feature_std = torch.from_numpy(std).to(device)
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "motion_part_slots.pt"
        history: list[dict[str, float]] = []
        best_val = math.inf
        rng = random.Random(self.config.seed)

        for epoch in range(max(1, self.config.epochs)):
            epoch_started = time.perf_counter()
            model.train()
            order = _topology_balanced_object_order(
                train_samples, enabled=self.config.topology_balanced_sampling
            )
            rng.shuffle(order)
            # Neighboring track counts limit padding while preserving a fresh
            # randomized order between similarly sized objects each epoch.
            order.sort(key=lambda index: len(train_samples[index]["features"]))
            losses = []
            batch_size = max(1, int(self.config.object_batch_size))
            for batch_start in range(0, len(order), batch_size):
                batch_started = time.perf_counter()
                prepared = []
                for sample_index in order[batch_start : batch_start + batch_size]:
                    sample = train_samples[sample_index]
                    indices = _training_track_indices(
                        sample,
                        self.config.track_dropout_ratio,
                        rng,
                        np,
                        view_dropout_probability=self.config.view_dropout_probability,
                        max_dropped_views=self.config.max_dropped_views,
                    )
                    values = sample["features"][indices].copy()
                    if self.config.geometry_augmentation:
                        values = _augment_geometry_features(
                            values, sample["embedding_dim"], rng, np,
                            noise_std=float(self.config.geometry_noise_std),
                            depth_bias_probability=float(self.config.depth_bias_probability),
                            depth_bias_std=float(self.config.depth_bias_std),
                            sampled_depth_spike_probability=float(
                                self.config.sampled_depth_spike_probability
                            ),
                        )
                    prepared.append((sample, indices, values))
                max_tracks = max(len(item[2]) for item in prepared)
                feature_batch = torch.zeros(
                    len(prepared), max_tracks, prepared[0][2].shape[1], dtype=torch.float32, device=device
                )
                padding_mask = torch.ones(len(prepared), max_tracks, dtype=torch.bool, device=device)
                for batch_index, (_, _, values) in enumerate(prepared):
                    count = len(values)
                    feature_batch[batch_index, :count] = torch.from_numpy(values).to(device)
                    padding_mask[batch_index, :count] = False
                batch_logits, batch_existence = model(
                    (feature_batch - feature_mean) / feature_std,
                    padding_mask=padding_mask,
                )
                if not bool(torch.isfinite(batch_logits).all() and torch.isfinite(batch_existence).all()):
                    object_ids = [str(item[0]["object_id"]) for item in prepared]
                    raise FloatingPointError(f"Non-finite slot predictions for objects: {object_ids}")
                batch_losses = []
                batch_rows = []
                for batch_index, (sample, sample_indices, values) in enumerate(prepared):
                    logits = batch_logits[batch_index, : len(values)]
                    existence = batch_existence[batch_index]
                    labels = sample["labels"][sample_indices]
                    target_slots = _hungarian_slot_targets(logits, labels, self.config.max_slots, torch)
                    target = torch.as_tensor(target_slots, dtype=torch.long, device=device)
                    assignment_loss = _part_balanced_cross_entropy(
                        logits, target, torch, enabled=self.config.part_balanced_assignment
                    )
                    probabilities = torch.softmax(logits, dim=-1)
                    dice_loss = _matched_slot_dice_loss(probabilities, target, torch)
                    pairwise_loss = _sampled_pairwise_assignment_loss(
                        probabilities, target, sample["references"][sample_indices], torch, rng,
                        max_pairs=self.config.pair_samples_per_object,
                    )
                    existence_target = torch.zeros(self.config.max_slots, device=device)
                    existence_target[torch.unique(target)] = 1.0
                    existence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        existence, existence_target
                    )
                    rigid_loss = _weighted_rigid_replay_loss(
                        probabilities, sample["references"][sample_indices], sample["points"][sample_indices],
                        sample["visibility"][sample_indices], torch, device,
                    )
                    sample_loss = (
                        assignment_loss + self.config.dice_loss_weight * dice_loss
                        + self.config.pairwise_loss_weight * pairwise_loss
                        + self.config.existence_loss_weight * existence_loss
                        + self.config.rigid_loss_weight * rigid_loss
                    )
                    batch_losses.append(sample_loss)
                    batch_rows.append((sample_loss, assignment_loss, dice_loss, pairwise_loss, existence_loss, rigid_loss))
                loss = torch.stack(batch_losses).mean()
                if not bool(torch.isfinite(loss)):
                    object_ids = [str(item[0]["object_id"]) for item in prepared]
                    raise FloatingPointError(f"Non-finite slot training loss for objects: {object_ids}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
                for values in batch_rows:
                    losses.append({
                        key: float(value.detach().cpu()) for key, value in zip(
                            ("total", "assignment", "dice", "pairwise", "existence", "rigid"), values
                        )
                    })
                batch_number = batch_start // batch_size + 1
                if batch_number % 10 == 0 or batch_start + batch_size >= len(order):
                    elapsed = time.perf_counter() - epoch_started
                    print(
                        json.dumps({
                            "event": "slot_training_progress",
                            "epoch": epoch + 1,
                            "batch": batch_number,
                            "batches": math.ceil(len(order) / batch_size),
                            "objects_processed": min(batch_start + batch_size, len(order)),
                            "objects_total": len(order),
                            "batch_time_s": time.perf_counter() - batch_started,
                            "elapsed_s": elapsed,
                            "objects_per_s": min(batch_start + batch_size, len(order)) / max(elapsed, 1e-9),
                            "peak_cuda_memory_bytes": (
                                int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else None
                            ),
                        }),
                        flush=True,
                    )
            val_loss, val_accuracy = _evaluate_slots(
                model, val_samples, feature_mean, feature_std, self.config.max_slots, torch, device
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "epoch_time_s": time.perf_counter() - epoch_started,
                    "train_loss": _mean_loss(losses, "total"),
                    "train_assignment_loss": _mean_loss(losses, "assignment"),
                    "train_dice_loss": _mean_loss(losses, "dice"),
                    "train_pairwise_loss": _mean_loss(losses, "pairwise"),
                    "train_existence_loss": _mean_loss(losses, "existence"),
                    "train_rigid_loss": _mean_loss(losses, "rigid"),
                    "val_assignment_loss": val_loss,
                    "val_assignment_accuracy": val_accuracy,
                }
            )
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": int(all_features.shape[1]),
                        "hidden_dim": self.config.hidden_dim,
                        "max_slots": self.config.max_slots,
                        "encoder_layers": self.config.encoder_layers,
                        "decoder_layers": self.config.decoder_layers,
                        "attention_heads": self.config.attention_heads,
                        "feature_mean": torch.from_numpy(mean),
                        "feature_std": torch.from_numpy(std),
                        "canonicalize_geometry": self.config.canonicalize_geometry,
                        "collapse_fixed_connected_labels": self.config.collapse_fixed_connected_labels,
                    },
                    checkpoint_path,
                )
        save_json(
            {
                "model_path": str(checkpoint_path),
                "validation_split": validation_split,
                "train_objects": [sample["object_id"] for sample in train_samples],
                "validation_objects": [sample["object_id"] for sample in val_samples],
                "best_val_assignment_loss": best_val,
                "history": history,
                "config": {
                    field: str(value) if isinstance(value := getattr(self.config, field), Path) else value
                    for field in self.config.__dataclass_fields__
                },
            },
            output_dir / "training_summary.json",
        )
        return checkpoint_path


class MotionPartSlotInferencer:
    def __init__(self, config: MotionPartSlotInferenceConfig) -> None:
        self.config = config

    def infer(self) -> Path:
        import time

        from ..benchmarks.runtime_profile import hardware_snapshot, synchronize_device

        total_started = time.perf_counter()
        np = _require_numpy()
        torch = _require_torch()
        input_started = time.perf_counter()
        artifact = load_json(self.config.tracks_path)
        artifact, quality_filter = _filter_inference_tracks(artifact, self.config)
        checkpoint = torch.load(Path(self.config.model_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        sample = _sample_from_artifact(
            artifact,
            load_cotracker_feature_map(self.config.features_npz),
            object_id=str(artifact.get("object_instance_id", Path(self.config.tracks_path).parent.name)),
            require_labels=False,
            canonicalize_geometry=bool(checkpoint.get("canonicalize_geometry", False)),
        )
        input_s = time.perf_counter() - input_started
        device = _resolve_device(torch, self.config.device)
        model_started = time.perf_counter()
        model = _build_slot_model(
            torch,
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            max_slots=int(checkpoint["max_slots"]),
            encoder_layers=int(checkpoint["encoder_layers"]),
            decoder_layers=int(checkpoint["decoder_layers"]),
            attention_heads=int(checkpoint["attention_heads"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        synchronize_device(torch, device)
        model_load_s = time.perf_counter() - model_started
        features = torch.from_numpy(sample["features"]).to(device)
        mean = checkpoint["feature_mean"].to(device)
        std = checkpoint["feature_std"].to(device)
        synchronize_device(torch, device)
        forward_started = time.perf_counter()
        with torch.no_grad():
            logits, existence = model((features - mean) / std)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            existence_probabilities = torch.sigmoid(existence).cpu().numpy()
        synchronize_device(torch, device)
        forward_s = time.perf_counter() - forward_started
        postprocess_started = time.perf_counter()
        active_slots = np.flatnonzero(existence_probabilities >= self.config.slot_existence_threshold).tolist()
        if not active_slots:
            active_slots = [int(np.argmax(existence_probabilities))]
        initial = np.asarray(active_slots, dtype=np.int64)[
            np.argmax(probabilities[:, active_slots], axis=1)
        ].astype(int).tolist()
        refined = list(initial)
        refinement = None
        if self.config.post_ransac_refine:
            refined, refinement = _post_ransac_refine(sample["tracks"], initial, self.config)
        slot_ids = _compact_ids(refined)
        relabeled_tracks = []
        for index, track in enumerate(sample["tracks"]):
            row = dict(track)
            row["original_part_id"] = int(track.get("original_part_id", track.get("part_id", 0)))
            row["part_id"] = slot_ids[index]
            row["part_name"] = f"motion_slot_{slot_ids[index]}"
            row["slot_initial_id"] = int(initial[index])
            row["slot_confidence"] = float(probabilities[index, initial[index]])
            relabeled_tracks.append(row)
        payload = dict(artifact)
        payload.update(
            {
                "estimator": "detr-motion-part-slot-decoder",
                "tracks": relabeled_tracks,
                "motion_segmentation": {
                    "mode": "motion-part-slots",
                    "model_path": str(Path(self.config.model_path).expanduser().resolve()),
                    "part_count": len(set(slot_ids)),
                    "slot_existence_probabilities": [float(value) for value in existence_probabilities],
                    "active_slots": active_slots,
                    "slot_existence_threshold": self.config.slot_existence_threshold,
                    "post_ransac_refine": self.config.post_ransac_refine,
                    "refinement": refinement,
                    "track_quality_filter": quality_filter,
                },
            }
        )
        if all("original_part_id" in track for track in sample["tracks"]):
            payload["slot_segmentation_evaluation"] = evaluate_slot_assignments(slot_ids, sample["labels"])
        output = Path(self.config.output_json).expanduser().resolve()
        payload["runtime_profile"] = {
            "scope": "motion_part_slot_inference",
            "input_and_feature_assembly_s": input_s,
            "model_load_s": model_load_s,
            "accelerator_forward_s": forward_s,
            "postprocess_before_serialization_s": None,
            "total_before_serialization_s": None,
            "device": device,
            "track_count": len(sample["tracks"]),
            "hardware": hardware_snapshot(torch),
        }
        payload["runtime_profile"]["postprocess_before_serialization_s"] = (
            time.perf_counter() - postprocess_started
        )
        payload["runtime_profile"]["total_before_serialization_s"] = (
            time.perf_counter() - total_started
        )
        save_json(payload, output)
        return output


def _filter_inference_tracks(
    artifact: dict[str, Any], config: MotionPartSlotInferenceConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optionally remove short-lived or discontinuous tracks before slot inference."""
    np = _require_numpy()
    tracks = [track for track in artifact.get("tracks", []) if isinstance(track, dict)]
    frame_count = max(1, int(artifact.get("frame_count", 0) or 0))
    kept = []
    dropped = {"too_few_visible_frames": 0, "low_visible_ratio": 0, "trajectory_jump": 0}
    max_jump = max(0.0, float(config.max_trajectory_jump_m))
    for track in tracks:
        trajectory = _trajectory_by_frame(track)
        visible_count = len(trajectory)
        reason = None
        if visible_count < max(0, int(config.min_visible_frames)):
            reason = "too_few_visible_frames"
        elif visible_count / frame_count < max(0.0, float(config.min_visible_ratio)):
            reason = "low_visible_ratio"
        elif max_jump > 0.0 and len(trajectory) >= 2:
            points = np.asarray([trajectory[index] for index in sorted(trajectory)], dtype=np.float64)
            if bool(np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > max_jump)):
                reason = "trajectory_jump"
        if reason is None:
            kept.append(track)
        else:
            dropped[reason] += 1
    if not kept:
        raise ValueError("Track quality filtering removed every track before slot inference.")
    filtered = dict(artifact)
    filtered["tracks"] = kept
    report = {
        "enabled": any((config.min_visible_frames, config.min_visible_ratio, config.max_trajectory_jump_m)),
        "input_track_count": len(tracks),
        "kept_track_count": len(kept),
        "dropped_track_count": len(tracks) - len(kept),
        "dropped_by_reason": dropped,
        "min_visible_frames": int(config.min_visible_frames),
        "min_visible_ratio": float(config.min_visible_ratio),
        "max_trajectory_jump_m": float(config.max_trajectory_jump_m),
    }
    return filtered, report


def _load_samples(
    rows: list[dict[str, str]], split: str, *, canonicalize_geometry: bool = True,
    collapse_fixed_connected_labels: bool = False, workers: int = 0,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split]

    def load(row: dict[str, str]) -> dict[str, Any]:
        return _sample_from_artifact(
            load_json(row["tracks_path"]),
            load_cotracker_feature_map(row["features_npz"]),
            object_id=row["object_id"],
            require_labels=True,
            canonicalize_geometry=canonicalize_geometry,
            collapse_fixed_connected_labels=collapse_fixed_connected_labels,
        )
    if int(workers) <= 1:
        return [load(row) for row in selected]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(load, selected))


def _sample_from_artifact(
    artifact: dict[str, Any], feature_map: dict[int, Any], *, object_id: str, require_labels: bool,
    canonicalize_geometry: bool = True, collapse_fixed_connected_labels: bool = False,
) -> dict[str, Any]:
    np = _require_numpy()
    tracks = [
        track for track in artifact.get("tracks", [])
        if isinstance(track, dict)
        and int(track.get("track_id", -1)) in feature_map
        and _trajectory_by_frame(track)
        and (not require_labels or "original_part_id" in track or "part_id" in track)
    ]
    if not tracks:
        raise ValueError(f"No aligned tracks/features for {object_id}.")
    max_frame = max(max(_trajectory_by_frame(track)) for track in tracks)
    points = np.zeros((len(tracks), max_frame + 1, 3), dtype=np.float32)
    visibility = np.zeros((len(tracks), max_frame + 1), dtype=np.float32)
    observation_quality = np.zeros((len(tracks), max_frame + 1), dtype=np.float32)
    track_quality = np.ones((len(tracks),), dtype=np.float32)
    features = []
    references_array = np.asarray([_reference_position(track) for track in tracks], dtype=np.float32)
    center = np.median(references_array, axis=0)
    scale = max(float(np.linalg.norm(np.max(references_array, axis=0) - np.min(references_array, axis=0))), 1e-3)
    for index, track in enumerate(tracks):
        trajectory = _trajectory_by_frame(track)
        samples_by_frame = {
            int(sample.get("frame_index", -1)): sample
            for sample in track.get("samples", []) or []
            if isinstance(sample, dict)
        }
        track_quality[index] = track_quality_score(track)
        reference = np.asarray(_reference_position(track), dtype=np.float32)
        for frame, point in trajectory.items():
            points[index, frame] = point
            visibility[index, frame] = 1.0
            observation_quality[index, frame] = observation_weight(
                track, samples_by_frame.get(int(frame), {})
            )
        frames = sorted(trajectory)
        endpoint = np.asarray(trajectory[frames[-1]], dtype=np.float32) - np.asarray(trajectory[frames[0]], dtype=np.float32)
        sampled = []
        for fraction in np.linspace(0.0, 1.0, 8):
            frame = frames[min(len(frames) - 1, int(round(fraction * (len(frames) - 1))))]
            sampled.extend((np.asarray(trajectory[frame], dtype=np.float32) - reference).tolist())
        if canonicalize_geometry:
            reference_feature = (reference - center) / scale
            endpoint_feature = endpoint / scale
            sampled_feature = (np.asarray(sampled, dtype=np.float32) / scale).tolist()
            motion_feature = _track_motion_m(track) / scale
        else:
            reference_feature = reference
            endpoint_feature = endpoint
            sampled_feature = sampled
            motion_feature = _track_motion_m(track)
        embedding = np.asarray(feature_map[int(track["track_id"])], dtype=np.float32)
        scalar = np.asarray(
            [*reference_feature.tolist(), *endpoint_feature.tolist(), motion_feature, len(frames) / (max_frame + 1), *sampled_feature],
            dtype=np.float32,
        )
        features.append(np.concatenate([embedding, scalar]))
    original_raw_labels = [
        int(track.get("original_part_id", track.get("part_id", 0))) for track in tracks
    ]
    raw_labels = list(original_raw_labels)
    label_ontology = "raw-simulator-parts"
    if collapse_fixed_connected_labels:
        segmentation = artifact.get("original_part_segmentation")
        if not isinstance(segmentation, dict):
            episode_path = artifact.get("input_episode_path")
            if episode_path:
                episode = load_json(Path(str(episode_path)).expanduser().resolve())
                segmentation = episode.get("metadata", {}).get("part_segmentation")
        if not isinstance(segmentation, dict):
            raise ValueError(
                f"Cannot collapse fixed-connected labels for {object_id}: "
                "neither original_part_segmentation nor episode metadata is available."
            )
        collapsed = collapse_fixed_connected_parts(segmentation)
        raw_to_kinematic = {
            int(raw): int(part_id)
            for raw, part_id in collapsed.get("raw_part_to_part_id", {}).items()
        }
        missing = sorted(set(raw_labels) - set(raw_to_kinematic))
        if missing:
            raise ValueError(
                f"Cannot collapse fixed-connected labels for {object_id}: "
                f"raw part IDs are absent from metadata: {missing}."
            )
        raw_labels = [raw_to_kinematic[label] for label in raw_labels]
        label_ontology = "maximal-fixed-joint-connected-components"
    label_map = {label: index for index, label in enumerate(sorted(set(raw_labels)))}
    raw_part_to_label = {
        raw_part: label_map[training_part]
        for raw_part, training_part in zip(original_raw_labels, raw_labels, strict=True)
    }
    view_indices = np.asarray(
        [int(track.get("view_index", 0)) for track in tracks], dtype=np.int64
    )
    view_motion_m = {
        int(view): float(np.median([
            _track_motion_m(tracks[index])
            for index in np.flatnonzero(view_indices == view).tolist()
        ]))
        for view in sorted(set(view_indices.tolist()))
    }
    return {
        "object_id": object_id,
        "tracks": tracks,
        "features": np.stack(features).astype(np.float32),
        "labels": np.asarray([label_map[label] for label in raw_labels], dtype=np.int64),
        "label_ontology": label_ontology,
        "points": points,
        "references": np.asarray([_reference_position(track) for track in tracks], dtype=np.float32),
        "reference_frames": np.asarray(
            [int(track.get("query_frame_index", 0)) for track in tracks], dtype=np.int64
        ),
        "view_indices": view_indices,
        "view_motion_m": view_motion_m,
        "visibility": visibility,
        "observation_quality": observation_quality,
        "track_quality": track_quality,
        "embedding_dim": int(len(np.asarray(feature_map[int(tracks[0]["track_id"])]))),
        "canonical_center_m": center.astype(np.float32),
        "canonical_scale_m": scale,
        "raw_part_to_label": raw_part_to_label,
    }


def _build_slot_model(torch: Any, *, input_dim: int, hidden_dim: int, max_slots: int, encoder_layers: int, decoder_layers: int, attention_heads: int) -> Any:
    class MotionPartSlotModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
            encoder_layer = torch.nn.TransformerEncoderLayer(hidden_dim, attention_heads, hidden_dim * 4, batch_first=True, norm_first=True)
            decoder_layer = torch.nn.TransformerDecoderLayer(hidden_dim, attention_heads, hidden_dim * 4, batch_first=True, norm_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, encoder_layers)
            self.decoder = torch.nn.TransformerDecoder(decoder_layer, decoder_layers)
            self.queries = torch.nn.Parameter(torch.randn(max_slots, hidden_dim) * 0.02)
            self.existence = torch.nn.Linear(hidden_dim, 1)

        def forward(
            self, features: Any, *, return_slots: bool = False, padding_mask: Any | None = None
        ) -> tuple[Any, ...]:
            single = features.ndim == 2
            if single:
                features = features.unsqueeze(0)
                if padding_mask is not None and padding_mask.ndim == 1:
                    padding_mask = padding_mask.unsqueeze(0)
            memory = self.encoder(self.projection(features), src_key_padding_mask=padding_mask)
            queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
            slots = self.decoder(queries, memory, memory_key_padding_mask=padding_mask)
            logits = torch.einsum("bnh,bkh->bnk", memory, slots) / math.sqrt(hidden_dim)
            existence = self.existence(slots).squeeze(-1)
            if single:
                logits = logits.squeeze(0)
                existence = existence.squeeze(0)
                slots = slots.squeeze(0)
            if return_slots:
                return logits, existence, slots
            return logits, existence
    return MotionPartSlotModel()


def _hungarian_slot_targets(logits: Any, labels: Any, max_slots: int, torch: Any) -> list[int]:
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Hungarian slot matching received non-finite logits.")
    probabilities = torch.softmax(logits.detach(), dim=-1).cpu()
    label_values = sorted(set(int(value) for value in labels.tolist()))
    cost = []
    for label in label_values:
        mask = torch.as_tensor(labels == label)
        cost.append([-float(probabilities[mask, slot].mean()) for slot in range(max_slots)])
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - training environments include scipy
        raise RuntimeError("Motion-part slot Hungarian matching requires scipy.") from exc
    row_indices, slot_indices = linear_sum_assignment(cost)
    mapping = {label_values[int(row)]: int(slot) for row, slot in zip(row_indices, slot_indices)}
    return [mapping[int(label)] for label in labels.tolist()]


def _part_balanced_cross_entropy(logits: Any, target: Any, torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return torch.nn.functional.cross_entropy(logits, target)
    counts = torch.bincount(target, minlength=logits.shape[-1]).float().clamp_min(1.0)
    weights = counts.sum() / (counts * max(1, int((counts > 0).sum())))
    per_track = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    return (per_track * weights[target]).sum() / weights[target].sum().clamp_min(1e-6)


def _matched_slot_dice_loss(probabilities: Any, target: Any, torch: Any) -> Any:
    losses = []
    for slot in torch.unique(target):
        expected = (target == slot).float()
        predicted = probabilities[:, slot]
        intersection = (predicted * expected).sum()
        losses.append(1.0 - (2.0 * intersection + 1e-6) / (predicted.sum() + expected.sum() + 1e-6))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0


def _sampled_pairwise_assignment_loss(
    probabilities: Any,
    target: Any,
    references: Any,
    torch: Any,
    rng: random.Random,
    *,
    max_pairs: int,
) -> Any:
    """Train same-part affinity, emphasizing spatially local cross-part boundaries."""
    count = int(target.shape[0])
    if count < 2 or max_pairs <= 0:
        return probabilities.sum() * 0.0
    reference = torch.as_tensor(references, dtype=torch.float32, device=probabilities.device)
    distances = torch.cdist(reference, reference)
    diagonal = torch.eye(count, dtype=torch.bool, device=probabilities.device)
    same = target[:, None] == target[None, :]
    positive_pairs = torch.nonzero(same & ~diagonal, as_tuple=False)
    negative_pairs = torch.nonzero(~same & ~diagonal, as_tuple=False)
    if negative_pairs.numel():
        negative_distance = distances[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_pairs = negative_pairs[torch.argsort(negative_distance)]
    half = max(1, max_pairs // 2)
    positive_pairs = _subsample_pair_rows(positive_pairs, half, rng, torch)
    negative_pairs = negative_pairs[:half]
    pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
    if not pairs.numel():
        return probabilities.sum() * 0.0
    predicted_same = (probabilities[pairs[:, 0]] * probabilities[pairs[:, 1]]).sum(-1).clamp(1e-5, 1.0 - 1e-5)
    expected_same = (target[pairs[:, 0]] == target[pairs[:, 1]]).float()
    return torch.nn.functional.binary_cross_entropy(predicted_same, expected_same)


def _subsample_pair_rows(rows: Any, limit: int, rng: random.Random, torch: Any) -> Any:
    if int(rows.shape[0]) <= limit:
        return rows
    indices = rng.sample(range(int(rows.shape[0])), limit)
    return rows[torch.as_tensor(indices, dtype=torch.long, device=rows.device)]


def _training_track_indices(
    sample: dict[str, Any], dropout_ratio: float, rng: random.Random, np: Any, *,
    view_dropout_probability: float = 0.0, max_dropped_views: int = 2,
) -> Any:
    """Drop tracks per GT part so augmentation never removes a complete small part."""
    labels = sample["labels"]
    eligible = np.arange(len(labels), dtype=np.int64)
    views = np.asarray(sample.get("view_indices", np.zeros(len(labels))), dtype=np.int64)
    unique_views = sorted(set(int(view) for view in views.tolist()))
    if (
        len(unique_views) > 1
        and max_dropped_views > 0
        and rng.random() < max(0.0, min(float(view_dropout_probability), 1.0))
    ):
        drop_count = rng.randint(1, min(int(max_dropped_views), len(unique_views) - 1))
        candidates = []
        for dropped in itertools.combinations(unique_views, drop_count):
            kept = [view for view in unique_views if view not in dropped]
            selected = np.flatnonzero(np.isin(views, kept))
            if all(int(np.sum(labels[selected] == label)) >= 3 for label in set(labels.tolist())):
                candidates.append((kept, selected))
        if len(unique_views) - drop_count == 1 and candidates:
            per_view_motion = sample.get("view_motion_m")
            if not isinstance(per_view_motion, dict):
                per_view_motion = {
                    view: float(np.median([
                        _track_motion_m(sample["tracks"][index])
                        for index in np.flatnonzero(views == view).tolist()
                    ]))
                    for view in unique_views
                }
            per_view_motion = {
                int(view): float(value) for view, value in per_view_motion.items()
            }
            minimum_motion = min(per_view_motion.values())
            non_minimum = [
                row for row in candidates
                if per_view_motion[row[0][0]] > minimum_motion + 1e-9
            ]
            if non_minimum:
                candidates = non_minimum
        if candidates:
            _, eligible = rng.choice(candidates)
    kept = []
    for label in sorted(set(labels.tolist())):
        indices = eligible[labels[eligible] == label].tolist()
        rng.shuffle(indices)
        keep_count = max(3, int(round(len(indices) * (1.0 - max(0.0, min(dropout_ratio, 0.8))))))
        kept.extend(indices[: min(len(indices), keep_count)])
    return np.asarray(sorted(kept), dtype=np.int64)


def _topology_balanced_object_order(samples: list[dict[str, Any]], *, enabled: bool) -> list[int]:
    if not enabled or not samples:
        return list(range(len(samples)))
    part_counts = [len(set(sample["labels"].tolist())) for sample in samples]
    baseline = max(1, min(part_counts))
    order = []
    for index, part_count in enumerate(part_counts):
        # Complex topologies are rare in the current simulator split. Repeating
        # them prevents the decoder from learning a refrigerator-is-three-parts
        # shortcut without changing any per-track supervision.
        order.extend([index] * max(1, int(math.ceil(part_count / baseline))))
    return order


def _augment_geometry_features(
    features: Any, embedding_dim: int, rng: random.Random, np: Any, *,
    noise_std: float = 0.003, depth_bias_probability: float = 0.0,
    depth_bias_std: float = 0.01, sampled_depth_spike_probability: float = 0.0,
) -> Any:
    """Apply a shared random rigid transform to canonical geometric token channels."""
    angle = rng.uniform(-math.pi, math.pi)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    geometry = features[:, embedding_dim:]
    geometry[:, 0:3] = geometry[:, 0:3] @ rotation.T
    geometry[:, 3:6] = geometry[:, 3:6] @ rotation.T
    sampled = geometry[:, 8:].reshape(len(features), -1, 3)
    geometry[:, 8:] = (sampled @ rotation.T).reshape(len(features), -1)
    generator = np.random.default_rng(rng.randrange(2**32))
    if noise_std > 0.0 and rng.random() < 0.5:
        geometry[:, 0:6] += generator.normal(0.0, noise_std, geometry[:, 0:6].shape)
    if rng.random() < depth_bias_probability:
        count = max(1, len(features) // 4)
        center = rng.randrange(len(features))
        distances = np.linalg.norm(geometry[:, 0:3] - geometry[center, 0:3], axis=-1)
        selected = np.argsort(distances)[:count]
        bias = float(generator.normal(0.0, depth_bias_std))
        geometry[selected, 2] += bias
        geometry[selected, 5] += bias
        selected_samples = geometry[selected, 8:].reshape(len(selected), -1, 3).copy()
        selected_samples[:, :, 2] += bias
        geometry[selected, 8:] = selected_samples.reshape(len(selected), -1)
    if sampled_depth_spike_probability > 0.0:
        sampled = geometry[:, 8:].reshape(len(features), -1, 3)
        mask = generator.random(sampled.shape[:2]) < sampled_depth_spike_probability
        sampled[:, :, 2] += mask * generator.normal(0.0, 3.0 * max(depth_bias_std, noise_std), mask.shape)
    return features


def _mean_loss(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row[key] for row in rows) / max(1, len(rows)))


def _weighted_rigid_replay_loss(
    probabilities: Any, references: Any, points: Any, visibility: Any, torch: Any, device: str
) -> Any:
    # torch.linalg.svd has no native MPS autograd kernel. Move this small 3x3
    # physical fitting calculation explicitly to CPU while preserving the
    # assignment-probability gradient across the device copy.
    compute_device = "cpu" if str(device).startswith("mps") else device
    probability_tensor = probabilities.to(compute_device)
    point_tensor = torch.as_tensor(points, dtype=torch.float32, device=compute_device)
    reference = torch.as_tensor(references, dtype=torch.float32, device=compute_device)
    visible_tensor = torch.as_tensor(visibility, dtype=torch.float32, device=compute_device)
    spatial_scale_sq = torch.sum((reference.max(0).values - reference.min(0).values) ** 2).clamp_min(1e-6)
    if point_tensor.shape[1] <= 1:
        return probability_tensor.sum() * 0.0

    # [K,T,N], excluding the reference frame. All slot/frame Kabsch fits are
    # solved in one batched SVD instead of hundreds of Python-level GPU calls.
    target_points = point_tensor[:, 1:].transpose(0, 1)
    frame_visibility = visible_tensor[:, 1:].transpose(0, 1)
    weights = probability_tensor.transpose(0, 1)[:, None, :] * frame_visibility[None, :, :]
    totals = weights.sum(-1).clamp_min(1e-6)
    valid = (weights > 1e-3).sum(-1) >= 3
    source_centers = torch.einsum("ktn,nd->ktd", weights, reference) / totals[..., None]
    target_centers = torch.einsum("ktn,tnd->ktd", weights, target_points) / totals[..., None]
    source = reference[None, None, :, :] - source_centers[:, :, None, :]
    target = target_points[None, :, :, :] - target_centers[:, :, None, :]
    covariance = torch.einsum("ktn,ktni,ktnj->ktij", weights, source, target)
    # The optimal Kabsch rotation is an inner fit. Differentiating through SVD
    # is numerically unstable for planar or nearly static parts because their
    # covariance has repeated singular values. Keep gradients through the
    # weighted centers and replay residual, but treat the fitted rotation as a
    # stop-gradient target.
    with torch.no_grad():
        u_matrix, _, vh_matrix = torch.linalg.svd(covariance.detach())
        rotation = vh_matrix.transpose(-1, -2) @ u_matrix.transpose(-1, -2)
        negative = torch.det(rotation) < 0
        if bool(negative.any()):
            correction = torch.eye(3, dtype=rotation.dtype, device=compute_device).expand_as(rotation).clone()
            correction[..., -1, -1] = torch.where(
                negative, -torch.ones_like(totals), torch.ones_like(totals)
            )
            rotation = vh_matrix.transpose(-1, -2) @ correction @ u_matrix.transpose(-1, -2)
    prediction = torch.matmul(source, rotation.transpose(-1, -2)) + target_centers[:, :, None, :]
    errors = (prediction - target_points[None, :, :, :]).square().sum(-1)
    per_fit = (weights * errors).sum(-1) / totals / spatial_scale_sq
    loss = per_fit[valid].mean() if bool(valid.any()) else probability_tensor.sum() * 0.0
    return loss.to(device)


def _evaluate_slots(model: Any, samples: list[dict[str, Any]], mean: Any, std: Any, max_slots: int, torch: Any, device: str) -> tuple[float, float]:
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for sample in samples:
            features = torch.from_numpy(sample["features"]).to(device)
            logits, _ = model((features - mean) / std)
            targets = torch.as_tensor(_hungarian_slot_targets(logits, sample["labels"], max_slots, torch), device=device)
            losses.append(float(torch.nn.functional.cross_entropy(logits, targets).cpu()))
            accuracies.append(float((logits.argmax(-1) == targets).float().mean().cpu()))
    return sum(losses) / max(1, len(losses)), sum(accuracies) / max(1, len(accuracies))


def evaluate_slot_assignments(predicted: Any, labels: Any) -> dict[str, Any]:
    """Evaluate arbitrary slot IDs without allowing one slot to cover many GT parts."""
    np = _require_numpy()
    predicted = np.asarray(predicted, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    pred_values = sorted(int(value) for value in np.unique(predicted))
    gt_values = sorted(int(value) for value in np.unique(labels))
    overlap = np.zeros((len(pred_values), len(gt_values)), dtype=np.int64)
    for pred_index, pred_value in enumerate(pred_values):
        for gt_index, gt_value in enumerate(gt_values):
            overlap[pred_index, gt_index] = int(np.sum((predicted == pred_value) & (labels == gt_value)))
    pred_sizes = overlap.sum(1)
    gt_sizes = overlap.sum(0)
    cluster_purities = np.max(overlap, axis=1) / np.maximum(pred_sizes, 1)
    gt_coverages = np.max(overlap, axis=0) / np.maximum(gt_sizes, 1)
    union = pred_sizes[:, None] + gt_sizes[None, :] - overlap
    iou = overlap / np.maximum(union, 1)
    matching = _best_rectangular_matching(iou)
    matched_rows = []
    for pred_index, gt_index in matching:
        intersection = int(overlap[pred_index, gt_index])
        precision = intersection / max(1, int(pred_sizes[pred_index]))
        recall = intersection / max(1, int(gt_sizes[gt_index]))
        matched_rows.append(
            {
                "pred_slot": pred_values[pred_index],
                "gt_part": gt_values[gt_index],
                "intersection": intersection,
                "iou": float(iou[pred_index, gt_index]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
            }
        )
    best_pred_per_gt = np.argmax(overlap, axis=0) if len(pred_values) else np.asarray([], dtype=int)
    gt_collision_count = int(len(best_pred_per_gt) - len(set(best_pred_per_gt.tolist())))
    pair_tp = pair_fp = pair_fn = 0
    for first in range(len(labels)):
        gt_same = labels[first + 1 :] == labels[first]
        pred_same = predicted[first + 1 :] == predicted[first]
        pair_tp += int(np.sum(gt_same & pred_same))
        pair_fp += int(np.sum(~gt_same & pred_same))
        pair_fn += int(np.sum(gt_same & ~pred_same))
    pair_precision = pair_tp / max(1, pair_tp + pair_fp)
    pair_recall = pair_tp / max(1, pair_tp + pair_fn)
    total_pairs = len(labels) * (len(labels) - 1) // 2
    pair_tn = max(0, total_pairs - pair_tp - pair_fp - pair_fn)
    rand_index = (pair_tp + pair_tn) / total_pairs if total_pairs else 1.0
    matched_gt = {row["gt_part"] for row in matched_rows}
    matched_pred = {row["pred_slot"] for row in matched_rows}
    gt_denominator = max(1, len(gt_values))
    return {
        "predicted_part_count": len(pred_values),
        "gt_part_count": len(gt_values),
        "part_count_exact": len(pred_values) == len(gt_values),
        "mean_cluster_purity": float(np.mean(cluster_purities)),
        "weighted_cluster_purity": float(np.sum(np.max(overlap, axis=1)) / max(1, len(labels))),
        "mean_gt_coverage": float(np.mean(gt_coverages)),
        "one_to_one_mean_iou": float(sum(row["iou"] for row in matched_rows) / gt_denominator),
        "one_to_one_mean_f1": float(sum(row["f1"] for row in matched_rows) / gt_denominator),
        "one_to_one_mean_recall": float(sum(row["recall"] for row in matched_rows) / gt_denominator),
        "pairwise_same_part_precision": float(pair_precision),
        "pairwise_same_part_recall": float(pair_recall),
        "pairwise_same_part_f1": float(2.0 * pair_precision * pair_recall / max(pair_precision + pair_recall, 1e-12)),
        "rand_index": float(rand_index),
        "adjusted_rand_index": _adjusted_rand_index(overlap),
        "normalized_mutual_information": _normalized_mutual_information(overlap),
        "gt_collision_count": gt_collision_count,
        "unmatched_gt_part_count": len(set(gt_values) - matched_gt),
        "undersegmented_gt_part_count": gt_collision_count,
        "oversegmented_pred_slot_count": len(set(pred_values) - matched_pred),
        "largest_cluster_ratio": float(np.max(pred_sizes) / max(1, len(labels))),
        "overlap_matrix": overlap.tolist(),
        "iou_matrix": iou.tolist(),
        "matching": matched_rows,
    }


def _adjusted_rand_index(overlap: Any) -> float:
    np = _require_numpy()
    choose_two = lambda values: np.sum(values * (values - 1) / 2.0)
    total = float(np.sum(overlap))
    if total < 2:
        return 1.0
    index = float(choose_two(overlap))
    row_pairs = float(choose_two(np.sum(overlap, axis=1)))
    column_pairs = float(choose_two(np.sum(overlap, axis=0)))
    total_pairs = total * (total - 1.0) / 2.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    return float((index - expected) / denominator) if abs(denominator) > 1e-12 else 1.0


def _normalized_mutual_information(overlap: Any) -> float:
    np = _require_numpy()
    total = float(np.sum(overlap))
    probabilities = overlap / max(total, 1.0)
    row_probabilities = np.sum(probabilities, axis=1)
    column_probabilities = np.sum(probabilities, axis=0)
    mutual_information = 0.0
    for row_index, column_index in zip(*np.nonzero(probabilities)):
        value = float(probabilities[row_index, column_index])
        mutual_information += value * math.log(
            value / (row_probabilities[row_index] * column_probabilities[column_index])
        )
    row_nonzero = row_probabilities[row_probabilities > 0]
    column_nonzero = column_probabilities[column_probabilities > 0]
    # Floating-point summation can produce a tiny negative entropy for a
    # constant partition (for example -2e-16). Clamp the mathematical lower
    # bound before taking the geometric mean.
    row_entropy = max(0.0, -float(np.sum(row_nonzero * np.log(row_nonzero))))
    column_entropy = max(0.0, -float(np.sum(column_nonzero * np.log(column_nonzero))))
    denominator = math.sqrt(row_entropy * column_entropy)
    if denominator > 1e-12:
        return float(mutual_information / denominator)
    # A constant prediction contains no information about a non-constant GT
    # partition (and vice versa). Only two identical trivial partitions have
    # perfect NMI when both entropies are zero.
    return 1.0 if row_entropy <= 1e-12 and column_entropy <= 1e-12 else 0.0


def _best_rectangular_matching(scores: Any) -> list[tuple[int, int]]:
    rows, columns = scores.shape
    if not rows or not columns:
        return []
    matched_rows, matched_columns = linear_sum_assignment(-scores)
    return list(zip(matched_rows.tolist(), matched_columns.tolist(), strict=True))


def _post_ransac_refine(tracks: list[dict[str, Any]], assignments: list[int], config: MotionPartSlotInferenceConfig) -> tuple[list[int], dict[str, Any]]:
    references = [_reference_position(track) for track in tracks]
    trajectories = [_trajectory_by_frame(track) for track in tracks]
    rng = random.Random(config.seed)
    models, rows = [], []
    for slot in sorted(set(assignments)):
        indices = [index for index, value in enumerate(assignments) if value == slot]
        if len(indices) < max(3, config.ransac_min_inliers):
            continue
        model = _best_rigid_ransac_model(
            candidate_indices=indices,
            references=references,
            trajectories=trajectories,
            iterations=config.ransac_iterations,
            sample_size=4,
            min_common_frames=3,
            inlier_threshold_m=config.ransac_inlier_threshold_m,
            spatial_link_m=float("inf"),
            rng=rng,
        )
        if model is None or len(model["inlier_indices"]) < config.ransac_min_inliers:
            continue
        model["model_index"] = len(models)
        model["initial_slot"] = slot
        models.append(model)
        rows.append({"initial_slot": slot, "input_count": len(indices), "inlier_count": len(model["inlier_indices"])})
    if not models:
        return assignments, {"models": [], "changed_track_count": 0}
    refined = list(assignments)
    for index in range(len(tracks)):
        model_index, residual = _best_model_assignment(index, models, references, trajectories)
        if model_index is None or residual is None or residual > config.ransac_inlier_threshold_m:
            continue
        best_slot = int(models[model_index]["initial_slot"])
        own_model = next((model for model in models if int(model["initial_slot"]) == assignments[index]), None)
        own_residual = None
        if own_model is not None:
            own_residual = _model_track_residual(
                index,
                own_model["transforms"],
                references,
                trajectories,
                min_common_frames=2,
            )
        # RANSAC is a conservative correction: never move a track merely because
        # another valid model is marginally better than its predicted slot.
        if own_residual is None or residual < 0.7 * own_residual:
            refined[index] = best_slot
    return refined, {"models": rows, "changed_track_count": sum(a != b for a, b in zip(assignments, refined))}


def _compact_ids(assignments: list[int]) -> list[int]:
    mapping = {value: index + 1 for index, value in enumerate(sorted(set(assignments)))}
    return [mapping[value] for value in assignments]


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _require_numpy() -> Any:
    import numpy as np
    return np


def _require_torch() -> Any:
    import torch
    return torch
