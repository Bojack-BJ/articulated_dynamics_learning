from __future__ import annotations

import json
import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from ..core.serialization import load_json, save_json
from ..perception.cotracker_features import load_cotracker_feature_map
from ..perception.motion_part_slots import (
    _build_slot_model,
    _hungarian_slot_targets,
    _matched_slot_dice_loss,
    _part_balanced_cross_entropy,
    _require_numpy,
    _require_torch,
    _sampled_pairwise_assignment_loss,
    _weighted_rigid_replay_loss,
    _resolve_device,
    _sample_from_artifact,
)
from ..perception.pairwise_affinity import load_pairwise_manifest
from .joint_inference import _joint_defs_from_mjcf
from .equivariant_relation_geometry import (
    aggregate_undirected_axes,
    build_equivariant_pair_geometry,
)
from .so3_augmentation import sample_so3


JOINT_TYPES = ("fixed", "revolute", "prismatic")


@dataclass(slots=True)
class SlotRelationTrainingConfig:
    manifest_path: str | Path
    slot_model_path: str | Path
    output_dir: str | Path
    epochs: int = 50
    object_batch_size: int = 1
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    edge_positive_weight: float = 4.0
    joint_type_loss_weight: float = 1.0
    axis_loss_weight: float = 2.0
    axis_line_loss_weight: float = 1.0
    joint_replay_loss_weight: float = 0.1
    slot_assignment_loss_weight: float = 1.0
    slot_dice_loss_weight: float = 0.5
    slot_pairwise_loss_weight: float = 0.25
    slot_pair_samples_per_object: int = 4096
    slot_rigid_loss_weight: float = 0.1
    slot_existence_loss_weight: float = 0.25
    slot_assignment_consistency_loss_weight: float = 0.0
    joint_type_balanced_loss: bool = True
    rotation_augmentation: bool = False
    rotation_augmentation_probability: float = 0.5
    rotation_augmentation_mode: str = "uniform_quaternion"
    rotation_augmentation_scope: str = "relation_geometry"
    slot_input_mode: str = "full"
    slot_geometry_representation: str = "raw"
    temporal_occlusion_augmentation: bool = False
    temporal_occlusion_probability: float = 0.5
    temporal_occlusion_min_fraction: float = 0.15
    temporal_occlusion_max_fraction: float = 0.45
    temporal_occlusion_track_fraction: float = 0.5
    trajectory_corruption_augmentation: bool = False
    trajectory_corruption_probability: float = 0.5
    trajectory_corruption_track_fraction: float = 0.25
    trajectory_drift_scale_fraction: float = 0.03
    trajectory_spike_probability: float = 0.15
    coherent_drift_probability: float = 0.0
    recovery_offset_probability: float = 0.0
    track_id_switch_probability: float = 0.0
    slot_contamination_probability: float = 0.0
    slot_contamination_fraction: float = 0.1
    excitation_weighted_axis_loss: bool = False
    min_axis_excitation: float = 0.01
    full_axis_excitation: float = 0.05
    hard_axis_focal_gamma: float = 0.0
    hard_axis_max_weight: float = 2.0
    axis_geometry_branch: bool = False
    axis_head_type: str = "direct"
    vector_pivot_parameterization: str = "analytic_plane_residual_v1"
    relation_train_scope: str = "all"
    geometry_encoder_type: str = "track_gru_average"
    trajectory_hidden_dim: int = 128
    trajectory_samples: int = 32
    quality_weighted_trajectories: bool = False
    robust_segment_weights: bool = False
    geometry_max_tracks: int = 64
    geometry_attention_heads: int = 4
    geometry_transformer_layers: int = 1
    axis_equivariance_loss_weight: float = 0.0
    axis_line_equivariance_loss_weight: float = 0.0
    edge_consistency_loss_weight: float = 0.0
    type_consistency_loss_weight: float = 0.0
    unfreeze_slot_backbone: bool = False
    slot_unfreeze_scope: str = "decoder"
    slot_learning_rate_scale: float = 0.1
    initial_relation_model_path: str | Path | None = None
    load_initial_slot_state: bool = True
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class SlotRelationInferenceConfig:
    tracks_path: str | Path
    features_npz: str | Path
    slot_model_path: str | Path
    relation_model_path: str | Path
    output_json: str | Path
    device: str = "auto"
    slot_existence_threshold: float = 0.5
    edge_threshold: float = 0.5
    min_axis_confidence: float = 0.0
    min_axis_observability: float = 0.0
    min_edge_observability: float = 0.0
    gate_selected_edges: bool = False
    quality_weighted_trajectories: bool | None = None
    robust_segment_weights: bool | None = None
    min_trajectory_quality: float = 0.0
    trajectory_assignment_override: str | Path | None = None


class SlotRelationTrainer:
    """Train a diagnostic ordered slot-pair joint proposal head.

    The slot network remains frozen. GT part IDs and MuJoCo joints are used only
    during training/evaluation to test whether its latent slots contain enough
    information for feedforward kinematic relation prediction.
    """

    def __init__(self, config: SlotRelationTrainingConfig) -> None:
        self.config = config

    def train(self) -> Path:
        np = _require_numpy()
        torch = _require_torch()
        rows = load_pairwise_manifest(self.config.manifest_path)
        slot_checkpoint_path = Path(self.config.slot_model_path).expanduser().resolve()
        slot_checkpoint = torch.load(slot_checkpoint_path, map_location="cpu", weights_only=True)
        device = _resolve_device(torch, self.config.device)
        torch.manual_seed(int(self.config.seed))
        random.seed(int(self.config.seed))
        slot_model = _load_slot_model(slot_checkpoint, torch, device)
        _configure_slot_trainability(slot_model, self.config)
        train_samples = _load_relation_samples(rows, "train", slot_checkpoint)
        validation_split = "val" if any(row["split"] == "val" for row in rows) else "train"
        val_samples = _load_relation_samples(rows, validation_split, slot_checkpoint)
        test_samples = _load_relation_samples(rows, "test", slot_checkpoint)
        feature_mean_np = slot_checkpoint["feature_mean"].cpu().numpy()
        train_samples = _apply_slot_input_mode(train_samples, self.config.slot_input_mode, feature_mean_np)
        val_samples = _apply_slot_input_mode(val_samples, self.config.slot_input_mode, feature_mean_np)
        test_samples = _apply_slot_input_mode(test_samples, self.config.slot_input_mode, feature_mean_np)
        train_samples = _apply_slot_geometry_representation(
            train_samples, self.config.slot_geometry_representation, np
        )
        val_samples = _apply_slot_geometry_representation(
            val_samples, self.config.slot_geometry_representation, np
        )
        test_samples = _apply_slot_geometry_representation(
            test_samples, self.config.slot_geometry_representation, np
        )
        if not train_samples:
            raise ValueError("No relation-head training samples with recoverable MuJoCo joints were found.")
        motion_summary_dim = int(slot_checkpoint["input_dim"]) - int(
            train_samples[0]["embedding_dim"]
        )
        relation_model = _build_relation_model(
            torch,
            slot_dim=int(slot_checkpoint["hidden_dim"]),
            hidden_dim=int(self.config.hidden_dim),
            motion_summary_dim=motion_summary_dim,
            axis_geometry_branch=bool(self.config.axis_geometry_branch),
            axis_head_type=str(self.config.axis_head_type),
            vector_pivot_parameterization=str(self.config.vector_pivot_parameterization),
            geometry_encoder_type=str(self.config.geometry_encoder_type),
            trajectory_hidden_dim=int(self.config.trajectory_hidden_dim),
            geometry_max_tracks=int(self.config.geometry_max_tracks),
            geometry_attention_heads=int(self.config.geometry_attention_heads),
            geometry_transformer_layers=int(self.config.geometry_transformer_layers),
        ).to(device)
        relation_model.trajectory_samples = int(self.config.trajectory_samples)
        relation_model.quality_weighted_trajectories = bool(
            self.config.quality_weighted_trajectories
        )
        relation_model.robust_segment_weights = bool(self.config.robust_segment_weights)
        inherited_slot_state = False
        if self.config.initial_relation_model_path is not None:
            initial = torch.load(
                Path(self.config.initial_relation_model_path).expanduser().resolve(),
                map_location="cpu",
                weights_only=True,
            )
            relation_model.load_state_dict(initial["state_dict"])
            if self.config.load_initial_slot_state and initial.get("slot_state_dict") is not None:
                slot_model.load_state_dict(initial["slot_state_dict"])
                inherited_slot_state = True
        if self.config.relation_train_scope == "vector_pivot_only":
            if self.config.axis_head_type != "vector_neuron":
                raise ValueError("vector_pivot_only requires axis_head_type=vector_neuron")
            for parameter in relation_model.parameters():
                parameter.requires_grad_(False)
            for parameter in relation_model.vector_pivot_weights.parameters():
                parameter.requires_grad_(True)
        elif self.config.relation_train_scope in {"frozen", "slot_only"}:
            if not self.config.unfreeze_slot_backbone:
                raise ValueError(
                    f"relation_train_scope={self.config.relation_train_scope} "
                    "requires --unfreeze-slot-backbone"
                )
            for parameter in relation_model.parameters():
                parameter.requires_grad_(False)
        elif self.config.relation_train_scope != "all":
            raise ValueError(f"Unsupported relation train scope: {self.config.relation_train_scope}")
        relation_parameters = [
            parameter for parameter in relation_model.parameters() if parameter.requires_grad
        ]
        parameter_groups: list[dict[str, Any]] = []
        if relation_parameters:
            parameter_groups.append({
                "params": relation_parameters,
                "lr": float(self.config.learning_rate),
            })
        if self.config.unfreeze_slot_backbone:
            parameter_groups.append({
                "params": [parameter for parameter in slot_model.parameters() if parameter.requires_grad],
                "lr": float(self.config.learning_rate) * float(self.config.slot_learning_rate_scale),
            })
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(self.config.weight_decay))
        if self.config.slot_geometry_representation == "raw":
            mean = slot_checkpoint["feature_mean"].to(device)
            std = slot_checkpoint["feature_std"].to(device)
        else:
            stacked_slot_features = np.concatenate(
                [sample["slot_features"] for sample in train_samples], axis=0
            )
            mean = torch.from_numpy(stacked_slot_features.mean(axis=0).astype(np.float32)).to(device)
            std = torch.from_numpy(
                np.maximum(stacked_slot_features.std(axis=0), 1e-6).astype(np.float32)
            ).to(device)
        joint_type_weights = _joint_type_weights(train_samples, torch, device)

        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "slot_relation_head.pt"
        best_score = math.inf
        best_slot_score = -math.inf
        best_axis_score = math.inf
        best_joint_score: tuple[float, float, float, float] | None = None
        best_states: dict[str, tuple[Any, Any]] = {}

        def snapshot_state() -> tuple[Any, Any]:
            relation_state = copy.deepcopy(relation_model.state_dict())
            slot_state = (
                copy.deepcopy(slot_model.state_dict())
                if self.config.unfreeze_slot_backbone or inherited_slot_state
                else None
            )
            return relation_state, slot_state

        history: list[dict[str, float]] = []
        training_started = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(max(1, int(self.config.epochs))):
            epoch_started = time.perf_counter()
            relation_model.train()
            slot_model.eval()
            if self.config.unfreeze_slot_backbone:
                if self.config.slot_unfreeze_scope == "all":
                    slot_model.train()
                else:
                    slot_model.decoder.train()
            order = list(range(len(train_samples)))
            random.shuffle(order)
            epoch_rows = []
            batch_size = max(1, int(self.config.object_batch_size))
            for batch_start in range(0, len(order), batch_size):
                samples = []
                for sample_index in order[batch_start : batch_start + batch_size]:
                    sample = train_samples[sample_index]
                    if (
                        self.config.temporal_occlusion_augmentation
                        and random.random() < float(self.config.temporal_occlusion_probability)
                    ):
                        sample = _augment_temporal_occlusion(
                            sample,
                            rng=random,
                            np=np,
                            min_fraction=float(self.config.temporal_occlusion_min_fraction),
                            max_fraction=float(self.config.temporal_occlusion_max_fraction),
                            track_fraction=float(self.config.temporal_occlusion_track_fraction),
                        )
                    if (
                        self.config.trajectory_corruption_augmentation
                        and random.random() < float(self.config.trajectory_corruption_probability)
                    ):
                        sample = _augment_trajectory_corruption(
                            sample,
                            rng=random,
                            np=np,
                            track_fraction=float(self.config.trajectory_corruption_track_fraction),
                            drift_scale_fraction=float(self.config.trajectory_drift_scale_fraction),
                            spike_probability=float(self.config.trajectory_spike_probability),
                            coherent_drift_probability=float(self.config.coherent_drift_probability),
                            recovery_offset_probability=float(self.config.recovery_offset_probability),
                            id_switch_probability=float(self.config.track_id_switch_probability),
                        )
                    if (
                        self.config.rotation_augmentation
                        and random.random() < float(self.config.rotation_augmentation_probability)
                    ):
                        sample = _augment_relation_sample(
                            sample, rng=random, np=np,
                            sampling_mode=self.config.rotation_augmentation_mode,
                            rotate_slot_geometry=(
                                self.config.rotation_augmentation_scope
                                == "slot_and_relation_geometry"
                            ),
                        )
                    samples.append(sample)
                losses = _relation_batch_loss(
                    samples, slot_model, relation_model,
                    mean, std, torch, device, self.config, joint_type_weights,
                )
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
                epoch_rows.append({key: float(value.detach().cpu()) for key, value in losses.items()})
            validation = _evaluate_relation_samples(
                val_samples, slot_model, relation_model, mean, std, torch, device,
            )
            validation_objective = _evaluate_training_objective(
                val_samples, slot_model, relation_model, mean, std, torch, device,
                self.config, joint_type_weights,
            )
            row = {
                "epoch": epoch + 1,
                "epoch_time_s": time.perf_counter() - epoch_started,
                **{f"train_{key}": _mean([item[key] for item in epoch_rows]) for key in epoch_rows[0]},
                **{f"val_{key}": float(value) for key, value in validation.items()},
                **{
                    f"val_objective_{key}": float(value)
                    for key, value in validation_objective.items()
                },
            }
            history.append(row)
            slot_score = float(validation.get("slot_ari", -math.inf))
            axis_score = float(validation.get("axis_error_deg", math.inf))
            joint_score = (
                float(validation.get("joint_type_accuracy", 0.0)),
                float(validation.get("edge_f1", 0.0)),
                slot_score,
                -axis_score,
            )
            if slot_score >= best_slot_score:
                best_slot_score = slot_score
                best_states["best_slot"] = snapshot_state()
            if math.isfinite(axis_score) and axis_score <= best_axis_score:
                best_axis_score = axis_score
                best_states["best_axis"] = snapshot_state()
            if best_joint_score is None or joint_score >= best_joint_score:
                best_joint_score = joint_score
                best_states["best_joint"] = snapshot_state()
            print(
                json.dumps(
                    {
                        "event": "relation_training_epoch",
                        **row,
                        "peak_cuda_memory_bytes": (
                            int(torch.cuda.max_memory_allocated(device))
                            if device.startswith("cuda")
                            else None
                        ),
                    }
                ),
                flush=True,
            )
            if validation_objective["total"] <= best_score:
                best_score = float(validation_objective["total"])
                torch.save(
                    {
                        "state_dict": relation_model.state_dict(),
                        "slot_state_dict": (
                            slot_model.state_dict()
                            if self.config.unfreeze_slot_backbone or inherited_slot_state
                            else None
                        ),
                        "slot_dim": int(slot_checkpoint["hidden_dim"]),
                        "hidden_dim": int(self.config.hidden_dim),
                        "motion_summary_dim": motion_summary_dim,
                        "axis_geometry_branch": bool(self.config.axis_geometry_branch),
                        "axis_head_type": str(self.config.axis_head_type),
                        "vector_pivot_parameterization": str(
                            self.config.vector_pivot_parameterization
                        ),
                        "relation_train_scope": str(self.config.relation_train_scope),
                        "rotation_augmentation": bool(
                            self.config.rotation_augmentation
                        ),
                        "rotation_augmentation_probability": float(
                            self.config.rotation_augmentation_probability
                        ),
                        "rotation_augmentation_mode": str(
                            self.config.rotation_augmentation_mode
                        ),
                        "rotation_augmentation_scope": str(
                            self.config.rotation_augmentation_scope
                        ),
                        "axis_equivariance_loss_weight": float(
                            self.config.axis_equivariance_loss_weight
                        ),
                        "axis_line_equivariance_loss_weight": float(
                            self.config.axis_line_equivariance_loss_weight
                        ),
                        "edge_consistency_loss_weight": float(
                            self.config.edge_consistency_loss_weight
                        ),
                        "type_consistency_loss_weight": float(
                            self.config.type_consistency_loss_weight
                        ),
                        "slot_assignment_consistency_loss_weight": float(
                            self.config.slot_assignment_consistency_loss_weight
                        ),
                        "slot_dice_loss_weight": float(
                            self.config.slot_dice_loss_weight
                        ),
                        "slot_pairwise_loss_weight": float(
                            self.config.slot_pairwise_loss_weight
                        ),
                        "slot_pair_samples_per_object": int(
                            self.config.slot_pair_samples_per_object
                        ),
                        "slot_rigid_loss_weight": float(
                            self.config.slot_rigid_loss_weight
                        ),
                        "slot_geometry_representation": str(
                            self.config.slot_geometry_representation
                        ),
                        "temporal_occlusion_augmentation": bool(
                            self.config.temporal_occlusion_augmentation
                        ),
                        "temporal_occlusion_probability": float(
                            self.config.temporal_occlusion_probability
                        ),
                        "temporal_occlusion_min_fraction": float(
                            self.config.temporal_occlusion_min_fraction
                        ),
                        "temporal_occlusion_max_fraction": float(
                            self.config.temporal_occlusion_max_fraction
                        ),
                        "temporal_occlusion_track_fraction": float(
                            self.config.temporal_occlusion_track_fraction
                        ),
                        "trajectory_corruption_augmentation": bool(
                            self.config.trajectory_corruption_augmentation
                        ),
                        "trajectory_corruption_probability": float(
                            self.config.trajectory_corruption_probability
                        ),
                        "trajectory_corruption_track_fraction": float(
                            self.config.trajectory_corruption_track_fraction
                        ),
                        "coherent_drift_probability": float(self.config.coherent_drift_probability),
                        "recovery_offset_probability": float(self.config.recovery_offset_probability),
                        "track_id_switch_probability": float(self.config.track_id_switch_probability),
                        "slot_contamination_probability": float(self.config.slot_contamination_probability),
                        "slot_contamination_fraction": float(self.config.slot_contamination_fraction),
                        "excitation_weighted_axis_loss": bool(
                            self.config.excitation_weighted_axis_loss
                        ),
                        "min_axis_excitation": float(self.config.min_axis_excitation),
                        "full_axis_excitation": float(self.config.full_axis_excitation),
                        "hard_axis_focal_gamma": float(self.config.hard_axis_focal_gamma),
                        "hard_axis_max_weight": float(self.config.hard_axis_max_weight),
                        "trajectory_drift_scale_fraction": float(
                            self.config.trajectory_drift_scale_fraction
                        ),
                        "trajectory_spike_probability": float(
                            self.config.trajectory_spike_probability
                        ),
                        "slot_feature_mean": mean.detach().cpu(),
                        "slot_feature_std": std.detach().cpu(),
                        "geometry_encoder_type": str(self.config.geometry_encoder_type),
                        "trajectory_hidden_dim": int(self.config.trajectory_hidden_dim),
                        "trajectory_samples": int(self.config.trajectory_samples),
                        "quality_weighted_trajectories": bool(
                            self.config.quality_weighted_trajectories
                        ),
                        "robust_segment_weights": bool(self.config.robust_segment_weights),
                        "geometry_max_tracks": int(self.config.geometry_max_tracks),
                        "geometry_attention_heads": int(self.config.geometry_attention_heads),
                        "geometry_transformer_layers": int(
                            self.config.geometry_transformer_layers
                        ),
                        "joint_types": JOINT_TYPES,
                        "slot_model_path": str(slot_checkpoint_path),
                        "unfreeze_slot_backbone": bool(self.config.unfreeze_slot_backbone),
                    },
                    checkpoint_path,
                )
        final_state = snapshot_state()
        best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for name, (relation_state, slot_state) in {
            **best_states,
            "final": final_state,
        }.items():
            variant_checkpoint = copy.deepcopy(best_checkpoint)
            variant_checkpoint["state_dict"] = relation_state
            variant_checkpoint["slot_state_dict"] = slot_state
            variant_checkpoint["selection"] = name
            torch.save(variant_checkpoint, output_dir / f"slot_relation_head_{name}.pt")
        relation_model.load_state_dict(best_checkpoint["state_dict"])
        if best_checkpoint.get("slot_state_dict") is not None:
            slot_model.load_state_dict(best_checkpoint["slot_state_dict"])
        test_metrics = _evaluate_relation_samples(
            test_samples, slot_model, relation_model, mean, std, torch, device,
        ) if test_samples else None
        save_json(
            {
                "model_path": str(checkpoint_path),
                "slot_model_path": str(slot_checkpoint_path),
                "validation_split": validation_split,
                "train_objects": [sample["object_id"] for sample in train_samples],
                "validation_objects": [sample["object_id"] for sample in val_samples],
                "test_objects": [sample["object_id"] for sample in test_samples],
                "best_validation_loss": best_score,
                "best_validation_axis_median_deg": min(
                    float(row["val_axis_error_median_deg"]) for row in history
                ),
                "test_metrics": test_metrics,
                "history": history,
                "runtime": {
                    "total_training_time_s": time.perf_counter() - training_started,
                    "mean_epoch_time_s": _mean([row["epoch_time_s"] for row in history]),
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.startswith("cuda")
                        else None
                    ),
                },
                "model": {
                    "relation_parameter_count": sum(
                        int(parameter.numel()) for parameter in relation_model.parameters()
                    ),
                    "relation_trainable_parameter_count": sum(
                        int(parameter.numel())
                        for parameter in relation_model.parameters()
                        if parameter.requires_grad
                    ),
                    "geometry_encoder_type": str(self.config.geometry_encoder_type),
                    "axis_head_type": str(self.config.axis_head_type),
                },
                "config": {
                    field: str(value) if isinstance(value := getattr(self.config, field), Path) else value
                    for field in self.config.__dataclass_fields__
                },
                "notes": [
                    (
                        "The part-slot backbone was fine-tuned with a reduced learning rate."
                        if self.config.unfreeze_slot_backbone
                        else "The part-slot backbone is frozen."
                    ),
                    "Pairwise predictions are unconstrained and may not form a legal kinematic tree.",
                ],
            },
            output_dir / "training_summary.json",
        )
        return checkpoint_path


def _configure_slot_trainability(slot_model: Any, config: SlotRelationTrainingConfig) -> None:
    for parameter in slot_model.parameters():
        parameter.requires_grad_(False)
    if not config.unfreeze_slot_backbone:
        return
    scope = str(config.slot_unfreeze_scope).lower()
    if scope not in {"decoder", "all"}:
        raise ValueError("slot_unfreeze_scope must be 'decoder' or 'all'")
    for name, parameter in slot_model.named_parameters():
        if scope == "all" or name == "queries" or name.startswith("decoder."):
            parameter.requires_grad_(True)


class SlotRelationInferencer:
    def __init__(self, config: SlotRelationInferenceConfig) -> None:
        self.config = config

    def infer(self) -> Path:
        import time

        from ..benchmarks.runtime_profile import hardware_snapshot, synchronize_device

        total_started = time.perf_counter()
        np = _require_numpy()
        torch = _require_torch()
        input_started = time.perf_counter()
        tracks_path = Path(self.config.tracks_path).expanduser().resolve()
        artifact = load_json(tracks_path)
        slot_checkpoint_path = Path(self.config.slot_model_path).expanduser().resolve()
        relation_checkpoint_path = Path(self.config.relation_model_path).expanduser().resolve()
        slot_checkpoint = torch.load(slot_checkpoint_path, map_location="cpu", weights_only=True)
        relation_checkpoint = torch.load(relation_checkpoint_path, map_location="cpu", weights_only=True)
        sample = _sample_from_artifact(
            artifact,
            load_cotracker_feature_map(self.config.features_npz),
            object_id=str(artifact.get("object_instance_id", tracks_path.parent.name)),
            require_labels=False,
            canonicalize_geometry=bool(slot_checkpoint.get("canonicalize_geometry", False)),
        )
        sample = _apply_slot_geometry_representation(
            [sample], str(relation_checkpoint.get("slot_geometry_representation", "raw")), np
        )[0]
        input_s = time.perf_counter() - input_started
        device = _resolve_device(torch, self.config.device)
        model_started = time.perf_counter()
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
            # Checkpoints predating this field used the restricted center-delta offset.
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
        ).to(device)
        relation_model.load_state_dict(relation_checkpoint["state_dict"])
        relation_model.eval()
        synchronize_device(torch, device)
        model_load_s = time.perf_counter() - model_started
        features = torch.from_numpy(sample["features"]).to(device)
        slot_features = torch.from_numpy(sample.get("slot_features", sample["features"])).to(device)
        slot_mean = relation_checkpoint.get("slot_feature_mean", slot_checkpoint["feature_mean"]).to(device)
        slot_std = relation_checkpoint.get("slot_feature_std", slot_checkpoint["feature_std"]).to(device)
        synchronize_device(torch, device)
        forward_started = time.perf_counter()
        with torch.no_grad():
            logits, existence_logits, slots = slot_model(
                (slot_features - slot_mean) / slot_std,
                return_slots=True,
            )
            assignment_tensor = torch.softmax(logits, dim=-1)
            assignment_override = None
            if self.config.trajectory_assignment_override is not None:
                override_path = Path(
                    self.config.trajectory_assignment_override
                ).expanduser().resolve()
                override_payload = load_json(override_path)
                override_by_track = {
                    int(track_id): int(slot_id)
                    for track_id, slot_id in override_payload.get("track_to_slot", {}).items()
                }
                missing = [
                    int(track["track_id"])
                    for track in sample["tracks"]
                    if int(track["track_id"]) not in override_by_track
                ]
                if missing:
                    raise ValueError(
                        "Trajectory assignment override is missing "
                        f"{len(missing)} inference tracks; first missing track_id={missing[0]}"
                    )
                override_slots = torch.as_tensor(
                    [override_by_track[int(track["track_id"])] for track in sample["tracks"]],
                    dtype=torch.long,
                    device=device,
                )
                if torch.any(override_slots < 0) or torch.any(override_slots >= logits.shape[-1]):
                    raise ValueError("Trajectory assignment override contains an invalid slot index")
                assignment_tensor = torch.nn.functional.one_hot(
                    override_slots, num_classes=logits.shape[-1]
                ).to(dtype=logits.dtype)
                assignment_override = str(override_path)
            motion_summary_dim = int(relation_checkpoint.get("motion_summary_dim", 0))
            slot_motion = (
                _slot_motion_summary(features, assignment_tensor, int(sample["embedding_dim"]), torch)
                if motion_summary_dim > 0
                else None
            )
            trajectory_tokens, trajectory_visibility = _trajectory_tensors(
                sample, torch, device,
                sample_count=int(relation_checkpoint.get("trajectory_samples", 32)),
                quality_weighted=(
                    bool(relation_checkpoint.get("quality_weighted_trajectories", False))
                    if self.config.quality_weighted_trajectories is None
                    else bool(self.config.quality_weighted_trajectories)
                ),
                min_quality=max(0.0, float(self.config.min_trajectory_quality)),
                robust_segment_weights=(
                    bool(relation_checkpoint.get("robust_segment_weights", False))
                    if self.config.robust_segment_weights is None
                    else bool(self.config.robust_segment_weights)
                ),
            )
            relation = relation_model(
                slots, slot_motion,
                trajectory_tokens=trajectory_tokens,
                trajectory_visibility=trajectory_visibility,
                slot_probabilities=assignment_tensor,
            )
            assignment = assignment_tensor.cpu().numpy()
            existence = torch.sigmoid(existence_logits).cpu().numpy()
            edge_probability = torch.sigmoid(relation["edge_logits"]).cpu().numpy()
            type_probability = torch.softmax(relation["type_logits"], dim=-1).cpu().numpy()
            axes = relation["axes"].cpu().numpy()
            pivots = relation["pivots"].cpu().numpy()
            axis_confidence = relation["axis_confidence"].cpu().numpy()
            candidate_observability_tensor = relation.get("candidate_observability")
            candidate_observability = (
                candidate_observability_tensor.cpu().numpy()
                if candidate_observability_tensor is not None
                else np.ones_like(axis_confidence)
            )
            geometry_valid_track_count = relation.get("geometry_valid_track_count")
            geometry_effective_track_count = relation.get(
                "geometry_effective_track_count"
            )
            geometry_pooling_entropy = relation.get("geometry_pooling_entropy")
        synchronize_device(torch, device)
        forward_s = time.perf_counter() - forward_started
        postprocess_started = time.perf_counter()
        active_slots = (
            sorted(set(int(value) for value in override_slots.cpu().tolist()))
            if self.config.trajectory_assignment_override is not None
            else np.flatnonzero(
                existence >= float(self.config.slot_existence_threshold)
            ).tolist()
        )
        if not active_slots:
            active_slots = [int(np.argmax(existence))]
        active_assignment = np.asarray(active_slots, dtype=np.int64)[
            np.argmax(assignment[:, active_slots], axis=1)
        ]
        pairs = []
        threshold_edges = []
        center = np.asarray(sample["canonical_center_m"], dtype=np.float64)
        scale = float(sample["canonical_scale_m"])
        for parent in active_slots:
            for child in active_slots:
                if parent == child:
                    continue
                type_index = int(np.argmax(type_probability[parent, child]))
                pivot_world = center + scale * pivots[parent, child]
                confidence = float(axis_confidence[parent, child])
                observability = float(candidate_observability[parent, child])
                axis_reliable = (
                    confidence >= float(self.config.min_axis_confidence)
                    and observability >= float(self.config.min_axis_observability)
                )
                axis_raw = [float(value) for value in axes[parent, child]]
                row = {
                    "parent_slot_id": int(parent),
                    "child_slot_id": int(child),
                    "edge_probability": float(edge_probability[parent, child]),
                    "joint_type": JOINT_TYPES[type_index],
                    "joint_type_probabilities": {
                        name: float(type_probability[parent, child, index])
                        for index, name in enumerate(JOINT_TYPES)
                    },
                    "axis_world": axis_raw if axis_reliable else None,
                    "axis_world_raw": axis_raw,
                    "axis_reliable": bool(axis_reliable),
                    "axis_confidence": confidence,
                    "axis_observability": observability,
                    "axis_abstention_reason": (
                        None
                        if axis_reliable
                        else "low_confidence_or_observability"
                    ),
                    "axis_line_point_world": [float(value) for value in pivot_world],
                }
                pairs.append(row)
                if (
                    row["edge_probability"] >= float(self.config.edge_threshold)
                    and observability >= float(self.config.min_edge_observability)
                ):
                    threshold_edges.append(row)
        # Gated decoding permits a legal forest instead of forcing K-1 edges
        # through uncertain or spurious active slots.
        selected_edges = decode_legal_relation_tree(
            active_slots, threshold_edges if self.config.gate_selected_edges else pairs
        )
        diagnostics = graph_legality_diagnostics(active_slots, selected_edges)
        threshold_diagnostics = graph_legality_diagnostics(
            active_slots, threshold_edges
        )
        output = Path(self.config.output_json).expanduser().resolve()
        payload = {
                "source": "constrained-slot-pair-relation-head",
                "object_instance_id": sample["object_id"],
                "tracks_path": str(tracks_path),
                "slot_model_path": str(slot_checkpoint_path),
                "relation_model_path": str(relation_checkpoint_path),
                "slot_existence_probabilities": [float(value) for value in existence],
                "active_slots": [int(value) for value in active_slots],
                "track_slot_assignments": [int(value) for value in active_assignment],
                "track_ids": [int(track["track_id"]) for track in sample["tracks"]],
                "pairwise_relations": pairs,
                "threshold_edges": threshold_edges,
                "selected_edges": selected_edges,
                "graph_diagnostics": diagnostics,
                "threshold_graph_diagnostics": threshold_diagnostics,
                "confidence_gating": {
                    "min_axis_confidence": float(self.config.min_axis_confidence),
                    "min_axis_observability": float(self.config.min_axis_observability),
                    "min_edge_observability": float(self.config.min_edge_observability),
                    "gate_selected_edges": bool(self.config.gate_selected_edges),
                    "abstained_axis_count": sum(
                        not bool(row["axis_reliable"]) for row in pairs
                    ),
                },
                "runtime_profile": {
                    "scope": "slot_and_relation_head_inference",
                    "input_and_feature_assembly_s": input_s,
                    "model_load_s": model_load_s,
                    "accelerator_forward_s": forward_s,
                    "postprocess_before_serialization_s": None,
                    "total_before_serialization_s": None,
                    "device": device,
                    "track_count": len(sample["tracks"]),
                    "active_slot_count": len(active_slots),
                    "hardware": hardware_snapshot(torch),
                },
                "geometry_diagnostics": (
                    {
                        "encoder_type": str(
                            relation_checkpoint.get(
                                "geometry_encoder_type", "track_gru_average"
                            )
                        ),
                        "effective_track_count_per_slot": (
                            geometry_effective_track_count.cpu().tolist()
                        ),
                        "valid_track_count_per_slot": (
                            geometry_valid_track_count.cpu().tolist()
                        ),
                        "pooling_entropy_per_slot": (
                            geometry_pooling_entropy.cpu().tolist()
                        ),
                    }
                    if geometry_valid_track_count is not None
                    and geometry_effective_track_count is not None
                    and geometry_pooling_entropy is not None
                    else None
                ),
                "quality_weighted_trajectories": (
                    bool(relation_checkpoint.get("quality_weighted_trajectories", False))
                    if self.config.quality_weighted_trajectories is None
                    else bool(self.config.quality_weighted_trajectories)
                ),
                "trajectory_assignment_override": assignment_override,
                "notes": [
                    "This diagnostic head predicts ordered edges independently.",
                    "No tree constraint, parent arbitration, or optimization refinement has been applied.",
                ],
            }
        payload["runtime_profile"]["postprocess_before_serialization_s"] = (
            time.perf_counter() - postprocess_started
        )
        payload["runtime_profile"]["total_before_serialization_s"] = (
            time.perf_counter() - total_started
        )
        save_json(payload, output)
        return output


def _load_relation_samples(
    rows: list[dict[str, str]], split: str, slot_checkpoint: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    sample_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    for row in rows:
        if row["split"] != split:
            continue
        cache_key = (
            row["tracks_path"], row["features_npz"], row.get("relation_gt_path", "")
        )
        if cache_key in sample_cache:
            cached = sample_cache[cache_key]
            if cached is not None:
                result.append({**cached, "object_id": row["object_id"]})
            continue
        artifact = load_json(row["tracks_path"])
        sample = _sample_from_artifact(
            artifact,
            load_cotracker_feature_map(row["features_npz"]),
            object_id=row["object_id"],
            require_labels=True,
            canonicalize_geometry=bool(slot_checkpoint.get("canonicalize_geometry", False)),
            collapse_fixed_connected_labels=bool(
                slot_checkpoint.get("collapse_fixed_connected_labels", False)
            ),
        )
        relations = (
            extract_manual_gt_relations(load_json(row["relation_gt_path"]), sample)
            if row.get("relation_gt_path")
            else extract_gt_relations(artifact, sample)
        )
        if relations:
            sample["gt_relations"] = relations
            sample_cache[cache_key] = sample
            result.append(sample)
        else:
            sample_cache[cache_key] = None
    return result


def extract_manual_gt_relations(
    annotation: dict[str, Any], sample: dict[str, Any]
) -> list[dict[str, Any]]:
    """Convert manually annotated world-space joints into training coordinates."""
    raw_part_to_label = sample["raw_part_to_label"]
    center = sample["canonical_center_m"]
    scale = max(float(sample["canonical_scale_m"]), 1e-6)
    relations = []
    for index, joint in enumerate(annotation.get("joints", [])):
        parent_raw = int(joint.get("parent_part_id", -1))
        child_raw = int(joint.get("child_part_id", -1))
        joint_type = str(joint.get("joint_type", "fixed"))
        axis = joint.get("axis")
        pivot = joint.get("pivot")
        if (
            parent_raw not in raw_part_to_label
            or child_raw not in raw_part_to_label
            or raw_part_to_label[parent_raw] == raw_part_to_label[child_raw]
            or joint_type not in JOINT_TYPES
            or not isinstance(axis, list)
            or len(axis) != 3
            or not isinstance(pivot, list)
            or len(pivot) != 3
        ):
            continue
        norm = math.sqrt(sum(float(value) ** 2 for value in axis))
        if norm < 1e-9:
            continue
        relations.append({
            "parent_label": int(raw_part_to_label[parent_raw]),
            "child_label": int(raw_part_to_label[child_raw]),
            "parent_part_id": parent_raw,
            "child_part_id": child_raw,
            "joint_name": str(joint.get("name", f"manual_joint_{index + 1}")),
            "joint_type": joint_type,
            "axis": [float(value) / norm for value in axis],
            "pivot": [
                (float(pivot[axis_index]) - float(center[axis_index])) / scale
                for axis_index in range(3)
            ],
            "annotation_source": str(
                joint.get("annotation_source", annotation.get("annotation_source", "manual"))
            ),
        })
    return relations


def extract_gt_relations(artifact: dict[str, Any], sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover simulator-only directed relation labels in canonical coordinates."""
    segmentation = artifact.get("original_part_segmentation") or artifact.get("part_segmentation")
    if not isinstance(segmentation, dict):
        return []
    episode_path_raw = artifact.get("input_episode_path") or artifact.get("episode_path")
    if not isinstance(episode_path_raw, str):
        return []
    episode_path = Path(episode_path_raw).expanduser()
    if not episode_path.is_absolute():
        episode_path = Path(sample.get("tracks_path", ".")).parent / episode_path
    if not episode_path.exists():
        return []
    episode = load_json(episode_path)
    model_path_raw = episode.get("metadata", {}).get("model_path")
    if not isinstance(model_path_raw, str):
        return []
    model_path = Path(model_path_raw).expanduser()
    if not model_path.is_absolute():
        model_path = episode_path.parent / model_path
    if not model_path.exists():
        return []
    joint_defs = _joint_defs_from_mjcf(model_path)
    parts = [part for part in segmentation.get("parts", []) if isinstance(part, dict)]
    body_to_part = {int(part.get("body_id", -1)): int(part.get("part_id", -1)) for part in parts}
    raw_part_to_label = sample["raw_part_to_label"]
    center = sample["canonical_center_m"]
    scale = max(float(sample["canonical_scale_m"]), 1e-6)
    relations = []
    for part in parts:
        child_raw = int(part.get("part_id", -1))
        parent_body_id = part.get("parent_body_id")
        parent_raw = (
            body_to_part.get(int(parent_body_id))
            if parent_body_id is not None
            else None
        )
        if child_raw not in raw_part_to_label or parent_raw not in raw_part_to_label:
            continue
        if raw_part_to_label[child_raw] == raw_part_to_label[parent_raw]:
            continue
        for joint_name in part.get("joint_names", []):
            joint = joint_defs.get(str(joint_name))
            if joint is None:
                continue
            joint_type = str(joint.get("joint_type", "fixed"))
            if joint_type not in JOINT_TYPES:
                joint_type = "fixed"
            relations.append(
                {
                    "parent_label": int(raw_part_to_label[parent_raw]),
                    "child_label": int(raw_part_to_label[child_raw]),
                    "parent_part_id": int(parent_raw),
                    "child_part_id": int(child_raw),
                    "joint_name": str(joint_name),
                    "joint_type": joint_type,
                    "axis": [float(value) for value in joint["axis_world"]],
                    "pivot": [
                        (float(joint["pivot_world"][axis]) - float(center[axis])) / scale
                        for axis in range(3)
                    ],
                }
            )
            break
    return relations


def _load_slot_model(checkpoint: dict[str, Any], torch: Any, device: str) -> Any:
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_relation_model(
    torch: Any, *, slot_dim: int, hidden_dim: int, motion_summary_dim: int = 0,
    axis_geometry_branch: bool = False,
    axis_head_type: str = "direct",
    vector_pivot_parameterization: str = "analytic_plane_residual_v1",
    geometry_encoder_type: str = "track_gru_average",
    trajectory_hidden_dim: int = 128,
    geometry_max_tracks: int = 64,
    geometry_attention_heads: int = 4,
    geometry_transformer_layers: int = 1,
) -> Any:
    if axis_head_type not in {"direct", "equivariant_proposal", "vector_neuron"}:
        raise ValueError(f"Unsupported axis head type: {axis_head_type}")
    if axis_head_type != "direct" and not axis_geometry_branch:
        raise ValueError(f"{axis_head_type} requires axis_geometry_branch=True")
    if vector_pivot_parameterization not in {
        "legacy_center_delta", "analytic_plane_residual_v1", "pure_plane_residual_v1"
    }:
        raise ValueError(
            f"Unsupported vector pivot parameterization: {vector_pivot_parameterization}"
        )
    if geometry_encoder_type not in {"track_gru_average", "track_gru_transformer"}:
        raise ValueError(f"Unsupported geometry encoder type: {geometry_encoder_type}")
    if (
        geometry_encoder_type == "track_gru_transformer"
        and trajectory_hidden_dim % geometry_attention_heads != 0
    ):
        raise ValueError("trajectory_hidden_dim must be divisible by geometry_attention_heads")

    class SlotRelationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.axis_geometry_branch = bool(axis_geometry_branch)
            self.axis_head_type = str(axis_head_type)
            self.vector_pivot_parameterization = str(vector_pivot_parameterization)
            self.geometry_encoder_type = str(geometry_encoder_type)
            self.geometry_max_tracks = max(1, int(geometry_max_tracks))
            self.trajectory_samples = 32
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear((slot_dim + motion_summary_dim) * 4, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
            )
            self.edge = torch.nn.Linear(hidden_dim, 1)
            self.joint_type = torch.nn.Linear(hidden_dim, len(JOINT_TYPES))
            if self.axis_head_type != "direct":
                self.equivariant_relation_backbone = torch.nn.Sequential(
                    torch.nn.Linear(slot_dim * 4 + 5, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.GELU(),
                )
            if self.axis_geometry_branch:
                self.trajectory_input = torch.nn.Sequential(
                    torch.nn.Linear(11, trajectory_hidden_dim),
                    torch.nn.LayerNorm(trajectory_hidden_dim),
                    torch.nn.GELU(),
                )
                self.trajectory_encoder = torch.nn.GRU(
                    trajectory_hidden_dim, trajectory_hidden_dim, batch_first=True,
                )
                if self.geometry_encoder_type == "track_gru_transformer":
                    self.cross_track_input = torch.nn.Sequential(
                        torch.nn.Linear(trajectory_hidden_dim + 1, trajectory_hidden_dim),
                        torch.nn.LayerNorm(trajectory_hidden_dim),
                        torch.nn.GELU(),
                    )
                    cross_track_layer = torch.nn.TransformerEncoderLayer(
                        d_model=trajectory_hidden_dim,
                        nhead=int(geometry_attention_heads),
                        dim_feedforward=trajectory_hidden_dim * 4,
                        dropout=0.0,
                        batch_first=True,
                        norm_first=True,
                    )
                    self.cross_track_encoder = torch.nn.TransformerEncoder(
                        cross_track_layer, num_layers=max(1, int(geometry_transformer_layers))
                    )
                self.geometry_backbone = torch.nn.Sequential(
                    torch.nn.Linear(trajectory_hidden_dim * 4, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.GELU(),
                )
            if self.axis_head_type == "direct":
                self.axis = torch.nn.Linear(hidden_dim, 3)
                self.pivot = torch.nn.Linear(hidden_dim, 3)
            elif self.axis_head_type == "equivariant_proposal":
                self.axis_candidate_weights = torch.nn.Linear(hidden_dim, 2)
                self.pivot_candidate_weights = torch.nn.Linear(hidden_dim, 4)
            else:
                self.vector_axis_weights = torch.nn.Linear(hidden_dim, 4)
                self.vector_pivot_weights = torch.nn.Linear(hidden_dim, 4)

        def forward(
            self, slots: Any, motion_summary: Any | None = None, *,
            trajectory_tokens: Any | None = None,
            trajectory_visibility: Any | None = None,
            slot_probabilities: Any | None = None,
        ) -> dict[str, Any]:
            single = slots.ndim == 2
            if single:
                slots = slots.unsqueeze(0)
                if motion_summary is not None:
                    motion_summary = motion_summary.unsqueeze(0)
                if trajectory_tokens is not None:
                    trajectory_tokens = trajectory_tokens.unsqueeze(0)
                    trajectory_visibility = trajectory_visibility.unsqueeze(0)
                    slot_probabilities = slot_probabilities.unsqueeze(0)
            tokens = (
                torch.cat([slots, motion_summary], dim=-1)
                if motion_summary is not None
                else slots
            )
            parent = tokens[:, :, None, :].expand(-1, -1, tokens.shape[1], -1)
            child = tokens[:, None, :, :].expand(-1, tokens.shape[1], -1, -1)
            pair = torch.cat([parent, child, child - parent, child * parent], dim=-1)
            slot_parent = slots[:, :, None, :].expand(-1, -1, slots.shape[1], -1)
            slot_child = slots[:, None, :, :].expand(-1, slots.shape[1], -1, -1)
            slot_pair = torch.cat(
                [slot_parent, slot_child, slot_child - slot_parent, slot_child * slot_parent],
                dim=-1,
            )
            hidden = self.backbone(pair)
            geometry_hidden = hidden
            pair_geometry = None
            if self.axis_geometry_branch:
                if (
                    trajectory_tokens is None
                    or trajectory_visibility is None
                    or slot_probabilities is None
                ):
                    raise ValueError(
                        "The axis geometry branch requires trajectory tokens, visibility, "
                        "and soft slot probabilities."
                    )
                batch, tracks, frames, _ = trajectory_tokens.shape
                flat_tokens = trajectory_tokens.reshape(batch * tracks, frames, 11)
                flat_visibility = trajectory_visibility.reshape(batch * tracks, frames)
                active_tracks = flat_visibility.any(dim=1)
                flat_motion = trajectory_tokens.new_zeros(
                    (batch * tracks, trajectory_hidden_dim)
                )
                if bool(active_tracks.any()):
                    encoded, _ = self.trajectory_encoder(
                        self.trajectory_input(flat_tokens[active_tracks])
                    )
                    active_visibility = flat_visibility[active_tracks].to(
                        encoded.dtype
                    ).unsqueeze(-1)
                    active_motion = (encoded * active_visibility).sum(dim=1)
                    active_motion = active_motion / active_visibility.sum(dim=1).clamp_min(1.0)
                    flat_motion[active_tracks] = active_motion
                track_motion = flat_motion.reshape(batch, tracks, trajectory_hidden_dim)
                weights = slot_probabilities.transpose(1, 2)
                valid_track_mask = trajectory_visibility.any(dim=2)
                hard_assignment = slot_probabilities.argmax(dim=-1)
                slot_ids = torch.arange(
                    slot_probabilities.shape[-1], device=slot_probabilities.device
                )
                valid_track_count = (
                    (
                        hard_assignment[:, :, None] == slot_ids[None, None, :]
                    )
                    & valid_track_mask[:, :, None]
                ).sum(dim=1).to(weights.dtype)
                if self.geometry_encoder_type == "track_gru_transformer":
                    part_motion, effective_track_count, pooling_entropy = (
                        self._cross_track_part_motion(
                            track_motion, weights, valid_track_mask
                        )
                    )
                else:
                    weights = weights * valid_track_mask[:, None, :].to(weights.dtype)
                    part_motion = weights @ track_motion
                    weight_sum = weights.sum(dim=2, keepdim=True)
                    part_motion = (
                        part_motion
                        / weight_sum.clamp_min(1e-6)
                    )
                    normalized_weights = torch.where(
                        weight_sum > 1e-6,
                        weights / weight_sum.clamp_min(1e-6),
                        torch.zeros_like(weights),
                    )
                    squared_weight_sum = normalized_weights.square().sum(dim=2)
                    effective_track_count = torch.where(
                        weight_sum.squeeze(-1) > 1e-6,
                        1.0 / squared_weight_sum.clamp_min(1e-6),
                        torch.zeros_like(squared_weight_sum),
                    )
                    pooling_entropy = self._normalized_entropy(
                        normalized_weights, valid_track_mask[:, None, :].expand_as(weights)
                    )
                geometry_parent = part_motion[:, :, None, :].expand(
                    -1, -1, part_motion.shape[1], -1
                )
                geometry_child = part_motion[:, None, :, :].expand(
                    -1, part_motion.shape[1], -1, -1
                )
                geometry_pair = torch.cat(
                    [
                        geometry_parent,
                        geometry_child,
                        geometry_child - geometry_parent,
                        geometry_child * geometry_parent,
                    ],
                    dim=-1,
                )
                geometry_hidden = self.geometry_backbone(geometry_pair)
                if self.axis_head_type != "direct":
                    pair_geometry = build_equivariant_pair_geometry(
                        trajectory_tokens,
                        trajectory_visibility,
                        slot_probabilities,
                        torch,
                    )
            if self.axis_head_type == "direct":
                axis = torch.nn.functional.normalize(
                    self.axis(geometry_hidden), dim=-1, eps=1e-6
                )
                pivot = self.pivot(geometry_hidden)
                axis_observable = torch.ones_like(axis[..., 0], dtype=torch.bool)
                axis_confidence = torch.ones_like(axis[..., 0])
            else:
                scalar_hidden = self.equivariant_relation_backbone(
                    torch.cat([slot_pair, pair_geometry["scalar_features"]], dim=-1)
                )
                candidates = pair_geometry["candidate_axes"]
                valid = pair_geometry["candidate_valid"]
                if self.axis_head_type == "equivariant_proposal":
                    candidate_logits = self.axis_candidate_weights(scalar_hidden)
                    axis, axis_confidence, axis_moment = aggregate_undirected_axes(
                        candidates[..., :2, :], candidate_logits, valid[..., :2], torch,
                        return_moment=True,
                    )
                    pivot_logits = self.pivot_candidate_weights(scalar_hidden)
                    pivot_weights = torch.softmax(
                        pivot_logits.masked_fill(~valid, -1e4), dim=-1
                    ) * valid.to(pivot_logits.dtype)
                    pivot_weights = pivot_weights / pivot_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    pivot = torch.sum(
                        pivot_weights[..., None] * pair_geometry["candidate_pivots"],
                        dim=-2,
                    )
                else:
                    vector_weights = self.vector_axis_weights(scalar_hidden)
                    # Candidate PCA/log axes are unoriented: their signs may
                    # independently flip after a coordinate rotation. Mixing
                    # them as signed vectors would break equivariance. Scalar
                    # VN gates therefore parameterize an unoriented second
                    # moment, whose principal eigenspace rotates exactly.
                    axis, axis_confidence, axis_moment = aggregate_undirected_axes(
                        candidates, vector_weights, valid, torch, return_moment=True
                    )
                    offset_weights = self.vector_pivot_weights(scalar_hidden)
                    gauge_axis = axis.detach()
                    if self.vector_pivot_parameterization == "legacy_center_delta":
                        center_valid = valid[..., 2].to(offset_weights.dtype)
                        offset = (
                            offset_weights[..., 2, None]
                            * center_valid[..., None]
                            * candidates[..., 2, :]
                        )
                        offset = offset - torch.sum(
                            offset * gauge_axis, dim=-1, keepdim=True
                        ) * gauge_axis
                        pivot = pair_geometry["pair_centers"] + offset
                    elif self.vector_pivot_parameterization == "analytic_plane_residual_v1":
                        pair_center = pair_geometry["pair_centers"]
                        analytic_pivot = pair_geometry["candidate_pivots"][..., 1, :]
                        analytic_valid = valid[..., 1]
                        base_pivot = torch.where(
                            analytic_valid[..., None], analytic_pivot, pair_center
                        )

                        # Build an axis-sign-invariant local plane using P=I-aa^T.
                        # Both basis vectors and their learned combination rotate with
                        # the object; neither introduces a preferred world direction.
                        center_delta = candidates[..., 2, :]
                        analytic_delta = analytic_pivot - pair_center
                        basis_1 = center_delta - torch.sum(
                            center_delta * gauge_axis, dim=-1, keepdim=True
                        ) * gauge_axis
                        basis_1_norm = torch.linalg.vector_norm(
                            basis_1, dim=-1, keepdim=True
                        )
                        unit_1 = basis_1 / basis_1_norm.clamp_min(1e-6)
                        basis_2 = analytic_delta - torch.sum(
                            analytic_delta * gauge_axis, dim=-1, keepdim=True
                        ) * gauge_axis
                        basis_2 = basis_2 - torch.sum(
                            basis_2 * unit_1, dim=-1, keepdim=True
                        ) * unit_1
                        basis_2_norm = torch.linalg.vector_norm(
                            basis_2, dim=-1, keepdim=True
                        )
                        unit_2 = basis_2 / basis_2_norm.clamp_min(1e-6)
                        unit_1 = torch.where(basis_1_norm > 1e-6, unit_1, torch.zeros_like(unit_1))
                        unit_2 = torch.where(basis_2_norm > 1e-6, unit_2, torch.zeros_like(unit_2))
                        local_scale = torch.maximum(
                            basis_1_norm, torch.linalg.vector_norm(
                                analytic_delta, dim=-1, keepdim=True
                            )
                        ).clamp(0.05, 1.0)
                        coefficients = 0.5 * torch.tanh(offset_weights[..., :2])
                        residual = local_scale * (
                            coefficients[..., 0, None] * unit_1
                            + coefficients[..., 1, None] * unit_2
                        )
                        pivot = base_pivot + residual
                    else:
                        pair_center = pair_geometry["pair_centers"]
                        raw_basis = pair_geometry["pivot_basis_vectors"]
                        basis_1 = raw_basis[..., 0, :] - torch.sum(
                            raw_basis[..., 0, :] * gauge_axis, dim=-1, keepdim=True
                        ) * gauge_axis
                        basis_1_norm = torch.linalg.vector_norm(
                            basis_1, dim=-1, keepdim=True
                        )
                        unit_1 = basis_1 / basis_1_norm.clamp_min(1e-6)
                        basis_2 = raw_basis[..., 1, :] - torch.sum(
                            raw_basis[..., 1, :] * gauge_axis, dim=-1, keepdim=True
                        ) * gauge_axis
                        basis_2 = basis_2 - torch.sum(
                            basis_2 * unit_1, dim=-1, keepdim=True
                        ) * unit_1
                        basis_2_norm = torch.linalg.vector_norm(
                            basis_2, dim=-1, keepdim=True
                        )
                        unit_2 = basis_2 / basis_2_norm.clamp_min(1e-6)
                        unit_1 = torch.where(
                            basis_1_norm > 1e-6, unit_1, torch.zeros_like(unit_1)
                        )
                        unit_2 = torch.where(
                            basis_2_norm > 1e-6, unit_2, torch.zeros_like(unit_2)
                        )
                        local_scale = torch.maximum(
                            basis_1_norm,
                            torch.linalg.vector_norm(raw_basis[..., 1, :], dim=-1, keepdim=True),
                        ).clamp(0.05, 1.0)
                        coefficients = 2.0 * torch.tanh(offset_weights[..., :2])
                        pivot = pair_center + local_scale * (
                            coefficients[..., 0, None] * unit_1
                            + coefficients[..., 1, None] * unit_2
                        )
                axis_observable = valid.any(dim=-1)
                axis = torch.where(axis_observable[..., None], axis, torch.zeros_like(axis))
            output = {
                "edge_logits": self.edge(
                    scalar_hidden if self.axis_head_type != "direct" else hidden
                ).squeeze(-1),
                "type_logits": self.joint_type(
                    scalar_hidden if self.axis_head_type != "direct" else hidden
                ),
                "axes": axis,
                "pivots": pivot,
                "axis_observable": axis_observable,
                "axis_confidence": axis_confidence,
            }
            if self.axis_head_type != "direct":
                output["axis_moments"] = axis_moment
            if self.axis_geometry_branch:
                output["geometry_valid_track_count"] = valid_track_count
                output["geometry_effective_track_count"] = effective_track_count
                output["geometry_pooling_entropy"] = pooling_entropy
            if pair_geometry is not None:
                output["candidate_valid"] = pair_geometry["candidate_valid"]
                output["candidate_residual"] = pair_geometry["candidate_residual"]
                output["candidate_observability"] = pair_geometry["observability"]
            return {key: value.squeeze(0) for key, value in output.items()} if single else output

        def _cross_track_part_motion(
            self, track_motion: Any, weights: Any, valid_track_mask: Any,
        ) -> tuple[Any, Any, Any]:
            batch, slots, tracks = weights.shape
            selected_count = min(self.geometry_max_tracks, tracks)
            ranking = weights.masked_fill(~valid_track_mask[:, None, :], float("-inf"))
            selected_weights, selected_indices = torch.topk(
                ranking, k=selected_count, dim=2, sorted=True
            )
            selected_valid = torch.isfinite(selected_weights)
            selected_weights = torch.where(
                selected_valid, selected_weights, torch.zeros_like(selected_weights)
            )
            expanded_motion = track_motion[:, None, :, :].expand(
                -1, slots, -1, -1
            )
            selected_motion = torch.gather(
                expanded_motion,
                dim=2,
                index=selected_indices[..., None].expand(
                    -1, -1, -1, track_motion.shape[-1]
                ),
            )
            selected_weight_sum = selected_weights.sum(dim=2, keepdim=True)
            normalized_weights = torch.where(
                selected_weight_sum > 1e-6,
                selected_weights / selected_weight_sum.clamp_min(1e-6),
                torch.zeros_like(selected_weights),
            )
            cross_track_input = torch.cat(
                [selected_motion, normalized_weights[..., None]], dim=-1
            )
            encoded = self.cross_track_encoder(
                self.cross_track_input(cross_track_input).reshape(
                    batch * slots, selected_count, track_motion.shape[-1]
                ),
                src_key_padding_mask=(~selected_valid).reshape(
                    batch * slots, selected_count
                ),
            ).reshape(batch, slots, selected_count, track_motion.shape[-1])
            part_motion = (
                encoded * normalized_weights[..., None]
            ).sum(dim=2)
            squared_weight_sum = normalized_weights.square().sum(dim=2)
            effective_count = torch.where(
                selected_weight_sum.squeeze(-1) > 1e-6,
                1.0 / squared_weight_sum.clamp_min(1e-6),
                torch.zeros_like(squared_weight_sum),
            )
            entropy = self._normalized_entropy(normalized_weights, selected_valid)
            return part_motion, effective_count, entropy

        @staticmethod
        def _normalized_entropy(weights: Any, valid: Any) -> Any:
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
            support = valid.sum(dim=-1).to(weights.dtype)
            denominator = support.clamp_min(2.0).log()
            return torch.where(support > 1.0, entropy / denominator, torch.zeros_like(entropy))

    return SlotRelationModel()


def _relation_targets(sample: dict[str, Any], logits: Any, max_slots: int, torch: Any, device: str) -> dict[str, Any]:
    labels = sample["labels"]
    target_slots = _hungarian_slot_targets(logits, labels, max_slots, torch)
    label_to_slot: dict[int, int] = {}
    for label, slot in zip(labels.tolist(), target_slots):
        label_to_slot[int(label)] = int(slot)
    edge = torch.zeros((max_slots, max_slots), dtype=torch.float32, device=device)
    joint_type = torch.zeros((max_slots, max_slots), dtype=torch.long, device=device)
    axis = torch.zeros((max_slots, max_slots, 3), dtype=torch.float32, device=device)
    pivot = torch.zeros((max_slots, max_slots, 3), dtype=torch.float32, device=device)
    relation_rows = []
    for relation in sample["gt_relations"]:
        parent = label_to_slot[int(relation["parent_label"])]
        child = label_to_slot[int(relation["child_label"])]
        edge[parent, child] = 1.0
        joint_type[parent, child] = JOINT_TYPES.index(str(relation["joint_type"]))
        axis[parent, child] = torch.as_tensor(relation["axis"], dtype=torch.float32, device=device)
        pivot[parent, child] = torch.as_tensor(relation["pivot"], dtype=torch.float32, device=device)
        relation_rows.append({**relation, "parent_slot": parent, "child_slot": child})
    diagonal = torch.eye(max_slots, dtype=torch.bool, device=device)
    active_slot = torch.zeros(max_slots, dtype=torch.bool, device=device)
    if label_to_slot:
        active_slot[torch.as_tensor(
            sorted(set(label_to_slot.values())), dtype=torch.long, device=device
        )] = True
    # Slot existence supervises unmatched proposals. Including every unmatched
    # slot pair in the edge BCE overwhelms the few true kinematic edges and
    # teaches the relation head to predict an empty graph.
    valid_pair = active_slot[:, None] & active_slot[None, :] & ~diagonal
    return {
        "edge": edge,
        "joint_type": joint_type,
        "axis": axis,
        "pivot": pivot,
        "valid_pair": valid_pair,
        "active_slot": active_slot,
        "positive": edge.bool(),
        "track_target": torch.as_tensor(target_slots, dtype=torch.long, device=device),
        "relations": relation_rows,
    }


def _augment_slot_probabilities(
    probabilities: Any, torch: Any, config: SlotRelationTrainingConfig,
) -> Any:
    """Mix a subset of track assignments toward another slot during training."""
    probability = float(config.slot_contamination_probability)
    fraction = float(config.slot_contamination_fraction)
    if probability <= 0.0 or fraction <= 0.0 or random.random() >= probability:
        return probabilities
    flat = probabilities.reshape(-1, probabilities.shape[-2], probabilities.shape[-1])
    augmented = []
    for rows in flat:
        count = max(1, int(round(rows.shape[0] * min(fraction, 1.0))))
        selected = torch.randperm(rows.shape[0], device=rows.device)[:count]
        # A cyclic slot shift simulates moving/static contamination without
        # introducing a preferred slot identity.
        strength = torch.zeros((rows.shape[0], 1), device=rows.device, dtype=rows.dtype)
        strength[selected] = 0.5 + 0.4 * torch.rand(
            (count, 1), device=rows.device, dtype=rows.dtype,
        )
        contaminated = torch.roll(rows, shifts=1, dims=-1)
        augmented.append(rows * (1.0 - strength) + contaminated * strength)
    return torch.stack(augmented, dim=0).reshape_as(probabilities)


def _relation_sample_loss(
    sample: dict[str, Any], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str, config: SlotRelationTrainingConfig,
    joint_type_weights: Any,
) -> dict[str, Any]:
    features = torch.from_numpy(sample["features"]).to(device)
    slot_features = torch.from_numpy(sample.get("slot_features", sample["features"])).to(device)
    if config.unfreeze_slot_backbone:
        logits, existence_logits, slots = slot_model((slot_features - mean) / std, return_slots=True)
    else:
        with torch.no_grad():
            logits, existence_logits, slots = slot_model((slot_features - mean) / std, return_slots=True)
    probabilities = torch.softmax(logits, dim=-1)
    probabilities = _augment_slot_probabilities(probabilities, torch, config)
    motion_summary = _slot_motion_summary(
        features, probabilities, int(sample["embedding_dim"]), torch
    )
    trajectory_tokens, trajectory_visibility = _trajectory_tensors(
        sample, torch, device, sample_count=int(config.trajectory_samples),
        quality_weighted=bool(config.quality_weighted_trajectories),
        robust_segment_weights=bool(config.robust_segment_weights),
    )
    prediction = relation_model(
        slots, motion_summary,
        trajectory_tokens=trajectory_tokens,
        trajectory_visibility=trajectory_visibility,
        slot_probabilities=probabilities,
    )
    losses = _relation_loss_from_outputs(
        sample, logits, existence_logits, slots, prediction,
        torch, device, config, joint_type_weights,
    )
    return _add_equivariance_loss(
        losses, sample, logits, prediction, slot_model, relation_model,
        mean, std, torch, device, config,
    )


def _relation_batch_loss(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str, config: SlotRelationTrainingConfig,
    joint_type_weights: Any,
) -> dict[str, Any]:
    """Run slot and relation forward passes for padded variable-size objects.

    Hungarian matching and the articulation replay objective remain per-object,
    because their labels and temporal reference groups are object-specific.
    """
    if len(samples) == 1:
        return _relation_sample_loss(
            samples[0], slot_model, relation_model, mean, std, torch, device,
            config, joint_type_weights,
        )
    feature_dim = int(samples[0]["features"].shape[1])
    max_tracks = max(int(sample["features"].shape[0]) for sample in samples)
    batch_features = torch.zeros(
        (len(samples), max_tracks, feature_dim), dtype=torch.float32, device=device,
    )
    batch_slot_features = torch.zeros_like(batch_features)
    padding_mask = torch.ones((len(samples), max_tracks), dtype=torch.bool, device=device)
    trajectory_rows = []
    trajectory_visibility_rows = []
    counts: list[int] = []
    for index, sample in enumerate(samples):
        count = int(sample["features"].shape[0])
        counts.append(count)
        batch_features[index, :count] = torch.from_numpy(sample["features"]).to(device)
        slot_features = sample.get("slot_features", sample["features"])
        batch_slot_features[index, :count] = torch.from_numpy(slot_features).to(device)
        padding_mask[index, :count] = False
        if relation_model.axis_geometry_branch:
            tokens, visible = _trajectory_tensors(
                sample, torch, device, sample_count=int(config.trajectory_samples),
                quality_weighted=bool(config.quality_weighted_trajectories),
                robust_segment_weights=bool(config.robust_segment_weights),
            )
            if tokens is None or visible is None:
                raise ValueError("Axis geometry branch requires points and visibility.")
            trajectory_rows.append(tokens)
            trajectory_visibility_rows.append(visible)
    if config.unfreeze_slot_backbone:
        logits, existence_logits, slots = slot_model(
            (batch_slot_features - mean) / std,
            return_slots=True,
            padding_mask=padding_mask,
        )
    else:
        with torch.no_grad():
            logits, existence_logits, slots = slot_model(
                (batch_slot_features - mean) / std,
                return_slots=True,
                padding_mask=padding_mask,
            )
    probabilities = torch.softmax(logits, dim=-1).masked_fill(padding_mask.unsqueeze(-1), 0.0)
    probabilities = _augment_slot_probabilities(probabilities, torch, config)
    motion_summary = _slot_motion_summary(
        batch_features, probabilities, int(samples[0]["embedding_dim"]), torch,
    )
    trajectory_tokens = trajectory_visibility = None
    if relation_model.axis_geometry_branch:
        trajectory_tokens = torch.zeros(
            (len(samples), max_tracks, int(config.trajectory_samples), 11),
            dtype=torch.float32,
            device=device,
        )
        trajectory_visibility = torch.zeros(
            (len(samples), max_tracks, int(config.trajectory_samples)),
            dtype=(torch.float32 if config.quality_weighted_trajectories else torch.bool),
            device=device,
        )
        for index, count in enumerate(counts):
            trajectory_tokens[index, :count] = trajectory_rows[index]
            trajectory_visibility[index, :count] = trajectory_visibility_rows[index]
    prediction = relation_model(
        slots, motion_summary,
        trajectory_tokens=trajectory_tokens,
        trajectory_visibility=trajectory_visibility,
        slot_probabilities=probabilities,
    )
    losses = []
    for index, sample in enumerate(samples):
        per_prediction = {key: value[index] for key, value in prediction.items()}
        per_losses = _relation_loss_from_outputs(
            sample,
            logits[index, :counts[index]],
            existence_logits[index],
            slots[index],
            per_prediction,
            torch,
            device,
            config,
            joint_type_weights,
        )
        losses.append(_add_equivariance_loss(
            per_losses, sample, logits[index, :counts[index]], per_prediction,
            slot_model, relation_model, mean, std, torch, device, config,
        ))
    return {
        key: torch.stack([loss[key] for loss in losses]).mean()
        for key in losses[0]
    }


def _target_axis_excitation(
    sample: dict[str, Any], target: dict[str, Any], axis_positive: Any,
    torch: Any, device: str,
) -> Any:
    """Measure GT-child full-path motion in canonical object units."""
    np = _require_numpy()
    count = int(axis_positive.sum().item())
    if count == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if "points" not in sample or "visibility" not in sample:
        return torch.ones(count, dtype=torch.float32, device=device)
    _, points = _canonical_replay_arrays(sample, np)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    labels = np.asarray(sample["labels"], dtype=np.int64)
    relation_by_pair = {
        (int(row["parent_slot"]), int(row["child_slot"])): row
        for row in target["relations"]
    }
    pair_indices = torch.nonzero(axis_positive, as_tuple=False).detach().cpu().numpy()
    values = []
    for parent, child in pair_indices:
        relation = relation_by_pair.get((int(parent), int(child)))
        child_label = int(relation["child_label"]) if relation is not None else -1
        selected = np.flatnonzero(labels == child_label)
        centers = []
        for frame in range(points.shape[1]):
            valid = selected[visibility[selected, frame]]
            if valid.size:
                centers.append(np.median(points[valid, frame], axis=0))
        if len(centers) < 2:
            values.append(0.0)
            continue
        centers = np.asarray(centers, dtype=np.float32)
        # Coordinate range is invariant to pull-return cancellation and robust to
        # one noisy endpoint after median aggregation across child tracks.
        extent = np.linalg.norm(
            np.percentile(centers, 95.0, axis=0) - np.percentile(centers, 5.0, axis=0)
        )
        values.append(float(extent))
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _relation_loss_from_outputs(
    sample: dict[str, Any], logits: Any, existence_logits: Any, slots: Any,
    prediction: dict[str, Any], torch: Any, device: str,
    config: SlotRelationTrainingConfig, joint_type_weights: Any,
) -> dict[str, Any]:
    target = _relation_targets(sample, logits, int(slots.shape[0]), torch, device)
    valid_pair = target["valid_pair"]
    edge_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            prediction["edge_logits"][valid_pair],
            target["edge"][valid_pair],
            pos_weight=torch.as_tensor(
                float(config.edge_positive_weight), device=device
            ),
        )
        if bool(valid_pair.any())
        else prediction["edge_logits"].sum() * 0.0
    )
    positive = target["positive"]
    if bool(positive.any()):
        positive_types = target["joint_type"][positive]
        type_loss = torch.nn.functional.cross_entropy(
            prediction["type_logits"][positive],
            positive_types,
            weight=joint_type_weights if config.joint_type_balanced_loss else None,
        )
        observable = prediction.get("axis_observable")
        axis_positive = positive & observable if observable is not None else positive
        axis_types = target["joint_type"][axis_positive]
        per_joint_weight = joint_type_weights[axis_types]
        per_joint_weight = per_joint_weight / per_joint_weight.mean().clamp_min(1e-6)
        if config.excitation_weighted_axis_loss:
            excitation = _target_axis_excitation(
                sample, target, axis_positive, torch, device
            )
            low = float(config.min_axis_excitation)
            high = max(float(config.full_axis_excitation), low + 1e-6)
            observability_weight = ((excitation - low) / (high - low)).clamp(0.0, 1.0)
            per_joint_weight = per_joint_weight * observability_weight
        dots = torch.sum(
            prediction["axes"][axis_positive] * target["axis"][axis_positive], dim=-1
        ).abs()
        # Direct angular supervision retains useful gradients in the 2-6 degree
        # regime where 1-cos(theta) becomes nearly flat.
        angular_error = torch.acos(dots.clamp(0.0, 1.0 - 1e-6))
        if bool(axis_positive.any()):
            if float(config.hard_axis_focal_gamma) > 0.0:
                normalized_error = (angular_error.detach() / (0.5 * math.pi)).clamp(0.0, 1.0)
                focal = 1.0 + normalized_error.pow(float(config.hard_axis_focal_gamma))
                focal = focal.clamp(max=max(1.0, float(config.hard_axis_max_weight)))
                per_joint_weight = per_joint_weight * focal
            weight_sum = per_joint_weight.sum()
            angular_loss = (
                (angular_error * per_joint_weight).sum() / weight_sum.clamp_min(1e-6)
                if bool(weight_sum > 0.0)
                else angular_error.sum() * 0.0
            )
            if "axis_moments" in prediction:
                target_axes = target["axis"][axis_positive]
                target_moments = torch.einsum(
                    "bi,bj->bij", target_axes, target_axes
                )
                moment_error = (
                    prediction["axis_moments"][axis_positive] - target_moments
                ).square().mean(dim=(-2, -1))
                moment_loss = (
                    (moment_error * per_joint_weight).sum() / weight_sum.clamp_min(1e-6)
                    if bool(weight_sum > 0.0)
                    else moment_error.sum() * 0.0
                )
                # Preserve angular loss as the reported forward value while
                # differentiating through the stable sign-invariant projector.
                axis_loss = angular_loss.detach() + moment_loss - moment_loss.detach()
            else:
                axis_loss = angular_loss
        else:
            axis_loss = prediction["axes"].sum() * 0.0
        delta = prediction["pivots"][axis_positive] - target["pivot"][axis_positive]
        line_distance = torch.linalg.vector_norm(
            torch.linalg.cross(delta, target["axis"][axis_positive], dim=-1), dim=-1
        )
        revolute = axis_types == JOINT_TYPES.index("revolute")
        line_loss = (
            (line_distance[revolute] * per_joint_weight[revolute]).sum()
            / per_joint_weight[revolute].sum().clamp_min(1e-6)
            if bool(revolute.any())
            else line_distance.sum() * 0.0
        )
        replay_prediction = prediction
        if "axis_moments" in prediction:
            replay_prediction = dict(prediction)
            replay_prediction["axes"] = prediction["axes"].detach()
        joint_replay_loss = _joint_model_replay_loss(
            replay_prediction, target, sample, torch, device
        )
    else:
        zero = prediction["edge_logits"].sum() * 0.0
        type_loss = axis_loss = line_loss = joint_replay_loss = zero
    if config.unfreeze_slot_backbone:
        slot_assignment_loss = _part_balanced_cross_entropy(
            logits, target["track_target"], torch, enabled=True
        )
        probabilities = torch.softmax(logits, dim=-1)
        slot_dice_loss = _matched_slot_dice_loss(
            probabilities, target["track_target"], torch
        )
        slot_pairwise_loss = _sampled_pairwise_assignment_loss(
            probabilities,
            target["track_target"],
            sample["references"],
            torch,
            random.Random(int(config.seed)),
            max_pairs=max(0, int(config.slot_pair_samples_per_object)),
        )
        slot_rigid_loss = _weighted_rigid_replay_loss(
            probabilities,
            sample["references"],
            sample["points"],
            sample["visibility"],
            torch,
            device,
        )
        existence_target = torch.zeros_like(existence_logits)
        existence_target[torch.unique(target["track_target"])] = 1.0
        slot_existence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            existence_logits, existence_target
        )
    else:
        slot_assignment_loss = logits.sum() * 0.0
        slot_dice_loss = logits.sum() * 0.0
        slot_pairwise_loss = logits.sum() * 0.0
        slot_rigid_loss = logits.sum() * 0.0
        slot_existence_loss = existence_logits.sum() * 0.0
    relation_total = (
        edge_loss
        + float(config.joint_type_loss_weight) * type_loss
        + float(config.axis_loss_weight) * axis_loss
        + float(config.axis_line_loss_weight) * line_loss
        + float(config.joint_replay_loss_weight) * joint_replay_loss
    )
    slot_total = (
        float(config.slot_assignment_loss_weight) * slot_assignment_loss
        + float(config.slot_dice_loss_weight) * slot_dice_loss
        + float(config.slot_pairwise_loss_weight) * slot_pairwise_loss
        + float(config.slot_rigid_loss_weight) * slot_rigid_loss
        + float(config.slot_existence_loss_weight) * slot_existence_loss
    )
    total = slot_total if config.relation_train_scope == "slot_only" else relation_total + slot_total
    return {
        "total": total,
        "edge": edge_loss,
        "type": type_loss,
        "axis": axis_loss,
        "axis_line": line_loss,
        "joint_replay": joint_replay_loss,
        "slot_assignment": slot_assignment_loss,
        "slot_dice": slot_dice_loss,
        "slot_pairwise": slot_pairwise_loss,
        "slot_rigid": slot_rigid_loss,
        "slot_existence": slot_existence_loss,
        "equivariance": prediction["edge_logits"].sum() * 0.0,
        "equivariance_axis_line": prediction["edge_logits"].sum() * 0.0,
        "equivariance_edge": prediction["edge_logits"].sum() * 0.0,
        "equivariance_type": prediction["edge_logits"].sum() * 0.0,
        "slot_consistency": prediction["edge_logits"].sum() * 0.0,
    }


def _add_equivariance_loss(
    losses: dict[str, Any], sample: dict[str, Any], augmented_logits: Any,
    augmented_prediction: dict[str, Any], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str,
    config: SlotRelationTrainingConfig,
) -> dict[str, Any]:
    axis_weight = float(config.axis_equivariance_loss_weight)
    line_weight = float(config.axis_line_equivariance_loss_weight)
    edge_weight = float(config.edge_consistency_loss_weight)
    type_weight = float(config.type_consistency_loss_weight)
    slot_weight = float(config.slot_assignment_consistency_loss_weight)
    source = sample.get("_augmentation_source")
    rotation = sample.get("_augmentation_rotation")
    if (
        max(axis_weight, line_weight, edge_weight, type_weight, slot_weight) <= 0.0
        or not isinstance(source, dict)
        or rotation is None
    ):
        return losses
    source_features = torch.from_numpy(source["features"]).to(device)
    source_slot_features = torch.from_numpy(
        source.get("slot_features", source["features"])
    ).to(device)
    source_logits, _, source_slots = slot_model(
        (source_slot_features - mean) / std, return_slots=True
    )
    source_probabilities = torch.softmax(source_logits, dim=-1)
    augmented_probabilities = torch.softmax(augmented_logits, dim=-1)
    source_trajectory_tokens, source_trajectory_visibility = _trajectory_tensors(
        source, torch, device, sample_count=int(config.trajectory_samples),
        quality_weighted=bool(config.quality_weighted_trajectories),
        robust_segment_weights=bool(config.robust_segment_weights),
    )
    source_prediction = relation_model(
        source_slots,
        _slot_motion_summary(
            source_features, source_probabilities, int(source["embedding_dim"]), torch
        ),
        trajectory_tokens=source_trajectory_tokens,
        trajectory_visibility=source_trajectory_visibility,
        slot_probabilities=source_probabilities,
    )
    target = _relation_targets(
        sample, augmented_logits, int(augmented_prediction["axes"].shape[0]), torch, device
    )
    positive = target["positive"]
    if bool(positive.any()):
        rotation_tensor = torch.as_tensor(rotation, dtype=torch.float32, device=device)
        angular_equivariance = _undirected_axis_equivariance_loss(
            source_prediction["axes"][positive],
            augmented_prediction["axes"][positive],
            rotation_tensor,
            torch,
        )
        if (
            "axis_moments" in source_prediction
            and "axis_moments" in augmented_prediction
        ):
            source_moment = source_prediction["axis_moments"][positive]
            expected_moment = (
                rotation_tensor @ source_moment @ rotation_tensor.T
            )
            moment_equivariance = torch.nn.functional.mse_loss(
                augmented_prediction["axis_moments"][positive], expected_moment
            )
            axis_equivariance = (
                angular_equivariance.detach()
                + moment_equivariance
                - moment_equivariance.detach()
            )
        else:
            axis_equivariance = angular_equivariance
    else:
        axis_equivariance = augmented_prediction["edge_logits"].sum() * 0.0
    rotation_tensor = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    source_pivot_rotated = source_prediction["pivots"] @ rotation_tensor.T
    pivot_delta = augmented_prediction["pivots"] - source_pivot_rotated
    line_distance = torch.linalg.vector_norm(
        torch.linalg.cross(
            pivot_delta, augmented_prediction["axes"].detach(), dim=-1
        ),
        dim=-1,
    )
    revolute_positive = positive & (
        target["joint_type"] == JOINT_TYPES.index("revolute")
    )
    line_equivariance = (
        line_distance[revolute_positive].mean()
        if bool(revolute_positive.any())
        else line_distance.sum() * 0.0
    )
    valid_pair = target["valid_pair"]
    if bool(valid_pair.any()):
        edge_equivariance = torch.nn.functional.mse_loss(
            torch.sigmoid(source_prediction["edge_logits"][valid_pair]),
            torch.sigmoid(augmented_prediction["edge_logits"][valid_pair]),
        )
        type_equivariance = torch.nn.functional.mse_loss(
            torch.softmax(source_prediction["type_logits"][valid_pair], dim=-1),
            torch.softmax(augmented_prediction["type_logits"][valid_pair], dim=-1),
        )
    else:
        edge_equivariance = augmented_prediction["edge_logits"].sum() * 0.0
        type_equivariance = augmented_prediction["type_logits"].sum() * 0.0
    result = dict(losses)
    result["equivariance"] = axis_equivariance
    result["equivariance_axis_line"] = line_equivariance
    result["equivariance_edge"] = edge_equivariance
    result["equivariance_type"] = type_equivariance
    result["slot_consistency"] = _permutation_aligned_slot_consistency_loss(
        source_probabilities, augmented_probabilities, torch
    )
    result["total"] = (
        result["total"]
        + axis_weight * axis_equivariance
        + line_weight * line_equivariance
        + edge_weight * edge_equivariance
        + type_weight * type_equivariance
        + slot_weight * result["slot_consistency"]
    )
    return result


def _permutation_aligned_slot_consistency_loss(
    source_probabilities: Any, rotated_probabilities: Any, torch: Any,
) -> Any:
    """Compare paired assignments after removing arbitrary slot permutations."""
    if source_probabilities.shape != rotated_probabilities.shape:
        raise ValueError("Paired slot assignments must have identical shapes.")
    overlap = source_probabilities.detach().T @ rotated_probabilities.detach()
    source_slots, rotated_slots = linear_sum_assignment(-overlap.cpu().numpy())
    aligned = torch.zeros_like(rotated_probabilities)
    aligned[:, torch.as_tensor(source_slots, device=aligned.device)] = rotated_probabilities[
        :, torch.as_tensor(rotated_slots, device=aligned.device)
    ]
    # MSE is bounded and does not encourage either branch to sharpen merely to
    # reduce the consistency term; supervised assignment losses retain purity.
    return torch.nn.functional.mse_loss(aligned, source_probabilities)


def _undirected_axis_equivariance_loss(
    source_axes: Any, rotated_axes: Any, rotation: Any, torch: Any,
) -> Any:
    expected = source_axes @ rotation.T
    expected = torch.nn.functional.normalize(expected, dim=-1, eps=1e-6)
    observed = torch.nn.functional.normalize(rotated_axes, dim=-1, eps=1e-6)
    dots = torch.sum(expected * observed, dim=-1).abs().clamp(0.0, 1.0)
    return (1.0 - dots).mean()


def _slot_motion_summary(features: Any, probabilities: Any, embedding_dim: int, torch: Any) -> Any:
    """Softly aggregate the complete canonical 3D trajectory descriptor per slot."""
    scalar = features[:, embedding_dim:]
    if features.ndim == 3:
        scalar = features[:, :, embedding_dim:]
        weights = probabilities.transpose(1, 2)
        return (weights @ scalar) / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
    weights = probabilities.T
    return (weights @ scalar) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _trajectory_tensors(
    sample: dict[str, Any], torch: Any, device: str, *, sample_count: int = 32,
    quality_weighted: bool = False,
    min_quality: float = 0.0,
    robust_segment_weights: bool = False,
) -> tuple[Any | None, Any | None]:
    """Build ordered per-track motion tokens from the full canonical trajectory."""
    if "points" not in sample or "visibility" not in sample:
        return None, None
    np = _require_numpy()
    references, points = _canonical_replay_arrays(sample, np)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    frame_count = int(points.shape[1])
    target_count = max(2, int(sample_count))
    indices = np.linspace(0, max(0, frame_count - 1), target_count).round().astype(np.int64)
    positions = points[:, indices] - references[:, None, :]
    visible = visibility[:, indices]
    observation_weights = visible.astype(np.float32)
    if quality_weighted and "observation_quality" in sample:
        quality = np.asarray(sample["observation_quality"], dtype=np.float32)
        if quality.shape == visibility.shape:
            observation_weights *= np.clip(quality[:, indices], 0.0, 1.0)
            if min_quality > 0.0:
                observation_weights[observation_weights < min_quality] = 0.0
    if robust_segment_weights:
        step = np.linalg.norm(positions[:, 1:] - positions[:, :-1], axis=-1)
        valid_step = visible[:, 1:] & visible[:, :-1]
        masked = np.where(valid_step, step, np.nan)
        median = np.nanmedian(masked, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(masked - median), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * np.nan_to_num(mad, nan=0.0), 0.005)
        residual = np.abs(step - np.nan_to_num(median, nan=0.0))
        huber = np.minimum(1.0, 0.005 / np.maximum(residual, 1e-8))
        hard_jump = valid_step & (step > np.maximum(0.02, np.nan_to_num(median, nan=0.0) + 6.0 * scale))
        observation_weights[:, 1:] *= np.where(valid_step, huber, 0.0)
        observation_weights[:, 1:][hard_jump] = 0.0
    positions = np.where(visible[..., None], positions, 0.0)
    velocity = np.zeros_like(positions)
    valid_steps = visible[:, 1:] & visible[:, :-1]
    velocity[:, 1:] = np.where(
        valid_steps[..., None], positions[:, 1:] - positions[:, :-1], 0.0
    )
    if robust_segment_weights:
        velocity[:, 1:][hard_jump] = 0.0
    time_values = np.linspace(0.0, 1.0, target_count, dtype=np.float32)
    time_channel = np.broadcast_to(
        time_values[None, :, None], (*positions.shape[:2], 1)
    )
    tokens = np.concatenate(
        [
            np.broadcast_to(
                references[:, None, :].astype(np.float32), positions.shape
            ),
            positions.astype(np.float32),
            velocity.astype(np.float32),
            visible[..., None].astype(np.float32),
            time_channel,
        ],
        axis=-1,
    )
    return (
        torch.as_tensor(tokens, dtype=torch.float32, device=device),
        torch.as_tensor(
            observation_weights if (quality_weighted or robust_segment_weights) else visible,
            dtype=(torch.float32 if (quality_weighted or robust_segment_weights) else torch.bool),
            device=device,
        ),
    )


def _apply_slot_input_mode(
    samples: list[dict[str, Any]], mode: str, feature_mean: Any,
) -> list[dict[str, Any]]:
    """Mask one slot-input branch at its training-distribution mean."""
    if mode not in {"geometry_only", "embedding_only"}:
        if mode == "full":
            return samples
        raise ValueError(f"Unsupported slot input mode: {mode}")
    result = []
    for sample in samples:
        updated = dict(sample)
        features = sample["features"].copy()
        embedding_dim = int(sample["embedding_dim"])
        if mode == "geometry_only":
            features[:, :embedding_dim] = feature_mean[:embedding_dim]
        else:
            features[:, embedding_dim:] = feature_mean[embedding_dim:]
        updated["slot_features"] = features
        result.append(updated)
    return result


def _apply_slot_geometry_representation(
    samples: list[dict[str, Any]], mode: str, np: Any,
) -> list[dict[str, Any]]:
    """Adapt only slot identity geometry; relation trajectories stay vector-valued."""
    if mode not in {"raw", "invariant_v1"}:
        raise ValueError(f"Unsupported slot geometry representation: {mode}")
    result = []
    for sample in samples:
        updated = dict(sample)
        source = sample.get("slot_features", sample["features"])
        features = np.asarray(source, dtype=np.float32).copy()
        if mode == "invariant_v1":
            embedding_dim = int(sample["embedding_dim"])
            geometry = features[:, embedding_dim:]
            if geometry.shape[1] != 32:
                raise ValueError(
                    f"invariant_v1 expects 32 geometry channels, got {geometry.shape[1]}"
                )
            reference = geometry[:, 0:3]
            endpoint = geometry[:, 3:6]
            sampled = geometry[:, 8:].reshape(len(features), 8, 3)
            eps = 1e-6

            reference_norm = np.linalg.norm(reference, axis=1)
            endpoint_norm = np.linalg.norm(endpoint, axis=1)
            reference_endpoint_cos = np.sum(reference * endpoint, axis=1) / np.maximum(
                reference_norm * endpoint_norm, eps
            )
            sampled_norm = np.linalg.norm(sampled, axis=2)
            sampled_endpoint_cos = np.sum(sampled * endpoint[:, None, :], axis=2) / np.maximum(
                sampled_norm * endpoint_norm[:, None], eps
            )
            sampled_steps = np.linalg.norm(
                np.diff(sampled, axis=1, prepend=np.zeros_like(sampled[:, :1])), axis=2
            )

            invariant = np.empty_like(geometry)
            invariant[:, 0:3] = np.stack(
                [reference_norm, endpoint_norm, reference_endpoint_cos], axis=1
            )
            invariant[:, 3:6] = np.stack(
                [sampled_norm.mean(axis=1), sampled_norm.max(axis=1), sampled_steps.sum(axis=1)],
                axis=1,
            )
            invariant[:, 6:8] = geometry[:, 6:8]
            invariant[:, 8:] = np.stack(
                [sampled_norm, sampled_endpoint_cos, sampled_steps], axis=2
            ).reshape(len(features), -1)
            features[:, embedding_dim:] = invariant
        updated["slot_features"] = features
        updated["_slot_geometry_representation"] = mode
        result.append(updated)
    return result


def _joint_type_weights(samples: list[dict[str, Any]], torch: Any, device: str) -> Any:
    counts = [0 for _ in JOINT_TYPES]
    for sample in samples:
        for relation in sample["gt_relations"]:
            counts[JOINT_TYPES.index(str(relation["joint_type"]))] += 1
    observed = [count for count in counts if count > 0]
    total = float(sum(observed))
    class_count = max(1, len(observed))
    weights = [total / (class_count * count) if count > 0 else 1.0 for count in counts]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _augment_relation_sample(
    sample: dict[str, Any], *, rng: Any, np: Any,
    sampling_mode: str = "uniform_quaternion",
    rotation: Any | None = None,
    rotate_slot_geometry: bool = False,
) -> dict[str, Any]:
    """Apply one shared random SO(3) transform to inputs and joint labels."""
    rotation = (
        np.asarray(rotation, dtype=np.float32)
        if rotation is not None
        else sample_so3(rng, mode=sampling_mode, np_module=np)
    )
    result = dict(sample)
    result["_augmentation_source"] = sample
    result["_augmentation_rotation"] = rotation
    # The frozen 2D/3D slot encoder is not SO(3)-equivariant. Preserve its
    # original input and rotate only the relation head's geometric evidence.
    result["slot_features"] = sample.get("slot_features", sample["features"])
    features = sample["features"].copy()
    offset = int(sample["embedding_dim"])
    geometry = features[:, offset:]
    geometry[:, 0:3] = geometry[:, 0:3] @ rotation.T
    geometry[:, 3:6] = geometry[:, 3:6] @ rotation.T
    sampled = geometry[:, 8:].reshape(len(features), -1, 3)
    geometry[:, 8:] = (sampled @ rotation.T).reshape(len(features), -1)
    result["features"] = features
    if rotate_slot_geometry:
        # Opaque embeddings remain unchanged, but every explicit geometric
        # descriptor consumed by the Slot encoder is now consistent.
        if sample.get("_slot_geometry_representation", "raw") == "raw":
            result["slot_features"] = features
    result["gt_relations"] = [
        {
            **relation,
            "axis": (np.asarray(relation["axis"], dtype=np.float32) @ rotation.T).tolist(),
            "pivot": (np.asarray(relation["pivot"], dtype=np.float32) @ rotation.T).tolist(),
        }
        for relation in sample["gt_relations"]
    ]
    references, points = _canonical_replay_arrays(sample, np)
    result["replay_references"] = references @ rotation.T
    result["replay_points"] = points @ rotation.T
    return result


def _augment_temporal_occlusion(
    sample: dict[str, Any], *, rng: Any, np: Any,
    min_fraction: float = 0.15, max_fraction: float = 0.45,
    track_fraction: float = 0.5,
) -> dict[str, Any]:
    """Hide contiguous track intervals while retaining evidence on both sides."""
    if "visibility" not in sample or "points" not in sample:
        return sample
    visibility = np.asarray(sample["visibility"], dtype=bool).copy()
    if visibility.ndim != 2 or visibility.shape[1] < 5:
        return sample
    track_count, frame_count = visibility.shape
    min_fraction = max(0.0, min(float(min_fraction), 0.9))
    max_fraction = max(min_fraction, min(float(max_fraction), 0.9))
    track_fraction = max(0.0, min(float(track_fraction), 1.0))
    selected_count = max(1, int(round(track_count * track_fraction)))
    selected = rng.sample(range(track_count), k=min(selected_count, track_count))
    block_fraction = rng.uniform(min_fraction, max_fraction)
    block_length = max(1, min(frame_count - 2, int(round(frame_count * block_fraction))))
    # A shared occlusion event with small per-track jitter models one occluder
    # crossing a spatially coherent group without identical cut boundaries.
    shared_start = rng.randint(1, max(1, frame_count - block_length - 1))
    jitter_limit = max(1, block_length // 8)
    changed = 0
    for track_index in selected:
        jitter = rng.randint(-jitter_limit, jitter_limit)
        start = max(1, min(shared_start + jitter, frame_count - block_length - 1))
        end = start + block_length
        candidate = visibility[track_index].copy()
        candidate[start:end] = False
        # Keep at least one valid observation before and after the synthetic gap.
        if candidate[:start].any() and candidate[end:].any():
            visibility[track_index] = candidate
            changed += 1
    if changed == 0:
        return sample
    result = dict(sample)
    result["visibility"] = visibility
    result["_temporal_occlusion"] = {
        "selected_tracks": changed,
        "block_length": block_length,
        "frame_count": frame_count,
    }
    return result


def _augment_trajectory_corruption(
    sample: dict[str, Any], *, rng: Any, np: Any,
    track_fraction: float = 0.25,
    drift_scale_fraction: float = 0.03,
    spike_probability: float = 0.15,
    coherent_drift_probability: float = 0.0,
    recovery_offset_probability: float = 0.0,
    id_switch_probability: float = 0.0,
) -> dict[str, Any]:
    """Add segment drift and isolated 3D jumps without changing relation labels."""
    if "visibility" not in sample or "points" not in sample:
        return sample
    points = np.asarray(sample["points"], dtype=np.float32).copy()
    visibility = np.asarray(sample["visibility"], dtype=bool)
    if points.ndim != 3 or visibility.shape != points.shape[:2]:
        return sample
    track_count, frame_count = visibility.shape
    selected_count = min(track_count, max(1, int(round(track_count * max(0.0, min(track_fraction, 1.0))))))
    coherent = rng.random() < coherent_drift_probability and "references" in sample
    if coherent:
        references = np.asarray(sample["references"], dtype=np.float32)
        anchor = rng.randrange(track_count)
        distances = np.linalg.norm(references - references[anchor], axis=-1)
        selected = np.argsort(distances)[:selected_count].tolist()
    else:
        selected = rng.sample(range(track_count), k=selected_count)
    scale = max(float(sample.get("canonical_scale_m", 1.0)), 1e-6)
    drift_limit = max(0.0, float(drift_scale_fraction)) * scale
    changed = 0
    spike_count = 0
    recovery_count = 0
    shared_direction = np.asarray(
        [rng.gauss(0.0, 1.0) for _ in range(3)], dtype=np.float32
    )
    shared_direction /= max(float(np.linalg.norm(shared_direction)), 1e-6)
    for track_index in selected:
        valid_frames = np.flatnonzero(visibility[track_index])
        if valid_frames.size < 4:
            continue
        start_offset = rng.randint(1, max(1, int(valid_frames.size) - 2))
        segment = valid_frames[start_offset:]
        if coherent:
            direction = shared_direction
        else:
            direction = np.asarray([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
        ramp = np.linspace(0.0, rng.uniform(0.25, 1.0) * drift_limit, len(segment), dtype=np.float32)
        points[track_index, segment] += ramp[:, None] * direction[None, :]
        if rng.random() < max(0.0, min(float(spike_probability), 1.0)):
            frame = int(rng.choice(segment.tolist()))
            spike = np.asarray([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=np.float32)
            spike /= max(float(np.linalg.norm(spike)), 1e-6)
            points[track_index, frame] += spike * rng.uniform(drift_limit, 3.0 * drift_limit)
            spike_count += 1
        if rng.random() < recovery_offset_probability:
            recovery_frame = int(rng.choice(segment.tolist()))
            points[track_index, recovery_frame:] += (
                direction * rng.uniform(0.25, 1.0) * drift_limit
            )
            recovery_count += 1
        changed += 1
    id_switch_count = 0
    if len(selected) >= 2 and rng.random() < id_switch_probability:
        first, second = rng.sample(selected, 2)
        common = np.flatnonzero(visibility[first] & visibility[second])
        if common.size >= 3:
            switch_frame = int(common[len(common) // 2])
            suffix = points[first, switch_frame:].copy()
            points[first, switch_frame:] = points[second, switch_frame:]
            points[second, switch_frame:] = suffix
            id_switch_count = 1
    if changed == 0:
        return sample
    result = dict(sample)
    result["points"] = points
    result["_trajectory_corruption"] = {
        "selected_tracks": changed,
        "spike_count": spike_count,
        "recovery_offset_count": recovery_count,
        "id_switch_count": id_switch_count,
        "drift_scale_fraction": drift_scale_fraction,
    }
    return result


def _canonical_replay_arrays(sample: dict[str, Any], np: Any) -> tuple[Any, Any]:
    if "replay_references" in sample and "replay_points" in sample:
        return sample["replay_references"], sample["replay_points"]
    center = np.asarray(sample["canonical_center_m"], dtype=np.float32)
    scale = max(float(sample["canonical_scale_m"]), 1e-6)
    references = (np.asarray(sample["references"], dtype=np.float32) - center) / scale
    points = (np.asarray(sample["points"], dtype=np.float32) - center) / scale
    return references, points


def _joint_model_replay_loss(
    prediction: dict[str, Any], target: dict[str, Any], sample: dict[str, Any],
    torch: Any, device: str,
) -> Any:
    """Replay child tracks under the predicted GT-type joint model.

    This differs from slot-level weighted Kabsch: it directly constrains the
    predicted axis and revolute axis line. Joint coordinates are fitted per
    frame and per track reference-frame group, so dynamically reseeded tracks
    do not share an invalid common zero configuration.
    """
    np = _require_numpy()
    references_np, points_np = _canonical_replay_arrays(sample, np)
    references = torch.as_tensor(references_np, dtype=torch.float32, device=device)
    points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
    visibility = torch.as_tensor(sample["visibility"], dtype=torch.bool, device=device)
    labels = torch.as_tensor(sample["labels"], dtype=torch.long, device=device)
    reference_frames = torch.as_tensor(sample["reference_frames"], dtype=torch.long, device=device)
    losses = []
    for relation in target["relations"]:
        parent, child = int(relation["parent_slot"]), int(relation["child_slot"])
        observable = prediction.get("axis_observable")
        if observable is not None and not bool(observable[parent, child]):
            continue
        child_mask = labels == int(relation["child_label"])
        axis = prediction["axes"][parent, child]
        pivot = prediction["pivots"][parent, child]
        joint_type = str(relation["joint_type"])
        for reference_frame in torch.unique(reference_frames[child_mask]).tolist():
            reference_group = child_mask & (reference_frames == int(reference_frame))
            for frame in range(points.shape[1]):
                if frame == int(reference_frame):
                    continue
                valid = reference_group & visibility[:, frame]
                if int(valid.sum().detach().cpu()) < 3:
                    continue
                source = references[valid]
                observed = points[valid, frame]
                if joint_type == "prismatic":
                    q_value = torch.mean(torch.sum((observed - source) * axis, dim=-1))
                    replay = source + q_value * axis
                elif joint_type == "revolute":
                    source_vector = source - pivot
                    observed_vector = observed - pivot
                    source_perp = source_vector - torch.sum(source_vector * axis, dim=-1, keepdim=True) * axis
                    observed_perp = observed_vector - torch.sum(observed_vector * axis, dim=-1, keepdim=True) * axis
                    cosine_vote = torch.sum(source_perp * observed_perp)
                    sine_vote = torch.sum(
                        torch.linalg.cross(source_perp, observed_perp, dim=-1) * axis
                    )
                    q_value = torch.atan2(sine_vote, cosine_vote)
                    cosine, sine = torch.cos(q_value), torch.sin(q_value)
                    replay_vector = (
                        source_vector * cosine
                        + torch.linalg.cross(axis.expand_as(source_vector), source_vector, dim=-1) * sine
                        + torch.sum(source_vector * axis, dim=-1, keepdim=True) * axis * (1.0 - cosine)
                    )
                    replay = pivot + replay_vector
                else:
                    replay = source
                squared_error = torch.sum((replay - observed) ** 2, dim=-1)
                losses.append(torch.sqrt(squared_error + 1e-8).mean())
    return torch.stack(losses).mean() if losses else prediction["edge_logits"].sum() * 0.0


def _evaluate_relation_samples(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str,
) -> dict[str, float]:
    if not samples:
        return {
            "loss": math.inf, "edge_f1": 0.0, "joint_type_accuracy": 0.0,
            "axis_error_deg": math.inf, "axis_error_deg_all_gt_pairs": math.inf,
            "axis_error_count": 0, "axis_error_all_gt_pair_count": 0,
            "axis_error_median_deg": math.inf, "axis_error_p75_deg": math.inf,
            "axis_error_p90_deg": math.inf,
            "detected_axis_error_deg": math.inf, "detected_axis_error_count": 0,
            "detected_axis_error_median_deg": math.inf,
            "axis_line_error_normalized": math.inf,
            "axis_line_error_count": 0,
            "revolute_type_accuracy": 0.0, "prismatic_type_accuracy": 0.0,
            "revolute_axis_error_deg": math.inf, "prismatic_axis_error_deg": math.inf,
            "revolute_axis_error_count": 0, "prismatic_axis_error_count": 0,
            "illegal_graph_rate": 1.0, "cycle_rate": 0.0, "multiple_parent_rate": 0.0,
            "raw_illegal_graph_rate": 1.0, "raw_cycle_rate": 0.0,
            "raw_multiple_parent_rate": 0.0,
            "slot_ari": 0.0, "slot_ri": 0.0,
            "geometry_effective_track_count": 0.0,
            "geometry_valid_track_count": 0.0,
            "geometry_pooling_entropy": 0.0,
            "candidate_coverage": 0.0,
            "axis_error_gt_10_count": 0, "axis_error_gt_30_count": 0,
            "axis_error_gt_60_count": 0, "axis_error_gt_80_count": 0,
        }
    relation_model.eval()
    slot_model.eval()
    losses, true_positive, false_positive, false_negative = [], 0, 0, 0
    type_correct, type_count = 0, 0
    axis_errors, all_axis_errors, detected_axis_errors, line_errors = [], [], [], []
    type_stats = {name: {"correct": 0, "count": 0, "axis_errors": []} for name in JOINT_TYPES}
    illegal_graphs = cycles = multiple_parents = 0
    raw_illegal_graphs = raw_cycles = raw_multiple_parents = 0
    geometry_valid_counts, geometry_effective_counts, geometry_entropies = [], [], []
    candidate_coverage_rows = []
    slot_ari_rows, slot_ri_rows = [], []
    with torch.no_grad():
        for sample in samples:
            features = torch.from_numpy(sample["features"]).to(device)
            slot_features = torch.from_numpy(
                sample.get("slot_features", sample["features"])
            ).to(device)
            logits, existence_logits, slots = slot_model(
                (slot_features - mean) / std, return_slots=True
            )
            probabilities = torch.softmax(logits, dim=-1)
            labels = sample.get("labels")
            if labels is not None:
                ari, ri = _clustering_pair_metrics(
                    logits.argmax(dim=-1).cpu().tolist(),
                    [int(value) for value in labels],
                )
                slot_ari_rows.append(ari)
                slot_ri_rows.append(ri)
            motion_summary = _slot_motion_summary(
                features, probabilities, int(sample["embedding_dim"]), torch
            )
            trajectory_tokens, trajectory_visibility = _trajectory_tensors(
                sample, torch, device,
                sample_count=int(getattr(relation_model, "trajectory_samples", 32)),
                quality_weighted=bool(
                    getattr(relation_model, "quality_weighted_trajectories", False)
                ),
                robust_segment_weights=bool(
                    getattr(relation_model, "robust_segment_weights", False)
                ),
            )
            prediction = relation_model(
                slots, motion_summary,
                trajectory_tokens=trajectory_tokens,
                trajectory_visibility=trajectory_visibility,
                slot_probabilities=probabilities,
            )
            if "geometry_valid_track_count" in prediction:
                geometry_effective_counts.extend(
                    prediction["geometry_effective_track_count"].cpu().flatten().tolist()
                )
                geometry_valid_counts.extend(
                    prediction["geometry_valid_track_count"].cpu().flatten().tolist()
                )
                geometry_entropies.extend(
                    prediction["geometry_pooling_entropy"].cpu().flatten().tolist()
                )
            target = _relation_targets(sample, logits, int(slots.shape[0]), torch, device)
            if "candidate_valid" in prediction and bool(target["positive"].any()):
                pair_valid = prediction["candidate_valid"].any(dim=-1)
                # The gate concerns whether true joints have usable geometric
                # proposals, not whether every non-edge slot pair happens to
                # admit some noisy candidate.
                candidate_coverage_rows.append(
                    float(pair_valid[target["positive"]].float().mean().cpu())
                )
            edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["edge_logits"][target["valid_pair"]], target["edge"][target["valid_pair"]]
            )
            predicted_edge = torch.sigmoid(prediction["edge_logits"]) >= 0.5
            true_positive += int((predicted_edge & target["positive"]).sum().cpu())
            false_positive += int((predicted_edge & ~target["positive"] & target["valid_pair"]).sum().cpu())
            false_negative += int((~predicted_edge & target["positive"]).sum().cpu())
            if bool(target["positive"].any()):
                pred_type = prediction["type_logits"][target["positive"]].argmax(-1)
                true_type = target["joint_type"][target["positive"]]
                type_matches = pred_type == true_type
                type_correct += int(type_matches.sum().cpu())
                type_count += int(true_type.numel())
                dots = torch.sum(
                    prediction["axes"][target["positive"]] * target["axis"][target["positive"]], dim=-1
                ).abs().clamp(0.0, 1.0)
                errors = torch.rad2deg(torch.acos(dots))
                all_axis_errors.extend(errors.cpu().tolist())
                axis_errors.extend(errors[type_matches].cpu().tolist())
                detected_type_matches = type_matches & predicted_edge[target["positive"]]
                detected_axis_errors.extend(errors[detected_type_matches].cpu().tolist())
                for type_index, type_name in enumerate(JOINT_TYPES):
                    type_mask = true_type == type_index
                    type_stats[type_name]["count"] += int(type_mask.sum().cpu())
                    type_stats[type_name]["correct"] += int((type_matches & type_mask).sum().cpu())
                    type_stats[type_name]["axis_errors"].extend(
                        errors[type_matches & type_mask].cpu().tolist()
                    )
                type_loss = torch.nn.functional.cross_entropy(
                    prediction["type_logits"][target["positive"]], true_type
                )
                axis_loss = (1.0 - dots).mean()
                delta = prediction["pivots"][target["positive"]] - target["pivot"][target["positive"]]
                line_distance = torch.linalg.vector_norm(
                    torch.linalg.cross(delta, target["axis"][target["positive"]], dim=-1), dim=-1
                )
                revolute = true_type == JOINT_TYPES.index("revolute")
                correctly_typed_revolute = revolute & type_matches
                if bool(correctly_typed_revolute.any()):
                    line_errors.extend(line_distance[correctly_typed_revolute].cpu().tolist())
                    line_loss = line_distance[revolute].mean()
                else:
                    line_loss = line_distance.sum() * 0.0
                losses.append(float((edge_loss + type_loss + axis_loss + line_loss).cpu()))
            else:
                losses.append(float(edge_loss.cpu()))
            active = torch.nonzero(
                torch.sigmoid(existence_logits) >= 0.5, as_tuple=False
            ).flatten().cpu().tolist()
            if not active:
                active = [int(torch.argmax(existence_logits).cpu())]
            pair_rows = [
                {
                    "parent_slot_id": parent,
                    "child_slot_id": child,
                    "edge_probability": float(
                        torch.sigmoid(prediction["edge_logits"][parent, child]).cpu()
                    ),
                }
                for parent in active for child in active if parent != child
            ]
            threshold_edges = [
                row for row in pair_rows if row["edge_probability"] >= 0.5
            ]
            raw_graph = graph_legality_diagnostics(active, threshold_edges)
            raw_illegal_graphs += int(not raw_graph["is_legal_tree"])
            raw_cycles += int(raw_graph["has_cycle"])
            raw_multiple_parents += int(bool(raw_graph["multiple_parent_slots"]))
            selected = decode_legal_relation_tree(active, pair_rows)
            graph = graph_legality_diagnostics(active, selected)
            illegal_graphs += int(not graph["is_legal_tree"])
            cycles += int(graph["has_cycle"])
            multiple_parents += int(bool(graph["multiple_parent_slots"]))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "loss": _mean(losses),
        "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "joint_type_accuracy": type_correct / max(1, type_count),
        "axis_error_deg": _mean(axis_errors) if axis_errors else math.inf,
        "axis_error_deg_all_gt_pairs": _mean(all_axis_errors) if all_axis_errors else math.inf,
        "axis_error_count": len(axis_errors),
        "axis_error_all_gt_pair_count": len(all_axis_errors),
        "axis_error_median_deg": _percentile(axis_errors, 0.5),
        "axis_error_p75_deg": _percentile(axis_errors, 0.75),
        "axis_error_p90_deg": _percentile(axis_errors, 0.9),
        "axis_error_gt_10_count": sum(error > 10.0 for error in axis_errors),
        "axis_error_gt_30_count": sum(error > 30.0 for error in axis_errors),
        "axis_error_gt_60_count": sum(error > 60.0 for error in axis_errors),
        "axis_error_gt_80_count": sum(error > 80.0 for error in axis_errors),
        "detected_axis_error_deg": (
            _mean(detected_axis_errors) if detected_axis_errors else math.inf
        ),
        "detected_axis_error_count": len(detected_axis_errors),
        "detected_axis_error_median_deg": _percentile(detected_axis_errors, 0.5),
        "axis_line_error_normalized": _mean(line_errors) if line_errors else math.inf,
        "axis_line_error_count": len(line_errors),
        "revolute_type_accuracy": (
            type_stats["revolute"]["correct"] / max(1, type_stats["revolute"]["count"])
        ),
        "prismatic_type_accuracy": (
            type_stats["prismatic"]["correct"] / max(1, type_stats["prismatic"]["count"])
        ),
        "revolute_axis_error_deg": (
            _mean(type_stats["revolute"]["axis_errors"])
            if type_stats["revolute"]["axis_errors"] else math.inf
        ),
        "revolute_axis_error_count": len(type_stats["revolute"]["axis_errors"]),
        "prismatic_axis_error_deg": (
            _mean(type_stats["prismatic"]["axis_errors"])
            if type_stats["prismatic"]["axis_errors"] else math.inf
        ),
        "prismatic_axis_error_count": len(type_stats["prismatic"]["axis_errors"]),
        "illegal_graph_rate": illegal_graphs / len(samples),
        "cycle_rate": cycles / len(samples),
        "multiple_parent_rate": multiple_parents / len(samples),
        "raw_illegal_graph_rate": raw_illegal_graphs / len(samples),
        "raw_cycle_rate": raw_cycles / len(samples),
        "raw_multiple_parent_rate": raw_multiple_parents / len(samples),
        "slot_ari": _mean(slot_ari_rows) if slot_ari_rows else 0.0,
        "slot_ri": _mean(slot_ri_rows) if slot_ri_rows else 0.0,
        "geometry_effective_track_count": (
            _mean(geometry_effective_counts) if geometry_effective_counts else 0.0
        ),
        "geometry_valid_track_count": (
            _mean(geometry_valid_counts) if geometry_valid_counts else 0.0
        ),
        "geometry_pooling_entropy": (
            _mean(geometry_entropies) if geometry_entropies else 0.0
        ),
        "candidate_coverage": (
            _mean(candidate_coverage_rows) if candidate_coverage_rows else 0.0
        ),
    }


def _evaluate_training_objective(
    samples: list[dict[str, Any]], slot_model: Any, relation_model: Any,
    mean: Any, std: Any, torch: Any, device: str,
    config: SlotRelationTrainingConfig, joint_type_weights: Any,
) -> dict[str, float]:
    """Evaluate exactly the optimized objective for checkpoint selection."""
    if not samples:
        return {"total": math.inf}
    slot_model.eval()
    relation_model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for sample in samples:
            losses = _relation_batch_loss(
                [sample], slot_model, relation_model, mean, std, torch, device,
                config, joint_type_weights,
            )
            rows.append({key: float(value.cpu()) for key, value in losses.items()})
    return {
        key: _mean([row[key] for row in rows])
        for key in rows[0]
    }


def _clustering_pair_metrics(
    predicted: list[int], target: list[int],
) -> tuple[float, float]:
    """Return permutation-invariant adjusted Rand and Rand indices."""
    pairs = [(p, t) for p, t in zip(predicted, target) if t >= 0]
    count = len(pairs)
    if count < 2:
        return 1.0, 1.0
    contingency: dict[tuple[int, int], int] = {}
    predicted_counts: dict[int, int] = {}
    target_counts: dict[int, int] = {}
    for pred, truth in pairs:
        contingency[(pred, truth)] = contingency.get((pred, truth), 0) + 1
        predicted_counts[pred] = predicted_counts.get(pred, 0) + 1
        target_counts[truth] = target_counts.get(truth, 0) + 1
    choose2 = lambda value: value * (value - 1) / 2.0
    true_positive = sum(choose2(value) for value in contingency.values())
    predicted_positive = sum(choose2(value) for value in predicted_counts.values())
    target_positive = sum(choose2(value) for value in target_counts.values())
    total_pairs = choose2(count)
    true_negative = total_pairs - predicted_positive - target_positive + true_positive
    rand_index = (true_positive + true_negative) / total_pairs
    expected = predicted_positive * target_positive / total_pairs
    maximum = 0.5 * (predicted_positive + target_positive)
    denominator = maximum - expected
    adjusted_rand = (
        (true_positive - expected) / denominator
        if abs(denominator) > 1e-12 else 1.0
    )
    return float(adjusted_rand), float(rand_index)


def decode_legal_relation_tree(
    active_slots: list[int], pairwise_relations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Decode a deterministic high-score branching over active slots.

    Pairwise edge logits are independent, so thresholding can assign multiple
    parents to one child or create a cycle. The active articulated structure is
    a tree; greedily accepting high-score edges subject to one-parent and
    acyclicity constraints produces a legal branching without changing logits.
    """
    nodes = sorted({int(slot) for slot in active_slots})
    if len(nodes) <= 1:
        return []
    node_set = set(nodes)
    ranked = sorted(
        (
            row for row in pairwise_relations
            if int(row["parent_slot_id"]) in node_set
            and int(row["child_slot_id"]) in node_set
            and int(row["parent_slot_id"]) != int(row["child_slot_id"])
        ),
        key=lambda row: (
            -float(row.get("edge_probability", 0.0)),
            int(row["parent_slot_id"]),
            int(row["child_slot_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    parent_of: dict[int, int] = {}
    for row in ranked:
        parent = int(row["parent_slot_id"])
        child = int(row["child_slot_id"])
        if child in parent_of:
            continue
        # Following parent pointers catches exactly the cycle introduced by
        # parent -> child before mutating the current forest.
        ancestor = parent
        creates_cycle = ancestor == child
        while not creates_cycle and ancestor in parent_of:
            ancestor = parent_of[ancestor]
            creates_cycle = ancestor == child
        if creates_cycle:
            continue
        selected.append(row)
        parent_of[child] = parent
        if len(selected) == len(nodes) - 1:
            break
    return selected


def graph_legality_diagnostics(active_slots: list[int], edges: list[dict[str, Any]]) -> dict[str, Any]:
    incoming = {int(slot): 0 for slot in active_slots}
    adjacency = {int(slot): [] for slot in active_slots}
    for edge in edges:
        parent = int(edge["parent_slot_id"])
        child = int(edge["child_slot_id"])
        incoming[child] = incoming.get(child, 0) + 1
        adjacency.setdefault(parent, []).append(child)
    roots = [slot for slot in active_slots if incoming.get(slot, 0) == 0]
    multiple_parent_slots = [slot for slot, count in incoming.items() if count > 1]
    has_cycle = _has_directed_cycle(active_slots, adjacency)
    expected_edges = max(0, len(active_slots) - 1)
    return {
        "node_count": len(active_slots),
        "edge_count": len(edges),
        "expected_tree_edge_count": expected_edges,
        "root_slots": roots,
        "multiple_parent_slots": multiple_parent_slots,
        "has_cycle": has_cycle,
        "is_legal_tree": (
            len(active_slots) > 0
            and len(edges) == expected_edges
            and len(roots) == 1
            and not multiple_parent_slots
            and not has_cycle
        ),
    }


def _has_directed_cycle(nodes: list[int], adjacency: dict[int, list[int]]) -> bool:
    state: dict[int, int] = {}

    def visit(node: int) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(child) for child in adjacency.get(node, [])):
            return True
        state[node] = 2
        return False

    return any(visit(int(node)) for node in nodes if state.get(int(node), 0) == 0)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without adding a NumPy dependency."""
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
