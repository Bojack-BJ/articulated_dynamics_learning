from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.models import (
    ArticulationArtifact,
    EpisodeInput,
    JointArtifact,
    PartArtifact,
    PipelineResult,
    PrimitiveBox,
)
from ..core.serialization import load_episode, load_json, save_articulation_artifact, save_json
from ..export.urdf import URDFExporter


@dataclass(slots=True)
class InferredArticulationPipelineConfig:
    episode_path: str | Path
    part_pose_path: str | Path
    joint_inference_path: str | Path
    output_dir: str | Path


class InferredArticulationPipeline:
    def __init__(self, exporter: URDFExporter | None = None) -> None:
        self.exporter = exporter or URDFExporter()

    def run(self, config: InferredArticulationPipelineConfig) -> PipelineResult:
        episode = load_episode(config.episode_path)
        part_pose_artifact = load_json(config.part_pose_path)
        joint_inference_artifact = load_json(config.joint_inference_path)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        articulation = self._build_articulation(episode, part_pose_artifact, joint_inference_artifact)
        articulation_artifact_path = output_dir / "articulation_artifact.json"
        save_articulation_artifact(articulation, articulation_artifact_path)

        urdf_package = self.exporter.export(episode, articulation, output_dir / "urdf")
        result = PipelineResult(
            episode_id=episode.object_instance_id,
            reconstruction_artifact_path="",
            articulation_artifact_path=str(articulation_artifact_path.resolve()),
            urdf_package=urdf_package,
        )
        save_json(
            {
                **result.to_dict(),
                "source": "inferred-articulation-pipeline",
                "part_pose_path": str(Path(config.part_pose_path).resolve()),
                "joint_inference_path": str(Path(config.joint_inference_path).resolve()),
            },
            output_dir / "pipeline_result.json",
        )
        return result

    def _build_articulation(
        self,
        episode: EpisodeInput,
        part_pose_artifact: dict[str, Any],
        joint_inference_artifact: dict[str, Any],
    ) -> ArticulationArtifact:
        parts_data = part_pose_artifact.get("parts", [])
        joints_data = joint_inference_artifact.get("joints", [])
        anchor_part_id = int(part_pose_artifact.get("anchor_part_id", 0))
        anchor_name = str(part_pose_artifact.get("anchor_part_name", "base_link"))

        joint_by_child_id = {
            int(raw_joint.get("child_part_id", -1)): raw_joint
            for raw_joint in joints_data
            if isinstance(raw_joint, dict)
        }

        parts: list[PartArtifact] = []
        part_name_map: dict[int, str] = {}
        for raw_part in parts_data:
            if not isinstance(raw_part, dict):
                continue
            part_id = int(raw_part.get("part_id", 0))
            part_name = str(raw_part.get("name", f"part_{part_id}"))
            part_name_map[part_id] = part_name
            reference_pose = dict(raw_part.get("canonical_frame", {}))
            reference_bounds = dict(raw_part.get("reference_local_bounds", {}))
            lower = [float(value) for value in reference_bounds.get("lower", [-0.05, -0.05, -0.05])]
            upper = [float(value) for value in reference_bounds.get("upper", [0.05, 0.05, 0.05])]
            size = [max(1e-3, upper[index] - lower[index]) for index in range(3)]

            if part_id == anchor_part_id:
                canonical_translation = [0.0, 0.0, 0.0]
                primitive_center = [0.0, 0.0, 0.0]
                role = "static"
            else:
                joint_data = joint_by_child_id.get(part_id)
                # For moving parts we place the link/body frame at the inferred
                # joint pivot so URDF and MJCF can reuse the same joint origin.
                # The box center then stays expressed relative to that pivot.
                canonical_translation = (
                    [float(value) for value in joint_data.get("pivot", [0.0, 0.0, 0.0])]
                    if joint_data is not None
                    else [float(value) for value in reference_pose.get("translation", [0.0, 0.0, 0.0])]
                )
                primitive_center = [float(value) for value in reference_pose.get("translation", canonical_translation)]
                role = "moving"

            confidence = self._part_confidence(raw_part)
            parts.append(
                PartArtifact(
                    name=part_name,
                    role=role,
                    primitive_boxes=[
                        PrimitiveBox(
                            name=f"{part_name}_proxy",
                            center=primitive_center,
                            size=size,
                            observed_ratio=min(1.0, float(raw_part.get("reference_point_count", 1)) / 1000.0),
                            generated=False,
                            metadata={"source": "part-pose-bounds"},
                        )
                    ],
                    canonical_pose={
                        "translation": canonical_translation,
                        "rotation_rpy": [0.0, 0.0, 0.0],
                    },
                    confidence=confidence,
                )
            )

        if anchor_part_id not in part_name_map:
            part_name_map[anchor_part_id] = anchor_name

        joints: list[JointArtifact] = []
        single_joint_state: list[Any] = []
        joint_state_tracks: dict[str, list[dict[str, Any]]] = {}
        for raw_joint in joints_data:
            if not isinstance(raw_joint, dict):
                continue
            joint_name = str(raw_joint.get("name", f"joint_{raw_joint.get('child_part_id', 0)}"))
            child_part_id = int(raw_joint.get("child_part_id", 0))
            joints.append(
                JointArtifact(
                    name=joint_name,
                    joint_type=str(raw_joint.get("joint_type", "fixed")),
                    parent=str(raw_joint.get("parent_name", part_name_map.get(anchor_part_id, anchor_name))),
                    child=str(raw_joint.get("child_name", part_name_map.get(child_part_id, f"part_{child_part_id}"))),
                    axis=[float(value) for value in raw_joint.get("axis", [0.0, 0.0, 1.0])],
                    origin=[float(value) for value in raw_joint.get("pivot", [0.0, 0.0, 0.0])],
                    limits=[float(value) for value in raw_joint.get("limits", [0.0, 0.0])],
                    confidence=float(raw_joint.get("confidence", 0.5)),
                    metadata={
                        "source": "joint-inference",
                        "parent_part_id": int(raw_joint.get("parent_part_id", anchor_part_id)),
                        "child_part_id": child_part_id,
                        "metrics": dict(raw_joint.get("metrics", {})),
                    },
                )
            )
            q_samples = [
                {
                    "frame_index": int(sample.get("frame_index", 0)),
                    "timestamp_s": float(sample.get("timestamp_s", 0.0)),
                    "q": float(sample.get("q", 0.0)),
                }
                for sample in raw_joint.get("q_samples", [])
                if isinstance(sample, dict)
            ]
            joint_state_tracks[joint_name] = q_samples

        if len(joints) == 1:
            from ..core.models import StateSample
            from .refit import differentiate_joint_signal

            q_samples = joint_state_tracks.get(joints[0].name, [])
            times = [sample["timestamp_s"] for sample in q_samples]
            values = [sample["q"] for sample in q_samples]
            qdots = differentiate_joint_signal(times, values) if q_samples else []
            single_joint_state = [
                StateSample(
                    timestamp_s=sample["timestamp_s"],
                    q=sample["q"],
                    qdot=qdot,
                    confidence=joints[0].confidence,
                )
                for sample, qdot in zip(q_samples, qdots)
            ]

        articulation = ArticulationArtifact(
            parts=parts,
            joints=joints,
            state=single_joint_state,
            fit_metrics={
                "initializer": "part-pose-plus-joint-inference",
                "anchor_part_id": anchor_part_id,
                "anchor_part_name": anchor_name,
                "joint_state_tracks": joint_state_tracks,
                "source_paths": {
                    "part_pose_artifact": part_pose_artifact.get("input_path"),
                    "joint_inference_artifact": joint_inference_artifact.get("input_path"),
                },
            },
            low_confidence_components=[
                joint.name for joint in joints if joint.confidence < 0.4
            ],
        )
        return articulation

    def _part_confidence(self, raw_part: dict[str, Any]) -> float:
        samples = raw_part.get("samples", [])
        valid = [float(sample.get("confidence", 0.0)) for sample in samples if bool(sample.get("valid", False))]
        if not valid:
            return 0.3
        return min(1.0, sum(valid) / len(valid))
