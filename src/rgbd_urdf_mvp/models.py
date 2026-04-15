from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Matrix4 = list[list[float]]
Vector3 = list[float]


@dataclass(slots=True)
class PrimitiveBox:
    name: str
    center: Vector3
    size: Vector3
    observed_ratio: float
    generated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PrimitiveBox":
        return PrimitiveBox(
            name=data["name"],
            center=list(data["center"]),
            size=list(data["size"]),
            observed_ratio=float(data.get("observed_ratio", 1.0)),
            generated=bool(data.get("generated", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class FrameObservation:
    timestamp_s: float
    rgb_path: str
    depth_path: str
    camera_pose: Matrix4
    mask_path: str | None = None
    part_mask_path: str | None = None
    rgb_paths_by_view: list[str] = field(default_factory=list)
    depth_paths_by_view: list[str] = field(default_factory=list)
    mask_paths_by_view: list[str] = field(default_factory=list)
    part_mask_paths_by_view: list[str] = field(default_factory=list)
    camera_poses_by_view: list[Matrix4] = field(default_factory=list)
    action_log: dict[str, Any] = field(default_factory=dict)
    joint_position_hint: float | None = None
    observation_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FrameObservation":
        return FrameObservation(
            timestamp_s=float(data["timestamp_s"]),
            rgb_path=data["rgb_path"],
            depth_path=data["depth_path"],
            mask_path=data.get("mask_path"),
            part_mask_path=data.get("part_mask_path"),
            camera_pose=[[float(v) for v in row] for row in data["camera_pose"]],
            rgb_paths_by_view=list(data.get("rgb_paths_by_view", [])),
            depth_paths_by_view=list(data.get("depth_paths_by_view", [])),
            mask_paths_by_view=list(data.get("mask_paths_by_view", [])),
            part_mask_paths_by_view=list(data.get("part_mask_paths_by_view", [])),
            camera_poses_by_view=[
                [[float(v) for v in row] for row in pose]
                for pose in data.get("camera_poses_by_view", [])
            ],
            action_log=dict(data.get("action_log", {})),
            joint_position_hint=(
                None
                if data.get("joint_position_hint") is None
                else float(data["joint_position_hint"])
            ),
            observation_confidence=float(data.get("observation_confidence", 1.0)),
        )


@dataclass(slots=True)
class EpisodeInput:
    object_instance_id: str
    category: str
    camera_intrinsics: dict[str, float]
    frames: list[FrameObservation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_dict() for frame in self.frames]
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EpisodeInput":
        return EpisodeInput(
            object_instance_id=data["object_instance_id"],
            category=data["category"],
            camera_intrinsics={
                key: float(value) for key, value in dict(data["camera_intrinsics"]).items()
            },
            frames=[FrameObservation.from_dict(frame) for frame in data["frames"]],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ReconstructionArtifact:
    canonical_mesh_path: str
    canonical_frame: dict[str, Any]
    primitives: list[PrimitiveBox]
    coverage: float
    confidence: float
    support_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primitives"] = [primitive.to_dict() for primitive in self.primitives]
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ReconstructionArtifact":
        return ReconstructionArtifact(
            canonical_mesh_path=data["canonical_mesh_path"],
            canonical_frame=dict(data["canonical_frame"]),
            primitives=[PrimitiveBox.from_dict(item) for item in data["primitives"]],
            coverage=float(data["coverage"]),
            confidence=float(data["confidence"]),
            support_metrics=dict(data.get("support_metrics", {})),
        )


@dataclass(slots=True)
class PartArtifact:
    name: str
    role: str
    primitive_boxes: list[PrimitiveBox]
    canonical_pose: dict[str, Any]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primitive_boxes"] = [primitive.to_dict() for primitive in self.primitive_boxes]
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PartArtifact":
        return PartArtifact(
            name=data["name"],
            role=data["role"],
            primitive_boxes=[PrimitiveBox.from_dict(item) for item in data["primitive_boxes"]],
            canonical_pose=dict(data["canonical_pose"]),
            confidence=float(data["confidence"]),
        )


@dataclass(slots=True)
class JointArtifact:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: Vector3
    origin: Vector3
    limits: list[float]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "JointArtifact":
        return JointArtifact(
            name=data["name"],
            joint_type=data["joint_type"],
            parent=data["parent"],
            child=data["child"],
            axis=list(data["axis"]),
            origin=list(data["origin"]),
            limits=[float(item) for item in data["limits"]],
            confidence=float(data["confidence"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class StateSample:
    timestamp_s: float
    q: float
    qdot: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "StateSample":
        return StateSample(
            timestamp_s=float(data["timestamp_s"]),
            q=float(data["q"]),
            qdot=float(data["qdot"]),
            confidence=float(data["confidence"]),
        )


@dataclass(slots=True)
class ArticulationArtifact:
    parts: list[PartArtifact]
    joints: list[JointArtifact]
    state: list[StateSample]
    fit_metrics: dict[str, Any] = field(default_factory=dict)
    low_confidence_components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parts"] = [part.to_dict() for part in self.parts]
        payload["joints"] = [joint.to_dict() for joint in self.joints]
        payload["state"] = [sample.to_dict() for sample in self.state]
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ArticulationArtifact":
        return ArticulationArtifact(
            parts=[PartArtifact.from_dict(item) for item in data["parts"]],
            joints=[JointArtifact.from_dict(item) for item in data["joints"]],
            state=[StateSample.from_dict(item) for item in data.get("state", [])],
            fit_metrics=dict(data.get("fit_metrics", {})),
            low_confidence_components=list(data.get("low_confidence_components", [])),
        )


@dataclass(slots=True)
class URDFPackage:
    urdf_path: str
    articulation_json_path: str
    part_mesh_paths: dict[str, str]
    mjcf_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    episode_id: str
    reconstruction_artifact_path: str
    articulation_artifact_path: str
    urdf_package: URDFPackage

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["urdf_package"] = self.urdf_package.to_dict()
        return payload
