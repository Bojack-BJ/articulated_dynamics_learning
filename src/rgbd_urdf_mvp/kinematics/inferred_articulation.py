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
from ..perception.part_pose import _local_bounds


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

        articulation = self._build_articulation(
            episode,
            part_pose_artifact,
            joint_inference_artifact,
            Path(config.part_pose_path).expanduser().resolve(),
        )
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
        part_pose_path: Path,
    ) -> ArticulationArtifact:
        parts_data = part_pose_artifact.get("parts", [])
        joints_data = joint_inference_artifact.get("joints", [])
        anchor_part_id = int(part_pose_artifact.get("anchor_part_id", 0))
        anchor_name = str(part_pose_artifact.get("anchor_part_name", "base_link"))
        pointcloud_geometry = self._load_reference_pointcloud_geometry(part_pose_artifact, part_pose_path)

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
            pointcloud_proxy = pointcloud_geometry.get(part_id)
            reference_bounds = (
                dict(pointcloud_proxy.get("reference_local_bounds", {}))
                if pointcloud_proxy is not None
                else dict(raw_part.get("reference_local_bounds", {}))
            )
            lower = [float(value) for value in reference_bounds.get("lower", [-0.05, -0.05, -0.05])]
            upper = [float(value) for value in reference_bounds.get("upper", [0.05, 0.05, 0.05])]
            size = [max(1e-3, upper[index] - lower[index]) for index in range(3)]

            if part_id == anchor_part_id:
                canonical_translation = [0.0, 0.0, 0.0]
                primitive_center = (
                    [float(value) for value in pointcloud_proxy["world_center"]]
                    if pointcloud_proxy is not None
                    else [0.0, 0.0, 0.0]
                )
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
                primitive_center = (
                    [float(value) for value in pointcloud_proxy["world_center"]]
                    if pointcloud_proxy is not None
                    else [float(value) for value in reference_pose.get("translation", canonical_translation)]
                )
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
                            metadata={
                                "source": (
                                    "pointcloud-reference-bounds"
                                    if pointcloud_proxy is not None
                                    else "part-pose-bounds"
                                ),
                                "point_count": (
                                    int(pointcloud_proxy["point_count"])
                                    if pointcloud_proxy is not None
                                    else int(raw_part.get("reference_point_count", 0) or 0)
                                ),
                            },
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
                        "prior": dict(raw_joint.get("prior", {})) if isinstance(raw_joint.get("prior"), dict) else None,
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

    def _load_reference_pointcloud_geometry(
        self,
        part_pose_artifact: dict[str, Any],
        part_pose_path: Path,
    ) -> dict[int, dict[str, Any]]:
        pointcloud_path = self._resolve_pointcloud_path(part_pose_artifact, part_pose_path)
        if pointcloud_path is None:
            return {}
        if not pointcloud_path.is_file():
            return {}
        part_specs = {
            int(raw_part.get("part_id", 0)): raw_part
            for raw_part in part_pose_artifact.get("parts", [])
            if isinstance(raw_part, dict) and int(raw_part.get("part_id", 0)) > 0
        }
        if not part_specs:
            return {}

        properties, rows = self._read_ascii_ply(pointcloud_path)
        required = {"x", "y", "z", "frame_index", "part_id"}
        if not required.issubset(properties):
            return {}
        idx = {name: properties.index(name) for name in properties}

        points_by_part: dict[int, list[list[float]]] = {part_id: [] for part_id in part_specs}
        reference_frame_by_part = {
            part_id: int(raw_part.get("reference_source_frame_index", raw_part.get("reference_frame_index", 0)))
            for part_id, raw_part in part_specs.items()
        }
        for items in rows:
            try:
                part_id = int(items[idx["part_id"]])
                frame_index = int(items[idx["frame_index"]])
            except (IndexError, ValueError):
                continue
            if part_id not in points_by_part or frame_index != reference_frame_by_part[part_id]:
                continue
            points_by_part[part_id].append(
                [
                    float(items[idx["x"]]),
                    float(items[idx["y"]]),
                    float(items[idx["z"]]),
                ]
            )

        geometry: dict[int, dict[str, Any]] = {}
        for part_id, points in points_by_part.items():
            if len(points) < 4:
                continue
            lower = [min(point[axis] for point in points) for axis in range(3)]
            upper = [max(point[axis] for point in points) for axis in range(3)]
            raw_part = part_specs[part_id]
            canonical_frame = dict(raw_part.get("canonical_frame", {}))
            rotation = canonical_frame.get("rotation_matrix") or [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            translation = [
                (lower[axis] + upper[axis]) / 2.0
                for axis in range(3)
            ]
            local_bounds = _local_bounds(points, rotation, translation)
            size = [
                max(0.01, float(local_bounds["upper"][axis]) - float(local_bounds["lower"][axis]))
                for axis in range(3)
            ]
            geometry[part_id] = {
                "world_center": translation,
                "reference_local_bounds": {
                    "lower": [-0.5 * value for value in size],
                    "upper": [0.5 * value for value in size],
                },
                "point_count": len(points),
            }
        return geometry

    def _resolve_pointcloud_path(self, part_pose_artifact: dict[str, Any], part_pose_path: Path) -> Path | None:
        pointcloud_path_raw = part_pose_artifact.get("pointcloud_path")
        if pointcloud_path_raw:
            pointcloud_path = Path(str(pointcloud_path_raw)).expanduser()
            if not pointcloud_path.is_absolute():
                pointcloud_path = part_pose_path.parent / pointcloud_path
            return pointcloud_path.resolve()

        input_path_raw = part_pose_artifact.get("input_path")
        candidate_dirs = [part_pose_path.parent]
        if input_path_raw:
            input_path = Path(str(input_path_raw)).expanduser()
            if not input_path.is_absolute():
                input_path = part_pose_path.parent / input_path
            candidate_dirs.append(input_path.resolve().parent)

        for candidate_dir in candidate_dirs:
            manifest_path = candidate_dir / "fusion_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            raw_pointcloud = manifest.get("pointcloud_4d_path")
            if not raw_pointcloud:
                continue
            pointcloud_path = Path(str(raw_pointcloud)).expanduser()
            if not pointcloud_path.is_absolute():
                pointcloud_path = manifest_path.parent / pointcloud_path
            return pointcloud_path.resolve()
        return None

    def _read_ascii_ply(self, path: Path) -> tuple[list[str], list[list[str]]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "ply":
            return [], []
        properties: list[str] = []
        vertex_count = 0
        header_end = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property"):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                header_end = index
                break
        if header_end is None:
            return [], []
        rows = [line.split() for line in lines[header_end + 1 : header_end + 1 + vertex_count] if line.strip()]
        return properties, rows

    def _part_confidence(self, raw_part: dict[str, Any]) -> float:
        samples = raw_part.get("samples", [])
        valid = [float(sample.get("confidence", 0.0)) for sample in samples if bool(sample.get("valid", False))]
        if not valid:
            return 0.3
        return min(1.0, sum(valid) / len(valid))
