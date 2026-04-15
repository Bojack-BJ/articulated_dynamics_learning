from __future__ import annotations

from pathlib import Path

from .categories import FULL_ROTATION_CATEGORIES, HINGE_CATEGORIES, PRISMATIC_CATEGORIES
from .geometry import primitive_volume
from .models import (
    ArticulationArtifact,
    EpisodeInput,
    JointArtifact,
    PartArtifact,
    PrimitiveBox,
    ReconstructionArtifact,
)


class CategoryPriorParticulateAdapter:
    """MVP articulation initializer with a clear replacement boundary for PARTICULATE."""

    def initialize(
        self,
        episode: EpisodeInput,
        reconstruction: ReconstructionArtifact,
        output_dir: str | Path,
    ) -> ArticulationArtifact:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if len(reconstruction.primitives) < 2:
            raise ValueError("Need at least two primitives to infer a single-DOF articulation.")

        sorted_primitives = sorted(
            reconstruction.primitives, key=primitive_volume, reverse=True
        )
        base_primitive = sorted_primitives[0]
        moving_primitive = sorted_primitives[1]
        extra_primitives = sorted_primitives[2:]

        base_part = PartArtifact(
            name="base_link",
            role="static",
            primitive_boxes=[base_primitive],
            canonical_pose={"translation": [0.0, 0.0, 0.0], "rotation_rpy": [0.0, 0.0, 0.0]},
            confidence=min(1.0, 0.5 * reconstruction.confidence + 0.5 * base_primitive.observed_ratio),
        )
        joint = self._build_joint(episode, moving_primitive)
        moving_part = PartArtifact(
            name="moving_link",
            role="moving",
            primitive_boxes=[moving_primitive],
            canonical_pose={"translation": list(joint.origin), "rotation_rpy": [0.0, 0.0, 0.0]},
            confidence=min(
                1.0,
                0.5 * reconstruction.confidence + 0.5 * moving_primitive.observed_ratio,
            ),
        )

        # Keep extra reconstructed components, but bias them into the nearest stable part so the tree stays valid.
        for primitive in extra_primitives:
            if abs(primitive.center[0] - moving_primitive.center[0]) < abs(
                primitive.center[0] - base_primitive.center[0]
            ):
                moving_part.primitive_boxes.append(primitive)
            else:
                base_part.primitive_boxes.append(primitive)

        articulation = ArticulationArtifact(
            parts=[base_part, moving_part],
            joints=[joint],
            state=[],
            fit_metrics={
                "initializer": "category-prior-particulate-adapter",
                "category_prior": episode.category,
                "canonical_support_mean": sum(
                    primitive.observed_ratio for primitive in reconstruction.primitives
                )
                / len(reconstruction.primitives),
            },
            low_confidence_components=[],
        )
        if joint.confidence < 0.4:
            articulation.low_confidence_components.append(joint.name)
        return articulation

    def _build_joint(self, episode: EpisodeInput, moving: PrimitiveBox) -> JointArtifact:
        default_limits = [0.0, 0.35]
        if episode.category in FULL_ROTATION_CATEGORIES:
            default_limits = [0.0, 6.283185307179586]
        elif episode.category in HINGE_CATEGORIES:
            default_limits = [0.0, 1.5708]
        limits = episode.metadata.get("joint_limits", default_limits)

        side_hinge_categories = {"door", "refrigerator", "microwave", "oven", "dishwasher", "toasteroven"}
        if episode.category in side_hinge_categories:
            origin = [
                moving.center[0] - moving.size[0] / 2.0,
                moving.center[1],
                moving.center[2],
            ]
            axis = [0.0, 0.0, 1.0]
            joint_type = "revolute"
        elif episode.category in FULL_ROTATION_CATEGORIES or episode.category in HINGE_CATEGORIES:
            origin = list(moving.center)
            axis = [0.0, 0.0, 1.0]
            joint_type = "revolute"
        elif episode.category in PRISMATIC_CATEGORIES:
            origin = list(moving.center)
            axis = [1.0, 0.0, 0.0]
            joint_type = "prismatic"
        else:
            origin = list(moving.center)
            axis = [0.0, 0.0, 1.0]
            joint_type = "revolute"

        confidence = min(1.0, 0.5 * moving.observed_ratio + 0.4)
        return JointArtifact(
            name=f"{episode.category}_joint",
            joint_type=joint_type,
            parent="base_link",
            child="moving_link",
            axis=axis,
            origin=origin,
            limits=[float(limits[0]), float(limits[1])],
            confidence=confidence,
            metadata={"source": "category-prior", "expected_dof": 1},
        )
