from __future__ import annotations

from pathlib import Path

from .categories import FULL_ROTATION_CATEGORIES, HINGE_CATEGORIES, PRISMATIC_CATEGORIES
from .geometry import primitive_volume, write_obj
from .models import EpisodeInput, PrimitiveBox, ReconstructionArtifact


DEFAULT_PRIMITIVES: dict[str, list[PrimitiveBox]] = {
    "door": [
        PrimitiveBox(
            name="frame",
            center=[0.0, 0.0, 1.05],
            size=[1.10, 0.12, 2.10],
            observed_ratio=0.95,
            generated=False,
        ),
        PrimitiveBox(
            name="panel",
            center=[0.02, 0.085, 1.00],
            size=[0.95, 0.05, 1.95],
            observed_ratio=0.92,
            generated=False,
        ),
        PrimitiveBox(
            name="hallucinated_handle_blob",
            center=[0.47, 0.14, 1.02],
            size=[0.08, 0.06, 0.08],
            observed_ratio=0.08,
            generated=True,
        ),
    ],
    "drawer": [
        PrimitiveBox(
            name="cabinet",
            center=[0.0, 0.0, 0.45],
            size=[1.20, 0.60, 0.90],
            observed_ratio=0.94,
            generated=False,
        ),
        PrimitiveBox(
            name="drawer_box",
            center=[0.12, 0.0, 0.36],
            size=[0.85, 0.42, 0.24],
            observed_ratio=0.88,
            generated=False,
        ),
        PrimitiveBox(
            name="hallucinated_pull_cluster",
            center=[0.60, 0.34, 0.38],
            size=[0.10, 0.06, 0.05],
            observed_ratio=0.10,
            generated=True,
        ),
    ],
    "refrigerator": [
        PrimitiveBox(
            name="cabinet",
            center=[0.0, 0.0, 0.95],
            size=[1.10, 0.80, 1.90],
            observed_ratio=0.94,
            generated=False,
        ),
        PrimitiveBox(
            name="door",
            center=[0.02, -0.40, 0.95],
            size=[1.05, 0.06, 1.85],
            observed_ratio=0.90,
            generated=False,
        ),
        PrimitiveBox(
            name="hallucinated_handle_cluster",
            center=[0.45, -0.45, 0.95],
            size=[0.10, 0.05, 0.20],
            observed_ratio=0.10,
            generated=True,
        ),
    ],
}


def _default_primitives_for_category(category: str) -> list[PrimitiveBox]:
    if category in DEFAULT_PRIMITIVES:
        return [PrimitiveBox.from_dict(item.to_dict()) for item in DEFAULT_PRIMITIVES[category]]
    if category in HINGE_CATEGORIES or category in FULL_ROTATION_CATEGORIES:
        return [PrimitiveBox.from_dict(item.to_dict()) for item in DEFAULT_PRIMITIVES["refrigerator"]]
    if category in PRISMATIC_CATEGORIES:
        return [PrimitiveBox.from_dict(item.to_dict()) for item in DEFAULT_PRIMITIVES["drawer"]]
    return [PrimitiveBox.from_dict(item.to_dict()) for item in DEFAULT_PRIMITIVES["door"]]


class GenerationFirstReconstructor:
    def __init__(self, min_support: float = 0.15) -> None:
        self.min_support = min_support

    def reconstruct(self, episode: EpisodeInput, output_dir: str | Path) -> ReconstructionArtifact:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        input_primitives = self._resolve_primitives(episode)
        cleaned_primitives, removed = self._cleanup_primitives(input_primitives)
        mesh_path = output_path / "canonical_mesh.obj"
        write_obj(mesh_path, cleaned_primitives)

        frame_confidences = [frame.observation_confidence for frame in episode.frames] or [0.0]
        mean_frame_confidence = sum(frame_confidences) / len(frame_confidences)
        coverage = sum(box.observed_ratio for box in cleaned_primitives) / max(len(cleaned_primitives), 1)
        confidence = min(1.0, 0.5 * coverage + 0.5 * mean_frame_confidence)

        return ReconstructionArtifact(
            canonical_mesh_path=str(mesh_path.resolve()),
            canonical_frame={
                "origin": [0.0, 0.0, 0.0],
                "up_axis": "z",
                "forward_axis": "x",
            },
            primitives=cleaned_primitives,
            coverage=coverage,
            confidence=confidence,
            support_metrics={
                "input_frame_count": len(episode.frames),
                "mean_frame_confidence": mean_frame_confidence,
                "removed_primitives": [box.to_dict() for box in removed],
                "generation_mode": "generation-first-with-depth-verification",
            },
        )

    def _resolve_primitives(self, episode: EpisodeInput) -> list[PrimitiveBox]:
        payload = episode.metadata.get("generated_primitives")
        if payload:
            return [PrimitiveBox.from_dict(item) for item in payload]
        return _default_primitives_for_category(episode.category)

    def _cleanup_primitives(
        self, primitives: list[PrimitiveBox]
    ) -> tuple[list[PrimitiveBox], list[PrimitiveBox]]:
        if not primitives:
            return [], []
        largest = max(primitives, key=primitive_volume)
        kept: list[PrimitiveBox] = []
        removed: list[PrimitiveBox] = []
        for primitive in primitives:
            if primitive.name == largest.name:
                kept.append(primitive)
                continue
            if primitive.observed_ratio >= self.min_support or not primitive.generated:
                kept.append(primitive)
            else:
                removed.append(primitive)
        kept.sort(key=primitive_volume, reverse=True)
        return kept, removed
