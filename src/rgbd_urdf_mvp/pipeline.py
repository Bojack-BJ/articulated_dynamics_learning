from __future__ import annotations

from pathlib import Path

from .core.models import EpisodeInput, PipelineResult
from .core.serialization import save_articulation_artifact, save_json, save_reconstruction_artifact
from .export.urdf import URDFExporter
from .kinematics.articulation_init import CategoryPriorParticulateAdapter
from .kinematics.refit import SlidingWindowRefitter
from .perception.reconstruction import GenerationFirstReconstructor


class RGBDToURDFPipeline:
    """Scaffold RGB-D pipeline with explicit feedforward vs optimization modes."""

    def __init__(
        self,
        reconstructor: GenerationFirstReconstructor | None = None,
        articulation_initializer: CategoryPriorParticulateAdapter | None = None,
        refitter: SlidingWindowRefitter | None = None,
        exporter: URDFExporter | None = None,
    ) -> None:
        self.reconstructor = reconstructor or GenerationFirstReconstructor()
        self.articulation_initializer = articulation_initializer or CategoryPriorParticulateAdapter()
        self.refitter = refitter or SlidingWindowRefitter()
        self.exporter = exporter or URDFExporter()

    def run(
        self,
        episode: EpisodeInput,
        output_dir: str | Path,
        path_mode: str = "optimization",
    ) -> PipelineResult:
        """Run the scaffold pipeline.

        `feedforward`:
            reconstruction -> articulation init -> export

        `optimization`:
            reconstruction -> articulation init -> temporal refit -> export
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Stage 1: build a canonical shape prior from the episode.
        reconstruction = self.reconstructor.reconstruct(episode, output_path / "reconstruction")
        reconstruction_artifact_path = output_path / "reconstruction_artifact.json"
        save_reconstruction_artifact(reconstruction, reconstruction_artifact_path)

        # Stage 2: estimate an initial articulated structure from that shape prior.
        articulation = self._estimate_articulation(
            episode=episode,
            reconstruction=reconstruction,
            output_path=output_path,
            path_mode=path_mode,
        )
        articulation.fit_metrics["pipeline_path"] = path_mode
        articulation_artifact_path = output_path / "articulation_artifact.json"
        save_articulation_artifact(articulation, articulation_artifact_path)

        # Stage 3: export the kinematic artifact into URDF/MJCF-friendly geometry.
        urdf_package = self.exporter.export(episode, articulation, output_path / "urdf")
        result = PipelineResult(
            episode_id=episode.object_instance_id,
            reconstruction_artifact_path=str(reconstruction_artifact_path.resolve()),
            articulation_artifact_path=str(articulation_artifact_path.resolve()),
            urdf_package=urdf_package,
        )
        save_json(
            {
                **result.to_dict(),
                "path_mode": path_mode,
            },
            output_path / "pipeline_result.json",
        )
        return result

    def _estimate_articulation(
        self,
        episode: EpisodeInput,
        reconstruction,
        output_path: Path,
        path_mode: str,
    ):
        # The initializer is shared by both paths so the replacement boundary
        # stays stable whether the user wants a fast feedforward pass or a
        # slower refinement-heavy pass.
        articulation = self.articulation_initializer.initialize(
            episode,
            reconstruction,
            output_path / "articulation_init",
        )
        if path_mode == "feedforward":
            return articulation
        if path_mode == "optimization":
            return self.refitter.refit(episode, articulation)
        raise ValueError(f"Unsupported pipeline path: {path_mode}")
