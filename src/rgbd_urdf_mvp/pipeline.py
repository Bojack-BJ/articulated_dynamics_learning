from __future__ import annotations

from pathlib import Path

from .articulation import CategoryPriorParticulateAdapter
from .models import EpisodeInput, PipelineResult
from .reconstruction import GenerationFirstReconstructor
from .refit import SlidingWindowRefitter
from .serialization import save_articulation_artifact, save_json, save_reconstruction_artifact
from .urdf import URDFExporter


class RGBDToURDFPipeline:
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

    def run(self, episode: EpisodeInput, output_dir: str | Path) -> PipelineResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        reconstruction = self.reconstructor.reconstruct(episode, output_path / "reconstruction")
        reconstruction_artifact_path = output_path / "reconstruction_artifact.json"
        save_reconstruction_artifact(reconstruction, reconstruction_artifact_path)

        articulation = self.articulation_initializer.initialize(
            episode, reconstruction, output_path / "articulation_init"
        )
        articulation = self.refitter.refit(episode, articulation)
        articulation_artifact_path = output_path / "articulation_artifact.json"
        save_articulation_artifact(articulation, articulation_artifact_path)

        urdf_package = self.exporter.export(episode, articulation, output_path / "urdf")
        result = PipelineResult(
            episode_id=episode.object_instance_id,
            reconstruction_artifact_path=str(reconstruction_artifact_path.resolve()),
            articulation_artifact_path=str(articulation_artifact_path.resolve()),
            urdf_package=urdf_package,
        )
        save_json(result.to_dict(), output_path / "pipeline_result.json")
        return result
