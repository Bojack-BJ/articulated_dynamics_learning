from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .categories import SUPPORTED_CATEGORIES
from .models import ArticulationArtifact, EpisodeInput, ReconstructionArtifact


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raw_path = str(path)
        stripped_path = raw_path.strip()
        if stripped_path != raw_path:
            hint = f"Path has leading/trailing whitespace: {raw_path!r}."
            if Path(stripped_path).exists():
                hint += f" Did you mean {stripped_path!r}?"
            raise FileNotFoundError(hint) from exc
        raise


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_episode(path: str | Path) -> EpisodeInput:
    return EpisodeInput.from_dict(load_json(path))


def save_reconstruction_artifact(artifact: ReconstructionArtifact, path: str | Path) -> None:
    save_json(artifact.to_dict(), path)


def save_articulation_artifact(artifact: ArticulationArtifact, path: str | Path) -> None:
    save_json(artifact.to_dict(), path)


def validate_episode(episode: EpisodeInput) -> list[str]:
    errors: list[str] = []
    if episode.category not in SUPPORTED_CATEGORIES:
        errors.append(f"Unsupported category: {episode.category}")
    if not episode.frames:
        errors.append("Episode must contain at least one frame")
    for index, frame in enumerate(episode.frames):
        if len(frame.camera_pose) != 4 or any(len(row) != 4 for row in frame.camera_pose):
            errors.append(f"Frame {index} camera_pose must be a 4x4 matrix")
        if frame.mask_path is not None and not isinstance(frame.mask_path, str):
            errors.append(f"Frame {index} mask_path must be a string when provided")
        if frame.part_mask_path is not None and not isinstance(frame.part_mask_path, str):
            errors.append(f"Frame {index} part_mask_path must be a string when provided")
        for view_index, pose in enumerate(frame.camera_poses_by_view):
            if len(pose) != 4 or any(len(row) != 4 for row in pose):
                errors.append(f"Frame {index} camera_poses_by_view[{view_index}] must be a 4x4 matrix")
        if any(not isinstance(path, str) for path in frame.mask_paths_by_view):
            errors.append(f"Frame {index} mask_paths_by_view entries must be strings")
        if any(not isinstance(path, str) for path in frame.part_mask_paths_by_view):
            errors.append(f"Frame {index} part_mask_paths_by_view entries must be strings")
        if frame.timestamp_s < 0.0:
            errors.append(f"Frame {index} timestamp must be non-negative")
    return errors
