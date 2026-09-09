"""Method-independent semantic domains for articulated-part evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rgbd_urdf_mvp.perception.part_segmentation import collapse_fixed_connected_parts


def remap_part_labels(labels: np.ndarray, part_id_map: dict[int, int]) -> np.ndarray:
    """Map raw body-part labels to fixed-connected kinematic-part labels."""
    values = np.asarray(labels, dtype=np.int64)
    output = values.copy()
    for raw_part_id, kinematic_part_id in part_id_map.items():
        output[values == int(raw_part_id)] = int(kinematic_part_id)
    return output


def build_kinematic_evaluation_domain(
    part_segmentation: dict[str, Any],
    frame_part_ids: Iterable[np.ndarray],
    *,
    min_union_points: int = 32,
    min_visible_frames: int = 2,
) -> dict[str, Any]:
    """Build all/observable domains without consulting method predictions.

    ``frame_part_ids`` must come from the fixed GT acquisition protocol. A part
    is observable when it has enough union support and appears in enough frames.
    This criterion is intentionally independent of CoTracker/TAPIP/AiM/ReArt.
    """
    ontology = collapse_fixed_connected_parts(part_segmentation)
    raw_map = {
        int(raw_id): int(part_id)
        for raw_id, part_id in ontology.get("raw_part_to_part_id", {}).items()
    }
    if not raw_map:
        raw_map = {
            int(part["part_id"]): int(part["part_id"])
            for part in ontology.get("parts", [])
        }
    all_part_ids = sorted(int(part["part_id"]) for part in ontology.get("parts", []))
    frames = [remap_part_labels(np.asarray(labels), raw_map) for labels in frame_part_ids]
    frame_count = len(frames)
    stats: list[dict[str, Any]] = []
    observable: list[int] = []
    for part_id in all_part_ids:
        per_frame_counts = [int(np.sum(labels == part_id)) for labels in frames]
        visible_frames = sum(count > 0 for count in per_frame_counts)
        union_points = sum(per_frame_counts)
        reasons: list[str] = []
        if union_points < int(min_union_points):
            reasons.append("insufficient_union_points")
        if visible_frames < int(min_visible_frames):
            reasons.append("insufficient_visible_frames")
        is_observable = not reasons
        if is_observable:
            observable.append(part_id)
        stats.append({
            "part_id": part_id,
            "union_point_count": union_points,
            "visible_frame_count": visible_frames,
            "visible_frame_ratio": float(visible_frames / max(1, frame_count)),
            "per_frame_point_counts": per_frame_counts,
            "observable": is_observable,
            "exclusion_reasons": reasons,
        })
    return {
        "schema": "kinematic-part-evaluation-domain-v1",
        "ontology": ontology.get("ontology"),
        "raw_part_id_to_kinematic_part_id": {
            str(raw_id): part_id for raw_id, part_id in sorted(raw_map.items())
        },
        "all_kinematic_part_ids": all_part_ids,
        "observable_kinematic_part_ids": observable,
        "selection": {
            "source": "fixed-gt-acquisition",
            "frame_count": frame_count,
            "min_union_points": int(min_union_points),
            "min_visible_frames": int(min_visible_frames),
            "uses_method_predictions": False,
            "uses_motion_magnitude": False,
        },
        "part_statistics": stats,
        "part_segmentation": ontology,
    }


def load_kinematic_evaluation_domain(
    path: Path,
    domain: str,
) -> tuple[dict[int, int], list[int]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema") != "kinematic-part-evaluation-domain-v1":
        raise ValueError(f"Unsupported kinematic evaluation domain: {path}")
    if domain not in {"all", "observable"}:
        raise ValueError(f"Expected domain all or observable, got {domain!r}")
    part_map = {
        int(raw_id): int(part_id)
        for raw_id, part_id in payload["raw_part_id_to_kinematic_part_id"].items()
    }
    key = (
        "all_kinematic_part_ids"
        if domain == "all"
        else "observable_kinematic_part_ids"
    )
    return part_map, [int(value) for value in payload[key]]
