#!/usr/bin/env python3
"""Evaluate predicted motion clusters against independent SAM pseudo-part masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_json", type=Path)
    parser.add_argument("pseudo_episode_json", type=Path)
    parser.add_argument("--tracking-episode-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--labeled-prediction-json", type=Path)
    parser.add_argument("--minimum-votes", type=int, default=2)
    return parser.parse_args()


def _track_pixels(track: dict[str, Any]) -> list[Any]:
    for key in ("pixels", "pixel_trajectory", "points_2d", "tracks_2d"):
        value = track.get(key)
        if isinstance(value, list):
            return value
    return []


def _visible(track: dict[str, Any], count: int) -> np.ndarray:
    for key in ("visibility", "visible", "visibilities"):
        value = track.get(key)
        if isinstance(value, list):
            array = np.asarray(value, dtype=bool)
            if array.size >= count:
                return array[:count]
    return np.ones(count, dtype=bool)


def main() -> None:
    args = parse_args()
    prediction_path = args.prediction_json.resolve()
    pseudo_path = args.pseudo_episode_json.resolve()
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    pseudo_root = pseudo_path.parent
    tracking_episode = json.loads(args.tracking_episode_json.resolve().read_text(encoding="utf-8"))

    masks: dict[tuple[int, int], np.ndarray] = {}
    for index, frame in enumerate(pseudo.get("frames", [])):
        paths_by_view = frame.get("part_mask_paths_by_view")
        if isinstance(paths_by_view, list):
            entries = enumerate(paths_by_view)
        else:
            view = int(frame.get("view_id", frame.get("camera_id", 0)))
            entries = [(view, frame.get("part_mask_path", frame.get("mask_path")))]
        for view, mask_value in entries:
            if mask_value is None:
                continue
            mask_path = Path(mask_value)
            if not mask_path.is_absolute():
                mask_path = pseudo_root / mask_path
            masks[(int(view), index)] = np.asarray(Image.open(mask_path), dtype=np.int64)

    predicted_labels: list[int] = []
    pseudo_labels: list[int] = []
    labeled_tracks: list[dict[str, Any]] = []
    vote_stats: list[dict[str, Any]] = []
    for track in prediction.get("tracks", []):
        samples = track.get("samples")
        if not isinstance(samples, list):
            pixels = _track_pixels(track)
            samples = [
                {"frame_index": index, "uv": pixel, "visible": bool(visible)}
                for index, (pixel, visible) in enumerate(zip(pixels, _visible(track, len(pixels))))
            ]
        if not samples:
            continue
        view = int(track.get("view_index", track.get("view_id", track.get("camera_id", 0))))
        votes: Counter[int] = Counter()
        for sample in samples:
            pixel = sample.get("uv")
            if not sample.get("visible", False) or pixel is None:
                continue
            frame_index = int(sample.get("frame_index", -1))
            if not 0 <= frame_index < len(tracking_episode.get("frames", [])):
                continue
            tracking_frame = tracking_episode["frames"][frame_index]
            rgb_by_view = tracking_frame.get("rgb_paths_by_view") or [tracking_frame.get("rgb_path")]
            if not 0 <= view < len(rgb_by_view) or rgb_by_view[view] is None:
                continue
            source_index = int(Path(rgb_by_view[view]).stem)
            mask = masks.get((view, source_index))
            if mask is None or len(pixel) < 2:
                continue
            x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                label = int(mask[y, x])
                if label > 0:
                    votes[label] += 1
        if not votes:
            continue
        label, count = votes.most_common(1)[0]
        if count < args.minimum_votes:
            continue
        predicted = int(track.get("part_id", track.get("pred_cluster", -1)))
        if predicted < 0:
            continue
        predicted_labels.append(predicted)
        pseudo_labels.append(label)
        enriched = dict(track)
        enriched["pseudo_part_id"] = label
        labeled_tracks.append(enriched)
        vote_stats.append({
            "track_id": int(track.get("track_id", len(vote_stats))),
            "pseudo_part_id": label,
            "winning_votes": count,
            "total_part_votes": int(sum(votes.values())),
        })

    if not labeled_tracks:
        raise ValueError("No prediction tracks could be labeled from the pseudo masks")
    metrics = evaluate_slot_assignments(
        np.asarray(predicted_labels, dtype=np.int64),
        np.asarray(pseudo_labels, dtype=np.int64),
    )
    output = {
        "evaluation_type": "sam_pseudo_part_track_evaluation",
        "warning": "SAM pseudo labels are independent diagnostic labels, not ground truth.",
        "prediction_json": str(prediction_path),
        "pseudo_episode_json": str(pseudo_path),
        "tracking_episode_json": str(args.tracking_episode_json.resolve()),
        "prediction_track_count": len(prediction.get("tracks", [])),
        "evaluated_track_count": len(labeled_tracks),
        "evaluated_track_ratio": len(labeled_tracks) / max(1, len(prediction.get("tracks", []))),
        "metrics": metrics,
        "vote_stats": vote_stats,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.labeled_prediction_json is not None:
        labeled_prediction = dict(prediction)
        labels_by_track = {int(track["track_id"]): track for track in labeled_tracks}
        labeled_prediction["tracks"] = [
            labels_by_track[int(track.get("track_id", -1))]
            for track in prediction.get("tracks", [])
            if int(track.get("track_id", -1)) in labels_by_track
        ]
        for track in labeled_prediction["tracks"]:
            track["original_part_id"] = int(track.pop("pseudo_part_id"))
        labeled_prediction["pseudo_label_metadata"] = {
            "source": "sam2_propagated_pseudo_parts",
            "warning": "original_part_id is a pseudo label in this diagnostic artifact, not GT.",
        }
        args.labeled_prediction_json.parent.mkdir(parents=True, exist_ok=True)
        args.labeled_prediction_json.write_text(
            json.dumps(labeled_prediction, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output_json": str(args.output_json.resolve()), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
