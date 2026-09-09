#!/usr/bin/env python3
"""Audit image-space observability for controlled AiM acquisition protocols."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def _bbox_metrics(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    height, width = mask.shape
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    return (
        max(bbox_width / width, bbox_height / height),
        (bbox_width * bbox_height) / (width * height),
    )


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def _trajectory(payload: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    names = sorted(payload["frames"][0]["action_log"]["joint_positions"])
    values = [
        [float(frame["action_log"]["joint_positions"][name]) for name in names]
        for frame in payload["frames"]
    ]
    return names, np.asarray(values, dtype=np.float64)


def _audit_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    parts = metadata["part_segmentation"]["parts"]
    moving_ids = {
        int(part["part_id"]) for part in parts if part.get("role") == "articulated"
    }
    per_observation = []
    centroids: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame_index, frame in enumerate(payload["frames"]):
        part_paths = frame.get("part_mask_paths_by_view") or [frame["part_mask_path"]]
        for view_index, relative_path in enumerate(part_paths):
            part_mask = _load_mask(path.parent / relative_path)
            object_mask = part_mask > 0
            moving_mask = np.isin(part_mask, list(moving_ids))
            object_linear, object_area = _bbox_metrics(object_mask)
            moving_linear, moving_area = _bbox_metrics(moving_mask)
            ys, xs = np.nonzero(moving_mask)
            centroid = (
                np.asarray([float(xs.mean()), float(ys.mean())])
                if len(xs)
                else np.asarray([np.nan, np.nan])
            )
            centroids.setdefault(view_index, []).append((frame_index, centroid))
            per_observation.append(
                {
                    "protocol": path.parent.name,
                    "frame_index": frame_index,
                    "view_index": view_index,
                    "object_bbox_linear_occupancy": object_linear,
                    "object_bbox_area_occupancy": object_area,
                    "moving_bbox_linear_occupancy": moving_linear,
                    "moving_bbox_area_occupancy": moving_area,
                    "moving_visible_pixels": int(moving_mask.sum()),
                    "moving_visible": bool(moving_mask.any()),
                    "object_touches_border": _touches_border(object_mask),
                    "moving_touches_border": _touches_border(moving_mask),
                }
            )
    displacements = []
    reference_displacements = []
    for samples in centroids.values():
        valid = [centroid for _, centroid in samples if np.isfinite(centroid).all()]
        displacements.extend(
            float(np.linalg.norm(current - previous))
            for previous, current in zip(valid, valid[1:])
        )
        if valid:
            reference_displacements.extend(
                float(np.linalg.norm(current - valid[0])) for current in valid
            )
    object_occupancy = [row["object_bbox_linear_occupancy"] for row in per_observation]
    moving_occupancy = [row["moving_bbox_linear_occupancy"] for row in per_observation]
    visible_pixels = [row["moving_visible_pixels"] for row in per_observation]
    summary = {
        "protocol": path.parent.name,
        "episode": str(path.resolve()),
        "frame_count": len(payload["frames"]),
        "view_count_per_frame": len(
            payload["frames"][0].get("camera_poses_by_view") or [None]
        ),
        "moving_part_ids": sorted(moving_ids),
        "object_bbox_occupancy_min": min(object_occupancy),
        "object_bbox_occupancy_mean": float(np.mean(object_occupancy)),
        "object_bbox_occupancy_max": max(object_occupancy),
        "moving_bbox_occupancy_min": min(moving_occupancy),
        "moving_bbox_occupancy_mean": float(np.mean(moving_occupancy)),
        "moving_bbox_occupancy_max": max(moving_occupancy),
        "moving_visible_ratio": float(
            np.mean([row["moving_visible"] for row in per_observation])
        ),
        "moving_visible_pixels_mean": float(np.mean(visible_pixels)),
        "moving_visible_pixels_min": min(visible_pixels),
        "object_border_touch_ratio": float(
            np.mean([row["object_touches_border"] for row in per_observation])
        ),
        "moving_border_touch_ratio": float(
            np.mean([row["moving_touches_border"] for row in per_observation])
        ),
        "moving_2d_step_displacement_median_px": (
            float(np.median(displacements)) if displacements else None
        ),
        "moving_2d_step_displacement_max_px": (
            max(displacements) if displacements else None
        ),
        "moving_2d_reference_displacement_median_px": (
            float(np.median(reference_displacements))
            if reference_displacements
            else None
        ),
        "moving_2d_reference_displacement_max_px": (
            max(reference_displacements) if reference_displacements else None
        ),
        "camera_fit_render_fraction": metadata.get("camera_fit_render_fraction"),
        "camera_distance": metadata.get("camera_distance"),
    }
    return summary, per_observation


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(
    episode_path: Path, payload: dict[str, Any], output_path: Path
) -> None:
    frames = payload["frames"]
    indices = np.linspace(0, len(frames) - 1, num=min(8, len(frames)), dtype=int)
    rows = []
    for frame_index in indices:
        paths = frames[frame_index].get("rgb_paths_by_view") or [
            frames[frame_index]["rgb_path"]
        ]
        images = [Image.open(episode_path.parent / path).convert("RGB") for path in paths]
        rows.append(images)
    tile_width = max(image.width for images in rows for image in images)
    tile_height = max(image.height for images in rows for image in images)
    column_count = max(len(images) for images in rows)
    sheet = Image.new(
        "RGB",
        (column_count * tile_width, len(rows) * tile_height),
        (12, 18, 24),
    )
    for row_index, images in enumerate(rows):
        for column_index, image in enumerate(images):
            sheet.paste(image, (column_index * tile_width, row_index * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_viewer(
    output_path: Path, episodes: list[tuple[Path, dict[str, Any]]]
) -> None:
    protocols = {}
    for episode_path, payload in episodes:
        protocols[episode_path.parent.name] = [
            [
                os.path.relpath(episode_path.parent / relative, output_path.parent)
                for relative in (
                    frame.get("rgb_paths_by_view") or [frame["rgb_path"]]
                )
            ]
            for frame in payload["frames"]
        ]
    data = json.dumps(protocols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>Storage-47024 observation sanity viewer</title>
<style>
body {{ margin: 0; color: #e8edf2; background: #0d1319; font: 16px sans-serif; }}
header {{ padding: 16px 24px; background: #17212b; position: sticky; top: 0; }}
main {{ padding: 20px; }}
img {{ max-width: 100%; max-height: 75vh; display: block; margin: auto; }}
label {{ margin-right: 20px; }}
</style>
<header>
  <label>Protocol <select id="protocol"></select></label>
  <label>Frame <input id="frame" type="range" min="0" value="0"> <span id="frameText"></span></label>
  <label>View <input id="view" type="range" min="0" value="0"> <span id="viewText"></span></label>
</header>
<main><img id="image"></main>
<script>
const data = {data};
const protocol = document.querySelector("#protocol");
const frame = document.querySelector("#frame");
const view = document.querySelector("#view");
for (const name of Object.keys(data)) protocol.add(new Option(name, name));
function render() {{
  const frames = data[protocol.value];
  frame.max = frames.length - 1;
  frame.value = Math.min(+frame.value, +frame.max);
  const views = frames[+frame.value];
  view.max = views.length - 1;
  view.value = Math.min(+view.value, +view.max);
  document.querySelector("#image").src = views[+view.value];
  document.querySelector("#frameText").textContent = `${{+frame.value + 1}} / ${{frames.length}}`;
  document.querySelector("#viewText").textContent = `${{+view.value + 1}} / ${{views.length}}`;
}}
protocol.onchange = () => {{ frame.value = 0; view.value = 0; render(); }};
frame.oninput = render;
view.oninput = render;
render();
</script>
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    summaries = []
    observations = []
    trajectories = []
    episodes = []
    for path in args.episodes:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes.append((path, payload))
        names, trajectory = _trajectory(payload)
        trajectories.append((path.parent.name, names, trajectory))
        summary, rows = _audit_episode(path)
        summaries.append(summary)
        observations.extend(rows)

    reference_name, reference_joints, reference = trajectories[0]
    trajectory_comparison = []
    for name, joints, trajectory in trajectories[1:]:
        if joints != reference_joints or trajectory.shape != reference.shape:
            difference = float("inf")
        else:
            difference = float(np.max(np.abs(trajectory - reference)))
        trajectory_comparison.append(
            {
                "reference": reference_name,
                "protocol": name,
                "max_abs_joint_trajectory_difference": difference,
                "identical": difference <= args.trajectory_tolerance,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "observability.csv", summaries)
    _write_csv(args.output / "observability_per_frame.csv", observations)
    _write_csv(args.output / "trajectory_validation.csv", trajectory_comparison)
    for path, payload in episodes:
        _write_contact_sheet(
            path,
            payload,
            args.output / "camera_contact_sheets" / f"{path.parent.name}.jpg",
        )
    _write_viewer(args.output / "viewers" / "index.html", episodes)
    (args.output / "observability.json").write_text(
        json.dumps(
            {
                "protocols": summaries,
                "trajectory_comparison": trajectory_comparison,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not all(row["identical"] for row in trajectory_comparison):
        raise RuntimeError("Observation protocols do not share an identical joint trajectory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
