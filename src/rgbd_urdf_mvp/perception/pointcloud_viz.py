from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..core.serialization import load_json


@dataclass(slots=True)
class PointCloudVisualizationConfig:
    input_path: str | Path
    output_html: str | Path | None = None
    max_points_per_frame: int = 4000
    point_radius_px: float = 2.0
    part_pose_path: str | Path | None = None
    part_track_path: str | Path | None = None
    joint_inference_path: str | Path | None = None
    canvas_width: int = 1200
    canvas_height: int = 860


def _parse_ascii_4d_ply(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Unsupported PLY file: {path}")

    header_end = None
    properties: list[str] = []
    vertex_count = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            parts = stripped.split()
            vertex_count = int(parts[-1])
        elif stripped.startswith("property"):
            parts = stripped.split()
            properties.append(parts[-1])
        elif stripped == "end_header":
            header_end = index
            break
    if header_end is None:
        raise ValueError(f"Missing end_header in {path}")

    expected = {"x", "y", "z"}
    if not expected.issubset(set(properties)):
        raise ValueError(f"PLY must contain x/y/z properties: {path}")

    property_indices = {name: properties.index(name) for name in properties}
    frame_groups: dict[int, dict[str, object]] = {}
    min_bounds = [math.inf, math.inf, math.inf]
    max_bounds = [-math.inf, -math.inf, -math.inf]
    min_time = math.inf
    max_time = -math.inf
    part_ids_present: set[int] = set()

    for raw in lines[header_end + 1 : header_end + 1 + vertex_count]:
        if not raw.strip():
            continue
        parts = raw.split()
        x_coord = float(parts[property_indices["x"]])
        y_coord = float(parts[property_indices["y"]])
        z_coord = float(parts[property_indices["z"]])
        time_s = (
            float(parts[property_indices["time"]])
            if "time" in property_indices
            else 0.0
        )
        frame_index = (
            int(parts[property_indices["frame_index"]])
            if "frame_index" in property_indices
            else 0
        )
        part_id = (
            int(parts[property_indices["part_id"]])
            if "part_id" in property_indices
            else 0
        )
        entry = frame_groups.setdefault(frame_index, {"time_s": time_s, "points": []})
        entry["points"].append([x_coord, y_coord, z_coord, part_id])
        entry["time_s"] = time_s
        if part_id > 0:
            part_ids_present.add(part_id)

        min_bounds[0] = min(min_bounds[0], x_coord)
        min_bounds[1] = min(min_bounds[1], y_coord)
        min_bounds[2] = min(min_bounds[2], z_coord)
        max_bounds[0] = max(max_bounds[0], x_coord)
        max_bounds[1] = max(max_bounds[1], y_coord)
        max_bounds[2] = max(max_bounds[2], z_coord)
        min_time = min(min_time, time_s)
        max_time = max(max_time, time_s)

    frames = [
        {"frame_index": frame_index, "time_s": frame_groups[frame_index]["time_s"], "points": frame_groups[frame_index]["points"]}
        for frame_index in sorted(frame_groups)
    ]
    if min_time is math.inf:
        min_time = 0.0
        max_time = 0.0
        min_bounds = [0.0, 0.0, 0.0]
        max_bounds = [0.0, 0.0, 0.0]

    metadata = {
        "frame_count": len(frames),
        "total_point_count": sum(len(frame["points"]) for frame in frames),
        "bounds": {"lower": min_bounds, "upper": max_bounds},
        "time_range_s": [min_time, max_time],
        "part_ids_present": sorted(part_ids_present),
    }
    return frames, metadata


def _sample_points(points: list[list[float]], max_points: int) -> list[list[float]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = max(1, math.ceil(len(points) / max_points))
    sampled = points[::stride]
    if len(sampled) > max_points:
        sampled = sampled[:max_points]
    return sampled


def _build_static_dynamic_payload(
    frames: list[dict[str, object]],
    bounds: dict[str, list[float]],
    voxel_size: float,
) -> tuple[list[list[float]], list[dict[str, object]], dict[str, int]]:
    if not frames:
        return [], [], {"static_voxel_count": 0, "dynamic_render_point_count": 0, "static_threshold": 0}

    if voxel_size <= 0.0:
        lower = bounds["lower"]
        upper = bounds["upper"]
        voxel_size = max(1e-3, max(upper[index] - lower[index] for index in range(3)) / 64.0)

    voxel_counts: dict[tuple[int, int, int], int] = {}
    voxel_sums: dict[tuple[int, int, int], list[float]] = {}
    voxel_part_counts: dict[tuple[int, int, int], dict[int, int]] = {}
    keyed_frames: list[list[tuple[list[float], tuple[int, int, int]]]] = []

    for frame in frames:
        keyed_points: list[tuple[list[float], tuple[int, int, int]]] = []
        frame_keys: set[tuple[int, int, int]] = set()
        for point in frame["points"]:
            key = tuple(int(math.floor(point[index] / voxel_size)) for index in range(3))
            keyed_points.append((point, key))
            frame_keys.add(key)
            if key not in voxel_sums:
                voxel_sums[key] = [point[0], point[1], point[2], 1.0]
            else:
                voxel_sums[key][0] += point[0]
                voxel_sums[key][1] += point[1]
                voxel_sums[key][2] += point[2]
                voxel_sums[key][3] += 1.0
            part_id = int(point[3]) if len(point) > 3 else 0
            bucket_part_counts = voxel_part_counts.setdefault(key, {})
            bucket_part_counts[part_id] = bucket_part_counts.get(part_id, 0) + 1
        keyed_frames.append(keyed_points)
        for key in frame_keys:
            voxel_counts[key] = voxel_counts.get(key, 0) + 1

    static_threshold = max(2, int(math.ceil(0.6 * len(frames))))
    static_keys = {key for key, count in voxel_counts.items() if count >= static_threshold}

    static_points: list[list[float]] = []
    for key in sorted(static_keys):
        sx, sy, sz, count = voxel_sums[key]
        label_counts = voxel_part_counts.get(key, {})
        static_part_id = 0
        if label_counts:
            static_part_id = max(
                sorted(label_counts),
                key=lambda candidate: (label_counts[candidate], candidate != 0, -candidate),
            )
        static_points.append([sx / count, sy / count, sz / count, static_part_id])

    dynamic_frames: list[dict[str, object]] = []
    dynamic_render_point_count = 0
    for frame, keyed_points in zip(frames, keyed_frames):
        dynamic_points = [point for point, key in keyed_points if key not in static_keys]
        dynamic_render_point_count += len(dynamic_points)
        dynamic_frames.append(
            {
                "frame_index": frame["frame_index"],
                "time_s": frame["time_s"],
                "points": dynamic_points,
            }
        )

    return static_points, dynamic_frames, {
        "static_voxel_count": len(static_keys),
        "dynamic_render_point_count": dynamic_render_point_count,
        "static_threshold": static_threshold,
    }


def _resolve_optional_artifact_path(
    input_path: Path,
    explicit_path: str | Path | None,
    default_name: str,
) -> Path | None:
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        return candidate if candidate.exists() else None
    candidate = input_path.parent / default_name
    return candidate if candidate.exists() else None


def _uniform_sample_items(items: list[dict[str, object]], max_items: int) -> list[dict[str, object]]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    stride = max(1, math.ceil(len(items) / max_items))
    sampled = items[::stride]
    if len(sampled) > max_items:
        sampled = sampled[:max_items]
    return sampled


def _overlay_frame_index(sample: dict[str, object]) -> int:
    return int(sample.get("source_frame_index", sample.get("frame_index", 0)))


def _matvec3(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vec[col] for col in range(3)) for row in range(3)]


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _norm3(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def _normalize3(vec: list[float], fallback: list[float] | None = None) -> list[float]:
    length = _norm3(vec)
    if length < 1e-9:
        return list(fallback) if fallback is not None else [0.0, 0.0, 1.0]
    return [value / length for value in vec]


def _load_part_pose_overlay_payload(
    input_path: Path,
    frames: list[dict[str, object]],
    bounds: dict[str, list[float]],
    part_pose_path: str | Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    resolved_part_pose_path = _resolve_optional_artifact_path(input_path, part_pose_path, "part_poses.json")
    if resolved_part_pose_path is None:
        return [], {
            "has_part_pose_overlays": False,
            "part_pose_overlay_count": 0,
            "part_pose_path": None,
            "part_pose_estimator": None,
        }

    part_pose_artifact = load_json(resolved_part_pose_path)
    max_extent = max(
        1e-3,
        *(bounds["upper"][axis] - bounds["lower"][axis] for axis in range(3)),
    )
    axis_length_m = 0.12 * max_extent
    frame_lookup = {int(frame["frame_index"]): [] for frame in frames}
    overlay_count = 0
    part_ids_present: set[int] = set()

    for part in part_pose_artifact.get("parts", []):
        if not isinstance(part, dict):
            continue
        part_id = int(part.get("part_id", 0))
        if part_id <= 0:
            continue
        part_name = str(part.get("name", f"part_{part_id}"))
        for sample in part.get("samples", []):
            if not isinstance(sample, dict) or not bool(sample.get("valid", False)):
                continue
            if "rotation_matrix" not in sample or "translation" not in sample:
                continue
            frame_index = _overlay_frame_index(sample)
            if frame_index not in frame_lookup:
                continue
            frame_lookup[frame_index].append(
                {
                    "part_id": part_id,
                    "part_name": part_name,
                    "translation": [float(value) for value in sample["translation"]],
                    "centroid_world": [float(value) for value in sample.get("centroid_world", sample["translation"])],
                    "rotation_matrix": [[float(value) for value in row] for row in sample["rotation_matrix"]],
                    "confidence": float(sample.get("confidence", 0.0)),
                    "source_frame_index": frame_index,
                    "line_length_m": axis_length_m,
                }
            )
            overlay_count += 1
            part_ids_present.add(part_id)

    return [
        {
            "frame_index": int(frame["frame_index"]),
            "items": frame_lookup.get(int(frame["frame_index"]), []),
        }
        for frame in frames
    ], {
        "has_part_pose_overlays": overlay_count > 0,
        "part_pose_overlay_count": overlay_count,
        "part_pose_path": str(resolved_part_pose_path),
        "part_pose_estimator": str(part_pose_artifact.get("estimator", "unknown")),
        "part_pose_axis_length_m": axis_length_m,
        "part_pose_part_ids": sorted(part_ids_present),
    }


def _load_track_flow_payload(
    input_path: Path,
    frames: list[dict[str, object]],
    part_track_path: str | Path | None,
    max_segments_per_frame: int = 240,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    resolved_part_track_path = _resolve_optional_artifact_path(input_path, part_track_path, "part_tracks.json")
    if resolved_part_track_path is None:
        return [], {
            "has_track_flows": False,
            "track_flow_segment_count": 0,
            "part_track_path": None,
            "part_track_estimator": None,
        }

    part_track_artifact = load_json(resolved_part_track_path)
    frame_lookup: dict[int, list[dict[str, object]]] = {int(frame["frame_index"]): [] for frame in frames}
    track_flow_count = 0
    part_ids_present: set[int] = set()

    for track in part_track_artifact.get("tracks", []):
        if not isinstance(track, dict):
            continue
        part_id = int(track.get("part_id", 0))
        if part_id <= 0:
            continue
        part_name = str(track.get("part_name", f"part_{part_id}"))
        samples = [sample for sample in track.get("samples", []) if isinstance(sample, dict)]
        previous_sample: dict[str, object] | None = None
        for sample in samples:
            if not bool(sample.get("visible", False)):
                previous_sample = None
                continue
            xyz_world = sample.get("xyz_world")
            if not isinstance(xyz_world, list):
                previous_sample = None
                continue
            if previous_sample is not None:
                previous_xyz = previous_sample.get("xyz_world")
                if isinstance(previous_xyz, list):
                    frame_index = _overlay_frame_index(sample)
                    if frame_index in frame_lookup:
                        frame_lookup[frame_index].append(
                            {
                                "part_id": part_id,
                                "part_name": part_name,
                                "track_id": int(track.get("track_id", 0)),
                                "start": [float(value) for value in previous_xyz],
                                "end": [float(value) for value in xyz_world],
                                "confidence": float(sample.get("confidence", 0.0)),
                            }
                        )
                        track_flow_count += 1
                        part_ids_present.add(part_id)
            previous_sample = sample

    return [
        {
            "frame_index": int(frame["frame_index"]),
            "items": _uniform_sample_items(frame_lookup.get(int(frame["frame_index"]), []), max_segments_per_frame),
        }
        for frame in frames
    ], {
        "has_track_flows": track_flow_count > 0,
        "track_flow_segment_count": track_flow_count,
        "part_track_path": str(resolved_part_track_path),
        "part_track_estimator": str(part_track_artifact.get("estimator", "unknown")),
        "track_flow_part_ids": sorted(part_ids_present),
    }


def _load_joint_overlay_payload(
    input_path: Path,
    frames: list[dict[str, object]],
    bounds: dict[str, list[float]],
    part_pose_path: str | Path | None,
    joint_inference_path: str | Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    resolved_part_pose_path = _resolve_optional_artifact_path(input_path, part_pose_path, "part_poses.json")
    resolved_joint_path = _resolve_optional_artifact_path(input_path, joint_inference_path, "joint_inference.json")
    if resolved_part_pose_path is None or resolved_joint_path is None:
        return [], {
            "has_joint_overlays": False,
            "joint_overlay_count": 0,
            "joint_inference_path": None,
            "part_pose_path": None,
        }

    part_pose_artifact = load_json(resolved_part_pose_path)
    joint_artifact = load_json(resolved_joint_path)
    sampled_frame_indices = [int(item) for item in part_pose_artifact.get("sampled_frame_indices", [])]
    q_frame_map: dict[int, int] = {index: source_index for index, source_index in enumerate(sampled_frame_indices)}
    part_samples_by_id: dict[int, dict[int, dict[str, object]]] = {}
    for part in part_pose_artifact.get("parts", []):
        if not isinstance(part, dict):
            continue
        part_id = int(part.get("part_id", 0))
        frame_samples: dict[int, dict[str, object]] = {}
        for sample in part.get("samples", []):
            if not isinstance(sample, dict) or not bool(sample.get("valid", False)):
                continue
            if "rotation_matrix" not in sample or "translation" not in sample:
                continue
            frame_samples[_overlay_frame_index(sample)] = {
                "rotation_matrix": [[float(value) for value in row] for row in sample["rotation_matrix"]],
                "translation": [float(value) for value in sample["translation"]],
            }
        part_samples_by_id[part_id] = frame_samples

    if not part_samples_by_id:
        return [], {
            "has_joint_overlays": False,
            "joint_overlay_count": 0,
            "joint_inference_path": str(resolved_joint_path),
            "part_pose_path": str(resolved_part_pose_path),
        }

    max_extent = max(
        1e-3,
        *(bounds["upper"][axis] - bounds["lower"][axis] for axis in range(3)),
    )
    line_length_m = 0.35 * max_extent
    overlays_by_frame: dict[int, list[dict[str, object]]] = {
        int(frame["frame_index"]): [] for frame in frames
    }
    overlay_count = 0

    for joint in joint_artifact.get("joints", []):
        if not isinstance(joint, dict):
            continue
        parent_part_id = int(joint.get("parent_part_id", 0))
        child_part_id = int(joint.get("child_part_id", 0))
        axis_parent = [float(value) for value in joint.get("axis", [0.0, 0.0, 1.0])]
        pivot_parent = [float(value) for value in joint.get("pivot", [0.0, 0.0, 0.0])]
        q_by_frame = {
            q_frame_map.get(int(sample.get("frame_index", 0)), int(sample.get("frame_index", 0))): float(sample.get("q", 0.0))
            for sample in joint.get("q_samples", [])
            if isinstance(sample, dict)
        }
        parent_track = part_samples_by_id.get(parent_part_id, {})
        if not parent_track:
            continue

        for frame in frames:
            frame_index = int(frame["frame_index"])
            parent_pose = parent_track.get(frame_index)
            if parent_pose is None:
                continue
            parent_rotation = parent_pose["rotation_matrix"]
            parent_translation = parent_pose["translation"]
            axis_world = _normalize3(_matvec3(parent_rotation, axis_parent), fallback=axis_parent)
            pivot_world = _add3(_matvec3(parent_rotation, pivot_parent), parent_translation)
            overlays_by_frame.setdefault(frame_index, []).append(
                {
                    "joint_name": str(joint.get("name", f"joint_{child_part_id}")),
                    "joint_type": str(joint.get("joint_type", "fixed")),
                    "parent_part_id": parent_part_id,
                    "parent_name": str(joint.get("parent_name", f"part_{parent_part_id}")),
                    "child_part_id": child_part_id,
                    "child_name": str(joint.get("child_name", f"part_{child_part_id}")),
                    "pivot_world": [float(value) for value in pivot_world],
                    "axis_world": [float(value) for value in axis_world],
                    "q": float(q_by_frame.get(frame_index, 0.0)),
                    "line_length_m": line_length_m,
                }
            )
            overlay_count += 1

    return [
        {
            "frame_index": int(frame["frame_index"]),
            "items": overlays_by_frame.get(int(frame["frame_index"]), []),
        }
        for frame in frames
    ], {
        "has_joint_overlays": overlay_count > 0,
        "joint_overlay_count": overlay_count,
        "joint_inference_path": str(resolved_joint_path),
        "part_pose_path": str(resolved_part_pose_path),
        "joint_overlay_line_length_m": line_length_m,
    }


def _load_visualization_payload(
    path: Path,
    max_points_per_frame: int,
    part_pose_path: str | Path | None = None,
    part_track_path: str | Path | None = None,
    joint_inference_path: str | Path | None = None,
) -> tuple[dict[str, object], Path]:
    if path.suffix.lower() == ".json":
        manifest = load_json(path)
        ply_path = Path(manifest["pointcloud_4d_path"])
        frames, ply_meta = _parse_ascii_4d_ply(ply_path)
        bounds = manifest.get("bounds", ply_meta["bounds"])
        meta = {
            "frame_count": int(manifest.get("frame_count", ply_meta["frame_count"])),
            "total_point_count": int(manifest.get("total_point_count", ply_meta["total_point_count"])),
            "bounds": bounds,
            "episode_path": manifest.get("episode_path"),
            "pose_sources_used": manifest.get("pose_sources_used", []),
            "per_frame_point_counts": manifest.get("per_frame_point_counts", []),
            "part_ids_present": manifest.get("part_ids_present", ply_meta.get("part_ids_present", [])),
            "part_point_counts": manifest.get("part_point_counts", {}),
            "part_segmentation": manifest.get("part_segmentation"),
            "source_type": "fusion_manifest",
            "input_path": str(path),
            "pointcloud_path": str(ply_path),
        }
        output_html = path.parent / "viewer.html"
    else:
        frames, ply_meta = _parse_ascii_4d_ply(path)
        meta = {
            **ply_meta,
            "source_type": "pointcloud_ply",
            "input_path": str(path),
            "pointcloud_path": str(path),
            "part_point_counts": {},
            "part_segmentation": None,
        }
        output_html = path.with_name(f"{path.stem}_viewer.html")

    prepared_frames = []
    for frame in frames:
        prepared_frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "time_s": float(frame["time_s"]),
                "points": _sample_points(frame["points"], max_points_per_frame),
            }
        )
    voxel_size = 0.0
    if path.suffix.lower() == ".json":
        manifest = load_json(path)
        config = manifest.get("config", {})
        if isinstance(config, dict):
            voxel_size = float(config.get("voxel_size_m", 0.0))
    static_points, dynamic_frames, static_meta = _build_static_dynamic_payload(
        prepared_frames,
        bounds=meta["bounds"],
        voxel_size=voxel_size,
    )
    part_pose_frames, part_pose_meta = _load_part_pose_overlay_payload(
        path,
        prepared_frames,
        bounds=meta["bounds"],
        part_pose_path=part_pose_path,
    )
    track_flow_frames, track_flow_meta = _load_track_flow_payload(
        path,
        prepared_frames,
        part_track_path=part_track_path,
    )
    joint_frames, joint_meta = _load_joint_overlay_payload(
        path,
        prepared_frames,
        bounds=meta["bounds"],
        part_pose_path=part_pose_path,
        joint_inference_path=joint_inference_path,
    )
    meta["render_point_count"] = sum(len(frame["points"]) for frame in prepared_frames)
    meta["max_points_per_frame"] = max_points_per_frame
    meta.update(static_meta)
    meta.update(part_pose_meta)
    meta.update(track_flow_meta)
    meta.update(joint_meta)
    merged_part_ids = sorted(
        {
            *(int(part_id) for part_id in meta.get("part_ids_present", [])),
            *(int(part_id) for part_id in meta.get("part_pose_part_ids", [])),
            *(int(part_id) for part_id in meta.get("track_flow_part_ids", [])),
        }
    )
    meta["part_ids_present"] = merged_part_ids
    meta["has_part_labels"] = bool(merged_part_ids)
    return {
        "meta": meta,
        "frames": prepared_frames,
        "static_points": static_points,
        "dynamic_frames": dynamic_frames,
        "part_pose_frames": part_pose_frames,
        "track_flow_frames": track_flow_frames,
        "joint_frames": joint_frames,
    }, output_html


def _build_html(payload: dict[str, object], config: PointCloudVisualizationConfig) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>4D Point Cloud Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101419;
      --panel: #182129;
      --panel-2: #22303b;
      --text: #e8f0f5;
      --muted: #90a4b2;
      --accent: #69d2a8;
      --accent-2: #6fb1ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1e2831 0%, var(--bg) 55%);
      color: var(--text);
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 20px;
      background: rgba(16, 20, 25, 0.86);
      border-right: 1px solid rgba(255,255,255,0.08);
      backdrop-filter: blur(10px);
    }}
    .main {{
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 22px;
      letter-spacing: 0.02em;
    }}
    .meta, .control-group {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .legend {{
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }}
    .legend-row {{
      display: grid;
      grid-template-columns: 14px 1fr auto;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.18);
    }}
    .meta-line {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin: 8px 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta-line strong {{
      color: var(--text);
      font-weight: 600;
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .value {{
      color: var(--text);
      font-size: 13px;
      margin-top: 6px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .subtoolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 6px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font-weight: 600;
      background: var(--panel-2);
      color: var(--text);
      cursor: pointer;
    }}
    button.primary {{
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #0c1216;
    }}
    canvas {{
      width: 100%;
      height: auto;
      aspect-ratio: {config.canvas_width} / {config.canvas_height};
      border-radius: 18px;
      background: linear-gradient(180deg, #0c1218 0%, #0a0f14 100%);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }}
    .timeline {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      background: rgba(16, 20, 25, 0.78);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
    }}
    .projection-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .projection-card {{
      background: rgba(16, 20, 25, 0.78);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 10px;
    }}
    .projection-card h2 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .projection-card canvas {{
      aspect-ratio: 1 / 1;
      border-radius: 12px;
      background: #0a0f14;
    }}
    code {{
      color: #d7edf8;
      word-break: break-all;
    }}
    @media (max-width: 1100px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      .projection-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>4D Point Cloud Viewer</h1>
      <div class="meta" id="meta"></div>
      <div class="control-group">
        <label for="frameSlider">Frame</label>
        <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" />
        <div class="value" id="frameValue"></div>
      </div>
      <div class="control-group">
        <label for="yawSlider">Yaw</label>
        <input id="yawSlider" type="range" min="-180" max="180" step="1" value="35" />
        <div class="value" id="yawValue"></div>
      </div>
      <div class="control-group">
        <label for="pitchSlider">Pitch</label>
        <input id="pitchSlider" type="range" min="-89" max="89" step="1" value="22" />
        <div class="value" id="pitchValue"></div>
      </div>
      <div class="control-group">
        <label for="scaleSlider">Zoom</label>
        <input id="scaleSlider" type="range" min="0.5" max="4.0" step="0.05" value="1.6" />
        <div class="value" id="scaleValue"></div>
      </div>
      <div class="control-group">
        <label for="radiusSlider">Point Radius</label>
        <input id="radiusSlider" type="range" min="0.5" max="6" step="0.25" value="{config.point_radius_px}" />
        <div class="value" id="radiusValue"></div>
      </div>
      <div class="control-group">
        <label for="ghostSlider">Ghost History</label>
        <input id="ghostSlider" type="range" min="0" max="8" step="1" value="0" />
        <div class="value" id="ghostValue"></div>
      </div>
      <div class="control-group">
        <label>Visible Parts</label>
        <div id="partFilterPanel" class="legend"></div>
      </div>
    </aside>
    <main class="main">
      <div class="toolbar">
        <button id="playButton" class="primary">Play</button>
        <button id="resetViewButton">Reset View</button>
      </div>
      <div class="subtoolbar">
        <button id="presetPerspective">Perspective</button>
        <button id="presetFront">Front</button>
        <button id="presetSide">Side</button>
        <button id="presetTop">Top</button>
      </div>
      <div class="subtoolbar">
        <button id="modeCurrent">Current</button>
        <button id="modeCurrentStatic">Current + Static</button>
        <button id="modeStatic">Static Only</button>
        <button id="modeDynamic">Dynamic Only</button>
      </div>
      <div class="subtoolbar">
        <button id="colorHeight">Color: Height</button>
        <button id="colorPart">Color: Part</button>
        <button id="togglePoses">Show Poses</button>
        <button id="toggleFlow">Show Flow</button>
        <button id="toggleJoints">Show Joints</button>
      </div>
      <canvas id="viewer" width="{config.canvas_width}" height="{config.canvas_height}"></canvas>
      <div class="projection-grid">
        <div class="projection-card">
          <h2>Front (X-Z)</h2>
          <canvas id="frontProjection" width="360" height="360"></canvas>
        </div>
        <div class="projection-card">
          <h2>Side (Y-Z)</h2>
          <canvas id="sideProjection" width="360" height="360"></canvas>
        </div>
        <div class="projection-card">
          <h2>Top (X-Y)</h2>
          <canvas id="topProjection" width="360" height="360"></canvas>
        </div>
      </div>
      <div class="timeline">
        <div class="status" id="status"></div>
        <div class="status">Drag on the canvas to rotate. Scroll to zoom.</div>
      </div>
    </main>
  </div>
  <script>
  const DATA = {data_json};
  const metaEl = document.getElementById("meta");
  const frameSlider = document.getElementById("frameSlider");
  const frameValue = document.getElementById("frameValue");
  const yawSlider = document.getElementById("yawSlider");
  const yawValue = document.getElementById("yawValue");
  const pitchSlider = document.getElementById("pitchSlider");
  const pitchValue = document.getElementById("pitchValue");
  const scaleSlider = document.getElementById("scaleSlider");
  const scaleValue = document.getElementById("scaleValue");
  const radiusSlider = document.getElementById("radiusSlider");
  const radiusValue = document.getElementById("radiusValue");
  const ghostSlider = document.getElementById("ghostSlider");
  const ghostValue = document.getElementById("ghostValue");
  const partFilterPanel = document.getElementById("partFilterPanel");
  const playButton = document.getElementById("playButton");
  const resetViewButton = document.getElementById("resetViewButton");
  const presetPerspective = document.getElementById("presetPerspective");
  const presetFront = document.getElementById("presetFront");
  const presetSide = document.getElementById("presetSide");
  const presetTop = document.getElementById("presetTop");
  const modeCurrent = document.getElementById("modeCurrent");
  const modeCurrentStatic = document.getElementById("modeCurrentStatic");
  const modeStatic = document.getElementById("modeStatic");
  const modeDynamic = document.getElementById("modeDynamic");
  const colorHeight = document.getElementById("colorHeight");
  const colorPart = document.getElementById("colorPart");
  const togglePoses = document.getElementById("togglePoses");
  const toggleFlow = document.getElementById("toggleFlow");
  const toggleJoints = document.getElementById("toggleJoints");
  const statusEl = document.getElementById("status");
  const canvas = document.getElementById("viewer");
  const ctx = canvas.getContext("2d");
  const projectionCanvases = {{
    front: document.getElementById("frontProjection"),
    side: document.getElementById("sideProjection"),
    top: document.getElementById("topProjection"),
  }};

  const frames = DATA.frames;
  const dynamicFrames = DATA.dynamic_frames || frames;
  const staticPoints = DATA.static_points || [];
  const partPoseFrames = DATA.part_pose_frames || [];
  const trackFlowFrames = DATA.track_flow_frames || [];
  const jointFrames = DATA.joint_frames || [];
  const meta = DATA.meta;
  const hasPartLabels = Boolean(meta.has_part_labels);
  const hasPartPoseOverlays = Boolean(meta.has_part_pose_overlays);
  const hasTrackFlows = Boolean(meta.has_track_flows);
  const hasJointOverlays = Boolean(meta.has_joint_overlays);
  const bounds = meta.bounds || {{ lower: [0,0,0], upper: [1,1,1] }};
  const center = bounds.lower.map((value, index) => (value + bounds.upper[index]) / 2);
  const extents = bounds.upper.map((value, index) => Math.max(1e-6, value - bounds.lower[index]));
  const baseScale = 0.42 * Math.min(canvas.width, canvas.height) / Math.max(...extents);
  let frameIndex = 0;
  let playing = false;
  let lastTick = 0;
  let viewMode = "current_static";
  let colorMode = hasPartLabels ? "part" : "height";
  let showPoses = hasPartPoseOverlays;
  let showFlow = hasTrackFlows;
  let showJoints = hasJointOverlays;
  const selectedPartIds = new Set((meta.part_ids_present || []).map((partId) => Number(partId)));

  function formatNumber(value) {{
    return Number(value).toFixed(3);
  }}

  function partColor(partId, alpha) {{
    const numericPartId = Number(partId || 0);
    if (!numericPartId) {{
      return `rgba(160, 174, 189, ${{alpha}})`;
    }}
    const hue = (numericPartId * 137.508) % 360;
    return `hsla(${{hue}}, 78%, 66%, ${{alpha}})`;
  }}

  function updateMeta() {{
    const partEntries = Object.entries(meta.part_point_counts || {{}});
    const legendHtml = partEntries.length
      ? `<div class="legend">${{partEntries.map(([partId, part]) => `
          <div class="legend-row">
            <span class="legend-swatch" style="background:${{partColor(Number(partId), 0.95)}}"></span>
            <span>${{part.name || `part_${{partId}}`}}</span>
            <strong>${{part.count}}</strong>
          </div>
        `).join("")}}</div>`
      : "";
    metaEl.innerHTML = `
      <div class="meta-line"><span>Source</span><strong>${{meta.source_type}}</strong></div>
      <div class="meta-line"><span>Frames</span><strong>${{meta.frame_count}}</strong></div>
      <div class="meta-line"><span>Total points</span><strong>${{meta.total_point_count}}</strong></div>
      <div class="meta-line"><span>Render points</span><strong>${{meta.render_point_count}}</strong></div>
      <div class="meta-line"><span>Static voxels</span><strong>${{meta.static_voxel_count || 0}}</strong></div>
      <div class="meta-line"><span>Input</span><strong><code>${{meta.input_path}}</code></strong></div>
      <div class="meta-line"><span>Point cloud</span><strong><code>${{meta.pointcloud_path}}</code></strong></div>
      <div class="meta-line"><span>Pose source</span><strong>${{(meta.pose_sources_used || []).join(", ") || "n/a"}}</strong></div>
      <div class="meta-line"><span>Part pose estimator</span><strong>${{meta.part_pose_estimator || "n/a"}}</strong></div>
      <div class="meta-line"><span>Track flow estimator</span><strong>${{meta.part_track_estimator || "n/a"}}</strong></div>
      <div class="meta-line"><span>Part ids</span><strong>${{(meta.part_ids_present || []).join(", ") || "none"}}</strong></div>
      <div class="meta-line"><span>Pose overlays</span><strong>${{meta.part_pose_overlay_count || 0}}</strong></div>
      <div class="meta-line"><span>Flow segments</span><strong>${{meta.track_flow_segment_count || 0}}</strong></div>
      <div class="meta-line"><span>Joint overlays</span><strong>${{meta.joint_overlay_count || 0}}</strong></div>
      <div class="meta-line"><span>Part poses</span><strong><code>${{meta.part_pose_path || "n/a"}}</code></strong></div>
      <div class="meta-line"><span>Part tracks</span><strong><code>${{meta.part_track_path || "n/a"}}</code></strong></div>
      <div class="meta-line"><span>Joint inference</span><strong><code>${{meta.joint_inference_path || "n/a"}}</code></strong></div>
      ${{legendHtml}}
    `;
  }}

  function isPartVisible(partId) {{
    const numericPartId = Number(partId || 0);
    if (!numericPartId) return true;
    if (!selectedPartIds.size) return true;
    return selectedPartIds.has(numericPartId);
  }}

  function rebuildPartFilters() {{
    const pointCountMap = meta.part_point_counts || {{}};
    const partIds = (meta.part_ids_present || []).map((partId) => Number(partId));
    if (!partIds.length) {{
      partFilterPanel.innerHTML = `<div class="value">No part labels</div>`;
      return;
    }}
    partFilterPanel.innerHTML = partIds.map((partId) => {{
      const part = pointCountMap[String(partId)] || {{}};
      const count = part.count ?? "-";
      const name = part.name || `part_${{partId}}`;
      return `
      <label class="legend-row" style="grid-template-columns: 18px 14px 1fr auto; cursor:pointer;">
        <input type="checkbox" data-part-id="${{partId}}" ${{isPartVisible(partId) ? "checked" : ""}} />
        <span class="legend-swatch" style="background:${{partColor(partId, 0.95)}}"></span>
        <span>${{name}}</span>
        <strong>${{count}}</strong>
      </label>
    `;
    }}).join("");
    partFilterPanel.querySelectorAll("input[type=checkbox]").forEach((el) => {{
      el.addEventListener("change", () => {{
        const partId = Number(el.dataset.partId || "0");
        if (el.checked) {{
          selectedPartIds.add(partId);
        }} else {{
          selectedPartIds.delete(partId);
        }}
        render();
      }});
    }});
  }}

  function syncControls() {{
    frameSlider.max = Math.max(0, frames.length - 1);
    frameSlider.value = frameIndex;
    frameValue.textContent = `Frame ${{frameIndex}} / ${{Math.max(0, frames.length - 1)}}`;
    yawValue.textContent = `${{yawSlider.value}} deg`;
    pitchValue.textContent = `${{pitchSlider.value}} deg`;
    scaleValue.textContent = `${{Number(scaleSlider.value).toFixed(2)}}x`;
    radiusValue.textContent = `${{Number(radiusSlider.value).toFixed(2)}} px`;
    ghostValue.textContent = `${{ghostSlider.value}} frames`;
  }}

  function rotate(point, yawRad, pitchRad) {{
    const x0 = point[0] - center[0];
    const y0 = point[1] - center[1];
    const z0 = point[2] - center[2];
    const cosY = Math.cos(yawRad);
    const sinY = Math.sin(yawRad);
    const cosP = Math.cos(pitchRad);
    const sinP = Math.sin(pitchRad);
    const x1 = cosY * x0 + sinY * z0;
    const z1 = -sinY * x0 + cosY * z0;
    const y1 = cosP * y0 - sinP * z1;
    const z2 = sinP * y0 + cosP * z1;
    return [x1, y1, z2];
  }}

  function project(rotated, scale) {{
    const depth = 3.2 + rotated[2] / Math.max(...extents);
    const perspective = 1.0 / Math.max(0.4, depth);
    return [
      canvas.width * 0.5 + rotated[0] * scale * perspective,
      canvas.height * 0.5 - rotated[1] * scale * perspective,
      perspective,
    ];
  }}

  function pointColor(zNorm, alpha) {{
    const r = Math.round(95 + 120 * zNorm);
    const g = Math.round(150 + 80 * (1 - zNorm));
    const b = Math.round(255 - 90 * zNorm);
    return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
  }}

  function pointFill(point, alpha, warm=false) {{
    if (colorMode === "part" && hasPartLabels) {{
      return partColor(point[3], alpha);
    }}
    const zNorm = (point[2] - bounds.lower[2]) / Math.max(1e-6, bounds.upper[2] - bounds.lower[2]);
    if (warm) {{
      return `rgba(255, ${{Math.round(180 - 90 * zNorm)}}, 110, ${{alpha}})`;
    }}
    return pointColor(zNorm, alpha);
  }}

  function projectPointForMain(point, yaw, pitch, scale) {{
    const rotated = rotate(point, yaw, pitch);
    const [sx, sy, perspective] = project(rotated, scale);
    return {{ sx, sy, perspective }};
  }}

  function jointEndpoints(item) {{
    const axis = item.axis_world || [0, 0, 1];
    const pivot = item.pivot_world || [0, 0, 0];
    const half = 0.5 * Number(item.line_length_m || meta.joint_overlay_line_length_m || 0.2);
    return {{
      start: [
        pivot[0] - axis[0] * half,
        pivot[1] - axis[1] * half,
        pivot[2] - axis[2] * half,
      ],
      end: [
        pivot[0] + axis[0] * half,
        pivot[1] + axis[1] * half,
        pivot[2] + axis[2] * half,
      ],
      pivot,
    }};
  }}

  function poseAxes(item) {{
    const origin = item.centroid_world || item.translation || [0, 0, 0];
    const rotation = item.rotation_matrix || [[1,0,0],[0,1,0],[0,0,1]];
    const scale = Number(item.line_length_m || meta.part_pose_axis_length_m || 0.1);
    return [
      {{
        axis: "x",
        color: "rgba(255,110,110,0.95)",
        start: origin,
        end: [
          origin[0] + rotation[0][0] * scale,
          origin[1] + rotation[1][0] * scale,
          origin[2] + rotation[2][0] * scale,
        ],
      }},
      {{
        axis: "y",
        color: "rgba(110,255,170,0.95)",
        start: origin,
        end: [
          origin[0] + rotation[0][1] * scale,
          origin[1] + rotation[1][1] * scale,
          origin[2] + rotation[2][1] * scale,
        ],
      }},
      {{
        axis: "z",
        color: "rgba(110,170,255,0.95)",
        start: origin,
        end: [
          origin[0] + rotation[0][2] * scale,
          origin[1] + rotation[1][2] * scale,
          origin[2] + rotation[2][2] * scale,
        ],
      }},
    ];
  }}

  function drawPartPosesMain(overlays, yaw, pitch, scale, radiusBase) {{
    if (!showPoses || !hasPartPoseOverlays) return;
    ctx.save();
    ctx.lineWidth = 1.35;
    for (const item of overlays) {{
      if (!isPartVisible(item.part_id)) continue;
      const origin = projectPointForMain(item.translation || [0,0,0], yaw, pitch, scale);
      for (const axis of poseAxes(item)) {{
        const start = projectPointForMain(axis.start, yaw, pitch, scale);
        const end = projectPointForMain(axis.end, yaw, pitch, scale);
        ctx.strokeStyle = axis.color;
        ctx.beginPath();
        ctx.moveTo(start.sx, start.sy);
        ctx.lineTo(end.sx, end.sy);
        ctx.stroke();
      }}
      ctx.fillStyle = partColor(item.part_id, 0.95);
      ctx.beginPath();
      ctx.arc(origin.sx, origin.sy, Math.max(1.8, radiusBase * 1.2), 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
  }}

  function drawTrackFlowsMain(flows, yaw, pitch, scale, radiusBase) {{
    if (!showFlow || !hasTrackFlows) return;
    ctx.save();
    for (const item of flows) {{
      if (!isPartVisible(item.part_id)) continue;
      const start = projectPointForMain(item.start, yaw, pitch, scale);
      const end = projectPointForMain(item.end, yaw, pitch, scale);
      const alpha = Math.max(0.15, Math.min(0.95, Number(item.confidence || 0.0)));
      ctx.strokeStyle = partColor(item.part_id, alpha);
      ctx.lineWidth = Math.max(0.8, radiusBase * 0.5);
      ctx.beginPath();
      ctx.moveTo(start.sx, start.sy);
      ctx.lineTo(end.sx, end.sy);
      ctx.stroke();
      ctx.fillStyle = partColor(item.part_id, alpha);
      ctx.beginPath();
      ctx.arc(end.sx, end.sy, Math.max(0.8, radiusBase * 0.75), 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
  }}

  function drawJointOverlaysMain(overlays, yaw, pitch, scale, radiusBase) {{
    if (!showJoints || !hasJointOverlays) return;
    ctx.save();
    ctx.lineWidth = 1.5;
    for (const item of overlays) {{
      const endpoints = jointEndpoints(item);
      const start = projectPointForMain(endpoints.start, yaw, pitch, scale);
      const end = projectPointForMain(endpoints.end, yaw, pitch, scale);
      const pivot = projectPointForMain(endpoints.pivot, yaw, pitch, scale);
      const color = partColor(item.child_part_id, 0.95);
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(start.sx, start.sy);
      ctx.lineTo(end.sx, end.sy);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pivot.sx, pivot.sy, Math.max(2.0, radiusBase * 1.4), 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
  }}

  function drawProjectionJointOverlays(pctx, axisA, axisB, pad, overlays) {{
    if (!showJoints || !hasJointOverlays) return;
    const aMin = bounds.lower[axisA];
    const aSpan = Math.max(1e-6, bounds.upper[axisA] - bounds.lower[axisA]);
    const bMin = bounds.lower[axisB];
    const bSpan = Math.max(1e-6, bounds.upper[axisB] - bounds.lower[axisB]);
    for (const item of overlays) {{
      const endpoints = jointEndpoints(item);
      const start = endpoints.start;
      const end = endpoints.end;
      const pivot = endpoints.pivot;
      const sx0 = pad + ((start[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
      const sy0 = pctx.canvas.height - pad - ((start[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
      const sx1 = pad + ((end[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
      const sy1 = pctx.canvas.height - pad - ((end[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
      const px = pad + ((pivot[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
      const py = pctx.canvas.height - pad - ((pivot[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
      pctx.strokeStyle = partColor(item.child_part_id, 0.95);
      pctx.lineWidth = 1.5;
      pctx.beginPath();
      pctx.moveTo(sx0, sy0);
      pctx.lineTo(sx1, sy1);
      pctx.stroke();
      pctx.fillStyle = partColor(item.child_part_id, 0.95);
      pctx.beginPath();
      pctx.arc(px, py, 3.0, 0, Math.PI * 2);
      pctx.fill();
    }}
  }}

  function drawProjectionPartPoses(pctx, axisA, axisB, pad, overlays) {{
    if (!showPoses || !hasPartPoseOverlays) return;
    const aMin = bounds.lower[axisA];
    const aSpan = Math.max(1e-6, bounds.upper[axisA] - bounds.lower[axisA]);
    const bMin = bounds.lower[axisB];
    const bSpan = Math.max(1e-6, bounds.upper[axisB] - bounds.lower[axisB]);
    for (const item of overlays) {{
      if (!isPartVisible(item.part_id)) continue;
      for (const axis of poseAxes(item)) {{
        const sx0 = pad + ((axis.start[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
        const sy0 = pctx.canvas.height - pad - ((axis.start[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
        const sx1 = pad + ((axis.end[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
        const sy1 = pctx.canvas.height - pad - ((axis.end[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
        pctx.strokeStyle = axis.color;
        pctx.lineWidth = 1.25;
        pctx.beginPath();
        pctx.moveTo(sx0, sy0);
        pctx.lineTo(sx1, sy1);
        pctx.stroke();
      }}
    }}
  }}

  function drawProjectionTrackFlows(pctx, axisA, axisB, pad, flows) {{
    if (!showFlow || !hasTrackFlows) return;
    const aMin = bounds.lower[axisA];
    const aSpan = Math.max(1e-6, bounds.upper[axisA] - bounds.lower[axisA]);
    const bMin = bounds.lower[axisB];
    const bSpan = Math.max(1e-6, bounds.upper[axisB] - bounds.lower[axisB]);
    for (const item of flows) {{
      if (!isPartVisible(item.part_id)) continue;
      const sx0 = pad + ((item.start[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
      const sy0 = pctx.canvas.height - pad - ((item.start[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
      const sx1 = pad + ((item.end[axisA] - aMin) / aSpan) * (pctx.canvas.width - 2 * pad);
      const sy1 = pctx.canvas.height - pad - ((item.end[axisB] - bMin) / bSpan) * (pctx.canvas.height - 2 * pad);
      pctx.strokeStyle = partColor(item.part_id, Math.max(0.15, Math.min(0.95, Number(item.confidence || 0.0))));
      pctx.lineWidth = 1.0;
      pctx.beginPath();
      pctx.moveTo(sx0, sy0);
      pctx.lineTo(sx1, sy1);
      pctx.stroke();
    }}
  }}

  function drawProjection(canvasEl, axisA, axisB, title) {{
    const pctx = canvasEl.getContext("2d");
    pctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    pctx.fillStyle = "#0b1015";
    pctx.fillRect(0, 0, canvasEl.width, canvasEl.height);

    const ghostFrames = viewMode === "static" ? 0 : Number(ghostSlider.value);
    const pad = 24;
    const aMin = bounds.lower[axisA];
    const aSpan = Math.max(1e-6, bounds.upper[axisA] - bounds.lower[axisA]);
    const bMin = bounds.lower[axisB];
    const bSpan = Math.max(1e-6, bounds.upper[axisB] - bounds.lower[axisB]);

    pctx.strokeStyle = "rgba(255,255,255,0.10)";
    pctx.strokeRect(pad, pad, canvasEl.width - 2 * pad, canvasEl.height - 2 * pad);
    pctx.fillStyle = "rgba(255,255,255,0.55)";
    pctx.font = "12px sans-serif";
    pctx.fillText(title, pad, 16);

    function drawProjectionPoints(points, alpha, warm=false) {{
      for (const point of points) {{
        if (!isPartVisible(point[3])) continue;
        const u = (point[axisA] - aMin) / aSpan;
        const v = (point[axisB] - bMin) / bSpan;
        const x = pad + u * (canvasEl.width - 2 * pad);
        const y = canvasEl.height - pad - v * (canvasEl.height - 2 * pad);
        pctx.fillStyle = pointFill(point, alpha, warm);
        pctx.fillRect(x, y, 2, 2);
      }}
    }}

    if (viewMode === "static" || viewMode === "current_static") {{
      drawProjectionPoints(staticPoints, 0.18, false);
    }}
    if (viewMode === "current" || viewMode === "current_static") {{
      for (let offset = ghostFrames; offset >= 0; offset -= 1) {{
        const current = frameIndex - offset;
        if (current < 0 || current >= frames.length) continue;
        const frame = frames[current];
        const alpha = offset === 0 ? 0.95 : 0.12 + 0.08 * (ghostFrames - offset);
        drawProjectionPoints(frame.points, alpha, false);
      }}
    }} else if (viewMode === "dynamic") {{
      const frame = dynamicFrames[frameIndex] || {{points: []}};
      drawProjectionPoints(frame.points, 0.95, true);
    }}
    const partPoseFrame = partPoseFrames[frameIndex] || {{items: []}};
    const trackFlowFrame = trackFlowFrames[frameIndex] || {{items: []}};
    const jointFrame = jointFrames[frameIndex] || {{items: []}};
    drawProjectionTrackFlows(pctx, axisA, axisB, pad, trackFlowFrame.items || []);
    drawProjectionPartPoses(pctx, axisA, axisB, pad, partPoseFrame.items || []);
    drawProjectionJointOverlays(pctx, axisA, axisB, pad, jointFrame.items || []);
  }}

  function render() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0b1015";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const yaw = Number(yawSlider.value) * Math.PI / 180.0;
    const pitch = Number(pitchSlider.value) * Math.PI / 180.0;
    const scale = baseScale * Number(scaleSlider.value);
    const radiusBase = Number(radiusSlider.value);
    const ghostFrames = viewMode === "static" ? 0 : Number(ghostSlider.value);
    const zMin = bounds.lower[2];
    const zSpan = Math.max(1e-6, bounds.upper[2] - bounds.lower[2]);

    if (viewMode === "static" || viewMode === "current_static") {{
      for (const point of staticPoints) {{
        if (!isPartVisible(point[3])) continue;
        const rotated = rotate(point, yaw, pitch);
        const [sx, sy, perspective] = project(rotated, scale);
        ctx.fillStyle = pointFill(point, 0.18);
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(0.6, radiusBase * perspective), 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    if (viewMode === "current" || viewMode === "current_static") {{
      for (let offset = ghostFrames; offset >= 0; offset -= 1) {{
        const current = frameIndex - offset;
        if (current < 0 || current >= frames.length) continue;
        const frame = frames[current];
        const ghostAlpha = offset === 0 ? 0.95 : 0.14 + 0.08 * (ghostFrames - offset);
        for (const point of frame.points) {{
          if (!isPartVisible(point[3])) continue;
          const rotated = rotate(point, yaw, pitch);
          const [sx, sy, perspective] = project(rotated, scale);
          ctx.fillStyle = pointFill(point, ghostAlpha);
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(0.6, radiusBase * perspective), 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
    }} else if (viewMode === "dynamic") {{
      const frame = dynamicFrames[frameIndex] || {{points: []}};
      for (const point of frame.points) {{
        if (!isPartVisible(point[3])) continue;
        const rotated = rotate(point, yaw, pitch);
        const [sx, sy, perspective] = project(rotated, scale);
        ctx.fillStyle = pointFill(point, 0.95, true);
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(0.6, radiusBase * perspective), 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    const frame = frames[frameIndex] || {{ time_s: 0, points: [] }};
    const dynamicFrame = dynamicFrames[frameIndex] || {{points: []}};
    const partPoseFrame = partPoseFrames[frameIndex] || {{items: []}};
    const trackFlowFrame = trackFlowFrames[frameIndex] || {{items: []}};
    const jointFrame = jointFrames[frameIndex] || {{items: []}};
    drawTrackFlowsMain(trackFlowFrame.items || [], yaw, pitch, scale, radiusBase);
    drawPartPosesMain(partPoseFrame.items || [], yaw, pitch, scale, radiusBase);
    drawJointOverlaysMain(jointFrame.items || [], yaw, pitch, scale, radiusBase);
    statusEl.textContent = `time=${{formatNumber(frame.time_s)}} s | frame=${{frameIndex}} | points=${{frame.points.length}} | dynamic=${{dynamicFrame.points.length}} | flow=${{(trackFlowFrame.items || []).length}} | poses=${{(partPoseFrame.items || []).length}} | joints=${{(jointFrame.items || []).length}} | pose_src=${{meta.part_pose_estimator || "n/a"}} | mode=${{viewMode}} | color=${{colorMode}}`;
    drawProjection(projectionCanvases.front, 0, 2, "X vs Z");
    drawProjection(projectionCanvases.side, 1, 2, "Y vs Z");
    drawProjection(projectionCanvases.top, 0, 1, "X vs Y");
  }}

  function applyPreset(name) {{
    if (name === "perspective") {{
      yawSlider.value = 35;
      pitchSlider.value = 22;
      scaleSlider.value = 1.6;
    }} else if (name === "front") {{
      yawSlider.value = 0;
      pitchSlider.value = 0;
      scaleSlider.value = 1.8;
    }} else if (name === "side") {{
      yawSlider.value = 90;
      pitchSlider.value = 0;
      scaleSlider.value = 1.8;
    }} else if (name === "top") {{
      yawSlider.value = 0;
      pitchSlider.value = -89;
      scaleSlider.value = 1.8;
    }}
    syncControls();
    render();
  }}

  function setMode(name) {{
    viewMode = name;
    render();
  }}

  function setColorMode(name) {{
    if (name === "part" && !hasPartLabels) return;
    colorMode = name;
    render();
  }}

  function animate(ts) {{
    if (playing && ts - lastTick > 120) {{
      frameIndex = (frameIndex + 1) % Math.max(1, frames.length);
      frameSlider.value = frameIndex;
      lastTick = ts;
      syncControls();
      render();
    }}
    requestAnimationFrame(animate);
  }}

  frameSlider.addEventListener("input", () => {{
    frameIndex = Number(frameSlider.value);
    syncControls();
    render();
  }});
  [yawSlider, pitchSlider, scaleSlider, radiusSlider, ghostSlider].forEach((el) => {{
    el.addEventListener("input", () => {{
      syncControls();
      render();
    }});
  }});
  playButton.addEventListener("click", () => {{
    playing = !playing;
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.classList.toggle("primary", !playing);
  }});
  resetViewButton.addEventListener("click", () => {{
    applyPreset("perspective");
    radiusSlider.value = {config.point_radius_px};
    ghostSlider.value = 0;
    syncControls();
    render();
  }});
  presetPerspective.addEventListener("click", () => applyPreset("perspective"));
  presetFront.addEventListener("click", () => applyPreset("front"));
  presetSide.addEventListener("click", () => applyPreset("side"));
  presetTop.addEventListener("click", () => applyPreset("top"));
  modeCurrent.addEventListener("click", () => setMode("current"));
  modeCurrentStatic.addEventListener("click", () => setMode("current_static"));
  modeStatic.addEventListener("click", () => setMode("static"));
  modeDynamic.addEventListener("click", () => setMode("dynamic"));
  colorHeight.addEventListener("click", () => setColorMode("height"));
  colorPart.addEventListener("click", () => setColorMode("part"));
  togglePoses.addEventListener("click", () => {{
    if (!hasPartPoseOverlays) return;
    showPoses = !showPoses;
    togglePoses.classList.toggle("primary", showPoses);
    render();
  }});
  toggleFlow.addEventListener("click", () => {{
    if (!hasTrackFlows) return;
    showFlow = !showFlow;
    toggleFlow.classList.toggle("primary", showFlow);
    render();
  }});
  toggleJoints.addEventListener("click", () => {{
    if (!hasJointOverlays) return;
    showJoints = !showJoints;
    toggleJoints.classList.toggle("primary", showJoints);
    render();
  }});

  if (!hasPartLabels) {{
    colorPart.disabled = true;
    colorPart.style.opacity = "0.45";
    colorPart.style.cursor = "default";
  }}
  if (!hasPartPoseOverlays) {{
    togglePoses.disabled = true;
    togglePoses.style.opacity = "0.45";
    togglePoses.style.cursor = "default";
  }} else {{
    togglePoses.classList.toggle("primary", showPoses);
  }}
  if (!hasTrackFlows) {{
    toggleFlow.disabled = true;
    toggleFlow.style.opacity = "0.45";
    toggleFlow.style.cursor = "default";
  }} else {{
    toggleFlow.classList.toggle("primary", showFlow);
  }}
  if (!hasJointOverlays) {{
    toggleJoints.disabled = true;
    toggleJoints.style.opacity = "0.45";
    toggleJoints.style.cursor = "default";
  }} else {{
    toggleJoints.classList.toggle("primary", showJoints);
  }}

  let drag = null;
  canvas.addEventListener("mousedown", (event) => {{
    drag = {{
      x: event.clientX,
      y: event.clientY,
      yaw: Number(yawSlider.value),
      pitch: Number(pitchSlider.value),
    }};
  }});
  window.addEventListener("mousemove", (event) => {{
    if (!drag) return;
    yawSlider.value = String(Math.max(-180, Math.min(180, drag.yaw + (event.clientX - drag.x) * 0.35)));
    pitchSlider.value = String(Math.max(-89, Math.min(89, drag.pitch + (event.clientY - drag.y) * 0.25)));
    syncControls();
    render();
  }});
  window.addEventListener("mouseup", () => {{
    drag = null;
  }});
  canvas.addEventListener("wheel", (event) => {{
    event.preventDefault();
    const current = Number(scaleSlider.value);
    const next = Math.max(0.5, Math.min(4.0, current * (event.deltaY > 0 ? 0.93 : 1.07)));
    scaleSlider.value = String(next);
    syncControls();
    render();
  }}, {{ passive: false }});

  updateMeta();
  rebuildPartFilters();
  syncControls();
  applyPreset("perspective");
  setMode("current_static");
  setColorMode(hasPartLabels ? "part" : "height");
  requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


class PointCloudViewerBuilder:
    def __init__(self, config: PointCloudVisualizationConfig) -> None:
        self.config = config

    def build(self) -> Path:
        input_path = Path(self.config.input_path).resolve()
        payload, default_output = _load_visualization_payload(
            input_path,
            max_points_per_frame=max(1, int(self.config.max_points_per_frame)),
            part_pose_path=self.config.part_pose_path,
            part_track_path=self.config.part_track_path,
            joint_inference_path=self.config.joint_inference_path,
        )
        output_html = (
            Path(self.config.output_html).resolve()
            if self.config.output_html is not None
            else default_output.resolve()
        )
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(_build_html(payload, self.config), encoding="utf-8")
        return output_html
