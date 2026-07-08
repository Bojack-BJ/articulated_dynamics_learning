from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json


@dataclass(slots=True)
class ObjectMaskFlowHtmlConfig:
    motion_tracks: str | Path
    output_html: str | Path | None = None
    joint_inference: str | Path | None = None
    evaluation_json: str | Path | None = None
    max_tracks: int = 1000
    frame_stride: int = 2
    trail_length: int = 10
    axis_remap: str = "x,z,-y"
    color_by: str = "pred_cluster"


class ObjectMaskFlowHtmlBuilder:
    """Build a lightweight Plotly HTML viewer for object-mask lifted tracks."""

    def build(self, config: ObjectMaskFlowHtmlConfig) -> Path:
        tracks_path = Path(config.motion_tracks).expanduser().resolve()
        output_html = (
            Path(config.output_html).expanduser().resolve()
            if config.output_html is not None
            else _default_output_html(tracks_path)
        )
        output_html.parent.mkdir(parents=True, exist_ok=True)
        transform = _AxisRemap.from_raw(config.axis_remap)
        artifact = load_json(tracks_path)
        raw_tracks = [track for track in artifact.get("tracks", []) if isinstance(track, dict)]
        tracks = _select_tracks(raw_tracks, max_tracks=max(1, int(config.max_tracks)))
        if not tracks:
            raise ValueError("No tracks found for object-mask flow HTML visualization.")

        frames = _sample_frame_indices(tracks, stride=max(1, int(config.frame_stride)))
        if not frames:
            raise ValueError("No visible track samples found for object-mask flow HTML visualization.")

        payload = {
            "source": "object-mask-flow-html",
            "motion_tracks": str(tracks_path),
            "axis_remap": config.axis_remap,
            "frame_stride": max(1, int(config.frame_stride)),
            "default_trail_length": max(0, int(config.trail_length)),
            "default_color_by": config.color_by
            if config.color_by
            in {
                "pred_cluster",
                "gt_part",
                "motion_magnitude",
                "track_quality",
                "timestep_quality",
                "step_length",
                "acceleration",
                "smooth_residual",
                "step_outlier_score",
                "acceleration_outlier_score",
                "direction_change_deg",
                "rigid_residual",
                "rigid_model_residual",
                "articulation_residual",
                "best_motion_type",
                "cluster_articulation_score",
            }
            else "pred_cluster",
            "tracks": [_track_payload(track, transform=transform) for track in tracks],
            "frame_indices": frames,
            "bounds": _bounds(tracks, transform=transform),
            "joints": _joint_payload(config.joint_inference, config.evaluation_json, transform=transform),
            "metadata": {
                "track_count_input": len(raw_tracks),
                "track_count_embedded": len(tracks),
                "max_tracks": max(1, int(config.max_tracks)),
                "has_original_part_id": any("original_part_id" in track for track in tracks),
            },
        }
        output_html.write_text(_build_html(payload), encoding="utf-8")
        return output_html


class _AxisRemap:
    def __init__(self, axes: list[tuple[int, float]]) -> None:
        self.axes = axes

    @classmethod
    def from_raw(cls, raw: str) -> "_AxisRemap":
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        if len(tokens) != 3:
            raise ValueError("--axis-remap must contain three comma-separated axes, e.g. x,z,-y")
        lookup = {"x": 0, "y": 1, "z": 2}
        axes = []
        seen = set()
        for token in tokens:
            sign = -1.0 if token.startswith("-") else 1.0
            name = token[1:] if token.startswith("-") else token
            if name not in lookup:
                raise ValueError(f"Unsupported axis token in --axis-remap: {token}")
            if name in seen:
                raise ValueError(f"Duplicate axis token in --axis-remap: {token}")
            seen.add(name)
            axes.append((lookup[name], sign))
        return cls(axes)

    def point(self, point: list[float]) -> list[float]:
        return [sign * float(point[index]) for index, sign in self.axes]

    def vector(self, vector: list[float]) -> list[float]:
        return [sign * float(vector[index]) for index, sign in self.axes]


def _select_tracks(tracks: list[dict[str, Any]], *, max_tracks: int) -> list[dict[str, Any]]:
    ranked = sorted(tracks, key=_track_motion, reverse=True)
    return ranked[:max_tracks]


def _default_output_html(tracks_path: Path) -> Path:
    if tracks_path.parent.name == "tracks":
        return tracks_path.parent.parent / "viewers" / "object_mask_flow_viewer.html"
    return tracks_path.parent / "viewers" / "object_mask_flow_viewer.html"


def _sample_frame_indices(tracks: list[dict[str, Any]], *, stride: int) -> list[int]:
    frames = set()
    for track in tracks:
        for sample in track.get("samples", []):
            if not _valid_sample(sample):
                continue
            try:
                frames.add(int(sample.get("frame_index", 0)))
            except (TypeError, ValueError):
                continue
    ordered = sorted(frames)
    return ordered[:: max(1, stride)]


def _track_payload(track: dict[str, Any], *, transform: _AxisRemap) -> dict[str, Any]:
    samples = []
    for sample in track.get("samples", []):
        if not _valid_sample(sample):
            continue
        try:
            frame_index = int(sample.get("frame_index", 0))
        except (TypeError, ValueError):
            continue
        point = transform.point([float(value) for value in sample["xyz_world"]])
        samples.append(
            {
                "frame_index": frame_index,
                "time_s": float(sample.get("timestamp_s", 0.0)),
                "xyz": point,
                "confidence": float(sample.get("confidence", 0.0)),
                "timestep_quality": _optional_float(sample.get("timestep_quality_score")),
                "step_length_m": _optional_float(sample.get("step_length_m")),
                "step_outlier_score": _optional_float(sample.get("step_outlier_score")),
                "acceleration_m": _optional_float(sample.get("acceleration_m")),
                "acceleration_outlier_score": _optional_float(sample.get("acceleration_outlier_score")),
                "jerk_m": _optional_float(sample.get("jerk_m")),
                "jerk_outlier_score": _optional_float(sample.get("jerk_outlier_score")),
                "smooth_residual_m": _optional_float(sample.get("smooth_residual_m")),
                "smooth_residual_score": _optional_float(sample.get("smooth_residual_score")),
                "rigid_residual_m": _optional_float(sample.get("rigid_residual_m")),
                "rigid_residual_score": _optional_float(sample.get("rigid_residual_score")),
                "articulation_residual_m": _optional_float(sample.get("articulation_residual_m")),
                "articulation_residual_score": _optional_float(
                    sample.get("articulation_residual_score", sample.get("residual_score"))
                ),
                "motion_model_type": str(sample.get("motion_model_type", "unknown")),
                "direction_change_deg": _optional_float(sample.get("direction_change_deg")),
                "direction_score": _optional_float(sample.get("direction_score")),
                "local_smoothness_score": _optional_float(sample.get("local_smoothness_score")),
                "step_consistency_score": _optional_float(sample.get("step_consistency_score")),
            }
        )
    samples = sorted(samples, key=lambda item: int(item["frame_index"]))
    pred_cluster = int(track.get("part_id", 0))
    gt_part = int(track.get("original_part_id", pred_cluster))
    quality = track.get("track_quality") or {}
    return {
        "track_id": int(track.get("track_id", len(samples))),
        "pred_cluster": pred_cluster,
        "gt_part": gt_part,
        "part_name": str(track.get("part_name", f"cluster_{pred_cluster}")),
        "motion_m": _track_motion(track),
        "track_quality": _optional_float(quality.get("track_quality_score")),
        "visible_ratio": _optional_float(quality.get("visible_ratio")),
        "temporal_smoothness_score": _optional_float(quality.get("temporal_smoothness_score")),
        "best_motion_type": str(quality.get("best_motion_type", track.get("best_motion_type", "unknown"))),
        "articulation_score": _optional_float(quality.get("articulation_score", track.get("articulation_score"))),
        "cluster_articulation_score": _optional_float(
            quality.get("cluster_articulation_score", track.get("cluster_articulation_score", quality.get("articulation_score")))
        ),
        "samples": samples,
    }


def _bounds(tracks: list[dict[str, Any]], *, transform: _AxisRemap) -> dict[str, list[float]]:
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    for track in tracks:
        for sample in track.get("samples", []):
            if not _valid_sample(sample):
                continue
            point = transform.point([float(value) for value in sample["xyz_world"]])
            for axis in range(3):
                lower[axis] = min(lower[axis], point[axis])
                upper[axis] = max(upper[axis], point[axis])
    if lower[0] is math.inf:
        return {"lower": [-1.0, -1.0, -1.0], "upper": [1.0, 1.0, 1.0]}
    margin = max(0.05, 0.04 * max(upper[axis] - lower[axis] for axis in range(3)))
    return {
        "lower": [value - margin for value in lower],
        "upper": [value + margin for value in upper],
    }


def _joint_payload(
    joint_inference_path: str | Path | None,
    evaluation_path: str | Path | None,
    *,
    transform: _AxisRemap,
) -> list[dict[str, Any]]:
    if joint_inference_path is None:
        return []
    joint_path = Path(joint_inference_path).expanduser().resolve()
    if not joint_path.exists():
        return []
    evaluation = {}
    if evaluation_path is not None:
        eval_path = Path(evaluation_path).expanduser().resolve()
        if eval_path.exists():
            evaluation = load_json(eval_path)
    eval_by_child: dict[int, dict[str, Any]] = {}
    if isinstance(evaluation, dict):
        for row in evaluation.get("per_joint", []):
            if isinstance(row, dict):
                try:
                    eval_by_child[int(row.get("child_part_id"))] = row
                except (TypeError, ValueError):
                    continue

    out = []
    artifact = load_json(joint_path)
    for joint in artifact.get("joints", []):
        if not isinstance(joint, dict):
            continue
        child_id = int(joint.get("child_part_id", -1))
        row = eval_by_child.get(child_id, {})
        out.append(
            {
                "name": str(joint.get("name", f"joint_{child_id}")),
                "joint_type": str(joint.get("joint_type", "unknown")),
                "parent_part_id": int(joint.get("parent_part_id", -1)),
                "child_part_id": child_id,
                "pivot": transform.point(_vec3(joint.get("pivot"), [0.0, 0.0, 0.0])),
                "axis": transform.vector(_vec3(joint.get("axis"), [0.0, 0.0, 1.0])),
                "axis_error_deg": row.get("axis_angle_error_deg"),
                "pivot_error_m": row.get("pivot_error_m"),
            }
        )
    return out


def _build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Object-Mask Flow Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1015;
      --panel: #131d26;
      --text: #e6eef5;
      --muted: #91a4b3;
      --accent: #6ee7b7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #1d2a34 0%, var(--bg) 52%);
      color: var(--text);
    }}
    .app {{ display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 100vh; }}
    aside {{ padding: 18px; background: rgba(10, 16, 21, 0.86); border-right: 1px solid rgba(255,255,255,0.08); }}
    main {{ padding: 16px; }}
    h1 {{ margin: 0 0 14px; font-size: 21px; }}
    .panel {{ margin-bottom: 14px; padding: 14px; border-radius: 14px; background: var(--panel); border: 1px solid rgba(255,255,255,0.08); }}
    label {{ display: block; margin-bottom: 8px; color: var(--muted); font-size: 13px; }}
    input[type="range"], select {{ width: 100%; }}
    input[type="checkbox"] {{ accent-color: var(--accent); }}
    input[type="range"] {{ accent-color: var(--accent); }}
    select {{ border: 1px solid rgba(255,255,255,0.12); background: #0f1720; color: var(--text); padding: 8px; border-radius: 10px; }}
    .value {{ margin-top: 6px; color: var(--text); font-size: 13px; }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 13px; margin: 7px 0; }}
    .row strong {{ color: var(--text); font-weight: 600; text-align: right; word-break: break-all; }}
    .checks {{ display: grid; gap: 8px; }}
    .checkline {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
    #plot {{ width: 100%; height: calc(100vh - 32px); min-height: 720px; border-radius: 18px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }}
    @media (max-width: 1000px) {{ .app {{ grid-template-columns: 1fr; }} #plot {{ height: 720px; }} }}
  </style>
</head>
<body>
<div class="app">
  <aside>
    <h1>Object-Mask Flow Viewer</h1>
    <div class="panel" id="meta"></div>
    <div class="panel">
      <label for="frameSlider">Frame</label>
      <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" />
      <div class="value" id="frameValue"></div>
    </div>
    <div class="panel">
      <label for="trackCountSlider">Visible track count</label>
      <input id="trackCountSlider" type="range" min="1" max="1" step="1" value="1" />
      <div class="value" id="trackCountValue"></div>
    </div>
    <div class="panel">
      <label for="trailSlider">Trail length</label>
      <input id="trailSlider" type="range" min="0" max="40" step="1" value="10" />
      <div class="value" id="trailValue"></div>
    </div>
    <div class="panel">
      <label for="colorBySelect">Color by</label>
      <select id="colorBySelect">
        <option value="pred_cluster">Predicted cluster</option>
        <option value="gt_part">GT part</option>
        <option value="motion_magnitude">Motion magnitude</option>
        <option value="track_quality">Track quality</option>
        <option value="timestep_quality">Timestep quality</option>
        <option value="step_length">Step length</option>
        <option value="acceleration">Acceleration</option>
        <option value="smooth_residual">Smooth residual</option>
        <option value="step_outlier_score">Step outlier score</option>
        <option value="acceleration_outlier_score">Acceleration outlier score</option>
        <option value="direction_change_deg">Direction change</option>
        <option value="rigid_residual">Rigid residual</option>
        <option value="rigid_model_residual">Rigid model residual</option>
        <option value="articulation_residual">Articulation residual</option>
        <option value="best_motion_type">Best motion type</option>
        <option value="cluster_articulation_score">Cluster articulation score</option>
      </select>
    </div>
    <div class="panel">
      <label for="qualityThresholdSlider">Low-quality timestep threshold</label>
      <input id="qualityThresholdSlider" type="range" min="0" max="1" step="0.01" value="0.35" />
      <div class="value" id="qualityThresholdValue"></div>
    </div>
    <div class="panel checks">
      <label>Layers</label>
      <div class="checkline"><input id="showTrails" type="checkbox" checked /> <span>Show trails</span></div>
      <div class="checkline"><input id="showJoints" type="checkbox" checked /> <span>Show joint axes</span></div>
      <div class="checkline"><input id="showAxes" type="checkbox" checked /> <span>Show coordinate axes</span></div>
      <div class="checkline"><input id="hideLowQualityTimesteps" type="checkbox" /> <span>Hide low-quality timesteps</span></div>
    </div>
  </aside>
  <main>
    <div id="plot"></div>
  </main>
</div>
<script>
const DATA = {data_json};
const tracks = DATA.tracks || [];
const frameIndices = DATA.frame_indices || [];
const joints = DATA.joints || [];
const bounds = DATA.bounds || {{lower: [-1,-1,-1], upper: [1,1,1]}};
const metadata = DATA.metadata || {{}};
const frameSlider = document.getElementById("frameSlider");
const frameValue = document.getElementById("frameValue");
const trackCountSlider = document.getElementById("trackCountSlider");
const trackCountValue = document.getElementById("trackCountValue");
const trailSlider = document.getElementById("trailSlider");
const trailValue = document.getElementById("trailValue");
const colorBySelect = document.getElementById("colorBySelect");
const qualityThresholdSlider = document.getElementById("qualityThresholdSlider");
const qualityThresholdValue = document.getElementById("qualityThresholdValue");
const showTrails = document.getElementById("showTrails");
const showJoints = document.getElementById("showJoints");
const showAxes = document.getElementById("showAxes");
const hideLowQualityTimesteps = document.getElementById("hideLowQualityTimesteps");
const meta = document.getElementById("meta");

frameSlider.max = Math.max(0, frameIndices.length - 1);
trackCountSlider.max = Math.max(1, tracks.length);
trackCountSlider.value = Math.min(tracks.length, Math.max(1, Math.min(300, tracks.length)));
trailSlider.value = Math.max(0, Number(DATA.default_trail_length || 10));
colorBySelect.value = DATA.default_color_by || "pred_cluster";

meta.innerHTML = `
  <div class="row"><span>Input tracks</span><strong>${{metadata.track_count_input || tracks.length}}</strong></div>
  <div class="row"><span>Embedded tracks</span><strong>${{tracks.length}}</strong></div>
  <div class="row"><span>Frames</span><strong>${{frameIndices.length}}</strong></div>
  <div class="row"><span>Axis remap</span><strong>${{DATA.axis_remap}}</strong></div>
  <div class="row"><span>Source</span><strong>${{DATA.motion_tracks}}</strong></div>
`;

function palette(index) {{
  const colors = ["#60a5fa", "#fb923c", "#34d399", "#f87171", "#c4b5fd", "#facc15", "#2dd4bf", "#f472b6", "#a3e635", "#94a3b8"];
  return colors[Math.abs(Number(index || 0)) % colors.length];
}}

function motionColor(value, maxValue) {{
  const t = maxValue <= 1e-12 ? 0 : Math.max(0, Math.min(1, value / maxValue));
  const r = Math.round(60 + 195 * t);
  const g = Math.round(180 * (1 - t));
  const b = Math.round(255 * (1 - t));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function scalarColor(value, minValue, maxValue, reverse=false) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "#64748b";
  const denom = Math.max(1e-12, Number(maxValue) - Number(minValue));
  let t = Math.max(0, Math.min(1, (Number(value) - Number(minValue)) / denom));
  if (reverse) t = 1 - t;
  const r = Math.round(239 * (1 - t) + 34 * t);
  const g = Math.round(68 * (1 - t) + 197 * t);
  const b = Math.round(68 * (1 - t) + 94 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function qualityColor(value, minValue, maxValue) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "#64748b";
  // Quality has semantic thresholds, unlike step/acceleration. Keep red reserved for
  // actually low-quality segments instead of percentile-normalizing everything.
  const low = 0.35;
  const high = 0.95;
  const t = Math.max(0, Math.min(1, (Number(value) - low) / Math.max(1e-12, high - low)));
  const stops = [
    [0.0, [239, 68, 68]],
    [0.5, [250, 204, 21]],
    [1.0, [34, 197, 94]],
  ];
  const left = t <= 0.5 ? stops[0] : stops[1];
  const right = t <= 0.5 ? stops[1] : stops[2];
  const u = (t - left[0]) / Math.max(1e-12, right[0] - left[0]);
  const r = Math.round(left[1][0] * (1 - u) + right[1][0] * u);
  const g = Math.round(left[1][1] * (1 - u) + right[1][1] * u);
  const b = Math.round(left[1][2] * (1 - u) + right[1][2] * u);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function sampleAtFrame(track, frameIndex) {{
  for (const sample of track.samples || []) {{
    const sampleFrame = Number(sample.frame_index);
    if (sampleFrame === frameIndex) return sample;
    if (sampleFrame > frameIndex) break;
  }}
  return null;
}}

function trailSamples(track, frameIndex, trailLength) {{
  const samples = [];
  for (const sample of track.samples || []) {{
    const sampleFrame = Number(sample.frame_index);
    if (sampleFrame <= frameIndex && sampleFrame >= frameIndex - trailLength) samples.push(sample);
    if (sampleFrame > frameIndex) break;
  }}
  return samples;
}}

function colorFor(track, colorBy, maxMotion) {{
  if (colorBy === "gt_part") return palette(track.gt_part);
  if (colorBy === "motion_magnitude") return motionColor(Number(track.motion_m || 0), maxMotion);
  if (colorBy === "track_quality") return scalarColor(track.track_quality, 0, 1);
  if (colorBy === "best_motion_type") return motionTypeColor(track.best_motion_type);
  if (colorBy === "cluster_articulation_score") return scalarColor(track.cluster_articulation_score ?? track.articulation_score, 0, 1);
  return palette(track.pred_cluster);
}}

function motionTypeColor(typeName) {{
  const name = String(typeName || "unknown");
  if (name === "static") return "#94a3b8";
  if (name === "prismatic") return "#22d3ee";
  if (name === "revolute") return "#f59e0b";
  return "#a78bfa";
}}

function sampleColor(track, sample, colorBy, maxMotion, scalarRanges) {{
  if (colorBy === "timestep_quality") return qualityColor(sample.timestep_quality, scalarRanges.qualityMin, scalarRanges.qualityMax);
  if (colorBy === "step_length") return scalarColor(sample.step_length_m, scalarRanges.stepMin, scalarRanges.stepMax, true);
  if (colorBy === "acceleration") return scalarColor(sample.acceleration_m, scalarRanges.accelMin, scalarRanges.accelMax, true);
  if (colorBy === "smooth_residual") return scalarColor(sample.smooth_residual_m, scalarRanges.smoothMin, scalarRanges.smoothMax, true);
  if (colorBy === "step_outlier_score") return qualityColor(sample.step_outlier_score, 0, 1);
  if (colorBy === "acceleration_outlier_score") return qualityColor(sample.acceleration_outlier_score, 0, 1);
  if (colorBy === "direction_change_deg") return scalarColor(sample.direction_change_deg, scalarRanges.directionMin, scalarRanges.directionMax, true);
  if (colorBy === "rigid_residual" || colorBy === "rigid_model_residual") return scalarColor(sample.rigid_residual_m, scalarRanges.rigidMin, scalarRanges.rigidMax, true);
  if (colorBy === "articulation_residual") return scalarColor(sample.articulation_residual_m, scalarRanges.articulationMin, scalarRanges.articulationMax, true);
  return colorFor(track, colorBy, maxMotion);
}}

function jointColor(joint, activeTracks, colorBy) {{
  const childId = Number(joint.child_part_id);
  const childTracks = activeTracks.filter((track) => Number(track.pred_cluster) === childId);
  if (colorBy === "gt_part" && childTracks.length) {{
    const counts = new Map();
    for (const track of childTracks) {{
      const gt = Number(track.gt_part);
      counts.set(gt, (counts.get(gt) || 0) + 1);
    }}
    let bestGt = Number(childTracks[0].gt_part);
    let bestCount = -1;
    for (const [gt, count] of counts.entries()) {{
      if (count > bestCount) {{
        bestGt = gt;
        bestCount = count;
      }}
    }}
    return palette(bestGt);
  }}
  if (colorBy === "motion_magnitude" && childTracks.length) {{
    const maxMotion = Math.max(1e-12, ...activeTracks.map((track) => Number(track.motion_m || 0)));
    const childMotion = Math.max(...childTracks.map((track) => Number(track.motion_m || 0)));
    return motionColor(childMotion, maxMotion);
  }}
  return palette(childId);
}}

function scalarRanges(activeTracks) {{
  const steps = [];
  const accels = [];
  const qualities = [];
  const smoothResiduals = [];
  const directionChanges = [];
  const rigidResiduals = [];
  const articulationResiduals = [];
  for (const track of activeTracks) {{
    for (const sample of track.samples || []) {{
      if (sample.step_length_m !== null && sample.step_length_m !== undefined) steps.push(Number(sample.step_length_m));
      if (sample.acceleration_m !== null && sample.acceleration_m !== undefined) accels.push(Number(sample.acceleration_m));
      if (sample.timestep_quality !== null && sample.timestep_quality !== undefined) qualities.push(Number(sample.timestep_quality));
      if (sample.smooth_residual_m !== null && sample.smooth_residual_m !== undefined) smoothResiduals.push(Number(sample.smooth_residual_m));
      if (sample.direction_change_deg !== null && sample.direction_change_deg !== undefined) directionChanges.push(Number(sample.direction_change_deg));
      if (sample.rigid_residual_m !== null && sample.rigid_residual_m !== undefined) rigidResiduals.push(Number(sample.rigid_residual_m));
      if (sample.articulation_residual_m !== null && sample.articulation_residual_m !== undefined) articulationResiduals.push(Number(sample.articulation_residual_m));
    }}
  }}
  const qualityRange = percentileRange(qualities, 0.05, 0.95, 0, 1);
  const stepRange = percentileRange(steps, 0.05, 0.98, 0, Math.max(1e-9, ...steps, 1e-9));
  const accelRange = percentileRange(accels, 0.05, 0.98, 0, Math.max(1e-9, ...accels, 1e-9));
  const smoothRange = percentileRange(smoothResiduals, 0.05, 0.98, 0, 1e-3);
  const directionRange = percentileRange(directionChanges, 0.05, 0.98, 0, 180);
  const rigidRange = percentileRange(rigidResiduals, 0.05, 0.98, 0, 1e-3);
  const articulationRange = percentileRange(articulationResiduals, 0.05, 0.98, 0, 1e-3);
  return {{
    qualityMin: qualityRange.min,
    qualityMax: qualityRange.max,
    stepMin: stepRange.min,
    stepMax: stepRange.max,
    accelMin: accelRange.min,
    accelMax: accelRange.max,
    smoothMin: smoothRange.min,
    smoothMax: smoothRange.max,
    directionMin: directionRange.min,
    directionMax: directionRange.max,
    rigidMin: rigidRange.min,
    rigidMax: rigidRange.max,
    articulationMin: articulationRange.min,
    articulationMax: articulationRange.max,
  }};
}}

function percentileRange(values, lowQ, highQ, fallbackMin, fallbackMax) {{
  const clean = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) return {{min: fallbackMin, max: fallbackMax}};
  const lowIndex = Math.floor(Math.max(0, Math.min(clean.length - 1, lowQ * (clean.length - 1))));
  const highIndex = Math.floor(Math.max(0, Math.min(clean.length - 1, highQ * (clean.length - 1))));
  const low = clean[lowIndex];
  const high = clean[highIndex];
  if (Math.abs(high - low) < 1e-9) {{
    return {{min: Math.max(0, low - 0.05), max: Math.min(1, high + 0.05)}};
  }}
  return {{min: low, max: high}};
}}

function buildPointTrace(activeTracks, frameIndex, colorBy, scalarRangePayload, qualityThreshold, hideLowQuality) {{
  const maxMotion = Math.max(1e-12, ...activeTracks.map((track) => Number(track.motion_m || 0)));
  const x = [], y = [], z = [], colors = [], hover = [];
  for (const track of activeTracks) {{
    const sample = sampleAtFrame(track, frameIndex);
    if (!sample) continue;
    if (hideLowQuality && sample.timestep_quality !== null && sample.timestep_quality !== undefined && Number(sample.timestep_quality) < qualityThreshold) continue;
    x.push(sample.xyz[0]); y.push(sample.xyz[1]); z.push(sample.xyz[2]);
    colors.push(sampleColor(track, sample, colorBy, maxMotion, scalarRangePayload));
    hover.push(`track=${{track.track_id}}<br>pred=${{track.pred_cluster}}<br>gt=${{track.gt_part}}<br>motion=${{Number(track.motion_m || 0).toFixed(4)}} m<br>track_q=${{track.track_quality ?? "n/a"}}<br>best_motion_type=${{track.best_motion_type ?? "n/a"}}<br>articulation_score=${{track.articulation_score ?? "n/a"}}<br>timestep_q=${{sample.timestep_quality ?? "n/a"}}<br>step=${{sample.step_length_m ?? "n/a"}}<br>step_score=${{sample.step_outlier_score ?? "n/a"}}<br>accel=${{sample.acceleration_m ?? "n/a"}}<br>accel_score=${{sample.acceleration_outlier_score ?? "n/a"}}<br>smooth_residual=${{sample.smooth_residual_m ?? "n/a"}}<br>dir_change=${{sample.direction_change_deg ?? "n/a"}}<br>rigid_residual=${{sample.rigid_residual_m ?? "n/a"}}<br>articulation_residual=${{sample.articulation_residual_m ?? "n/a"}}<br>motion_model=${{sample.motion_model_type ?? "n/a"}}<br>frame=${{sample.frame_index}}`);
  }}
  return {{
    type: "scatter3d",
    mode: "markers",
    name: "current points",
    x, y, z,
    text: hover,
    hoverinfo: "text",
    marker: {{ size: 3.2, color: colors, opacity: 0.9 }},
  }};
}}

function shouldHideSample(sample, qualityThreshold, hideLowQuality) {{
  return hideLowQuality
    && sample.timestep_quality !== null
    && sample.timestep_quality !== undefined
    && Number(sample.timestep_quality) <= qualityThreshold;
}}

function segmentColor(track, prevSample, sample, colorBy, maxMotion, scalarRangePayload) {{
  if (colorBy === "timestep_quality" || colorBy === "step_length" || colorBy === "acceleration" || colorBy === "articulation_residual" || colorBy === "rigid_model_residual") {{
    return sampleColor(track, sample, colorBy, maxMotion, scalarRangePayload);
  }}
  return colorFor(track, colorBy, maxMotion);
}}

function buildTrailTraces(activeTracks, frameIndex, colorBy, trailLength, scalarRangePayload, qualityThreshold, hideLowQuality) {{
  if (!showTrails.checked || trailLength <= 0) return [];
  const maxMotion = Math.max(1e-12, ...activeTracks.map((track) => Number(track.motion_m || 0)));
  const groups = new Map();
  for (const track of activeTracks) {{
    const samples = trailSamples(track, frameIndex, trailLength);
    if (samples.length < 2) continue;
    for (let idx = 1; idx < samples.length; idx++) {{
      const prevSample = samples[idx - 1];
      const sample = samples[idx];
      if (shouldHideSample(sample, qualityThreshold, hideLowQuality)) continue;
      const color = segmentColor(track, prevSample, sample, colorBy, maxMotion, scalarRangePayload);
      if (!groups.has(color)) groups.set(color, {{x: [], y: [], z: [], text: []}});
      const group = groups.get(color);
      group.x.push(prevSample.xyz[0], sample.xyz[0], null);
      group.y.push(prevSample.xyz[1], sample.xyz[1], null);
      group.z.push(prevSample.xyz[2], sample.xyz[2], null);
      const hover = `track=${{track.track_id}}<br>pred=${{track.pred_cluster}}<br>gt=${{track.gt_part}}<br>frame=${{prevSample.frame_index}}→${{sample.frame_index}}<br>timestep_q=${{sample.timestep_quality ?? "n/a"}}<br>step=${{sample.step_length_m ?? "n/a"}}<br>step_score=${{sample.step_outlier_score ?? "n/a"}}<br>accel=${{sample.acceleration_m ?? "n/a"}}<br>accel_score=${{sample.acceleration_outlier_score ?? "n/a"}}<br>smooth_residual=${{sample.smooth_residual_m ?? "n/a"}}<br>dir_change=${{sample.direction_change_deg ?? "n/a"}}<br>rigid_residual=${{sample.rigid_residual_m ?? "n/a"}}<br>articulation_residual=${{sample.articulation_residual_m ?? "n/a"}}<br>motion_model=${{sample.motion_model_type ?? "n/a"}}`;
      group.text.push(hover, hover, null);
    }}
  }}
  const traces = [];
  for (const [color, group] of groups.entries()) {{
    traces.push({{
      type: "scatter3d",
      mode: "lines",
      name: `trail ${{color}}`,
      x: group.x, y: group.y, z: group.z,
      text: group.text,
      line: {{ color, width: 3 }},
      opacity: 0.35,
      hoverinfo: "text",
      showlegend: false,
    }});
  }}
  return traces;
}}

function buildJointTraces(activeTracks, colorBy) {{
  if (!showJoints.checked) return [];
  const traces = [];
  for (const joint of joints) {{
    const p = joint.pivot || [0,0,0];
    const a = joint.axis || [0,0,1];
    const color = jointColor(joint, activeTracks, colorBy);
    const segment = clippedAxisSegment(p, a);
    const start = segment.start;
    const end = segment.end;
    const nearest = segment.nearest;
    traces.push({{
      type: "scatter3d",
      mode: "lines+markers",
      name: joint.name || "joint",
      x: [start[0], nearest[0], end[0]],
      y: [start[1], nearest[1], end[1]],
      z: [start[2], nearest[2], end[2]],
      line: {{ color, width: 8 }},
      marker: {{ size: [2, 5, 2], color: [color, "#ffffff", color] }},
      text: [`${{joint.name}} axis<br>child=${{joint.child_part_id}}`, `${{joint.name}} axis near object<br>pivot=[${{p.map((v) => Number(v).toFixed(3)).join(", ")}}]<br>child=${{joint.child_part_id}}<br>type=${{joint.joint_type}}<br>axis_err=${{joint.axis_error_deg ?? "n/a"}}<br>pivot_err=${{joint.pivot_error_m ?? "n/a"}}`, `${{joint.name}} axis<br>child=${{joint.child_part_id}}`],
      hoverinfo: "text",
    }});
    traces.push({{
      type: "scatter3d",
      mode: "markers",
      name: `${{joint.name || "joint"}} pivot`,
      x: [p[0]],
      y: [p[1]],
      z: [p[2]],
      marker: {{ size: 5, color: color, symbol: "diamond", line: {{color: "#ffffff", width: 2}} }},
      text: [`${{joint.name}} actual pivot<br>child=${{joint.child_part_id}}<br>type=${{joint.joint_type}}<br>pivot_err=${{joint.pivot_error_m ?? "n/a"}}`],
      hoverinfo: "text",
      showlegend: false,
    }});
  }}
  return traces;
}}

function clippedAxisSegment(pivot, axis) {{
  const center = [
    0.5 * (bounds.lower[0] + bounds.upper[0]),
    0.5 * (bounds.lower[1] + bounds.upper[1]),
    0.5 * (bounds.lower[2] + bounds.upper[2]),
  ];
  const extents = [
    bounds.upper[0] - bounds.lower[0],
    bounds.upper[1] - bounds.lower[1],
    bounds.upper[2] - bounds.lower[2],
  ];
  const diagonal = Math.max(1e-6, Math.hypot(extents[0], extents[1], extents[2]));
  const len = 0.95 * diagonal;
  const axisNorm = Math.max(1e-9, Math.hypot(axis[0], axis[1], axis[2]));
  const unit = [axis[0] / axisNorm, axis[1] / axisNorm, axis[2] / axisNorm];
  const centerDelta = [center[0] - pivot[0], center[1] - pivot[1], center[2] - pivot[2]];
  const t = centerDelta[0] * unit[0] + centerDelta[1] * unit[1] + centerDelta[2] * unit[2];
  const nearest = [pivot[0] + t * unit[0], pivot[1] + t * unit[1], pivot[2] + t * unit[2]];
  return {{
    start: [nearest[0] - 0.5 * len * unit[0], nearest[1] - 0.5 * len * unit[1], nearest[2] - 0.5 * len * unit[2]],
    nearest,
    end: [nearest[0] + 0.5 * len * unit[0], nearest[1] + 0.5 * len * unit[1], nearest[2] + 0.5 * len * unit[2]],
  }};
}}

function buildAxisTraces() {{
  if (!showAxes.checked) return [];
  const length = 0.25 * Math.max(
    bounds.upper[0] - bounds.lower[0],
    bounds.upper[1] - bounds.lower[1],
    bounds.upper[2] - bounds.lower[2],
  );
  return [
    {{type: "scatter3d", mode: "lines", name: "X", x: [0,length], y: [0,0], z: [0,0], line: {{color:"#ef4444", width:5}}}},
    {{type: "scatter3d", mode: "lines", name: "Y", x: [0,0], y: [0,length], z: [0,0], line: {{color:"#22c55e", width:5}}}},
    {{type: "scatter3d", mode: "lines", name: "Z", x: [0,0], y: [0,0], z: [0,length], line: {{color:"#3b82f6", width:5}}}},
  ];
}}

function render() {{
  const framePosition = Number(frameSlider.value);
  const frameIndex = Number(frameIndices[framePosition] || 0);
  const trackCount = Math.min(Number(trackCountSlider.value), tracks.length);
  const trailLength = Number(trailSlider.value);
  const colorBy = colorBySelect.value;
  const qualityThreshold = Number(qualityThresholdSlider.value);
  const hideLowQuality = hideLowQualityTimesteps.checked;
  const activeTracks = tracks.slice(0, trackCount);
  const scalarRangePayload = scalarRanges(activeTracks);
  frameValue.textContent = `${{framePosition + 1}} / ${{frameIndices.length}} (frame ${{frameIndex}})`;
  trackCountValue.textContent = `${{trackCount}} / ${{tracks.length}} embedded tracks`;
  trailValue.textContent = `${{trailLength}} frames`;
  qualityThresholdValue.textContent = `${{qualityThreshold.toFixed(2)}}`;
  const data = [
    buildPointTrace(activeTracks, frameIndex, colorBy, scalarRangePayload, qualityThreshold, hideLowQuality),
    ...buildTrailTraces(activeTracks, frameIndex, colorBy, trailLength, scalarRangePayload, qualityThreshold, hideLowQuality),
    ...buildJointTraces(activeTracks, colorBy),
    ...buildAxisTraces(),
  ];
  const layout = {{
    paper_bgcolor: "#0b1015",
    plot_bgcolor: "#0b1015",
    font: {{color: "#e6eef5"}},
    margin: {{l: 0, r: 0, t: 28, b: 0}},
    title: `frame ${{frameIndex}} | tracks ${{trackCount}} | color=${{colorBy}}`,
    showlegend: true,
    scene: {{
      xaxis: {{range: [bounds.lower[0], bounds.upper[0]], title: "X", gridcolor: "#1f2a33", zerolinecolor: "#475569"}},
      yaxis: {{range: [bounds.lower[1], bounds.upper[1]], title: "Y", gridcolor: "#1f2a33", zerolinecolor: "#475569"}},
      zaxis: {{range: [bounds.lower[2], bounds.upper[2]], title: "Z", gridcolor: "#1f2a33", zerolinecolor: "#475569"}},
      aspectmode: "data",
    }},
  }};
  Plotly.react("plot", data, layout, {{responsive: true, displaylogo: false}});
}}

for (const element of [frameSlider, trackCountSlider, trailSlider, colorBySelect, qualityThresholdSlider, showTrails, showJoints, showAxes, hideLowQualityTimesteps]) {{
  element.addEventListener("input", render);
  element.addEventListener("change", render);
}}

if (!window.Plotly) {{
  document.getElementById("plot").innerHTML = "<div style='padding:24px;color:#fca5a5'>Plotly failed to load. Check network access or use a local Plotly bundle.</div>";
}} else {{
  render();
}}
</script>
</body>
</html>
"""


def _valid_sample(sample: Any) -> bool:
    return (
        isinstance(sample, dict)
        and bool(sample.get("visible", False))
        and bool(sample.get("depth_valid", True))
        and isinstance(sample.get("xyz_world"), list)
        and len(sample["xyz_world"]) == 3
    )


def _sample_track_point(track: dict[str, Any], mode: str) -> list[float] | None:
    samples = [
        sample for sample in track.get("samples", [])
        if _valid_sample(sample)
    ]
    if not samples:
        return None
    samples = sorted(samples, key=lambda item: int(item.get("frame_index", 0)))
    if mode == "last":
        sample = samples[-1]
    else:
        sample = samples[0]
    return [float(value) for value in sample["xyz_world"]]


def _track_motion(track: dict[str, Any]) -> float:
    first = _sample_track_point(track, "first")
    last = _sample_track_point(track, "last")
    if first is None or last is None:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, last)))


def _vec3(raw: Any, fallback: list[float]) -> list[float]:
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    return list(fallback)


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
