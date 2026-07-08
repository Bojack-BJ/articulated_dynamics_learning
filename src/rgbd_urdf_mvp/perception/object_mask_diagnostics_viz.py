from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json


@dataclass(slots=True)
class ObjectMaskDiagnosticsVisualizationConfig:
    motion_tracks: str | Path
    output_dir: str | Path
    joint_inference: str | Path | None = None
    evaluation_json: str | Path | None = None
    local_split_summary: str | Path | None = None
    candidate_eval: str | Path | None = None
    frame: str = "first"
    top_n_candidates: int = 8
    axis_length_m: float = 0.5
    viz_frame: str = "world"
    axis_remap: str = "x,z,-y"
    flow_min_motion_m: float = 0.005
    flow_max_tracks: int = 2000
    flow_subsample: int = 1
    flow_scale: float = 1.0
    make_matplotlib: bool = False
    plot_projections: str = "xz,xy"
    make_animation: bool = False
    animation_fps: int = 12
    animation_max_tracks: int = 1000
    animation_color_by: str = "pred_cluster"


class ObjectMaskDiagnosticsVisualizer:
    """Write diagnostic PLY files for object-mask motion segmentation."""

    def build(self, config: ObjectMaskDiagnosticsVisualizationConfig) -> Path:
        tracks_path = Path(config.motion_tracks).expanduser().resolve()
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ply_dir = output_dir / "ply"
        plots_dir = output_dir / "plots"
        metadata_dir = output_dir / "metadata"
        ply_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        transform = _VizTransform.from_remap(config.axis_remap)
        track_artifact = load_json(tracks_path)
        tracks = [track for track in track_artifact.get("tracks", []) if isinstance(track, dict)]
        if not tracks:
            raise ValueError("No tracks found for object-mask diagnostic visualization.")

        outputs: dict[str, str] = {}
        pred_points = _track_points(tracks, mode=config.frame, color_mode="pred", transform=transform)
        outputs["clusters_pred"] = str(_write_vertex_ply(ply_dir / f"clusters_pred_{config.frame}.ply", pred_points))
        gt_points = _track_points(tracks, mode=config.frame, color_mode="gt", transform=transform)
        outputs["clusters_gt"] = str(_write_vertex_ply(ply_dir / f"clusters_gt_{config.frame}.ply", gt_points))
        motion_points = _track_points(tracks, mode=config.frame, color_mode="motion", transform=transform)
        outputs["clusters_motion"] = str(_write_vertex_ply(ply_dir / f"clusters_motion_{config.frame}.ply", motion_points))
        largest_points = _largest_cluster_points(tracks, mode=config.frame, transform=transform)
        if largest_points:
            outputs["largest_cluster_gt"] = str(_write_vertex_ply(ply_dir / f"largest_cluster_gt_{config.frame}.ply", largest_points))

        evaluation = _load_optional_json(config.evaluation_json)
        if evaluation is not None:
            purity_points = _purity_error_points(tracks, evaluation, mode=config.frame, transform=transform)
            outputs["purity_error"] = str(_write_vertex_ply(ply_dir / f"purity_error_{config.frame}.ply", purity_points))

        joint_artifact = _load_optional_json(config.joint_inference)
        if joint_artifact is not None:
            outputs["joints_pred_vs_gt"] = str(
                _write_joint_ply(
                    ply_dir / "joints_pred_vs_gt_3d.ply",
                    tracks=tracks,
                    joint_artifact=joint_artifact,
                    evaluation=evaluation,
                    axis_length_m=float(config.axis_length_m),
                    mode=config.frame,
                    transform=transform,
                )
            )

        flow_edges = _flow_edges(
            tracks,
            color_mode="pred",
            transform=transform,
            min_motion_m=float(config.flow_min_motion_m),
            max_tracks=int(config.flow_max_tracks),
            subsample=max(1, int(config.flow_subsample)),
            scale=float(config.flow_scale),
        )
        outputs["flow_arrows_pred_cluster"] = str(_write_ply(ply_dir / "flow_arrows_pred_cluster.ply", flow_edges["vertices"], flow_edges["edges"]))
        flow_edges_gt = _flow_edges(
            tracks,
            color_mode="gt",
            transform=transform,
            min_motion_m=float(config.flow_min_motion_m),
            max_tracks=int(config.flow_max_tracks),
            subsample=max(1, int(config.flow_subsample)),
            scale=float(config.flow_scale),
        )
        outputs["flow_arrows_gt_part"] = str(_write_ply(ply_dir / "flow_arrows_gt_part.ply", flow_edges_gt["vertices"], flow_edges_gt["edges"]))
        flow_edges_motion = _flow_edges(
            tracks,
            color_mode="motion",
            transform=transform,
            min_motion_m=float(config.flow_min_motion_m),
            max_tracks=int(config.flow_max_tracks),
            subsample=max(1, int(config.flow_subsample)),
            scale=float(config.flow_scale),
        )
        outputs["flow_arrows_motion_magnitude"] = str(_write_ply(ply_dir / "flow_arrows_motion_magnitude.ply", flow_edges_motion["vertices"], flow_edges_motion["edges"]))
        largest_flow = _focused_flow_edges(
            tracks,
            part_ids=_largest_cluster_ids(tracks),
            color_mode="gt",
            transform=transform,
            min_motion_m=0.0,
            max_tracks=int(config.flow_max_tracks),
            scale=float(config.flow_scale),
        )
        if largest_flow["vertices"]:
            outputs["largest_cluster_flow"] = str(_write_ply(ply_dir / "largest_cluster_flow.ply", largest_flow["vertices"], largest_flow["edges"]))

        local_split = _load_optional_json(config.local_split_summary)
        candidate_eval = _load_optional_json(config.candidate_eval)
        if local_split is not None:
            candidate_points = _local_candidate_points(
                local_split,
                candidate_eval,
                mode=config.frame,
                top_n=max(1, int(config.top_n_candidates)),
                transform=transform,
            )
            if candidate_points:
                outputs["local_split_candidates_topn"] = str(
                    _write_vertex_ply(ply_dir / f"local_split_candidates_top{int(config.top_n_candidates)}_{config.frame}.ply", candidate_points)
                )

        plot_outputs = {}
        if config.make_matplotlib:
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_outputs = _write_matplotlib_overview(
                plots_dir,
                tracks,
                evaluation=evaluation,
                transform=transform,
                projections=_parse_projections(config.plot_projections),
                flow_min_motion_m=float(config.flow_min_motion_m),
                flow_scale=float(config.flow_scale),
            )
            outputs.update(plot_outputs)
        if config.make_animation:
            plots_dir.mkdir(parents=True, exist_ok=True)
            animation_outputs = _write_matplotlib_animations(
                plots_dir,
                tracks,
                transform=transform,
                projections=_parse_projections(config.plot_projections),
                color_by=str(config.animation_color_by),
                fps=max(1, int(config.animation_fps)),
                max_tracks=max(1, int(config.animation_max_tracks)),
                flow_scale=float(config.flow_scale),
            )
            outputs.update(animation_outputs)

        color_maps = {
            "pred_cluster": _color_map_for_ids(sorted({int(track.get("part_id", 0)) for track in tracks})),
            "gt_part": _color_map_for_ids(sorted({int(track.get("original_part_id", track.get("part_id", 0))) for track in tracks})),
        }
        save_json(color_maps["pred_cluster"], metadata_dir / "cluster_color_map.json")
        save_json(color_maps["gt_part"], metadata_dir / "gt_color_map.json")
        manifest_path = output_dir / "object_mask_diagnostics_viz.json"
        save_json(
            {
                "source": "object-mask-diagnostics-visualization",
                "motion_tracks": str(tracks_path),
                "frame": config.frame,
                "viz_frame": config.viz_frame,
                "axis_remap": config.axis_remap,
                "outputs": outputs,
                "metadata": {
                    "ply_dir": str(ply_dir),
                    "plots_dir": str(plots_dir),
                    "metadata_dir": str(metadata_dir),
                    "color_maps": color_maps,
                },
                "notes": [
                    "PLY outputs are diagnostic only and do not affect inference.",
                    "clusters_pred colors by predicted cluster id.",
                    "clusters_gt colors by original_part_id when available.",
                    "purity_error colors non-dominant GT tracks inside each predicted cluster in red.",
                    "joints_pred_vs_gt draws predicted axes in red and GT axes in green.",
                    "All PLY and PNG outputs use the same axis remap transform.",
                    "Animation outputs are 2D projected temporal replays and are diagnostic only.",
                ],
            },
            manifest_path,
        )
        return manifest_path


class _VizTransform:
    def __init__(self, axes: list[tuple[int, float]]) -> None:
        self.axes = axes

    @classmethod
    def from_remap(cls, raw: str) -> "_VizTransform":
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        if len(tokens) != 3:
            raise ValueError("--axis-remap must contain three comma-separated axes, e.g. x,z,-y")
        axis_index = {"x": 0, "y": 1, "z": 2}
        axes = []
        seen = set()
        for token in tokens:
            sign = -1.0 if token.startswith("-") else 1.0
            name = token[1:] if token.startswith("-") else token
            if name not in axis_index:
                raise ValueError(f"Unsupported axis token in --axis-remap: {token}")
            if name in seen:
                raise ValueError(f"Duplicate axis token in --axis-remap: {token}")
            seen.add(name)
            axes.append((axis_index[name], sign))
        return cls(axes)

    def point(self, point: list[float]) -> list[float]:
        return [sign * float(point[index]) for index, sign in self.axes]

    def vector(self, vector: list[float]) -> list[float]:
        return [sign * float(vector[index]) for index, sign in self.axes]


def _track_points(tracks: list[dict[str, Any]], *, mode: str, color_mode: str, transform: _VizTransform) -> list[dict[str, Any]]:
    motions = [_track_motion(track) for track in tracks]
    max_motion = max(motions, default=0.0)
    points = []
    for track, motion in zip(tracks, motions):
        point = _sample_track_point(track, mode)
        if point is None:
            continue
        if color_mode == "pred":
            color = _palette(int(track.get("part_id", 0)))
        elif color_mode == "gt":
            color = _palette(int(track.get("original_part_id", track.get("part_id", 0))))
        elif color_mode == "motion":
            color = _motion_color(motion, max_motion)
        else:
            color = (180, 180, 180)
        points.append(_vertex(transform.point(point), color))
    return points


def _largest_cluster_ids(tracks: list[dict[str, Any]]) -> list[int]:
    counts: dict[int, int] = {}
    for track in tracks:
        part_id = int(track.get("part_id", 0))
        counts[part_id] = counts.get(part_id, 0) + 1
    if not counts:
        return []
    return [max(sorted(counts), key=lambda part_id: counts[part_id])]


def _largest_cluster_points(tracks: list[dict[str, Any]], *, mode: str, transform: _VizTransform) -> list[dict[str, Any]]:
    largest_ids = set(_largest_cluster_ids(tracks))
    if not largest_ids:
        return []
    out = []
    for track in tracks:
        if int(track.get("part_id", 0)) not in largest_ids:
            continue
        point = _sample_track_point(track, mode)
        if point is not None:
            out.append(_vertex(transform.point(point), _palette(int(track.get("original_part_id", track.get("part_id", 0))))))
    return out


def _purity_error_points(tracks: list[dict[str, Any]], evaluation: dict[str, Any], *, mode: str, transform: _VizTransform) -> list[dict[str, Any]]:
    dominant: dict[int, int] = {}
    for item in evaluation.get("overlap", {}).get("per_cluster", []):
        if not isinstance(item, dict):
            continue
        try:
            dominant[int(item["pred_cluster_id"])] = int(item["dominant_gt_part_id"])
        except (TypeError, ValueError, KeyError):
            continue
    out = []
    for track in tracks:
        point = _sample_track_point(track, mode)
        if point is None:
            continue
        pred = int(track.get("part_id", 0))
        gt = int(track.get("original_part_id", pred))
        color = (235, 60, 60) if pred in dominant and gt != dominant[pred] else (120, 190, 255)
        out.append(_vertex(transform.point(point), color))
    return out


def _write_joint_ply(
    path: Path,
    *,
    tracks: list[dict[str, Any]],
    joint_artifact: dict[str, Any],
    evaluation: dict[str, Any] | None,
    axis_length_m: float,
    mode: str,
    transform: _VizTransform,
) -> Path:
    vertices = []
    edges = []
    eval_by_child = {}
    if isinstance(evaluation, dict):
        for row in evaluation.get("per_joint", []):
            if isinstance(row, dict):
                try:
                    eval_by_child[int(row.get("child_part_id"))] = row
                except (TypeError, ValueError):
                    continue

    for joint_index, joint in enumerate(joint_artifact.get("joints", [])):
        if not isinstance(joint, dict):
            continue
        parent_id = int(joint.get("parent_part_id", -1))
        child_id = int(joint.get("child_part_id", -1))
        for track in tracks:
            part_id = int(track.get("part_id", 0))
            if part_id not in {parent_id, child_id}:
                continue
            point = _sample_track_point(track, mode)
            if point is None:
                continue
            color = (90, 150, 240) if part_id == parent_id else (245, 160, 55)
            vertices.append(_vertex(transform.point(point), color))

        pred_axis = _vec3(joint.get("axis"), [0.0, 0.0, 1.0])
        pred_pivot = _vec3(joint.get("pivot"), [0.0, 0.0, 0.0])
        _append_axis(vertices, edges, transform.point(pred_pivot), transform.vector(pred_axis), axis_length_m, (240, 50, 50))
        eval_row = eval_by_child.get(child_id, {})
        gt_axis = _vec3(eval_row.get("ground_truth_axis"), None)
        gt_pivot = _vec3(eval_row.get("ground_truth_pivot"), None)
        if gt_axis is not None and gt_pivot is not None:
            _append_axis(vertices, edges, transform.point(gt_pivot), transform.vector(gt_axis), axis_length_m, (50, 210, 80))

        # Add a tiny per-joint offset marker so coincident pivots remain visible.
        pred_pivot_viz = transform.point(pred_pivot)
        marker = [pred_pivot_viz[0], pred_pivot_viz[1], pred_pivot_viz[2] + 0.01 * (joint_index + 1)]
        vertices.append(_vertex(marker, (255, 255, 255)))

    return _write_ply(path, vertices, edges)


def _local_candidate_points(
    local_split: dict[str, Any],
    candidate_eval: dict[str, Any] | None,
    *,
    mode: str,
    top_n: int,
    transform: _VizTransform,
) -> list[dict[str, Any]]:
    rows = []
    if isinstance(candidate_eval, dict):
        rows = [
            row for row in candidate_eval.get("rows", [])
            if isinstance(row, dict) and row.get("track_path") and row.get("part_id") is not None
        ]
        rows = sorted(rows, key=lambda row: float(row.get("joint_selection_score_no_gt") or row.get("selection_score_no_gt") or 0.0), reverse=True)
    if not rows:
        for candidate in local_split.get("candidates", []):
            if isinstance(candidate, dict) and candidate.get("status") == "written":
                for cluster in candidate.get("after", {}).get("clusters", []):
                    if isinstance(cluster, dict):
                        rows.append({"track_path": candidate.get("output_json"), "part_id": cluster.get("part_id")})
    out = []
    for rank, row in enumerate(rows[:top_n]):
        path = Path(str(row.get("track_path", ""))).expanduser()
        if not path.exists():
            continue
        tracks = [track for track in load_json(path).get("tracks", []) if isinstance(track, dict)]
        part_id = int(row.get("part_id", -1))
        color = _palette(rank + 1)
        for track in tracks:
            if int(track.get("part_id", 0)) != part_id:
                continue
            point = _sample_track_point(track, mode)
            if point is not None:
                out.append(_vertex(transform.point(point), color))
    return out


def _flow_edges(
    tracks: list[dict[str, Any]],
    *,
    color_mode: str,
    transform: _VizTransform,
    min_motion_m: float,
    max_tracks: int,
    subsample: int,
    scale: float,
) -> dict[str, list[dict[str, Any]]]:
    selected = tracks[:: max(1, int(subsample))]
    if len(selected) > max_tracks:
        step = max(1, math.ceil(len(selected) / max(1, max_tracks)))
        selected = selected[::step]
    max_motion = max((_track_motion(track) for track in selected), default=0.0)
    vertices: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for track in selected:
        start = _sample_track_point(track, "first")
        end = _sample_track_point(track, "last")
        if start is None or end is None:
            continue
        motion = _distance(start, end)
        if motion < min_motion_m:
            continue
        if color_mode == "pred":
            color = _palette(int(track.get("part_id", 0)))
        elif color_mode == "gt":
            color = _palette(int(track.get("original_part_id", track.get("part_id", 0))))
        else:
            color = _motion_color(motion, max_motion)
        start_viz = transform.point(start)
        end_scaled = [start[i] + (end[i] - start[i]) * scale for i in range(3)]
        end_viz = transform.point(end_scaled)
        first_index = len(vertices)
        vertices.append(_vertex(start_viz, color))
        vertices.append(_vertex(end_viz, color))
        edges.append({"v1": first_index, "v2": first_index + 1, "red": color[0], "green": color[1], "blue": color[2]})
    return {"vertices": vertices, "edges": edges}


def _focused_flow_edges(
    tracks: list[dict[str, Any]],
    *,
    part_ids: list[int],
    color_mode: str,
    transform: _VizTransform,
    min_motion_m: float,
    max_tracks: int,
    scale: float,
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(int(value) for value in part_ids)
    subset = [track for track in tracks if int(track.get("part_id", 0)) in allowed]
    return _flow_edges(
        subset,
        color_mode=color_mode,
        transform=transform,
        min_motion_m=min_motion_m,
        max_tracks=max_tracks,
        subsample=1,
        scale=scale,
    )


def _write_matplotlib_overview(
    plots_dir: Path,
    tracks: list[dict[str, Any]],
    *,
    evaluation: dict[str, Any] | None,
    transform: _VizTransform,
    projections: list[str],
    flow_min_motion_m: float,
    flow_scale: float,
) -> dict[str, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    projections = projections or ["xz", "xy"]
    outputs: dict[str, str] = {}
    for projection in projections:
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        panels = [
            ("Pred clusters", "pred"),
            ("GT parts", "gt"),
            ("Motion magnitude", "motion"),
            ("Purity error", "purity"),
        ]
        for axis, (title, mode) in zip(axes.flatten(), panels):
            _plot_projection_panel(axis, tracks, projection=projection, color_mode=mode, transform=transform, evaluation=evaluation, flow_scale=flow_scale, min_motion_m=flow_min_motion_m)
            axis.set_title(f"{title} ({projection.upper()})")
        figure.tight_layout()
        path = plots_dir / f"overview_{projection}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs[f"overview_{projection}"] = str(path)
    return outputs


def _write_matplotlib_animations(
    plots_dir: Path,
    tracks: list[dict[str, Any]],
    *,
    transform: _VizTransform,
    projections: list[str],
    color_by: str,
    fps: int,
    max_tracks: int,
    flow_scale: float,
) -> dict[str, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception:
        return {}
    selected = _select_animation_tracks(tracks, max_tracks=max_tracks)
    frame_indices = _track_frame_indices(selected)
    if not selected or not frame_indices:
        return {}

    projections = projections or ["xz", "xy"]
    color_by = color_by if color_by in {"pred_cluster", "gt_part", "motion_magnitude"} else "pred_cluster"
    bounds = {projection: _projection_bounds(selected, projection=projection, transform=transform) for projection in projections}
    outputs: dict[str, str] = {}
    for projection in projections:
        figure, axis = plt.subplots(figsize=(8, 8))
        xlim, ylim = bounds[projection]

        def update(frame_index: int) -> None:
            axis.clear()
            axis.set_title(f"Track replay {projection.upper()} | frame {frame_index} | color={color_by}")
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(projection[0].upper())
            axis.set_ylabel(projection[1].upper())
            axis.grid(True, alpha=0.2)
            _draw_animation_frame(
                axis,
                selected,
                frame_index=frame_index,
                projection=projection,
                transform=transform,
                color_by=color_by,
                flow_scale=flow_scale,
            )

        animation = FuncAnimation(figure, update, frames=frame_indices, interval=1000.0 / max(1, fps), repeat=True)
        stem = f"track_replay_{projection}_{color_by}"
        mp4_path = plots_dir / f"{stem}.mp4"
        gif_path = plots_dir / f"{stem}.gif"
        try:
            animation.save(mp4_path, writer="ffmpeg", fps=fps, dpi=140)
            outputs[f"{stem}_mp4"] = str(mp4_path)
        except Exception:
            try:
                animation.save(gif_path, writer="pillow", fps=fps, dpi=100)
                outputs[f"{stem}_gif"] = str(gif_path)
            except Exception:
                pass
        plt.close(figure)
    return outputs


def _draw_animation_frame(
    axis: Any,
    tracks: list[dict[str, Any]],
    *,
    frame_index: int,
    projection: str,
    transform: _VizTransform,
    color_by: str,
    flow_scale: float,
) -> None:
    frame_points_x = []
    frame_points_y = []
    frame_colors = []
    max_motion = max((_track_motion_until(track, frame_index) for track in tracks), default=0.0)
    for track in tracks:
        start = _sample_track_point(track, "first")
        current = _sample_track_point_at_frame(track, frame_index)
        if start is None or current is None:
            continue
        motion = _distance(start, current)
        color = _animation_color(track, color_by=color_by, motion=motion, max_motion=max_motion)
        start_viz = transform.point(start)
        current_scaled = [start[i] + (current[i] - start[i]) * flow_scale for i in range(3)]
        current_viz = transform.point(current_scaled)
        x0, y0 = _project(start_viz, projection)
        x1, y1 = _project(current_viz, projection)
        if motion > 1e-6:
            axis.plot([x0, x1], [y0, y1], color=color, alpha=0.3, linewidth=0.6)
        frame_points_x.append(x1)
        frame_points_y.append(y1)
        frame_colors.append(color)
    if frame_points_x:
        axis.scatter(frame_points_x, frame_points_y, s=8, c=frame_colors, alpha=0.8, linewidths=0.0)


def _select_animation_tracks(tracks: list[dict[str, Any]], *, max_tracks: int) -> list[dict[str, Any]]:
    movable = [track for track in tracks if _track_motion(track) > 1e-6]
    selected = movable if movable else tracks
    selected = sorted(selected, key=_track_motion, reverse=True)
    if len(selected) <= max_tracks:
        return selected
    step = max(1, math.ceil(len(selected) / max_tracks))
    return selected[::step][:max_tracks]


def _track_frame_indices(tracks: list[dict[str, Any]]) -> list[int]:
    indices = set()
    for track in tracks:
        for sample in track.get("samples", []):
            if isinstance(sample, dict) and bool(sample.get("visible", False)) and bool(sample.get("depth_valid", True)):
                try:
                    indices.add(int(sample.get("frame_index", 0)))
                except (TypeError, ValueError):
                    continue
    return sorted(indices)


def _projection_bounds(tracks: list[dict[str, Any]], *, projection: str, transform: _VizTransform) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = []
    ys = []
    for track in tracks:
        for sample in track.get("samples", []):
            if not isinstance(sample, dict) or not bool(sample.get("visible", False)) or not bool(sample.get("depth_valid", True)):
                continue
            point = sample.get("xyz_world")
            if not isinstance(point, list) or len(point) != 3:
                continue
            x, y = _project(transform.point([float(value) for value in point]), projection)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)
    x_margin = max(0.05, 0.08 * (max(xs) - min(xs) + 1e-9))
    y_margin = max(0.05, 0.08 * (max(ys) - min(ys) + 1e-9))
    return (min(xs) - x_margin, max(xs) + x_margin), (min(ys) - y_margin, max(ys) + y_margin)


def _animation_color(track: dict[str, Any], *, color_by: str, motion: float, max_motion: float) -> tuple[float, float, float]:
    if color_by == "gt_part":
        return _rgb01(_palette(int(track.get("original_part_id", track.get("part_id", 0)))))
    if color_by == "motion_magnitude":
        return _rgb01(_motion_color(motion, max_motion))
    return _rgb01(_palette(int(track.get("part_id", 0))))


def _plot_projection_panel(
    axis: Any,
    tracks: list[dict[str, Any]],
    *,
    projection: str,
    color_mode: str,
    transform: _VizTransform,
    evaluation: dict[str, Any] | None,
    flow_scale: float,
    min_motion_m: float,
) -> None:
    dominant = {}
    if isinstance(evaluation, dict):
        for item in evaluation.get("overlap", {}).get("per_cluster", []):
            if isinstance(item, dict):
                try:
                    dominant[int(item["pred_cluster_id"])] = int(item["dominant_gt_part_id"])
                except (TypeError, ValueError, KeyError):
                    pass
    motions = [_track_motion(track) for track in tracks]
    max_motion = max(motions, default=0.0)
    for track, motion in zip(tracks, motions):
        start = _sample_track_point(track, "first")
        end = _sample_track_point(track, "last")
        if start is None or end is None:
            continue
        if color_mode == "pred":
            color = _rgb01(_palette(int(track.get("part_id", 0))))
        elif color_mode == "gt":
            color = _rgb01(_palette(int(track.get("original_part_id", track.get("part_id", 0)))))
        elif color_mode == "motion":
            color = _rgb01(_motion_color(motion, max_motion))
        else:
            pred = int(track.get("part_id", 0))
            gt = int(track.get("original_part_id", pred))
            color = _rgb01((235, 60, 60) if pred in dominant and gt != dominant[pred] else (120, 190, 255))
        start_viz = transform.point(start)
        end_scaled = [start[i] + (end[i] - start[i]) * flow_scale for i in range(3)]
        end_viz = transform.point(end_scaled)
        x0, y0 = _project(start_viz, projection)
        x1, y1 = _project(end_viz, projection)
        axis.scatter([x0], [y0], s=4, color=color, alpha=0.65)
        if motion >= min_motion_m:
            axis.arrow(x0, y0, x1 - x0, y1 - y0, color=color, alpha=0.45, width=0.0005, head_width=0.01, length_includes_head=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(projection[0].upper())
    axis.set_ylabel(projection[1].upper())
    axis.grid(True, alpha=0.2)


def _parse_projections(raw: str) -> list[str]:
    allowed = {"xy", "xz", "yz"}
    if raw == "all":
        return ["xy", "xz", "yz"]
    out = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return [item for item in out if item in allowed]


def _project(point: list[float], projection: str) -> tuple[float, float]:
    axes = {"x": 0, "y": 1, "z": 2}
    return float(point[axes[projection[0]]]), float(point[axes[projection[1]]])


def _rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in color)


def _color_map_for_ids(ids: list[int]) -> dict[str, dict[str, int]]:
    return {
        str(item): {"red": color[0], "green": color[1], "blue": color[2]}
        for item, color in ((item, _palette(item)) for item in ids)
    }


def _append_axis(vertices: list[dict[str, Any]], edges: list[dict[str, Any]], pivot: list[float], axis: list[float], length: float, color: tuple[int, int, int]) -> None:
    unit = _normalize(axis)
    half = 0.5 * max(1e-6, float(length))
    start = [pivot[i] - half * unit[i] for i in range(3)]
    end = [pivot[i] + half * unit[i] for i in range(3)]
    start_index = len(vertices)
    vertices.append(_vertex(start, color))
    vertices.append(_vertex(end, color))
    edges.append({"v1": start_index, "v2": start_index + 1, "red": color[0], "green": color[1], "blue": color[2]})


def _write_vertex_ply(path: Path, vertices: list[dict[str, Any]]) -> Path:
    return _write_ply(path, vertices, [])


def _write_ply(path: Path, vertices: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if edges:
        lines.extend(
            [
                f"element edge {len(edges)}",
                "property int vertex1",
                "property int vertex2",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    lines.append("end_header")
    for vertex in vertices:
        lines.append(
            f"{vertex['x']:.9f} {vertex['y']:.9f} {vertex['z']:.9f} {int(vertex['red'])} {int(vertex['green'])} {int(vertex['blue'])}"
        )
    for edge in edges:
        lines.append(f"{edge['v1']} {edge['v2']} {edge['red']} {edge['green']} {edge['blue']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sample_track_point(track: dict[str, Any], mode: str) -> list[float] | None:
    samples = [
        sample for sample in track.get("samples", [])
        if isinstance(sample, dict)
        and bool(sample.get("visible", False))
        and bool(sample.get("depth_valid", True))
        and isinstance(sample.get("xyz_world"), list)
        and len(sample["xyz_world"]) == 3
    ]
    if not samples:
        point = track.get("reference_xyz_world")
        return [float(value) for value in point] if isinstance(point, list) and len(point) == 3 else None
    samples = sorted(samples, key=lambda item: int(item.get("frame_index", 0)))
    if mode == "last":
        sample = samples[-1]
    elif mode == "median":
        sample = samples[len(samples) // 2]
    else:
        sample = samples[0]
    return [float(value) for value in sample["xyz_world"]]


def _sample_track_point_at_frame(track: dict[str, Any], frame_index: int) -> list[float] | None:
    for sample in track.get("samples", []):
        if (
            not isinstance(sample, dict)
            or not bool(sample.get("visible", False))
            or not bool(sample.get("depth_valid", True))
            or not isinstance(sample.get("xyz_world"), list)
            or len(sample["xyz_world"]) != 3
        ):
            continue
        try:
            if int(sample.get("frame_index", 0)) != int(frame_index):
                continue
        except (TypeError, ValueError):
            continue
        return [float(value) for value in sample["xyz_world"]]
    return None


def _track_motion_until(track: dict[str, Any], frame_index: int) -> float:
    first = _sample_track_point(track, "first")
    current = _sample_track_point_at_frame(track, frame_index)
    if first is None or current is None:
        return 0.0
    return _distance(first, current)


def _track_motion(track: dict[str, Any]) -> float:
    first = _sample_track_point(track, "first")
    last = _sample_track_point(track, "last")
    if first is None or last is None:
        return 0.0
    return _distance(first, last)


def _vertex(point: list[float], color: tuple[int, int, int]) -> dict[str, Any]:
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2]), "red": color[0], "green": color[1], "blue": color[2]}


def _palette(index: int) -> tuple[int, int, int]:
    colors = [
        (96, 165, 250),
        (251, 146, 60),
        (52, 211, 153),
        (248, 113, 113),
        (196, 181, 253),
        (250, 204, 21),
        (45, 212, 191),
        (244, 114, 182),
        (163, 230, 53),
        (148, 163, 184),
    ]
    return colors[int(index) % len(colors)]


def _motion_color(value: float, max_value: float) -> tuple[int, int, int]:
    t = 0.0 if max_value <= 1e-12 else max(0.0, min(1.0, value / max_value))
    return (int(60 + 195 * t), int(180 * (1.0 - t)), int(255 * (1.0 - t)))


def _vec3(raw: Any, fallback: list[float] | None) -> list[float] | None:
    if isinstance(raw, list) and len(raw) == 3:
        return [float(value) for value in raw]
    return list(fallback) if fallback is not None else None


def _normalize(vec: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vec))
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [value / length for value in vec]


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    return load_json(candidate) if candidate.exists() else None
