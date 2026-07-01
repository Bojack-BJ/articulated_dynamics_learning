from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json


@dataclass(slots=True)
class DynamicsOptimizationPlotConfig:
    input_path: str | Path
    output_svg: str | Path | None = None
    log_loss: bool = True
    width: int = 1280
    height: int = 760


class DynamicsOptimizationPlotter:
    def plot(self, config: DynamicsOptimizationPlotConfig) -> Path:
        input_path = Path(config.input_path).expanduser().resolve()
        artifact = load_json(input_path)
        history = [
            item
            for item in artifact.get("optimizer", {}).get("history", [])
            if isinstance(item, dict) and math.isfinite(float(item.get("loss", 0.0)))
        ]
        if not history:
            raise ValueError("dynamics_identification artifact has no optimizer.history entries.")

        output_svg = (
            Path(config.output_svg).expanduser().resolve()
            if config.output_svg is not None
            else input_path.with_name("optimization_history.svg")
        )
        output_svg.parent.mkdir(parents=True, exist_ok=True)

        width = max(720, int(config.width))
        margin_left = 86
        margin_right = 36
        margin_top = 96
        panel_gap = 62
        panel_height = 210
        loss_series = self._loss_series(history)
        param_series = self._parameter_series(history)
        gt_values = self._ground_truth_parameter_values(artifact)
        panel_count = len(loss_series) + len(param_series)
        dynamic_height = margin_top + panel_count * panel_height + max(0, panel_count - 1) * panel_gap + 72
        height = max(max(480, int(config.height)), dynamic_height)

        parts = [
            _svg_header(width, height),
            f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
            f'<text x="{margin_left}" y="34" class="title">Dynamics Optimization History</text>',
            f'<text x="{margin_left}" y="54" class="subtitle">{_escape(str(input_path))}</text>',
        ]

        rect_y = margin_top
        for name, values in loss_series.items():
            rect = _Rect(margin_left, rect_y, width - margin_left - margin_right, panel_height)
            parts.extend(
                self._plot_panel(
                    rect=rect,
                    title=_short_name(name),
                    series_name=name,
                    values=values,
                    color="#2563eb",
                    log_scale=bool(config.log_loss),
                    y_label="log value" if bool(config.log_loss) else "value",
                )
            )
            rect_y += panel_height + panel_gap

        for index, (name, values) in enumerate(param_series.items()):
            rect = _Rect(margin_left, rect_y, width - margin_left - margin_right, panel_height)
            parts.extend(
                self._plot_panel(
                    rect=rect,
                    title=_short_name(name),
                    series_name="estimate",
                    values=values,
                    color=_palette(index + 1)[index],
                    log_scale=False,
                    y_label="value",
                    ground_truth=gt_values.get(name),
                )
            )
            rect_y += panel_height + panel_gap

        parts.extend(self._summary_text(artifact, margin_left, height - 40))
        parts.append("</svg>\n")
        output_svg.write_text("\n".join(parts), encoding="utf-8")
        return output_svg

    def _loss_series(self, history: list[dict[str, Any]]) -> dict[str, list[float]]:
        total = [float(item.get("loss", 0.0)) for item in history]
        q_mse = [
            float(item.get("loss_breakdown", {}).get("q_mse", 0.0))
            for item in history
        ]
        qdot_mse = [
            float(item.get("loss_breakdown", {}).get("qdot_mse", 0.0))
            for item in history
        ]
        prior = [
            float(item.get("loss_breakdown", {}).get("prior_penalty", 0.0))
            for item in history
        ]
        series = {"total_loss": total, "q_mse": q_mse, "qdot_mse": qdot_mse}
        if any(value > 0.0 for value in prior):
            series["prior_penalty"] = prior
        return series

    def _parameter_series(self, history: list[dict[str, Any]]) -> dict[str, list[float]]:
        names: list[str] = []
        for item in history:
            values = item.get("parameter_values", {})
            if not isinstance(values, dict):
                continue
            for name in values:
                if name not in names:
                    names.append(str(name))

        series: dict[str, list[float]] = {name: [] for name in names}
        last_values = {name: 0.0 for name in names}
        for item in history:
            values = item.get("parameter_values", {})
            if isinstance(values, dict):
                for name in names:
                    if name in values:
                        last_values[name] = float(values[name])
            for name in names:
                series[name].append(last_values[name])
        return series

    def _ground_truth_parameter_values(self, artifact: dict[str, Any]) -> dict[str, float]:
        comparison = artifact.get("ground_truth_comparison", {})
        if not isinstance(comparison, dict) or not bool(comparison.get("available", False)):
            return {}
        values: dict[str, float] = {}
        for part in comparison.get("parts", []):
            if not isinstance(part, dict):
                continue
            name = str(part.get("name", "")).strip()
            if not name:
                continue
            if "ground_truth_mass_kg" in part:
                values[f"body::{name}::mass"] = float(part["ground_truth_mass_kg"])
        for joint in comparison.get("joints", []):
            if not isinstance(joint, dict):
                continue
            name = str(joint.get("name", "")).strip()
            if not name:
                continue
            if "ground_truth_damping" in joint:
                values[f"joint::{name}::damping"] = float(joint["ground_truth_damping"])
            if "ground_truth_frictionloss" in joint:
                values[f"joint::{name}::frictionloss"] = float(joint["ground_truth_frictionloss"])
        return values

    def _plot_panel(
        self,
        rect: "_Rect",
        title: str,
        series_name: str,
        values: list[float],
        color: str,
        log_scale: bool,
        y_label: str,
        ground_truth: float | None = None,
    ) -> list[str]:
        if not values:
            return []
        sample_count = len(values)
        transformed_values = [_transform_value(value, log_scale) for value in values]
        all_values = [
            value
            for value in transformed_values
            if math.isfinite(value)
        ]
        transformed_gt = None
        if ground_truth is not None:
            transformed_gt = _transform_value(float(ground_truth), log_scale)
            if math.isfinite(transformed_gt):
                all_values.append(transformed_gt)
        y_min = min(all_values) if all_values else 0.0
        y_max = max(all_values) if all_values else 1.0
        if abs(y_max - y_min) < 1e-12:
            y_min -= 1.0
            y_max += 1.0
        y_padding = 0.08 * (y_max - y_min)
        y_min -= y_padding
        y_max += y_padding

        lines = [
            f'<text x="{rect.x}" y="{rect.y - 20}" class="panel-title">{_escape(title)}</text>',
            f'<text x="{rect.x}" y="{rect.y - 5}" class="axis-label">y: {_escape(y_label)}</text>',
            f'<rect x="{rect.x}" y="{rect.y}" width="{rect.width}" height="{rect.height}" class="panel-bg"/>',
        ]
        lines.extend(_grid(rect, y_min, y_max))
        points = _polyline_points(transformed_values, rect, y_min, y_max, sample_count)
        if points:
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_y = rect.y + 18
        legend_x = rect.x + rect.width - 390
        lines.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 32}" y="{legend_y}" class="legend">'
            f'{_escape(series_name)}: {_format_number(values[-1])}</text>'
        )
        if transformed_gt is not None and math.isfinite(transformed_gt):
            y = rect.y + rect.height * (1.0 - (transformed_gt - y_min) / max(y_max - y_min, 1e-12))
            gt_color = "#0f172a"
            lines.append(
                f'<line x1="{rect.x}" y1="{y:.2f}" x2="{rect.x + rect.width}" y2="{y:.2f}" '
                f'stroke="{gt_color}" stroke-width="2" stroke-dasharray="8 6"/>'
            )
            gt_legend_y = legend_y + 20
            lines.append(
                f'<line x1="{legend_x}" y1="{gt_legend_y - 4}" x2="{legend_x + 24}" y2="{gt_legend_y - 4}" '
                f'stroke="{gt_color}" stroke-width="2" stroke-dasharray="8 6"/>'
            )
            lines.append(
                f'<text x="{legend_x + 32}" y="{gt_legend_y}" class="legend">'
                f'ground truth: {_format_number(float(ground_truth))}</text>'
            )
        lines.append(f'<text x="{rect.x + rect.width - 64}" y="{rect.y + rect.height + 26}" class="axis-label">step</text>')
        lines.append(f'<text x="{rect.x}" y="{rect.y + rect.height + 26}" class="axis-label">0</text>')
        lines.append(f'<text x="{rect.x + rect.width / 2 - 12}" y="{rect.y + rect.height + 26}" class="axis-label">{max(0, sample_count // 2)}</text>')
        lines.append(f'<text x="{rect.x + rect.width - 24}" y="{rect.y + rect.height + 26}" class="axis-label">{max(0, sample_count - 1)}</text>')
        return lines

    def _summary_text(self, artifact: dict[str, Any], x: int, y: int) -> list[str]:
        trajectory = artifact.get("fit_metrics", {}).get("trajectory_error", {})
        gt_summary = artifact.get("ground_truth_comparison", {}).get("summary", {})
        values = []
        if isinstance(trajectory, dict):
            values.append(f"q_rmse={_format_number(float(trajectory.get('q_rmse', 0.0)))}")
            values.append(f"qdot_rmse={_format_number(float(trajectory.get('qdot_rmse', 0.0)))}")
        if isinstance(gt_summary, dict) and "effective_inertia_rel_error_mean" in gt_summary:
            values.append(
                "effective_inertia_rel_error="
                f"{_format_number(float(gt_summary.get('effective_inertia_rel_error_mean', 0.0)))}"
            )
        if not values:
            return []
        return [f'<text x="{x}" y="{y}" class="subtitle">{" | ".join(_escape(value) for value in values)}</text>']


class _Rect:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def _svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 22px ui-sans-serif, system-ui, sans-serif; fill: #0f172a; }}
  .subtitle {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #64748b; }}
  .panel-title {{ font: 700 16px ui-sans-serif, system-ui, sans-serif; fill: #0f172a; }}
  .axis-label {{ font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #64748b; }}
  .tick {{ font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #64748b; }}
  .legend {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #334155; }}
  .panel-bg {{ fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 12; }}
  .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
</style>"""


def _grid(rect: _Rect, y_min: float, y_max: float) -> list[str]:
    lines: list[str] = []
    for index in range(5):
        frac = index / 4.0
        y = rect.y + rect.height * frac
        value = y_max - frac * (y_max - y_min)
        lines.append(f'<line x1="{rect.x}" y1="{y:.2f}" x2="{rect.x + rect.width}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{rect.x - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{_format_number(value)}</text>')
    return lines


def _polyline_points(values: list[float], rect: _Rect, y_min: float, y_max: float, sample_count: int) -> str:
    points: list[str] = []
    denominator = max(1, sample_count - 1)
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        x = rect.x + rect.width * index / denominator
        y = rect.y + rect.height * (1.0 - (value - y_min) / max(y_max - y_min, 1e-12))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _transform_value(value: float, log_scale: bool) -> float:
    if not log_scale:
        return float(value)
    return math.log10(max(float(value), 1e-12))


def _format_number(value: float) -> str:
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _short_name(name: str) -> str:
    return name.replace("body::", "").replace("joint::", "").replace("::", "/")


def _palette(count: int) -> list[str]:
    base = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
    ]
    return [base[index % len(base)] for index in range(max(1, count))]
