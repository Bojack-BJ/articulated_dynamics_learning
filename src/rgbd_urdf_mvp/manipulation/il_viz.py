from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ILDatasetVisualizationConfig:
    dataset_path: str | Path
    output_svg: str | Path | None = None
    max_trajectories: int = 32
    title: str | None = None


class ILDatasetVisualizer:
    def visualize(self, config: ILDatasetVisualizationConfig) -> Path:
        dataset_path = Path(config.dataset_path).expanduser().resolve()
        data = np.load(dataset_path)
        obs = data["obs"]
        action = data["action"]
        response = data["free_response"]
        output_svg = (
            Path(config.output_svg).expanduser().resolve()
            if config.output_svg is not None
            else dataset_path.with_suffix(".svg")
        )
        output_svg.parent.mkdir(parents=True, exist_ok=True)

        max_count = min(max(1, int(config.max_trajectories)), int(response.shape[0]))
        indices = np.linspace(0, int(response.shape[0]) - 1, max_count, dtype=int)
        target = obs[:, 2]
        q0 = obs[:, 0]
        time = response[:, :, 0]
        q = response[:, :, 1]
        qdot = response[:, :, 2]

        width, height = 1180, 760
        plot_x, plot_y, plot_w, plot_h = 70, 80, 760, 470
        side_x, side_y, side_w, side_h = 880, 90, 230, 360
        q_min = float(min(np.min(q[indices]), np.min(target[indices]), np.min(q0[indices])))
        q_max = float(max(np.max(q[indices]), np.max(target[indices]), np.max(q0[indices])))
        if abs(q_max - q_min) < 1e-9:
            q_min -= 1.0
            q_max += 1.0
        t_min, t_max = 0.0, float(np.max(time[indices]))
        if t_max <= 0.0:
            t_max = 1.0

        def sx(t: float) -> float:
            return plot_x + (float(t) - t_min) / (t_max - t_min) * plot_w

        def sy(value: float) -> float:
            return plot_y + plot_h - (float(value) - q_min) / (q_max - q_min) * plot_h

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .muted{fill:#64748b;font-size:13px}.axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.line{fill:none;stroke-width:1.7;opacity:.7}.target{stroke:#ef4444;stroke-width:1;stroke-dasharray:4 4;opacity:.35}.bar{fill:#3b82f6;opacity:.75}</style>",
            f'<text x="60" y="38" font-size="24" font-weight="700">{html.escape(config.title or "Ballistic IL Dataset")}</text>',
            f'<text x="60" y="60" class="muted">{html.escape(str(dataset_path))}</text>',
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#cbd5e1"/>',
        ]
        for i in range(6):
            y = plot_y + i * plot_h / 5
            value = q_max - i * (q_max - q_min) / 5
            parts.append(f'<line class="grid" x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{plot_x - 10}" y="{y + 4:.1f}" text-anchor="end" class="muted">{value:.2f}</text>')
        for i in range(6):
            x = plot_x + i * plot_w / 5
            value = t_min + i * (t_max - t_min) / 5
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}"/>')
            parts.append(f'<text x="{x:.1f}" y="{plot_y + plot_h + 22}" text-anchor="middle" class="muted">{value:.2f}s</text>')
        colors = ["#2563eb", "#16a34a", "#7c3aed", "#f97316", "#0891b2", "#4f46e5"]
        for draw_i, idx in enumerate(indices):
            color = colors[draw_i % len(colors)]
            path = " ".join(
                ("M" if j == 0 else "L") + f"{sx(time[idx, j]):.1f},{sy(q[idx, j]):.1f}"
                for j in range(response.shape[1])
            )
            parts.append(f'<path class="line" d="{path}" stroke="{color}"/>')
            y_target = sy(target[idx])
            parts.append(f'<line class="target" x1="{plot_x}" y1="{y_target:.1f}" x2="{plot_x + plot_w}" y2="{y_target:.1f}"/>')
        parts.extend(
            [
                f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 52}" text-anchor="middle" class="muted">time after release</text>',
                f'<text x="22" y="{plot_y + plot_h / 2}" transform="rotate(-90 22 {plot_y + plot_h / 2})" text-anchor="middle" class="muted">joint position q</text>',
            ]
        )

        force_values = action[:, 6] if action.shape[1] >= 8 else action[:, 0]
        label = "teacher force" if action.shape[1] >= 8 else "teacher initial qvel"
        parts.append(f'<text x="{side_x}" y="{side_y - 28}" font-size="16" font-weight="700">{html.escape(label)} distribution</text>')
        parts.append(f'<rect x="{side_x}" y="{side_y}" width="{side_w}" height="{side_h}" fill="white" stroke="#cbd5e1"/>')
        hist, edges = np.histogram(force_values, bins=16)
        max_hist = max(1, int(hist.max()))
        bar_w = side_w / len(hist)
        for i, count in enumerate(hist):
            h = side_h * float(count) / max_hist
            x = side_x + i * bar_w
            y = side_y + side_h - h
            parts.append(f'<rect class="bar" x="{x + 1:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}"/>')
        parts.append(f'<text x="{side_x}" y="{side_y + side_h + 22}" class="muted">{float(force_values.min()):.2f}</text>')
        parts.append(f'<text x="{side_x + side_w}" y="{side_y + side_h + 22}" text-anchor="end" class="muted">{float(force_values.max()):.2f}</text>')

        summary = {
            "episodes": int(response.shape[0]),
            "shown": int(max_count),
            "q_final_error_mean": float(np.mean(np.abs(q[:, -1] - target))),
            "qdot_final_abs_mean": float(np.mean(np.abs(qdot[:, -1]))),
            "action_min": float(force_values.min()),
            "action_max": float(force_values.max()),
            "action_mean": float(force_values.mean()),
        }
        parts.append(f'<text x="70" y="640" font-size="15" font-weight="700">Summary</text>')
        parts.append(f'<text x="70" y="666" class="muted">{html.escape(json.dumps(summary, indent=2))}</text>')
        parts.append("</svg>")
        output_svg.write_text("\n".join(parts), encoding="utf-8")
        return output_svg

