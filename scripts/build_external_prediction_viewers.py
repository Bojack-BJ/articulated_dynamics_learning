#!/usr/bin/env python3
"""Build lightweight part/axis viewers for external prediction adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=20_000)
    args = parser.parse_args()

    built = []
    for object_dir in sorted((args.suite_root / "per_object").glob("partnet_*")):
        for method in ("dta", "artgs", "videoartgs", "paris"):
            adapter = object_dir / method / "adapter"
            prediction_path = adapter / "predictions.json"
            if not prediction_path.exists():
                continue
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            ply_path = _prediction_ply(prediction_path, prediction, method)
            if not ply_path.exists():
                continue
            points, colors = _read_ascii_ply(ply_path)
            points, colors = _downsample(points, colors, args.max_points)
            axes = _axes(prediction, method, colors)
            output = object_dir / method / "viewer.html"
            output.write_text(
                _html(object_dir.name, method, points, colors, axes),
                encoding="utf-8",
            )
            metrics_path = object_dir / method / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics.setdefault("artifacts", {})
                metrics["artifacts"].update(
                    {
                        "predictions": str(prediction_path.resolve()),
                        "labeled_point_cloud": str(ply_path.resolve()),
                        "viewer_html": str(output.resolve()),
                    }
                )
                if method == "dta":
                    metrics["kinematics"] = {
                        "support": "dual_hypothesis_visualization_only",
                        "joint_type_selection": "unsupported_by_released_non_gt_path",
                        "joint_hypotheses": prediction.get("joint_hypotheses", {}),
                    }
                elif method == "artgs":
                    metrics["kinematics"] = {
                        "support": "native_joint_predictions",
                        "joint_types": prediction.get("joint_types", []),
                        "joints": prediction.get("joints", []),
                    }
                elif method == "videoartgs":
                    metrics.setdefault("kinematics", {})
                    metrics["kinematics"]["native_joints"] = prediction.get(
                        "joints", []
                    )
                else:
                    metrics.setdefault("kinematics", {})
                    metrics["kinematics"]["native_joints"] = prediction.get(
                        "joints", []
                    )
                metrics_path.write_text(
                    json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
                )
            built.append({"object_id": object_dir.name, "method": method, "viewer": str(output)})
    print(json.dumps({"built": len(built), "viewers": built}, indent=2))
    return 0


def _prediction_ply(path: Path, prediction: dict[str, Any], method: str) -> Path:
    if method in {"dta", "videoartgs", "paris"}:
        candidate = prediction["labeled_point_cloud"]
    else:
        candidate = prediction["states"][0]["ply"]
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return candidate_path
    return path.parent / candidate_path.name


def _read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    properties = []
    count = 0
    with path.open(encoding="ascii") as stream:
        for line in stream:
            tokens = line.strip().split()
            if tokens[:2] == ["element", "vertex"]:
                count = int(tokens[2])
            elif tokens[:1] == ["property"]:
                properties.append(tokens[-1])
            elif tokens[:1] == ["end_header"]:
                break
        rows = [
            [float(value) for value in stream.readline().split()]
            for _ in range(count)
        ]
    values = np.asarray(rows, dtype=float)
    lookup = {name: index for index, name in enumerate(properties)}
    points = values[:, [lookup["x"], lookup["y"], lookup["z"]]]
    colors = values[:, [lookup["red"], lookup["green"], lookup["blue"]]].astype(int)
    return points, colors


def _downsample(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[indices], colors[indices]


def _axes(
    prediction: dict[str, Any], method: str, colors: np.ndarray
) -> list[dict[str, Any]]:
    palette = [tuple(int(v) for v in row) for row in np.unique(colors, axis=0)]
    output = []
    if method == "dta":
        hypotheses = prediction.get("joint_hypotheses", {})
        for motion_type in ("revolute", "prismatic"):
            for index, joint in enumerate(hypotheses.get(motion_type, [])):
                output.append(
                    _axis_payload(
                        joint,
                        f"part_{index + 1}:{motion_type} hypothesis",
                        palette[(index + 1) % len(palette)],
                        motion_type,
                        hypothesis=True,
                    )
                )
    else:
        for index, joint in enumerate(prediction.get("joints", [])):
            motion_type = (
                "revolute"
                if joint.get("type") in {"r", "revolute"}
                else "prismatic"
            )
            output.append(
                _axis_payload(
                    joint,
                    f"part_{index + 1}:{motion_type}",
                    palette[(index + 1) % len(palette)],
                    motion_type,
                    hypothesis=False,
                )
            )
    return output


def _axis_payload(
    joint: dict[str, Any],
    name: str,
    color: tuple[int, int, int],
    motion_type: str,
    *,
    hypothesis: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "origin": joint.get("axis_position", [0.0, 0.0, 0.0]),
        "direction": joint.get("axis_direction", [1.0, 0.0, 0.0]),
        "color": f"rgb({color[0]},{color[1]},{color[2]})",
        "motion_type": motion_type,
        "hypothesis": hypothesis,
    }


def _html(
    object_id: str,
    method: str,
    points: np.ndarray,
    colors: np.ndarray,
    axes: list[dict[str, Any]],
) -> str:
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    center = 0.5 * (bounds_min + bounds_max)
    traces: list[dict[str, Any]] = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "predicted parts",
            "x": points[:, 0].tolist(),
            "y": points[:, 1].tolist(),
            "z": points[:, 2].tolist(),
            "marker": {
                "size": 2.4,
                "opacity": 0.85,
                "color": [
                    f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors
                ],
            },
        }
    ]
    for axis in axes:
        direction = np.asarray(axis["direction"], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction /= norm
        origin = np.asarray(axis["origin"], dtype=float)
        if axis["motion_type"] == "prismatic" and np.linalg.norm(origin) < 1e-9:
            origin = center
        extent = 0.65 * diag
        endpoints = np.vstack([origin - extent * direction, origin + extent * direction])
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": axis["name"],
                "x": endpoints[:, 0].tolist(),
                "y": endpoints[:, 1].tolist(),
                "z": endpoints[:, 2].tolist(),
                "line": {
                    "color": axis["color"],
                    "width": 8 if not axis["hypothesis"] else 5,
                    "dash": (
                        "dash"
                        if axis["hypothesis"] and axis["motion_type"] == "prismatic"
                        else "solid"
                    ),
                },
            }
        )
    payload = json.dumps(traces, separators=(",", ":"))
    note = (
        "DTA displays both released revolute and prismatic hypotheses; "
        "the non-GT path does not select one."
        if method == "dta"
        else f"{method} native predicted parts and joint axes."
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{object_id} {method}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
html,body,#plot{{width:100%;height:100%;margin:0;background:#091016;color:#e8f0f3}}
#note{{position:fixed;z-index:2;left:18px;top:18px;max-width:420px;padding:12px 14px;
background:rgba(14,27,35,.9);border:1px solid #30434f;border-radius:9px;
font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body><div id="note"><b>{object_id} / {method}</b><br>{note}</div>
<div id="plot"></div><script>
const traces={payload};
Plotly.newPlot("plot",traces,{{
  paper_bgcolor:"#091016",plot_bgcolor:"#091016",font:{{color:"#e8f0f3"}},
  margin:{{l:0,r:0,t:0,b:0}},showlegend:true,
  scene:{{aspectmode:"data",xaxis:{{title:"X"}},yaxis:{{title:"Y"}},zaxis:{{title:"Z"}}}}
}},{{responsive:true,scrollZoom:true}});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
