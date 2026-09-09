#!/usr/bin/env python3
"""Summarize legacy, final VN, and analytic joint predictions on real scenes."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs/real_scene_dense_filtered_review_v1")
SCENES = ("scene29", "scene30", "scene31", "scene32", "scene34", "drawer_hand", "drawer_hand2")


def neural_edges(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [{
        "parent": edge.get("parent_slot_id"),
        "child": edge.get("child_slot_id"),
        "type": edge.get("joint_type"),
        "edge_probability": edge.get("edge_probability"),
        "axis": edge.get("axis_world"),
        "pivot": edge.get("axis_line_point_world"),
    } for edge in payload.get("selected_edges", [])]


def analytic_edges(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [{
        "parent": joint.get("parent_part_id"),
        "child": joint.get("child_part_id"),
        "type": joint.get("joint_type"),
        "edge_probability": joint.get("confidence"),
        "axis": joint.get("axis"),
        "pivot": joint.get("pivot"),
    } for joint in payload.get("joints", [])]


def main() -> None:
    rows = []
    for scene in SCENES:
        directory = ROOT / scene / "kinematics"
        methods = {
            "legacy": neural_edges(directory / "joint_inference_neural.json"),
            "vn_final": neural_edges(directory / "joint_inference_neural_vn_final.json"),
            "analytic": analytic_edges(directory / "joint_inference_analytic.json"),
        }
        for method, edges in methods.items():
            rows.append({
                "scene": scene,
                "method": method,
                "edge_count": len(edges),
                "joint_types": "+".join(edge["type"] or "unknown" for edge in edges),
                "mean_edge_confidence": (
                    sum(float(edge["edge_probability"]) for edge in edges) / len(edges)
                    if edges else None
                ),
                "edges_json": json.dumps(edges, separators=(",", ":")),
            })
    output = ROOT / "relation_head_comparison.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Real-scene relation-head comparison", "",
        "| Scene | Legacy | Final VN | Analytic |", "|---|---|---|---|",
    ]
    for scene in SCENES:
        scene_rows = {row["method"]: row for row in rows if row["scene"] == scene}
        def cell(method: str) -> str:
            row = scene_rows[method]
            return f"{row['edge_count']} edges: {row['joint_types'] or 'none'}"
        lines.append(f"| {scene} | {cell('legacy')} | {cell('vn_final')} | {cell('analytic')} |")
    lines.extend([
        "", "The table reports native predictions without manual correction. Analytic estimates are diagnostics, not GT.",
    ])
    (ROOT / "relation_head_comparison.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
