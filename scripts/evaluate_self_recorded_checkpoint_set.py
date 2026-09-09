#!/usr/bin/env python3
"""Evaluate compact self-recorded predictions against local manual annotations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs/balanced_real_generalization_v1/self_recorded_eval_compact"
GT = {
    "scene29": ROOT / "data/track2art_real_20260814/scenes/scene29_case/annotations/real_scene_gt_annotations.json",
    "scene30": ROOT / "data/track2art_real_20260814/scenes/scene30_small_box/annotations/real_scene_gt_annotations.json",
    "scene31": ROOT / "data/track2art_real_20260814/scenes/scene31_small_fridge/annotations/real_scene_gt_annotations.json",
    "scene32": ROOT / "data/track2art_real_20260814/scenes/scene32/annotations/real_scene_gt_annotations.json",
    "scene34": ROOT / "data/track2art_real_20260814/scenes/scene34/annotations/real_scene_gt_annotations.json",
    "drawer_hand": ROOT / "data/track2art_real_new_20260830/extracted/drawer_hand/annotations/real_scene_gt_annotations.json",
    "drawer_hand2": ROOT / "data/track2art_real_new_20260830/extracted/drawer_hand2/annotations/real_scene_gt_annotations.json",
}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    rows = []
    for checkpoint in sorted(path for path in PREDICTIONS.iterdir() if path.is_dir()):
        for scene, annotation in GT.items():
            scene_dir = checkpoint / scene
            output = scene_dir / "metrics.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_real_scene_annotations.py"),
                    str(scene_dir / "slots_compact.json"),
                    str(scene_dir / "joints.json"),
                    str(annotation),
                    "--output-json",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            payload = json.loads(output.read_text())
            seg = payload["segmentation"]
            kin = payload["kinematics"]
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "scene": scene,
                    "slot_ari": seg["adjusted_rand_index"],
                    "slot_iou": seg["one_to_one_mean_iou"],
                    "purity": seg["weighted_cluster_purity"],
                    "part_count_exact": seg["part_count_exact"],
                    "joint_coverage": kin["directed_joint_coverage"],
                    "type_accuracy": kin["joint_type_accuracy"],
                    "axis_error_deg": kin["type_correct_axis_error_deg_mean"],
                }
            )

    summary = []
    for checkpoint in sorted({row["checkpoint"] for row in rows}):
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        summary.append(
            {
                "checkpoint": checkpoint,
                "scene_count": len(selected),
                "mean_slot_ari": mean([row["slot_ari"] for row in selected]),
                "mean_slot_iou": mean([row["slot_iou"] for row in selected]),
                "mean_purity": mean([row["purity"] for row in selected]),
                "part_count_exact_rate": mean([float(row["part_count_exact"]) for row in selected]),
                "mean_joint_coverage": mean([row["joint_coverage"] for row in selected]),
                "mean_type_accuracy": mean([row["type_accuracy"] for row in selected]),
                "mean_axis_error_deg": mean(
                    [row["axis_error_deg"] for row in selected if row["axis_error_deg"] is not None]
                ),
                "axis_scene_count": sum(row["axis_error_deg"] is not None for row in selected),
            }
        )

    output = PREDICTIONS / "summary.json"
    output.write_text(json.dumps({"summary": summary, "per_scene": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
