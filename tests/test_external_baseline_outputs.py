import json
from pathlib import Path

import pytest

from rgbd_urdf_mvp.benchmarks.external_baseline_outputs import (
    adapt_gaussianart_output,
    adapt_paris_output,
    adapt_videoartgs_output,
)


def test_adapt_paris_output_preserves_native_metrics(tmp_path: Path) -> None:
    save = tmp_path / "raw" / "run" / "save"
    save.mkdir(parents=True)
    (save / "it5000_test_motion.json").write_text(
        json.dumps(
            {
                "type": "rotate",
                "gt_type": "rotate",
                "R_axis_d": [0.0, 0.0, 1.0],
            }
        )
    )
    (save / "it5000_val_metrics.json").write_text(
        json.dumps(
            {
                "nvs": {"psnr": 29.1, "ssim": 0.97},
                "motion": {"ang_err": 8.0, "pos_err": 0.1, "geo_dist": 2.0},
            }
        )
    )

    result = adapt_paris_output(tmp_path)

    assert result["status"] == "success_native_metrics"
    assert result["kinematics"]["predicted_joint_type"] == "revolute"
    assert result["kinematics"]["axis_angle_error_deg"] == pytest.approx(8.0)
    assert result["geometry"]["novel_view_psnr"] == pytest.approx(29.1)
    assert result["segmentation"] == {}


def test_adapt_gaussianart_output_marks_oracle_inputs(tmp_path: Path) -> None:
    (tmp_path / "results.txt").write_text(
        "\n".join(
            [
                "The best: 87000",
                "Parts num: 2",
                "Angle mean: 74.25",
                "Distance mean: 0.0",
                "Theta diff mean: 0.38",
            ]
        )
        + "\n"
    )

    result = adapt_gaussianart_output(tmp_path)

    assert result["status"] == "success_native_metrics"
    assert result["oracle"]["used_for_inference"] is True
    assert result["segmentation"]["predicted_part_count"] == 2
    assert result["kinematics"]["axis_angle_error_deg"] == pytest.approx(74.25)
    assert result["kinematics"]["axis_angle_error_deg_native_all"] == pytest.approx(
        74.25
    )
    assert result["kinematics"]["axis_position_error_native_x10"] == pytest.approx(0.0)
    assert "no type-correct conditioning" in result["metric_support"]["kinematics"]


def test_adapt_videoartgs_output_preserves_native_predictions(tmp_path: Path) -> None:
    output = tmp_path / "scene" / "final" / "train" / "ours_20000"
    meshes = output / "meshes"
    meshes.mkdir(parents=True)
    (output / "joint_info.json").write_text(
        json.dumps(
            [
                {"joint": "heavy", "name": "base"},
                {
                    "joint": "hinge",
                    "name": "door",
                    "jointData": {
                        "axis": {
                            "direction": [0.0, 0.0, 1.0],
                            "origin": [0.0, 0.0, 0.0],
                        }
                    },
                },
            ]
        )
    )
    (meshes / "part_0.ply").write_text("ply\n")
    (meshes / "part_1.ply").write_text("ply\n")
    (meshes / "whole_mesh.ply").write_text("ply\n")

    result = adapt_videoartgs_output(tmp_path)

    assert result["status"] == "success_native_export"
    assert result["segmentation"]["predicted_part_count"] == 2
    assert result["kinematics"]["predicted_joint_count"] == 1
    assert result["geometry"]["predicted_part_mesh_count"] == 2
    assert result["artifacts"]["selected_iteration"] == 20000
