from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.perception.baseline_viewer import (
    BaselineComparisonViewerBuilder,
    BaselineViewerConfig,
    _overlap_payload,
    _read_ply,
    _reart_joint_axes,
    _reart_pose_trajectory,
    _trajectory_point_major,
    _AxisRemap,
)


def _write_ply(
    path: Path,
    points: np.ndarray,
    *,
    part_ids: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    properties = ["property float x", "property float y", "property float z"]
    if part_ids is not None:
        properties.append("property int part_id")
    if colors is not None:
        properties.extend(
            ["property uchar red", "property uchar green", "property uchar blue"]
        )
    rows = []
    for index, point in enumerate(points):
        values = [*point]
        if part_ids is not None:
            values.append(int(part_ids[index]))
        if colors is not None:
            values.extend(int(value) for value in colors[index])
        rows.append(" ".join(str(value) for value in values))
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points)}",
                *properties,
                "end_header",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_ascii_ply_reader_preserves_rgb_columns(tmp_path: Path) -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    colors = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = tmp_path / "colored.ply"
    _write_ply(path, points, colors=colors)

    loaded_points, loaded_colors, extra = _read_ply(path)

    np.testing.assert_allclose(loaded_points, points)
    np.testing.assert_array_equal(loaded_colors, colors)
    assert extra == {}


def test_overlap_payload_retains_unmatched_points() -> None:
    result = _overlap_payload(
        np.asarray([0, 0, 1, 2]),
        np.asarray([5, -1, 6, -1]),
    )

    assert result["pred_ids"] == [0, 1, 2]
    assert result["gt_ids"] == [5, 6]
    assert result["count"] == [[1, 0], [0, 1], [0, 0]]
    assert result["unmatched_count"] == 2


def test_reart_trajectory_layout_is_normalized() -> None:
    time_major = np.zeros((4, 512, 3), dtype=float)
    point_major = _trajectory_point_major(time_major)
    assert point_major.shape == (512, 4, 3)
    assert _trajectory_point_major(point_major).shape == (512, 4, 3)


def test_reart_pose_trajectory_replays_predicted_part_transforms() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    labels = np.asarray([0, 1], dtype=int)
    poses = np.repeat(np.eye(4)[None, None], 2, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    poses[0, 1, 1, 3] = 0.5
    poses[1, 0, 2, 3] = 0.25

    trajectory = _reart_pose_trajectory(
        points, labels, poses, canonical_index=1, frame_count=3
    )

    assert trajectory is not None
    np.testing.assert_allclose(trajectory[0, 0], [0, 0, 0])
    np.testing.assert_allclose(trajectory[1, 0], [1, 0.5, 0])
    np.testing.assert_allclose(trajectory[:, 1], points)
    np.testing.assert_allclose(trajectory[0, 2], [0, 0, 0.25])


def test_reart_joint_axes_converts_revolute_relative_poses() -> None:
    poses = np.repeat(np.eye(4)[None, None], 5, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    pivot = np.asarray([0.5, 0.0, 0.0])
    for frame, angle in enumerate(np.linspace(0.0, 0.6, 5)):
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        poses[frame, 1, :3, :3] = rotation
        poses[frame, 1, :3, 3] = pivot - rotation @ pivot
    result = _reart_joint_axes(
        poses,
        [[0, 1]],
        points=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float),
        predicted=np.asarray([0, 1]),
        transform=_AxisRemap.from_raw("x,y,z"),
    )

    assert len(result) == 1
    assert result[0]["joint_type"] == "revolute"
    assert result[0]["source"] == "converted_from_native_part_poses"
    np.testing.assert_allclose(np.abs(result[0]["axis"]), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(result[0]["origin"][:2], pivot[:2], atol=1e-6)


def test_reart_joint_axes_converts_prismatic_relative_poses() -> None:
    poses = np.repeat(np.eye(4)[None, None], 5, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    for frame, displacement in enumerate(np.linspace(0.0, 0.4, 5)):
        poses[frame, 1, :3, 3] = [0.0, displacement, 0.0]
    result = _reart_joint_axes(
        poses,
        [[0, 1]],
        points=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float),
        predicted=np.asarray([0, 1]),
        transform=_AxisRemap.from_raw("x,z,-y"),
    )

    assert len(result) == 1
    assert result[0]["joint_type"] == "prismatic"
    np.testing.assert_allclose(np.abs(result[0]["axis"]), [0, 0, 1], atol=1e-6)


def test_builder_reads_official_reart_pickle(tmp_path: Path) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_ply(reference, points, part_ids=np.asarray([0, 1]))
    poses = np.repeat(np.eye(4)[None, None], 1, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    result = tmp_path / "result.pkl"
    with result.open("wb") as stream:
        pickle.dump(
            {
                "cano_pc": points,
                "pred_cano_part": np.asarray([0, 1]),
                "pred_pose_list": poses,
                "cano_idx": 0,
                "complete_pc_list": np.stack([points, points]),
            },
            stream,
        )

    BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=tmp_path / "viewer.html",
            reference_ply=reference,
            reart_prediction=result,
        )
    )

    data = json.loads((tmp_path / "viewer_data" / "reart.json").read_text())
    assert len(data["trajectory"][0]) == 31
    assert data["pred_part_id"] == [0, 1]


def test_builder_exposes_failed_reart_inputs_without_fake_labels(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_ply(reference, points, part_ids=np.asarray([0, 1]))
    frames = []
    for index in range(4):
        path = tmp_path / "reart_input" / f"frame_{index:04d}.ply"
        _write_ply(path, points + [0, index * 0.1, 0])
        frames.append(
            {
                "source_frame_index": index * 10,
                "output_path": str(path),
            }
        )
    manifest = tmp_path / "reart_input" / "reart_sequence_manifest.json"
    manifest.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    status = tmp_path / "pilot_status.json"
    status.write_text(
        json.dumps({"status": "failed", "failure_stage": "official_optimization"}),
        encoding="utf-8",
    )

    BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=tmp_path / "viewer.html",
            reference_ply=reference,
            reart_sequence_manifest=manifest,
            reart_status=status,
        )
    )

    data = json.loads((tmp_path / "viewer_data" / "reart.json").read_text())
    assert data["metrics"]["status"] == "failed"
    assert data["pred_part_id"] == [-1, -1]
    assert len(data["geometry_frames"]) == 31
    assert len(data["trajectory"][0]) == 31


def test_builder_creates_self_contained_aim_viewer_deterministically(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray(
        [[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [1.1, 0, 0]], dtype=float
    )
    gt = np.asarray([0, 0, 1, 1], dtype=int)
    _write_ply(reference, points, part_ids=gt)

    aim = tmp_path / "aim"
    initial_colors = np.asarray(
        [[10, 10, 10], [10, 10, 10], [20, 20, 20], [20, 20, 20]]
    )
    final_colors = np.asarray(
        [[30, 30, 30], [30, 30, 30], [40, 40, 40], [40, 40, 40]]
    )
    _write_ply(
        aim / "seg_init_dual" / "segmented_point.ply",
        points,
        colors=initial_colors,
    )
    _write_ply(
        aim / "motion_seg_final" / "segmented_point.ply",
        points,
        colors=final_colors,
    )
    for time, offset in ((0.0, 0.0), (0.5, 0.05), (1.0, 0.1)):
        moved = points.copy()
        moved[2:, 1] += offset
        _write_ply(
            aim / f"motion_traj_t={time:.1f}" / "point_cloud_seq_0.ply",
            moved,
            colors=final_colors,
        )
    _write_ply(aim / "sub_0_point_cloud_end.ply", points[2:], colors=final_colors[2:])
    (aim / "segmentation.log").write_text("2 2\n", encoding="utf-8")

    first = tmp_path / "viewer_first.html"
    second = tmp_path / "viewer_second.html"
    builder = BaselineComparisonViewerBuilder()
    common = dict(
        reference_ply=reference,
        aim_run=aim,
        max_points=3,
        random_seed=9,
    )
    builder.build(BaselineViewerConfig(output_html=first, **common))
    first_data = json.loads((tmp_path / "viewer_data" / "aim.json").read_text())
    builder.build(BaselineViewerConfig(output_html=second, **common))
    second_data = json.loads((tmp_path / "viewer_data" / "aim.json").read_text())

    assert first_data["points"] == second_data["points"]
    assert first_data["visualized_point_count"] == 3
    assert first_data["gt_mapping_valid"] == [True, True, True]
    assert first_data["dynamic_flag"].count(1) >= 1
    assert "premerge_component" in first_data["available_fields"]
    html = first.read_text(encoding="utf-8")
    assert "__PAYLOAD__" not in html
    assert "AiM post-merge RANSAC" in html
    assert "GT-mapped only" in html


def test_builder_prefers_dense_aim_trajectory(tmp_path: Path) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_ply(reference, points, part_ids=np.asarray([0, 1]))
    aim = tmp_path / "aim"
    colors = np.asarray([[10, 10, 10], [20, 20, 20]])
    _write_ply(aim / "seg_init_dual" / "segmented_point.ply", points, colors=colors)
    _write_ply(aim / "motion_seg_final" / "segmented_point.ply", points, colors=colors)
    trajectory = np.stack(
        [points + [0, time, 0] for time in np.linspace(0, 0.4, 5)], axis=0
    )
    np.savez(
        aim / "dense_trajectory.npz",
        trajectory=trajectory,
        times=np.linspace(0, 1, 5),
    )

    BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=tmp_path / "viewer.html",
            reference_ply=reference,
            aim_run=aim,
            timeline_steps=5,
        )
    )

    data = json.loads((tmp_path / "viewer_data" / "aim.json").read_text())
    assert data["trajectory_times"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(data["trajectory"][0]) == 5


def test_builder_loads_temporal_gt_geometry_and_uses_equal_bounds(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=float)
    labels = np.asarray([0, 1], dtype=int)
    _write_ply(reference, points, part_ids=labels)
    frames = tmp_path / "frames"
    for index in range(3):
        moved = points.copy()
        moved[1, 1] = index * 0.5
        _write_ply(frames / f"frame_{index:04d}.ply", moved, part_ids=labels)

    BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=tmp_path / "viewer.html",
            reference_ply=reference,
            gt_frames_dir=frames,
            timeline_steps=5,
        )
    )

    data = json.loads((tmp_path / "viewer_data" / "gt.json").read_text())
    assert len(data["geometry_frames"]) == 5
    assert data["geometry_frames"][-1]["source_frame_index"] == 2
    html = (tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert 'aspectmode:"cube"' in html
    assert "Show GT mesh replay" in html
    assert "Show GT point-cloud overlap" in html


def test_builder_loads_hybrid_reart_and_existing_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference.ply"
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_ply(reference, points, part_ids=np.asarray([0, 1]))
    hybrid = tmp_path / "hybrid.json"
    hybrid.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "part_id": index,
                        "original_part_id": index,
                        "samples": [
                            {
                                "visible": True,
                                "frame_index": 0,
                                "xyz_world": point.tolist(),
                            }
                        ],
                    }
                    for index, point in enumerate(points)
                ]
            }
        ),
        encoding="utf-8",
    )
    reart = tmp_path / "reart.npz"
    np.savez(
        reart,
        points=np.stack([points, points + [0, 0.1, 0]], axis=0),
        labels=np.asarray([0, 1]),
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "primary": {
                    "covered_only_metrics": {
                        "one_to_one_mean_iou": 0.75,
                        "adjusted_rand_index": 0.5,
                        "rand_index": 0.8,
                    }
                },
                "predicted_part_count": 2,
                "gt_part_count": 2,
                "thresholds": [
                    {"distance_ratio_bbox": 0.02, "geometry_coverage": 0.9}
                ],
            }
        ),
        encoding="utf-8",
    )

    BaselineComparisonViewerBuilder().build(
        BaselineViewerConfig(
            output_html=tmp_path / "viewer.html",
            reference_ply=reference,
            hybrid_tracks=hybrid,
            hybrid_metrics=metrics,
            reart_prediction=reart,
            reart_metrics=metrics,
            timeline_steps=2,
        )
    )

    hybrid_data = json.loads((tmp_path / "viewer_data" / "hybrid.json").read_text())
    reart_data = json.loads((tmp_path / "viewer_data" / "reart.json").read_text())
    assert hybrid_data["metrics"]["point_iou"] == 0.75
    assert reart_data["metrics"]["geometry_coverage"] == 0.9
    assert reart_data["trajectory"][0] == [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
