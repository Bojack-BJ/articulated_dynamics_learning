from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_post_spectral_merge_diagnostic.py"
    spec = importlib.util.spec_from_file_location("run_post_spectral_merge_diagnostic", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _track(track_id: int, part_id: int, x_offset: float) -> dict:
    return {
        "track_id": track_id,
        "part_id": part_id,
        "part_name": f"motion_part_{part_id}",
        "reference_xyz_world": [x_offset, 0.0, 0.0],
        "samples": [
            {"frame_index": 0, "xyz_world": [x_offset, 0.0, 0.0], "visible": True, "depth_valid": True},
            {"frame_index": 1, "xyz_world": [x_offset, 0.1, 0.0], "visible": True, "depth_valid": True},
            {"frame_index": 2, "xyz_world": [x_offset, 0.2, 0.0], "visible": True, "depth_valid": True},
        ],
    }


def _payload() -> dict:
    return {
        "tracks": [_track(0, 1, 0.0), _track(1, 2, 0.1), _track(2, 3, 1.5)],
        "part_track_counts": {
            "1": {"name": "motion_part_1", "count": 1},
            "2": {"name": "motion_part_2", "count": 1},
            "3": {"name": "motion_part_3", "count": 1},
        },
    }


def _revolute_arc_track(track_id: int, part_id: int, radius: float, y_offset: float = 0.0) -> dict:
    points = []
    for angle in [0.0, 0.15, 0.30, 0.45]:
        points.append([radius * __import__("math").cos(angle), y_offset + radius * __import__("math").sin(angle), 0.0])
    return {
        "track_id": track_id,
        "part_id": part_id,
        "part_name": f"motion_part_{part_id}",
        "reference_xyz_world": points[0],
        "samples": [
            {"frame_index": idx, "xyz_world": point, "visible": True, "depth_valid": True}
            for idx, point in enumerate(points)
        ],
    }


def _linear_track(track_id: int, part_id: int, x_offset: float, y_offset: float = 0.0) -> dict:
    points = [[x_offset + 0.08 * idx, y_offset, 0.0] for idx in range(4)]
    return {
        "track_id": track_id,
        "part_id": part_id,
        "part_name": f"motion_part_{part_id}",
        "reference_xyz_world": points[0],
        "samples": [
            {"frame_index": idx, "xyz_world": point, "visible": True, "depth_valid": True}
            for idx, point in enumerate(points)
        ],
    }


def test_same_revolute_like_features_produce_high_merge_score() -> None:
    module = _load_module()
    components = module.merge_score_no_gt(
        {
            "shared_hinge_score": 0.95,
            "joint_stability_score": 0.9,
            "joint_replay_delta": -0.02,
            "rigid_rmse_delta": -0.01,
            "complexity_reduction_bonus": 0.2,
            "spatial_extent_gain": 0.4,
            "parent_pollution_delta": 0.0,
            "instability_penalty": 0.0,
        }
    )
    assert components["merge_score_no_gt"] > 0.55


def test_different_motion_model_features_produce_low_merge_score() -> None:
    module = _load_module()
    components = module.merge_score_no_gt(
        {
            "shared_hinge_score": 0.0,
            "joint_stability_score": 0.2,
            "joint_replay_delta": 0.10,
            "rigid_rmse_delta": 0.08,
            "complexity_reduction_bonus": 0.1,
            "spatial_extent_gain": 0.0,
            "parent_pollution_delta": 0.08,
            "instability_penalty": 0.9,
        }
    )
    assert components["merge_score_no_gt"] < 0.25


def test_merge_clusters_does_not_modify_baseline_payload() -> None:
    module = _load_module()
    payload = _payload()
    merged = module._merge_clusters(payload, 1, 2)
    assert [track["part_id"] for track in payload["tracks"]] == [1, 2, 3]
    assert [track["part_id"] for track in merged["tracks"]] == [1, 1, 3]


def test_greedy_mode_writes_postmerge_tracks(tmp_path: Path) -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _revolute_arc_track(1, 1, 1.0),
            _revolute_arc_track(2, 2, 1.1),
            _revolute_arc_track(3, 2, 1.2),
        ]
    }
    before_joints = {
        "joints": [
            {"child_part_id": 1, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
            {"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
        ]
    }
    args = Namespace(
        merge_max_iterations=1,
        merge_score_threshold=-1.0,
        dry_run=True,
        generate_viewer=False,
    )
    postmerge = module._run_greedy(
        Path.cwd(),
        payload,
        before_joints,
        {},
        module._cluster_stats(payload),
        tmp_path,
        args,
    )
    assert postmerge is not None
    saved = json.loads(postmerge.read_text(encoding="utf-8"))
    assert saved["post_spectral_merge"]["mode"] == "greedy"
    assert len(saved["post_spectral_merge"]["accepted_merges"]) == 1


def test_moving_cluster_is_not_allowed_to_merge_into_base() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _track(0, 1, 0.0),
            _track(1, 1, 0.1),
            _track(2, 2, 1.0),
            _track(3, 2, 1.1),
        ]
    }
    before_joints = {
        "anchor_part_id": 1,
        "joints": [{"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]}],
    }
    stats = module._cluster_stats(payload)
    module._annotate_cluster_roles(stats, before_joints, {})
    guard = module._base_absorption_guard(2, stats, before_joints)
    assert guard["base_merge_allowed"] is False
    assert guard["moving_evidence_score"] > 0.0


def test_low_motion_hinge_patch_prefers_attach_over_base_merge() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _track(0, 1, 0.0),
            _revolute_arc_track(1, 2, 1.0),
            _revolute_arc_track(2, 2, 1.1),
            _revolute_arc_track(3, 3, 0.15),
        ]
    }
    before_joints = {
        "anchor_part_id": 1,
        "joints": [{"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]}],
    }
    stats = module._cluster_stats(payload)
    module._annotate_cluster_roles(stats, before_joints, {})
    attach = module._patch_attach_features(3, 2, stats, before_joints)
    guard = module._base_absorption_guard(3, stats, before_joints)
    assert attach["attach_score_no_gt"] > 0.45
    assert guard["base_merge_allowed"] is False


def test_local_prismatic_patch_gets_degeneracy_penalty_near_revolute() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _revolute_arc_track(1, 2, 1.0),
            _revolute_arc_track(2, 2, 1.1),
            _revolute_arc_track(3, 3, 0.9),
        ]
    }
    before_joints = {
        "joints": [
            {"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
            {"child_part_id": 3, "joint_type": "prismatic", "axis": [1, 0, 0]},
        ]
    }
    stats = module._cluster_stats(payload)
    module._annotate_cluster_roles(stats, before_joints, {})
    degen = module._local_prismatic_degeneracy(3, stats, before_joints)
    assert degen["local_prismatic_degeneracy_penalty"] > 0.45
    assert degen["prismatic_degeneracy_reason"]


def test_distant_prismatic_patch_is_not_attached_to_revolute_child() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _revolute_arc_track(1, 2, 1.0),
            _revolute_arc_track(2, 2, 1.1),
            _linear_track(3, 3, 2.0),
            _linear_track(4, 3, 2.2),
        ]
    }
    before_joints = {
        "joints": [
            {"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
            {"child_part_id": 3, "joint_type": "prismatic", "axis": [1, 0, 0]},
        ]
    }
    stats = module._cluster_stats(payload)
    attach = module._patch_attach_features(3, 2, stats, before_joints)
    assert attach["attach_allowed"] is False
    assert attach["attach_score_no_gt"] == 0.0


def test_child_child_score_prefers_same_revolute_over_base_absorption() -> None:
    module = _load_module()
    child = module.merge_score_no_gt(
        {
            "candidate_type": "child_child_merge",
            "shared_joint_score": 0.8,
            "joint_stability_score": 0.8,
            "joint_replay_delta": -0.01,
            "coverage_delta": 0.1,
            "complexity_reduction_bonus": 0.2,
        }
    )
    base = module.merge_score_no_gt(
        {
            "candidate_type": "child_base_merge",
            "moving_evidence_score": 0.8,
            "base_absorption_penalty": 0.9,
            "rigid_rmse_delta": 0.0,
        }
    )
    assert child["merge_score_no_gt"] > base["merge_score_no_gt"]


def test_shared_revolute_model_links_different_radius_door_clusters() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _revolute_arc_track(1, 2, 0.35),
            _revolute_arc_track(2, 2, 0.45),
            _revolute_arc_track(3, 3, 1.05),
            _revolute_arc_track(4, 3, 1.20),
        ]
    }
    before_joints = {
        "joints": [
            {"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
            {"child_part_id": 3, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
        ]
    }
    stats = module._cluster_stats(payload)
    module._annotate_cluster_roles(stats, before_joints, {})
    features = module._shared_joint_model_features(2, 3, stats, before_joints)
    assert features["best_group_motion_model"] == "revolute"
    assert features["shared_q_consistency"] > 0.85
    assert features["radius_displacement_consistency"] > 0.75
    assert features["revolute_group_score"] > features["prismatic_group_score"]


def test_large_radius_revolute_patch_gets_prismatic_degeneracy_near_hinge() -> None:
    module = _load_module()
    payload = {
        "tracks": [
            _revolute_arc_track(1, 2, 0.35),
            _revolute_arc_track(2, 2, 0.45),
            _revolute_arc_track(3, 3, 1.20),
            _revolute_arc_track(4, 3, 1.35),
            _linear_track(5, 4, 2.0, 2.0),
        ]
    }
    before_joints = {
        "joints": [
            {"child_part_id": 2, "joint_type": "revolute", "axis": [0, 0, 1], "pivot": [0, 0, 0]},
            {"child_part_id": 3, "joint_type": "prismatic", "axis": [1, 0, 0]},
        ]
    }
    stats = module._cluster_stats(payload)
    module._annotate_cluster_roles(stats, before_joints, {})
    degen = module._local_prismatic_degeneracy(3, stats, before_joints)
    attach = module._patch_attach_features(3, 2, stats, before_joints)
    assert degen["local_prismatic_degeneracy_penalty"] > 0.45
    assert degen["nearby_revolute_child"] == 2
    assert attach["best_group_motion_model"] == "revolute"
