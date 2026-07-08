from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_routing_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_em_lite_routing_diagnostic.py"
    spec = importlib.util.spec_from_file_location("run_em_lite_routing_diagnostic", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_transforms_are_monotonic() -> None:
    module = _load_routing_module()
    assert module._exp_decay_score(0.0, 0.02) > module._exp_decay_score(0.05, 0.02)
    assert module._parent_pollution_penalty(0.05) > module._parent_pollution_penalty(0.001)
    assert module._motion_compatibility_score(1.0) > module._motion_compatibility_score(-1.0)


def test_ignore_beats_merge_parent_when_parent_pollution_is_high() -> None:
    module = _load_routing_module()
    u_stats = {"track_count": 80, "bbox_diag_m": 0.5, "mean_motion_m": 0.12}
    merge_parent = module._score_components_v2(
        "merge_parent",
        u_stats,
        {"track_count": 400, "bbox_diag_m": 1.0},
        {"track_count": 480, "bbox_diag_m": 1.2},
        {
            "rigid_rmse_delta": 0.08,
            "joint_replay_delta": 0.04,
            "parent_rigid_rmse_delta": 0.08,
            "largest_cluster_ratio_after": 0.8,
        },
    )
    ignore = module._score_components_v2(
        "ignore",
        u_stats,
        None,
        None,
        {
            "rigid_rmse_delta": 0.0,
            "joint_replay_delta": 0.0,
            "parent_rigid_rmse_delta": 0.0,
            "largest_cluster_ratio_after": 0.6,
        },
    )
    assert ignore["route_score_no_gt_v2"] > merge_parent["route_score_no_gt_v2"]


def test_merge_sibling_beats_ignore_when_motion_and_deltas_are_good() -> None:
    module = _load_routing_module()
    u_stats = {
        "track_count": 90,
        "bbox_diag_m": 0.45,
        "mean_motion_m": 0.10,
        "mean_displacement": [0.1, 0.0, 0.0],
    }
    merge_sibling = module._score_components_v2(
        "merge_sibling",
        u_stats,
        {"track_count": 80, "bbox_diag_m": 0.4},
        {"track_count": 170, "bbox_diag_m": 0.7},
        {
            "motion_compatibility_raw": 0.95,
            "rigid_rmse_delta": 0.001,
            "joint_replay_delta": 0.0,
            "parent_rigid_rmse_delta": 0.0,
            "largest_cluster_ratio_after": 0.5,
        },
    )
    ignore = module._score_components_v2(
        "ignore",
        u_stats,
        None,
        None,
        {
            "motion_compatibility_raw": 0.95,
            "rigid_rmse_delta": 0.0,
            "joint_replay_delta": 0.0,
            "parent_rigid_rmse_delta": 0.0,
            "largest_cluster_ratio_after": 0.5,
        },
    )
    assert merge_sibling["route_score_no_gt_v2"] > ignore["route_score_no_gt_v2"]


def test_v3_merge_sibling_rewards_stable_effective_coverage() -> None:
    module = _load_routing_module()
    v2_components = {
        "motion_compatibility_score": 0.95,
        "rigid_delta_score": 0.95,
        "joint_replay_delta_score": 0.95,
        "coverage_gain_score": 0.8,
        "anti_degeneracy_score": 0.9,
        "joint_replay_worsening_penalty": 0.0,
        "ignore_safety_score": 0.8,
        "lost_motion_usefulness_penalty": 0.6,
        "low_motion_base_like_score": 0.0,
        "parent_pollution_penalty": 0.0,
        "moving_cluster_to_base_penalty": 0.0,
    }
    stable = module._score_components_v3("merge_sibling", v2_components, {"joint_stability_score_after": 0.95})
    unstable = module._score_components_v3("merge_sibling", v2_components, {"joint_stability_score_after": 0.05})
    assert stable["route_score_no_gt_v3"] > unstable["route_score_no_gt_v3"]
    assert stable["effective_coverage_gain_score"] > unstable["effective_coverage_gain_score"]


def test_v3_penalizes_unstable_joint_parameters() -> None:
    module = _load_routing_module()
    v2_components = {
        "motion_compatibility_score": 0.95,
        "rigid_delta_score": 0.95,
        "joint_replay_delta_score": 0.95,
        "coverage_gain_score": 0.8,
        "anti_degeneracy_score": 0.9,
        "joint_replay_worsening_penalty": 0.0,
        "ignore_safety_score": 0.8,
        "lost_motion_usefulness_penalty": 0.6,
        "low_motion_base_like_score": 0.0,
        "parent_pollution_penalty": 0.0,
        "moving_cluster_to_base_penalty": 0.0,
    }
    stable = module._score_components_v3(
        "merge_sibling",
        v2_components,
        {"joint_stability_score_after": 0.9, "axis_std_deg_after": 2.0, "pivot_std_m_after": 0.01},
    )
    unstable = module._score_components_v3(
        "merge_sibling",
        v2_components,
        {"joint_stability_score_after": 0.9, "axis_std_deg_after": 30.0, "pivot_std_m_after": 0.20},
    )
    assert stable["route_score_no_gt_v3"] > unstable["route_score_no_gt_v3"]
    assert unstable["instability_penalty"] > stable["instability_penalty"]


def test_stability_aware_diagnostic_penalizes_instability() -> None:
    module = _load_routing_module()
    stable = {
        "route_score_diagnostic": 1.0,
        "joint_stability_score": 0.9,
        "effective_coverage_gain_score": 0.5,
        "instability_penalty": 0.1,
    }
    unstable = {
        "route_score_diagnostic": 1.0,
        "joint_stability_score": 0.9,
        "effective_coverage_gain_score": 0.5,
        "instability_penalty": 0.9,
    }
    assert module._diagnostic_score_stability_aware(stable) > module._diagnostic_score_stability_aware(unstable)


def test_stability_aware_diagnostic_rewards_joint_stability() -> None:
    module = _load_routing_module()
    high_stability = {
        "route_score_diagnostic": 1.0,
        "joint_stability_score": 0.9,
        "effective_coverage_gain_score": 0.5,
        "instability_penalty": 0.1,
    }
    low_stability = {
        "route_score_diagnostic": 1.0,
        "joint_stability_score": 0.1,
        "effective_coverage_gain_score": 0.5,
        "instability_penalty": 0.1,
    }
    assert module._diagnostic_score_stability_aware(high_stability) > module._diagnostic_score_stability_aware(
        low_stability
    )


def test_severe_instability_category_triggers_on_axis_std() -> None:
    module = _load_routing_module()
    chosen = {"route_option": "ignore", "route": "ignore"}
    raw_best = {
        "route_option": "merge_sibling_4",
        "route": "merge_sibling",
        "joint_stability_score": 0.8,
        "axis_std_deg_after": 20.0,
        "pivot_std_m_after": 0.01,
        "instability_penalty": 0.2,
    }
    stable_best = raw_best
    category, _ = module._classify_v3_failure(chosen, raw_best, stable_best)
    assert category == "raw_diagnostic_route_severely_unstable"


def test_severe_instability_category_triggers_on_pivot_std() -> None:
    module = _load_routing_module()
    chosen = {"route_option": "ignore", "route": "ignore"}
    raw_best = {
        "route_option": "merge_sibling_4",
        "route": "merge_sibling",
        "joint_stability_score": 0.8,
        "axis_std_deg_after": 2.0,
        "pivot_std_m_after": 0.2,
        "instability_penalty": 0.2,
    }
    stable_best = raw_best
    category, _ = module._classify_v3_failure(chosen, raw_best, stable_best)
    assert category == "raw_diagnostic_route_severely_unstable"


def test_stable_merge_sibling_ignored_by_v3_is_true_miss() -> None:
    module = _load_routing_module()
    chosen = {"route_option": "ignore", "route": "ignore"}
    raw_best = {
        "route_option": "merge_sibling_4",
        "route": "merge_sibling",
        "joint_stability_score": 0.8,
        "axis_std_deg_after": 2.0,
        "pivot_std_m_after": 0.02,
        "instability_penalty": 0.1,
    }
    stable_best = raw_best
    category, _ = module._classify_v3_failure(chosen, raw_best, stable_best)
    assert category == "true_v3_miss"


def test_stability_topk_selects_only_high_v2_target_routes() -> None:
    module = _load_routing_module()
    rows = [
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 2, "route_score_no_gt_v2": 0.9},
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 3, "route_score_no_gt_v2": 0.7},
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 4, "route_score_no_gt_v2": 0.1},
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": None, "route_score_no_gt_v2": 1.0},
    ]
    assert module._select_rows_for_stability(rows, "topk", 2, 0.15) == {0, 1}


def test_stability_borderline_selects_routes_within_margin() -> None:
    module = _load_routing_module()
    rows = [
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 2, "route_score_no_gt_v2": 0.9},
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 3, "route_score_no_gt_v2": 0.82},
        {"object_id": "obj", "unmatched_cluster_id": 1, "target_part_id": 4, "route_score_no_gt_v2": 0.7},
    ]
    assert module._select_rows_for_stability(rows, "borderline", 2, 0.1) == {0, 1}


def test_skipped_stability_applies_neutral_v3_fields() -> None:
    module = _load_routing_module()
    row = {
        "route": "merge_sibling",
        "route_score_no_gt_v2": 0.5,
        "motion_compatibility_score": 0.5,
        "rigid_delta_score": 0.5,
        "joint_replay_delta_score": 0.5,
        "coverage_gain_score": 0.5,
        "anti_degeneracy_score": 0.5,
        "joint_replay_worsening_penalty": 0.0,
        "ignore_safety_score": 0.5,
        "lost_motion_usefulness_penalty": 0.5,
        "low_motion_base_like_score": 0.5,
        "parent_pollution_penalty": 0.0,
        "moving_cluster_to_base_penalty": 0.0,
        "route_score_diagnostic": 0.0,
    }
    module._apply_route_stability_fields(row, None, module._neutral_skipped_stability("test skip"))
    assert row["stability_available"] is False
    assert row["stability_skipped_reason"] == "test skip"
    assert row["joint_stability_score"] == 0.5
