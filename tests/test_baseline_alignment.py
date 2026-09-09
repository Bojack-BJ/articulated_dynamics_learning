from rgbd_urdf_mvp.benchmarks.baseline_alignment import (
    CORE_OBJECT_IDS,
    METHOD_CONTRACTS,
    build_aligned_rows,
    method_applicable,
)


def test_core_tier_balances_complexity_and_categories() -> None:
    assert len(CORE_OBJECT_IDS) == 12
    assert len(set(CORE_OBJECT_IDS)) == 12


def test_two_part_methods_reject_multi_part_objects() -> None:
    assert method_applicable("paris", 2, 1)
    assert method_applicable("ditto", 2, 1)
    assert not method_applicable("paris", 3, 2)
    assert not method_applicable("ditto", 2, 2)
    assert not method_applicable("ditto", 2, None)
    assert method_applicable("dta", 6, 5)


def test_oracle_metrics_are_explicit() -> None:
    assert "gt_part_count" in METHOD_CONTRACTS["dta"].oracle_inputs
    assert "part_semantic_initialization" in METHOD_CONTRACTS["gaussianart"].oracle_inputs
    assert "edge_f1" not in METHOD_CONTRACTS["aim"].predicted_metrics


def test_manifest_expands_protocol_and_metric_contracts() -> None:
    rows = build_aligned_rows(
        [
            {
                "object_id": "partnet_102018",
                "category": "oven",
                "gt_part_count": "2",
            },
            {
                "object_id": "partnet_103069",
                "category": "coffeemachine",
                "gt_part_count": "6",
                "gt_joint_count": "5",
            },
        ]
    )
    assert len(rows) == 2 * len(METHOD_CONTRACTS)
    paris_complex = next(
        row
        for row in rows
        if row["object_id"] == "partnet_103069" and row["method"] == "paris"
    )
    assert paris_complex["applicable"] is False
    dta = next(
        row
        for row in rows
        if row["object_id"] == "partnet_103069" and row["method"] == "dta"
    )
    assert "gt_part_count" in dta["oracle_inputs"]
    assert "core12" in dta["suite_tiers"]
    assert dta["gt_joint_count_source"] == "manifest_gt"


def test_missing_joint_count_remains_unknown() -> None:
    rows = build_aligned_rows(
        [{"object_id": "partnet_9388", "category": "door", "gt_part_count": "2"}]
    )
    paris = next(row for row in rows if row["method"] == "paris")
    assert paris["gt_joint_count"] == ""
    assert paris["gt_joint_count_source"] == "unknown"
    assert paris["applicable"] is False
