from scripts.build_external_baseline_suite import (
    _common_success_summary,
    _complexity_summary,
    _gt_domain_issues,
    _mean_valid_gt,
)
from rgbd_urdf_mvp.benchmarks.baseline_alignment import CORE16_OBJECT_IDS


def _result(method: str, object_id: str, count: int, status: str = "success"):
    return {
        "method": method,
        "object_id": object_id,
        "category": None,
        "status": status,
        "applicable": True,
        "_manifest_gt_part_count": count,
        # Deliberately disagree with the manifest count to verify that
        # complexity grouping never depends on a partial observed GT domain.
        "segmentation": {
            "gt_part_count": 2,
            "point_iou": 0.5,
            "ari": 0.4,
            "ri": 0.8,
            "undersegmented": False,
        },
    }


def test_core16_has_frozen_complexity_distribution():
    counts = {
        "2": sum(
            object_id
            in {
                "partnet_102018",
                "partnet_10849",
                "partnet_10944",
                "partnet_46556",
                "partnet_9388",
            }
            for object_id in CORE16_OBJECT_IDS
        ),
        "3-4": 5,
        ">=5": 6,
    }
    assert len(CORE16_OBJECT_IDS) == 16
    assert counts == {"2": 5, "3-4": 5, ">=5": 6}


def test_complexity_summary_uses_manifest_count():
    rows = _complexity_summary(
        [_result("ours_hybrid", "partnet_complex", 5)]
    )
    ours = {
        row["complexity_bucket"]: row
        for row in rows
        if row["method"] == "ours_hybrid"
    }
    assert ours["2"]["success_n"] == 0
    assert ours[">=5"]["success_n"] == 1


def test_common_success_keeps_failures_out_of_conditional_intersection():
    rows = []
    for method in ("ours_hybrid", "aim_aligned", "reart"):
        rows.append(_result(method, "shared", 2))
        rows.append(
            _result(
                method,
                "failed_for_reart",
                3,
                "failed" if method == "reart" else "success",
            )
        )
    summary = _common_success_summary(rows)
    non_oracle = [
        row
        for row in summary
        if row["comparison"] == "non_oracle_part_discovery"
    ]
    assert {row["common_success_n"] for row in non_oracle} == {1}
    assert {row["object_ids"] for row in non_oracle} == {"shared"}


def test_invalid_partial_gt_domain_is_excluded_from_metric_mean():
    valid = _result("aim_aligned", "valid", 3)
    invalid = _result("aim_aligned", "partial", 3)
    valid["segmentation"]["point_iou"] = 0.8
    invalid["segmentation"]["point_iou"] = 0.1
    invalid["segmentation"]["gt_domain_valid"] = False
    invalid["segmentation"]["observed_gt_part_count"] = 2
    assert _mean_valid_gt([valid, invalid], "point_iou") == 0.8
    assert _gt_domain_issues([invalid]) == [
        {
            "object_id": "partial",
            "category": None,
            "method": "aim_aligned",
            "manifest_gt_part_count": 3,
            "observed_gt_part_count": 2,
            "missing_gt_part_count": 1,
            "required_action": "rerun_multiframe_union_gt_evaluation",
        }
    ]
