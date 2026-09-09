from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_aligned_external_baselines.py"
SPEC = importlib.util.spec_from_file_location("aligned_external", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rand_index_from_contingency_matches_perfect_partition() -> None:
    assert MODULE.rand_index_from_contingency([[3, 0], [0, 2]]) == 1.0


def test_coverage_at_ratio_selects_requested_threshold() -> None:
    payload = {
        "thresholds": [
            {"distance_ratio_bbox": 0.02, "geometry_coverage": 0.6},
            {"distance_ratio_bbox": 1.0, "geometry_coverage": 1.0},
        ]
    }

    assert MODULE._coverage_at_ratio(payload, 0.02) == 0.6


def test_report_rows_keep_failures_and_sort_by_complexity() -> None:
    objects = [
        {"object_id": "partnet_complex", "category": "cabinet"},
        {"object_id": "partnet_simple", "category": "door"},
    ]
    ours = {
        "partnet_complex": _metrics(5, 4),
        "partnet_simple": _metrics(2, 2),
    }
    aim = {"partnet_simple": _metrics(2, 2)}
    reart = {"partnet_simple": _metrics(2, 3)}
    reart_failures = {"partnet_complex": {"failure_stage": "projection"}}

    manifest, rows, failures = MODULE.build_report_rows(
        objects, ours, aim, reart, reart_failures
    )

    assert [row["object_id"] for row in rows] == ["partnet_simple", "partnet_complex"]
    complex_manifest = next(row for row in manifest if row["object_id"] == "partnet_complex")
    assert complex_manifest["available_reart_native"] is True
    complex_row = next(row for row in rows if row["object_id"] == "partnet_complex")
    assert complex_row["reart_status"] == "failed"
    assert complex_row["aim_status"] == "missing"
    assert {(row["method"], row["status"]) for row in failures} >= {
        ("ReArt", "failed"),
        ("AiM", "missing"),
    }


def test_observed_gt_count_does_not_invalidate_protocol_specific_result() -> None:
    objects = [{"object_id": "partnet_a", "category": "cabinet"}]
    ours = {"partnet_a": _metrics(4, 4)}
    reart = {"partnet_a": _metrics(3, 3)}

    _manifest, rows, failures = MODULE.build_report_rows(
        objects, ours, {}, reart, {}
    )

    assert rows[0]["reart_status"] == "success"
    assert rows[0]["reart_gt_domain_valid"] is True
    assert rows[0]["reart_observed_gt_parts"] == 3
    assert rows[0]["reart_part_count_error"] == -1
    assert not any(row["method"] == "ReArt" for row in failures)


def _metrics(gt_parts: int, predicted_parts: int) -> dict[str, object]:
    return {
        "gt_part_count": gt_parts,
        "predicted_part_count": predicted_parts,
        "point_iou": 0.5,
        "ari": 0.4,
        "ri": 0.8,
        "unmatched_gt_part_count": max(0, gt_parts - predicted_parts),
        "undersegmented": predicted_parts < gt_parts,
        "largest_cluster_ratio": 0.6,
        "geometry_coverage": None,
    }
