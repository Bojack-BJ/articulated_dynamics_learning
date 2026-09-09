from __future__ import annotations

import unittest

from rgbd_urdf_mvp.benchmarks.aim_overlap import (
    AIM_OBJECTS,
    METRIC_DISCLAIMER,
    axis_distribution,
    joint_success,
    select_explicit_objects,
    sequence_plan,
    validate_object_metadata,
)


class AimOverlapBenchmarkTest(unittest.TestCase):
    def test_explicit_selection_ignores_categories(self) -> None:
        rows = [{"object_id": spec.object_id, "category": "not-allowed"} for spec in reversed(AIM_OBJECTS)]
        self.assertEqual([row["object_id"] for row in select_explicit_objects(rows)], [s.object_id for s in AIM_OBJECTS])

    def test_expected_counts(self) -> None:
        for spec in AIM_OBJECTS:
            metadata = {
                "total_part_count": spec.total_part_count,
                "movable_joints": (
                    [{"joint_type": "revolute"}] * spec.revolute_count
                    + [{"joint_type": "prismatic"}] * spec.prismatic_count
                ),
            }
            validate_object_metadata(spec, metadata)

    def test_sequence_plan_contains_every_isolated_joint_and_combined(self) -> None:
        joints = [{"name": "door"}, {"name": "drawer"}]
        plans = sequence_plan(joints)
        self.assertEqual([plan["name"] for plan in plans], ["isolated_door", "isolated_drawer", "combined_sequential"])
        self.assertEqual(plans[-1]["joint_names"], ["door", "drawer"])

    def test_joint_success_penalizes_missing_and_wrong_type(self) -> None:
        rows = [
            {"edge_detected": True, "type_correct": True, "axis_error_deg": 4.0},
            {"edge_detected": False, "type_correct": True, "axis_error_deg": 1.0},
            {"edge_detected": True, "type_correct": False, "axis_error_deg": 1.0},
        ]
        self.assertAlmostEqual(joint_success(rows, 5.0), 1.0 / 3.0)

    def test_axis_distribution_ignores_missing_values(self) -> None:
        result = axis_distribution([{"axis_error_deg": 1.0}, {"axis_error_deg": 9.0}, {"axis_error_deg": None}])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mean_deg"], 5.0)

    def test_disclaimer_forbids_direct_metric_ranking(self) -> None:
        self.assertIn("not identical", METRIC_DISCLAIMER)
        self.assertIn("must not be used for direct ranking", METRIC_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
