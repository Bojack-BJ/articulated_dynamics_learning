from __future__ import annotations

import unittest

import numpy as np

from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    build_kinematic_evaluation_domain,
    remap_part_labels,
)
from rgbd_urdf_mvp.perception.part_segmentation import collapse_fixed_connected_parts


def _legacy_segmentation() -> dict:
    return {
        "provider": "mujoco-body-geom-prior",
        "version": 1,
        "parts": [
            {
                "part_id": 1,
                "body_id": 1,
                "body_name": "base",
                "role": "base",
                "parent_body_id": None,
                "geom_ids": [10],
                "geom_names": ["base_geom"],
                "visible_geom_ids": [10],
                "joint_names": [],
            },
            {
                "part_id": 2,
                "body_id": 2,
                "body_name": "trim",
                "role": "fixed_child",
                "parent_body_id": 1,
                "geom_ids": [11],
                "geom_names": ["trim_geom"],
                "visible_geom_ids": [11],
                "joint_names": [],
            },
            {
                "part_id": 3,
                "body_id": 3,
                "body_name": "door",
                "role": "articulated",
                "parent_body_id": 1,
                "geom_ids": [12],
                "geom_names": ["door_geom"],
                "visible_geom_ids": [12],
                "joint_names": ["hinge"],
            },
            {
                "part_id": 4,
                "body_id": 4,
                "body_name": "handle",
                "role": "fixed_child",
                "parent_body_id": 3,
                "geom_ids": [13],
                "geom_names": ["handle_geom"],
                "visible_geom_ids": [13],
                "joint_names": [],
            },
        ],
    }


class KinematicPartDomainTest(unittest.TestCase):
    def test_fixed_descendants_collapse_into_nearest_movable_anchor(self) -> None:
        result = collapse_fixed_connected_parts(_legacy_segmentation())
        self.assertEqual(result["ontology"], "maximal-fixed-joint-connected-components")
        self.assertEqual(len(result["parts"]), 2)
        self.assertEqual(result["raw_part_to_part_id"], {"1": 1, "2": 1, "3": 2, "4": 2})
        self.assertEqual(result["parts"][0]["geom_ids"], [10, 11])
        self.assertEqual(result["parts"][1]["geom_ids"], [12, 13])

    def test_observable_domain_does_not_delete_small_part_from_all_domain(self) -> None:
        result = build_kinematic_evaluation_domain(
            _legacy_segmentation(),
            [np.asarray([1, 1, 2, 3]), np.asarray([1, 2, 3])],
            min_union_points=3,
            min_visible_frames=2,
        )
        self.assertEqual(result["all_kinematic_part_ids"], [1, 2])
        self.assertEqual(result["observable_kinematic_part_ids"], [1])
        self.assertFalse(result["selection"]["uses_method_predictions"])
        self.assertEqual(
            result["part_statistics"][1]["exclusion_reasons"],
            ["insufficient_union_points"],
        )

    def test_label_remapping_is_deterministic(self) -> None:
        labels = np.asarray([1, 2, 3, 4, 0])
        remapped = remap_part_labels(labels, {1: 1, 2: 1, 3: 2, 4: 2})
        np.testing.assert_array_equal(remapped, np.asarray([1, 1, 2, 2, 0]))


if __name__ == "__main__":
    unittest.main()
