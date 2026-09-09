from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_aim_multiframe_union,
    evaluate_aim_pointcloud_iou,
    evaluate_track_pointcloud_iou,
    read_original_part_ids,
)


def _write_ply(path: Path, rows: list[tuple[float, ...]], properties: list[str]) -> None:
    types = {"x": "float", "y": "float", "z": "float", "part_id": "ushort",
             "red": "uchar", "green": "uchar", "blue": "uchar"}
    lines = ["ply", "format ascii 1.0", f"element vertex {len(rows)}"]
    lines.extend(f"property {types[name]} {name}" for name in properties)
    lines.append("end_header")
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_binary_rgb_ply(
    path: Path, rows: list[tuple[float, float, float, int, int, int]]
) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(rows)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = b"".join(struct.pack("<fffBBB", *row) for row in rows)
    path.write_bytes(header + body)


class AimPointcloudIouTest(unittest.TestCase):
    def test_permuted_palette_labels_match_gt_parts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            predicted = root / "predicted.ply"
            _write_ply(reference, [
                (0, 0, 0, 4), (0, 1, 0, 4), (1, 0, 0, 9), (1, 1, 0, 9),
            ], ["x", "y", "z", "part_id"])
            _write_ply(predicted, [
                (0, 0, 0, 200, 10, 10), (0, 1, 0, 200, 10, 10),
                (1, 0, 0, 10, 200, 10), (1, 1, 0, 10, 200, 10),
            ], ["x", "y", "z", "red", "green", "blue"])
            result = evaluate_aim_pointcloud_iou(
                predicted, reference, distance_ratios=(0.01,), primary_distance_ratio=0.01
            )
            self.assertEqual(result["primary"]["geometry_coverage"], 1.0)
            self.assertEqual(
                result["primary"]["coverage_aware_metrics"]["one_to_one_mean_iou"], 1.0
            )
            self.assertEqual(result["primary"]["covered_only_metrics"]["rand_index"], 1.0)

    def test_reference_labels_are_collapsed_before_metric_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            predicted = root / "predicted.ply"
            _write_ply(
                reference,
                [(0, 0, 0, 1), (0, 1, 0, 2), (1, 0, 0, 3), (1, 1, 0, 4)],
                ["x", "y", "z", "part_id"],
            )
            _write_ply(
                predicted,
                [
                    (0, 0, 0, 200, 10, 10),
                    (0, 1, 0, 200, 10, 10),
                    (1, 0, 0, 10, 200, 10),
                    (1, 1, 0, 10, 200, 10),
                ],
                ["x", "y", "z", "red", "green", "blue"],
            )
            result = evaluate_aim_pointcloud_iou(
                predicted,
                reference,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
                reference_part_id_map={1: 1, 2: 1, 3: 2, 4: 2},
                reference_part_ids=(1, 2),
            )
            self.assertEqual(result["gt_part_count"], 2)
            self.assertEqual(
                result["primary"]["coverage_aware_metrics"]["one_to_one_mean_iou"],
                1.0,
            )

    def test_all_domain_counts_unobserved_kinematic_part_as_zero_iou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            predicted = root / "predicted.ply"
            _write_ply(
                reference,
                [(0, 0, 0, 1), (1, 0, 0, 1)],
                ["x", "y", "z", "part_id"],
            )
            _write_ply(
                predicted,
                [(0, 0, 0, 200, 10, 10), (1, 0, 0, 200, 10, 10)],
                ["x", "y", "z", "red", "green", "blue"],
            )
            result = evaluate_aim_pointcloud_iou(
                predicted,
                reference,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
                reference_part_ids=(1, 2),
            )
            self.assertEqual(result["gt_part_count"], 2)
            self.assertEqual(result["reference_domain_missing_part_ids"], [2])
            self.assertEqual(
                result["primary"]["coverage_aware_metrics"]["one_to_one_mean_iou"],
                0.5,
            )

    def test_reads_binary_little_endian_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            predicted = root / "predicted.ply"
            _write_ply(
                reference,
                [(0, 0, 0, 4), (1, 0, 0, 9)],
                ["x", "y", "z", "part_id"],
            )
            _write_binary_rgb_ply(
                predicted,
                [(0, 0, 0, 200, 10, 10), (1, 0, 0, 10, 200, 10)],
            )
            result = evaluate_aim_pointcloud_iou(
                predicted,
                reference,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
            )
            self.assertEqual(result["predicted_part_count"], 2)
            self.assertEqual(result["primary"]["geometry_coverage"], 1.0)

    def test_uncovered_reference_points_reduce_coverage_aware_iou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            predicted = root / "predicted.ply"
            _write_ply(reference, [
                (0, 0, 0, 1), (0, 1, 0, 1), (10, 0, 0, 2), (10, 1, 0, 2),
            ], ["x", "y", "z", "part_id"])
            _write_ply(predicted, [
                (0, 0, 0, 10, 20, 30), (0, 1, 0, 10, 20, 30),
            ], ["x", "y", "z", "red", "green", "blue"])
            result = evaluate_aim_pointcloud_iou(
                predicted, reference, distance_ratios=(0.02,), primary_distance_ratio=0.02
            )
            primary = result["primary"]
            self.assertEqual(primary["geometry_coverage"], 0.5)
            self.assertEqual(primary["coverage_aware_metrics"]["one_to_one_mean_iou"], 0.5)
            self.assertEqual(primary["covered_only_metrics"]["one_to_one_mean_iou"], 1.0)

    def test_multiframe_union_recovers_parts_visible_at_different_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_pairs = []
            for frame_index, (offset, gt_part, color) in enumerate(
                ((0.0, 4, (200, 10, 10)), (5.0, 9, (10, 200, 10)))
            ):
                reference = root / f"reference_{frame_index}.ply"
                predicted = root / f"predicted_{frame_index}.ply"
                _write_ply(
                    reference,
                    [(offset, 0, 0, gt_part), (offset, 1, 0, gt_part)],
                    ["x", "y", "z", "part_id"],
                )
                _write_ply(
                    predicted,
                    [
                        (offset, 0, 0, *color),
                        (offset, 1, 0, *color),
                    ],
                    ["x", "y", "z", "red", "green", "blue"],
                )
                frame_pairs.append((predicted, reference))
            result = evaluate_aim_multiframe_union(
                frame_pairs,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
            )
            self.assertEqual(result["frame_count"], 2)
            self.assertEqual(result["gt_part_count"], 2)
            self.assertEqual(result["predicted_part_count"], 2)
            self.assertEqual(
                result["primary"]["covered_only_metrics"]["one_to_one_mean_iou"],
                1.0,
            )

    def test_multiframe_union_maps_filtered_component_labels_spatially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            trajectory = root / "trajectory.ply"
            components = root / "components.ply"
            _write_ply(
                reference,
                [(0, 0, 0, 4), (1, 0, 0, 9)],
                ["x", "y", "z", "part_id"],
            )
            _write_ply(
                trajectory,
                [
                    (0, 0, 0, 1, 1, 1),
                    (0.5, 0, 0, 1, 1, 1),
                    (1, 0, 0, 1, 1, 1),
                ],
                ["x", "y", "z", "red", "green", "blue"],
            )
            _write_ply(
                components,
                [
                    (0, 0, 0, 200, 10, 10),
                    (1, 0, 0, 10, 200, 10),
                ],
                ["x", "y", "z", "red", "green", "blue"],
            )
            result = evaluate_aim_multiframe_union(
                [(trajectory, reference)],
                component_label_ply=components,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
            )
            transfer = result["primary"]["per_frame"][0]["component_label_transfer"]
            self.assertEqual(transfer["mode"], "source-time-index-map-propagated")
            self.assertEqual(transfer["labeled_trajectory_point_count"], 2)
            self.assertEqual(result["predicted_part_count"], 2)
            self.assertEqual(
                result["primary"]["covered_only_metrics"]["one_to_one_mean_iou"],
                1.0,
            )

    def test_filtered_component_index_map_follows_moving_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            components = root / "components.ply"
            _write_ply(
                components,
                [(0, 0, 0, 200, 10, 10), (1, 0, 0, 10, 200, 10)],
                ["x", "y", "z", "red", "green", "blue"],
            )
            frame_pairs = []
            for frame_index, offset in enumerate((0.0, 10.0)):
                trajectory = root / f"trajectory_{frame_index}.ply"
                reference = root / f"reference_{frame_index}.ply"
                _write_ply(
                    trajectory,
                    [
                        (offset, 0, 0, 1, 1, 1),
                        (50, 0, 0, 1, 1, 1),
                        (1 + offset, 0, 0, 1, 1, 1),
                    ],
                    ["x", "y", "z", "red", "green", "blue"],
                )
                _write_ply(
                    reference,
                    [(offset, 0, 0, 4), (1 + offset, 0, 0, 9)],
                    ["x", "y", "z", "part_id"],
                )
                frame_pairs.append((trajectory, reference))
            result = evaluate_aim_multiframe_union(
                frame_pairs,
                component_label_ply=components,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
            )
            self.assertEqual(result["frame_count"], 2)
            self.assertEqual(
                result["primary"]["covered_only_metrics"]["one_to_one_mean_iou"],
                1.0,
            )

    def test_track_json_uses_requested_source_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.ply"
            tracks = root / "tracks.json"
            _write_ply(reference, [
                (0, 0, 0, 4), (0, 1, 0, 4), (1, 0, 0, 9), (1, 1, 0, 9),
                (10, 10, 10, 15),
            ], ["x", "y", "z", "part_id"])
            payload = {"tracks": []}
            for track_id, (xyz, part_id) in enumerate([
                ((0, 0, 0), 12), ((0, 1, 0), 12), ((1, 0, 0), 7), ((1, 1, 0), 7),
            ]):
                payload["tracks"].append({
                    "track_id": track_id,
                    "part_id": part_id,
                    "original_part_id": 4 if part_id == 12 else 9,
                    "samples": [
                        {
                            "source_frame_index": 4,
                            "visible": True,
                            "xyz_world": list(xyz),
                        },
                        {
                            "source_frame_index": 8,
                            "visible": True,
                            "xyz_world": [value + 20 for value in xyz],
                        },
                    ],
                })
            tracks.write_text(json.dumps(payload), encoding="utf-8")
            result = evaluate_track_pointcloud_iou(
                tracks,
                reference,
                source_frame_index=4,
                distance_ratios=(0.01,),
                primary_distance_ratio=0.01,
                reference_part_ids=(4, 9),
            )
            self.assertEqual(result["source_frame_index"], 4)
            self.assertEqual(result["selected_reference_part_ids"], [4, 9])
            self.assertEqual(result["primary"]["geometry_coverage"], 1.0)
            self.assertEqual(
                result["primary"]["coverage_aware_metrics"]["one_to_one_mean_iou"], 1.0
            )

    def test_reads_part_mask_seed_ids_as_simulation_gt_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracks = Path(directory) / "tracks.json"
            tracks.write_text(json.dumps({
                "object_mask_tracking_mode": False,
                "part_segmentation": {"provider": "mujoco-body-geom-prior"},
                "tracks": [{"part_id": 7}, {"part_id": 3}, {"part_id": 7}],
            }), encoding="utf-8")
            self.assertEqual(read_original_part_ids(tracks), [3, 7])

    def test_does_not_treat_object_mask_cluster_ids_as_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracks = Path(directory) / "tracks.json"
            tracks.write_text(json.dumps({
                "object_mask_tracking_mode": True,
                "tracks": [{"part_id": 7}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "simulation GT part labels"):
                read_original_part_ids(tracks)


if __name__ == "__main__":
    unittest.main()
