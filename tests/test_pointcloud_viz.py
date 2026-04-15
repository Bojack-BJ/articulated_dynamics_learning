from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.perception.pointcloud_viz import PointCloudViewerBuilder, PointCloudVisualizationConfig


class PointCloudVisualizationTests(unittest.TestCase):
    def test_build_viewer_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ply_path = root / "pointcloud_4d.ply"
            ply_path.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 4",
                        "property float x",
                        "property float y",
                        "property float z",
                        "property float time",
                        "property int frame_index",
                        "property ushort part_id",
                        "end_header",
                        "0 0 0 0.0 0 1",
                        "1 0 0 0.0 0 1",
                        "0 1 0 0.1 1 2",
                        "0 0 1 0.1 1 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "fusion_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pointcloud_4d_path": str(ply_path),
                        "frame_count": 2,
                        "total_point_count": 4,
                        "bounds": {"lower": [0, 0, 0], "upper": [1, 1, 1]},
                        "pose_sources_used": ["recorded-per-view"],
                        "part_ids_present": [1, 2],
                        "part_point_counts": {
                            "1": {"name": "base", "count": 2},
                            "2": {"name": "door", "count": 2},
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            html_path = PointCloudViewerBuilder(
                PointCloudVisualizationConfig(input_path=manifest_path, max_points_per_frame=2)
            ).build()

            html = html_path.read_text(encoding="utf-8")
            self.assertIn("4D Point Cloud Viewer", html)
            self.assertIn("recorded-per-view", html)
            self.assertIn("\"frame_count\":2", html)
            self.assertIn("canvas", html)
            self.assertIn("Color: Part", html)
            self.assertIn("base", html)
            self.assertIn("door", html)

    def test_build_viewer_includes_joint_overlays_from_sibling_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ply_path = root / "pointcloud_4d.ply"
            ply_path.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 4",
                        "property float x",
                        "property float y",
                        "property float z",
                        "property float time",
                        "property int frame_index",
                        "property ushort part_id",
                        "end_header",
                        "0 0 0 0.0 0 1",
                        "1 0 0 0.0 0 1",
                        "0.5 0 0 0.1 1 2",
                        "0.5 1 0 0.1 1 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = root / "fusion_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pointcloud_4d_path": str(ply_path),
                        "frame_count": 2,
                        "total_point_count": 4,
                        "bounds": {"lower": [0, 0, 0], "upper": [1, 1, 1]},
                        "pose_sources_used": ["recorded-per-view"],
                        "part_ids_present": [1, 2],
                        "part_point_counts": {
                            "1": {"name": "base", "count": 2},
                            "2": {"name": "door", "count": 2},
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "part_poses.json").write_text(
                json.dumps(
                    {
                        "parts": [
                            {
                                "part_id": 1,
                                "name": "base",
                                "samples": [
                                    {
                                        "frame_index": 0,
                                        "valid": True,
                                        "translation": [0.0, 0.0, 0.0],
                                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    },
                                    {
                                        "frame_index": 1,
                                        "valid": True,
                                        "translation": [0.0, 0.0, 0.0],
                                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    },
                                ],
                            }
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "joint_inference.json").write_text(
                json.dumps(
                    {
                        "joints": [
                            {
                                "name": "door_joint",
                                "parent_part_id": 1,
                                "parent_name": "base",
                                "child_part_id": 2,
                                "child_name": "door",
                                "joint_type": "revolute",
                                "axis": [0.0, 0.0, 1.0],
                                "pivot": [0.5, 0.0, 0.0],
                                "q_samples": [
                                    {"frame_index": 0, "q": 0.0},
                                    {"frame_index": 1, "q": 0.3},
                                ],
                            }
                        ]
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            html_path = PointCloudViewerBuilder(
                PointCloudVisualizationConfig(input_path=manifest_path, max_points_per_frame=4)
            ).build()
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Show Joints", html)
            self.assertIn("door_joint", html)
            self.assertIn("\"has_joint_overlays\":true", html)
            self.assertIn("joint_overlay_count", html)


if __name__ == "__main__":
    unittest.main()
