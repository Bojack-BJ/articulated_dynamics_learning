from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import mujoco

from rgbd_urdf_mvp.cli_parser import build_parser
from rgbd_urdf_mvp.sim.gapartnet_adapter import GAPartNetMJCFAdapter, GAPartNetMJCFConfig
from rgbd_urdf_mvp.sim.gapartnet_dataset import GAPartNetDatasetConfig, GAPartNetDatasetPreparer


class GAPartNetAdapterTests(unittest.TestCase):
    def test_cli_accepts_gapartnet_conversion(self) -> None:
        args = build_parser().parse_args(
            ["convert-gapartnet-mjcf", "asset", "--output-mjcf", "model.xml", "--density-kg-m3", "42"]
        )
        self.assertEqual(args.command, "convert-gapartnet-mjcf")
        self.assertEqual(args.density_kg_m3, 42.0)

    def test_cli_accepts_gapartnet_dataset_preparation(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare-gapartnet-recordings",
                "dataset.zip",
                "--output-dir",
                "prepared",
                "--object-id",
                "7304",
                "--object-id",
                "12042",
            ]
        )
        self.assertEqual(args.command, "prepare-gapartnet-recordings")
        self.assertEqual(args.object_ids, ["7304", "12042"])

    def test_converts_thin_mesh_urdf_to_loadable_proxy_mjcf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh_dir = root / "textured_objs"
            mesh_dir.mkdir()
            (mesh_dir / "panel.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n",
                encoding="utf-8",
            )
            (root / "mobility_annotation_gapartnet.urdf").write_text(
                """<?xml version="1.0"?>
<robot name="thin_panel">
  <link name="base"><visual><geometry><mesh filename="textured_objs/panel.obj"/></geometry></visual></link>
  <link name="door"><visual><geometry><mesh filename="textured_objs/panel.obj"/></geometry></visual></link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="door"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="1.57"/>
  </joint>
</robot>
""",
                encoding="utf-8",
            )
            output = root / "model.xml"
            GAPartNetMJCFAdapter().convert(GAPartNetMJCFConfig(root, output))

            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(model.njnt, 1)
            self.assertEqual(model.nbody, 3)
            self.assertEqual(model.ngeom, 4)
            visual_geoms = [index for index in range(model.ngeom) if model.geom_group[index] == 1]
            collision_geoms = [index for index in range(model.ngeom) if model.geom_group[index] == 3]
            self.assertEqual(len(visual_geoms), 2)
            self.assertEqual(len(collision_geoms), 2)
            self.assertTrue(all(model.geom_contype[index] == 0 for index in visual_geoms))
            manifest = json.loads(output.with_suffix(".conversion.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["geometry_mode"],
                "original-visual-mesh-with-axis-aligned-box-collision-proxy",
            )
            self.assertEqual(manifest["link_count"], 2)

    def test_prepares_selected_single_dof_asset_and_batch_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            object_dir = source / "partnet_mobility_part" / "7304"
            mesh_dir = object_dir / "textured_objs"
            mesh_dir.mkdir(parents=True)
            (mesh_dir / "panel.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 0.1\nf 1 2 3\nf 1 3 4\n",
                encoding="utf-8",
            )
            (object_dir / "meta.json").write_text('{"model_cat":"Microwave"}', encoding="utf-8")
            (object_dir / "bounding_box.json").write_text(
                '{"min":[0,0,0],"max":[1,1,0.1]}', encoding="utf-8"
            )
            (object_dir / "mobility_annotation_gapartnet.urdf").write_text(
                """<robot name="test">
<link name="base"><visual><geometry><mesh filename="textured_objs/panel.obj"/></geometry></visual></link>
<link name="door"><visual><geometry><mesh filename="textured_objs/panel.obj"/></geometry></visual></link>
<joint name="door_joint" type="revolute"><parent link="base"/><child link="door"/>
<axis xyz="0 1 0"/><limit lower="0" upper="1.5"/></joint></robot>""",
                encoding="utf-8",
            )
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for path in object_dir.rglob("*"):
                    if path.is_file():
                        output.write(path, path.relative_to(source).as_posix())

            result = GAPartNetDatasetPreparer().prepare(
                GAPartNetDatasetConfig(archive, root / "prepared", object_ids=("7304",))
            )
            self.assertEqual(result["object_count"], 1)
            self.assertTrue(Path(result["batch_manifest_path"]).exists())
            self.assertEqual(result["objects"][0]["category"], "microwave")
            mujoco.MjModel.from_xml_path(result["objects"][0]["model_path"])


if __name__ == "__main__":
    unittest.main()
