from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.sim.usda_to_obj import MeshData, write_obj


class USDAtoOBJTests(unittest.TestCase):
    def test_write_obj_allows_declared_but_empty_uvs(self) -> None:
        mesh = MeshData(
            name="NoUVMesh",
            points=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            face_vertex_counts=[3],
            face_vertex_indices=[0, 1, 2],
            st=[],
            st_indices=None,
            st_interpolation="faceVarying",
            xform_translate=None,
            xform_scale=None,
            xform_orient=None,
            xform_rotate_xyz_deg=None,
            xform_rotate_zyx_deg=None,
            xform_order=None,
            material_binding=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "mesh.obj"

            write_obj(mesh, out_path)

            text = out_path.read_text(encoding="utf-8")
            self.assertIn("f 1 2 3", text)
            self.assertNotIn("vt ", text)


if __name__ == "__main__":
    unittest.main()
