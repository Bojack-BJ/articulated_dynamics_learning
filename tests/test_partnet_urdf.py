from pathlib import Path

from rgbd_urdf_mvp.benchmarks.partnet_urdf import (
    load_partnet_joint_metadata,
    resolve_partnet_urdf,
)


def test_load_partnet_joint_metadata_excludes_fixed_joints(tmp_path: Path) -> None:
    urdf = tmp_path / "mobility.urdf"
    urdf.write_text(
        """
<robot name="sample">
  <joint name="fixed" type="fixed"/>
  <joint name="door" type="revolute"/>
  <joint name="drawer" type="prismatic"/>
  <joint name="wheel" type="continuous"/>
</robot>
""".strip(),
        encoding="utf-8",
    )
    metadata = load_partnet_joint_metadata(urdf)
    assert metadata.joint_count == 3
    assert metadata.joint_names == ("door", "drawer", "wheel")
    assert metadata.joint_types == ("revolute", "prismatic", "continuous")


def test_resolve_partnet_urdf_accepts_prefixed_object_id(tmp_path: Path) -> None:
    urdf = tmp_path / "9388" / "mobility.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot/>", encoding="utf-8")
    assert resolve_partnet_urdf(tmp_path, "partnet_9388") == urdf
