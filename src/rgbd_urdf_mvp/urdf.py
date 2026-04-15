from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

from .geometry import combine_bounds, shifted_boxes, size_from_bounds, write_obj
from .models import ArticulationArtifact, EpisodeInput, PartArtifact, PrimitiveBox, URDFPackage
from .serialization import save_json


def _format_xyz(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _box_mass(size: list[float], density_kg_m3: float = 30.0) -> float:
    return max(0.05, density_kg_m3 * size[0] * size[1] * size[2])


def _box_inertia(mass: float, size: list[float]) -> tuple[float, float, float]:
    x_len, y_len, z_len = size
    ixx = mass * (y_len * y_len + z_len * z_len) / 12.0
    iyy = mass * (x_len * x_len + z_len * z_len) / 12.0
    izz = mass * (x_len * x_len + y_len * y_len) / 12.0
    return ixx, iyy, izz


class URDFExporter:
    def export(
        self,
        episode: EpisodeInput,
        articulation: ArticulationArtifact,
        output_dir: str | Path,
    ) -> URDFPackage:
        output_path = Path(output_dir)
        mesh_dir = output_path / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        part_mesh_paths: dict[str, str] = {}
        robot = ET.Element("robot", name=episode.object_instance_id)
        collision_summary: dict[str, list[dict[str, object]]] = {}

        for part in articulation.parts:
            mesh_path = mesh_dir / f"{part.name}.obj"
            link_frame = list(part.canonical_pose.get("translation", [0.0, 0.0, 0.0]))
            local_boxes = shifted_boxes(part.primitive_boxes, link_frame)
            write_obj(mesh_path, local_boxes)
            part_mesh_paths[part.name] = str(mesh_path.resolve())
            collision_summary[part.name] = self._append_link(robot, part, local_boxes, mesh_path)

        for joint in articulation.joints:
            self._append_joint(robot, joint)

        tree = ET.ElementTree(robot)
        urdf_path = output_path / f"{episode.object_instance_id}.urdf"
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

        articulation_json_path = output_path / "articulation.json"
        save_json(
            {
                "episode_id": episode.object_instance_id,
                "category": episode.category,
                "articulation": articulation.to_dict(),
                "collision_proxies": collision_summary,
            },
            articulation_json_path,
        )
        mjcf_path = output_path / f"{episode.object_instance_id}.mjcf.xml"
        self._write_mjcf_stub(episode, articulation, mjcf_path)

        return URDFPackage(
            urdf_path=str(urdf_path.resolve()),
            articulation_json_path=str(articulation_json_path.resolve()),
            part_mesh_paths=part_mesh_paths,
            mjcf_path=str(mjcf_path.resolve()),
            metadata={"collision_proxy_count": sum(len(items) for items in collision_summary.values())},
        )

    def _append_link(
        self,
        robot: ET.Element,
        part: PartArtifact,
        local_boxes: list[PrimitiveBox],
        mesh_path: Path,
    ) -> list[dict[str, object]]:
        link = ET.SubElement(robot, "link", name=part.name)

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        visual_geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            visual_geometry,
            "mesh",
            filename=str(Path("meshes") / mesh_path.name),
            scale="1 1 1",
        )

        lower, upper = combine_bounds(local_boxes)
        bbox_size = size_from_bounds(lower, upper)
        bbox_center = [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)]
        mass = _box_mass(bbox_size)
        ixx, iyy, izz = _box_inertia(mass, bbox_size)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_format_xyz(bbox_center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{ixx:.6f}",
            ixy="0.0",
            ixz="0.0",
            iyy=f"{iyy:.6f}",
            iyz="0.0",
            izz=f"{izz:.6f}",
        )

        collision_boxes = sorted(
            local_boxes, key=lambda item: item.size[0] * item.size[1] * item.size[2], reverse=True
        )[:3]
        proxies: list[dict[str, object]] = []
        for index, box in enumerate(collision_boxes):
            collision = ET.SubElement(link, "collision", name=f"{part.name}_proxy_{index}")
            ET.SubElement(collision, "origin", xyz=_format_xyz(box.center), rpy="0 0 0")
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", size=_format_xyz(box.size))
            proxies.append({"name": box.name, "center": list(box.center), "size": list(box.size)})
        return proxies

    def _append_joint(self, robot: ET.Element, joint) -> None:
        joint_node = ET.SubElement(robot, "joint", name=joint.name, type=joint.joint_type)
        ET.SubElement(joint_node, "parent", link=joint.parent)
        ET.SubElement(joint_node, "child", link=joint.child)
        ET.SubElement(joint_node, "origin", xyz=_format_xyz(joint.origin), rpy="0 0 0")
        ET.SubElement(joint_node, "axis", xyz=_format_xyz(joint.axis))
        if joint.joint_type in {"revolute", "prismatic"}:
            ET.SubElement(
                joint_node,
                "limit",
                lower=f"{joint.limits[0]:.6f}",
                upper=f"{joint.limits[1]:.6f}",
                effort=f"{max(1.0, abs(joint.limits[1] - joint.limits[0]) * 10.0):.6f}",
                velocity=f"{max(0.1, math.pi if joint.joint_type == 'revolute' else 1.0):.6f}",
            )

    def _write_mjcf_stub(
        self,
        episode: EpisodeInput,
        articulation: ArticulationArtifact,
        mjcf_path: Path,
    ) -> None:
        root = ET.Element("mujoco", model=episode.object_instance_id)
        ET.SubElement(root, "compiler", angle="radian", coordinate="local")
        worldbody = ET.SubElement(root, "worldbody")

        part_map = {part.name: part for part in articulation.parts}
        base_part = part_map.get("base_link")
        moving_part = part_map.get("moving_link")
        joint = articulation.joints[0] if articulation.joints else None
        if base_part is None or moving_part is None or joint is None:
            ET.ElementTree(root).write(mjcf_path, encoding="utf-8", xml_declaration=True)
            return

        base_body = ET.SubElement(worldbody, "body", name=base_part.name, pos="0 0 0")
        for box in base_part.primitive_boxes[:3]:
            ET.SubElement(
                base_body,
                "geom",
                type="box",
                pos=_format_xyz(box.center),
                size=_format_xyz([dimension / 2.0 for dimension in box.size]),
                rgba="0.7 0.7 0.7 1",
            )

        moving_body = ET.SubElement(base_body, "body", name=moving_part.name, pos=_format_xyz(joint.origin))
        ET.SubElement(
            moving_body,
            "joint",
            name=joint.name,
            type="hinge" if joint.joint_type == "revolute" else "slide",
            axis=_format_xyz(joint.axis),
            range=_format_xyz(joint.limits),
        )
        link_frame = list(moving_part.canonical_pose.get("translation", [0.0, 0.0, 0.0]))
        local_boxes = shifted_boxes(moving_part.primitive_boxes, link_frame)
        for box in local_boxes[:3]:
            ET.SubElement(
                moving_body,
                "geom",
                type="box",
                pos=_format_xyz(box.center),
                size=_format_xyz([dimension / 2.0 for dimension in box.size]),
                rgba="0.8 0.3 0.3 1",
            )

        ET.ElementTree(root).write(mjcf_path, encoding="utf-8", xml_declaration=True)
