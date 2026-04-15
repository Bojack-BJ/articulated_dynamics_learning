from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

from ..core.geometry import combine_bounds, shifted_boxes, size_from_bounds, write_obj
from ..core.models import ArticulationArtifact, EpisodeInput, PartArtifact, PrimitiveBox, URDFPackage
from ..core.serialization import save_json


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
        if not part_map:
            ET.ElementTree(root).write(mjcf_path, encoding="utf-8", xml_declaration=True)
            return

        children_by_parent: dict[str, list] = {}
        child_names = set()
        for joint in articulation.joints:
            if joint.parent not in part_map or joint.child not in part_map:
                continue
            children_by_parent.setdefault(joint.parent, []).append(joint)
            child_names.add(joint.child)

        roots = [part for part in articulation.parts if part.name not in child_names]
        if not roots:
            roots = list(articulation.parts[:1])

        visited: set[str] = set()
        # URDF already stores an explicit link/joint graph. The MJCF stub
        # rebuilds that graph recursively as nested MuJoCo bodies.
        for root_part in roots:
            self._append_mjcf_body(
                parent_node=worldbody,
                part=root_part,
                part_map=part_map,
                children_by_parent=children_by_parent,
                visited=visited,
                body_pos=list(root_part.canonical_pose.get("translation", [0.0, 0.0, 0.0])),
                depth=0,
            )

        ET.ElementTree(root).write(mjcf_path, encoding="utf-8", xml_declaration=True)

    def _append_mjcf_body(
        self,
        parent_node: ET.Element,
        part: PartArtifact,
        part_map: dict[str, PartArtifact],
        children_by_parent: dict[str, list],
        visited: set[str],
        body_pos: list[float],
        depth: int,
    ) -> None:
        if part.name in visited:
            return
        visited.add(part.name)

        body = ET.SubElement(parent_node, "body", name=part.name, pos=_format_xyz(body_pos))
        self._append_mjcf_geoms(body, part, depth)

        for joint in children_by_parent.get(part.name, []):
            child_part = part_map.get(joint.child)
            if child_part is None:
                continue
            child_body = ET.SubElement(body, "body", name=child_part.name, pos=_format_xyz(joint.origin))
            joint_type = self._mjcf_joint_type(joint.joint_type)
            if joint_type is not None:
                ET.SubElement(
                    child_body,
                    "joint",
                    name=joint.name,
                    type=joint_type,
                    axis=_format_xyz(joint.axis),
                    range=_format_xyz(joint.limits),
                )
            self._append_mjcf_geoms(child_body, child_part, depth + 1)
            visited.add(child_part.name)
            for nested_joint in children_by_parent.get(child_part.name, []):
                nested_part = part_map.get(nested_joint.child)
                if nested_part is None:
                    continue
                self._append_mjcf_child_subtree(
                    child_body,
                    nested_part,
                    nested_joint,
                    part_map,
                    children_by_parent,
                    visited,
                    depth + 2,
                )

    def _append_mjcf_child_subtree(
        self,
        parent_body: ET.Element,
        part: PartArtifact,
        joint,
        part_map: dict[str, PartArtifact],
        children_by_parent: dict[str, list],
        visited: set[str],
        depth: int,
    ) -> None:
        if part.name in visited:
            return
        body = ET.SubElement(parent_body, "body", name=part.name, pos=_format_xyz(joint.origin))
        joint_type = self._mjcf_joint_type(joint.joint_type)
        if joint_type is not None:
            ET.SubElement(
                body,
                "joint",
                name=joint.name,
                type=joint_type,
                axis=_format_xyz(joint.axis),
                range=_format_xyz(joint.limits),
            )
        self._append_mjcf_geoms(body, part, depth)
        visited.add(part.name)
        for nested_joint in children_by_parent.get(part.name, []):
            nested_part = part_map.get(nested_joint.child)
            if nested_part is None:
                continue
            self._append_mjcf_child_subtree(
                body,
                nested_part,
                nested_joint,
                part_map,
                children_by_parent,
                visited,
                depth + 1,
            )

    def _append_mjcf_geoms(self, body: ET.Element, part: PartArtifact, depth: int) -> None:
        link_frame = list(part.canonical_pose.get("translation", [0.0, 0.0, 0.0]))
        local_boxes = shifted_boxes(part.primitive_boxes, link_frame)
        rgba = self._mjcf_rgba(part, depth)
        for box in local_boxes[:3]:
            ET.SubElement(
                body,
                "geom",
                type="box",
                pos=_format_xyz(box.center),
                size=_format_xyz([dimension / 2.0 for dimension in box.size]),
                rgba=rgba,
            )

    def _mjcf_joint_type(self, joint_type: str) -> str | None:
        if joint_type == "revolute":
            return "hinge"
        if joint_type == "prismatic":
            return "slide"
        return None

    def _mjcf_rgba(self, part: PartArtifact, depth: int) -> str:
        if part.role == "static":
            return "0.70 0.70 0.70 1"
        palette = [
            "0.83 0.36 0.36 1",
            "0.36 0.67 0.87 1",
            "0.73 0.58 0.29 1",
            "0.41 0.74 0.52 1",
        ]
        return palette[depth % len(palette)]
