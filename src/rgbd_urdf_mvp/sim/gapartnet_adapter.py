from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from ..core.xml_utils import write_xml_tree


@dataclass
class GAPartNetMJCFConfig:
    asset_dir: Path
    output_mjcf: Path
    urdf_name: str = "mobility_annotation_gapartnet.urdf"
    density_kg_m3: float = 30.0
    minimum_proxy_size_m: float = 0.002
    add_floor: bool = False
    source_up_axis: str = "y"


@dataclass
class _BoxProxy:
    center: list[float]
    size: list[float]
    source_mesh: str


@dataclass
class _VisualMesh:
    name: str
    path: Path
    scale: list[float]
    translation: list[float]
    rotation: list[float]


def _numbers(text: str | None, fallback: list[float]) -> list[float]:
    if not text:
        return list(fallback)
    values = [float(value) for value in text.split()]
    return values if len(values) == len(fallback) else list(fallback)


def _format(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _rpy_to_quat_wxyz(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _rotate_rpy(point: list[float], rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = point
    return [
        (cy * cp) * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
        (sy * cp) * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
        (-sp) * x + (cp * sr) * y + (cp * cr) * z,
    ]


def _obj_vertices(path: Path) -> list[list[float]]:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return vertices


def _mesh_proxy(asset_dir: Path, visual: ET.Element, minimum_size: float) -> _BoxProxy | None:
    geometry = visual.find("geometry")
    mesh = geometry.find("mesh") if geometry is not None else None
    if mesh is None or not mesh.get("filename"):
        return None
    mesh_path = (asset_dir / str(mesh.get("filename"))).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing GAPartNet mesh: {mesh_path}")

    scale = _numbers(mesh.get("scale"), [1.0, 1.0, 1.0])
    origin = visual.find("origin")
    translation = _numbers(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
    rotation = _numbers(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
    transformed: list[list[float]] = []
    for vertex in _obj_vertices(mesh_path):
        scaled = [vertex[axis] * scale[axis] for axis in range(3)]
        rotated = _rotate_rpy(scaled, rotation)
        transformed.append([rotated[axis] + translation[axis] for axis in range(3)])

    lower = [min(point[axis] for point in transformed) for axis in range(3)]
    upper = [max(point[axis] for point in transformed) for axis in range(3)]
    size = [max(minimum_size, upper[axis] - lower[axis]) for axis in range(3)]
    center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
    return _BoxProxy(center=center, size=size, source_mesh=str(mesh.get("filename")))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _link_visual_meshes(asset_dir: Path, link_name: str, link: ET.Element) -> list[_VisualMesh]:
    meshes: list[_VisualMesh] = []
    for index, visual in enumerate(link.findall("visual")):
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or not mesh.get("filename"):
            continue
        path = (asset_dir / str(mesh.get("filename"))).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing GAPartNet visual mesh: {path}")
        origin = visual.find("origin")
        meshes.append(
            _VisualMesh(
                name=_safe_name(f"{link_name}_visual_{index}"),
                path=path,
                scale=_numbers(mesh.get("scale"), [1.0, 1.0, 1.0]),
                translation=_numbers(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
                rotation=_numbers(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
            )
        )
    return meshes


def _link_proxies(asset_dir: Path, link: ET.Element, minimum_size: float) -> list[_BoxProxy]:
    proxies = [
        proxy
        for visual in link.findall("visual")
        if (proxy := _mesh_proxy(asset_dir, visual, minimum_size)) is not None
    ]
    if proxies:
        return proxies
    proxies = [
        proxy
        for collision in link.findall("collision")
        if (proxy := _mesh_proxy(asset_dir, collision, minimum_size)) is not None
    ]
    return proxies or [_BoxProxy(center=[0.0, 0.0, 0.0], size=[0.05, 0.05, 0.05], source_mesh="placeholder")]


def _aggregate_inertial(proxies: list[_BoxProxy], density: float) -> tuple[float, list[float], list[float]]:
    raw_masses = [density * box.size[0] * box.size[1] * box.size[2] for box in proxies]
    mass = max(0.05, sum(raw_masses))
    scale = mass / max(1e-12, sum(raw_masses))
    masses = [value * scale for value in raw_masses]
    center = [sum(value * box.center[axis] for value, box in zip(masses, proxies)) / mass for axis in range(3)]
    inertia = [0.0, 0.0, 0.0]
    for value, box in zip(masses, proxies):
        x, y, z = box.size
        local = [value * (y * y + z * z) / 12.0, value * (x * x + z * z) / 12.0, value * (x * x + y * y) / 12.0]
        dx, dy, dz = [box.center[axis] - center[axis] for axis in range(3)]
        offsets = [value * (dy * dy + dz * dz), value * (dx * dx + dz * dz), value * (dx * dx + dy * dy)]
        inertia = [inertia[axis] + local[axis] + offsets[axis] for axis in range(3)]
    return mass, center, [max(1e-8, value) for value in inertia]


class GAPartNetMJCFAdapter:
    """Convert a GAPartNet URDF into a MuJoCo-loadable proxy MJCF.

    GAPartNet's thin collision meshes can fail MuJoCo convex-hull compilation.
    This adapter preserves the kinematic graph but represents each source mesh
    with a link-local box for deterministic collision and inertial setup.
    """

    def convert(self, config: GAPartNetMJCFConfig) -> Path:
        asset_dir = Path(config.asset_dir).resolve()
        urdf_path = asset_dir / config.urdf_name
        urdf_root = ET.parse(urdf_path).getroot()
        links = {str(link.get("name")): link for link in urdf_root.findall("link")}
        joints = list(urdf_root.findall("joint"))
        children: dict[str, list[ET.Element]] = {}
        child_names: set[str] = set()
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None or not parent.get("link") or not child.get("link"):
                continue
            children.setdefault(str(parent.get("link")), []).append(joint)
            child_names.add(str(child.get("link")))

        roots = [name for name in links if name not in child_names]
        if not roots:
            raise ValueError(f"No root link found in {urdf_path}")
        proxies = {
            name: _link_proxies(asset_dir, link, max(1e-6, config.minimum_proxy_size_m))
            for name, link in links.items()
        }
        visual_meshes = {
            name: _link_visual_meshes(asset_dir, name, link)
            for name, link in links.items()
        }

        mjcf = ET.Element("mujoco", model=asset_dir.name)
        ET.SubElement(mjcf, "compiler", angle="radian", coordinate="local")
        ET.SubElement(mjcf, "option", timestep="0.004166667", gravity="0 0 -9.81")
        visual = ET.SubElement(mjcf, "visual")
        ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.8 0.8 0.8", specular="0.1 0.1 0.1")
        assets = ET.SubElement(mjcf, "asset")
        for meshes in visual_meshes.values():
            for mesh in meshes:
                ET.SubElement(
                    assets,
                    "mesh",
                    name=mesh.name,
                    file=str(mesh.path),
                    scale=_format(mesh.scale),
                    inertia="shell",
                )
        world = ET.SubElement(mjcf, "worldbody")
        if config.add_floor:
            ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", rgba="0.35 0.38 0.42 1")

        visited: set[str] = set()
        for root_name in roots:
            self._append_link(
                world, root_name, None, links, children, proxies, visual_meshes, config, visited, depth=0
            )

        output = Path(config.output_mjcf)
        write_xml_tree(ET.ElementTree(mjcf), output)
        manifest = {
            "source_asset_dir": str(asset_dir),
            "source_urdf": str(urdf_path),
            "output_mjcf": str(output.resolve()),
            "geometry_mode": "original-visual-mesh-with-axis-aligned-box-collision-proxy",
            "dynamics_mode": "geometry_initialized_placeholder",
            "density_kg_m3": config.density_kg_m3,
            "link_count": len(links),
            "joint_count": len(joints),
            "proxy_count": sum(len(values) for values in proxies.values()),
            "visual_mesh_count": sum(len(values) for values in visual_meshes.values()),
            "warnings": [
                "Visual mesh paths are absolute and the MJCF depends on the extracted GAPartNet asset directory.",
                "Mass and inertia are initialization values, not ground-truth dynamics.",
            ],
        }
        manifest_path = output.with_suffix(".conversion.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output

    def _append_link(
        self,
        parent_node: ET.Element,
        link_name: str,
        incoming_joint: ET.Element | None,
        links: dict[str, ET.Element],
        children: dict[str, list[ET.Element]],
        proxies: dict[str, list[_BoxProxy]],
        visual_meshes: dict[str, list[_VisualMesh]],
        config: GAPartNetMJCFConfig,
        visited: set[str],
        depth: int,
    ) -> None:
        if link_name in visited:
            return
        visited.add(link_name)
        body_attributes = {"name": link_name}
        if incoming_joint is not None:
            origin = incoming_joint.find("origin")
            body_attributes["pos"] = _format(_numbers(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]))
            body_attributes["quat"] = _format(
                _rpy_to_quat_wxyz(
                    _numbers(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
                )
            )
        elif config.source_up_axis == "y":
            # PartNet/GAPartNet assets are Y-up. Rotate the complete kinematic
            # subtree into MuJoCo's Z-up world without changing joint-local data.
            body_attributes["quat"] = _format(_rpy_to_quat_wxyz([math.pi * 0.5, 0.0, 0.0]))
        body = ET.SubElement(parent_node, "body", **body_attributes)

        if incoming_joint is not None:
            kind = str(incoming_joint.get("type", "fixed")).lower()
            if kind in {"revolute", "continuous", "prismatic"}:
                axis_node = incoming_joint.find("axis")
                limit_node = incoming_joint.find("limit")
                lower = float(limit_node.get("lower", "-3.141592654" if kind == "continuous" else "-1")) if limit_node is not None else (-math.pi if kind == "continuous" else -1.0)
                upper = float(limit_node.get("upper", "3.141592654" if kind == "continuous" else "1")) if limit_node is not None else (math.pi if kind == "continuous" else 1.0)
                ET.SubElement(
                    body,
                    "joint",
                    name=str(incoming_joint.get("name", f"{link_name}_joint")),
                    type="slide" if kind == "prismatic" else "hinge",
                    axis=_format(_numbers(axis_node.get("xyz") if axis_node is not None else None, [1.0, 0.0, 0.0])),
                    range=_format([lower, upper]),
                    damping="0.1",
                    frictionloss="0",
                )

        link_proxies = proxies[link_name]
        mass, center, inertia = _aggregate_inertial(link_proxies, max(1e-6, config.density_kg_m3))
        ET.SubElement(body, "inertial", pos=_format(center), mass=f"{mass:.9g}", diaginertia=_format(inertia))
        palette = ["0.62 0.69 0.72 1", "0.23 0.75 0.66 1", "0.96 0.55 0.24 1", "0.64 0.52 0.94 1"]
        for mesh in visual_meshes[link_name]:
            ET.SubElement(
                body,
                "geom",
                name=f"{mesh.name}_geom",
                type="mesh",
                mesh=mesh.name,
                pos=_format(mesh.translation),
                quat=_format(_rpy_to_quat_wxyz(mesh.rotation)),
                contype="0",
                conaffinity="0",
                group="1",
                rgba=palette[depth % len(palette)],
            )
        for index, proxy in enumerate(link_proxies):
            ET.SubElement(
                body,
                "geom",
                name=f"{link_name}_proxy_{index}",
                type="box",
                pos=_format(proxy.center),
                size=_format([value * 0.5 for value in proxy.size]),
                rgba="0 0 0 0",
                group="3",
                friction="0.8 0.02 0.001",
            )

        for joint in children.get(link_name, []):
            child_node = joint.find("child")
            child_name = str(child_node.get("link")) if child_node is not None else ""
            if child_name in links:
                self._append_link(
                    body,
                    child_name,
                    joint,
                    links,
                    children,
                    proxies,
                    visual_meshes,
                    config,
                    visited,
                    depth + 1,
                )
