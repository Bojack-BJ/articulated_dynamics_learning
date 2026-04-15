#!/usr/bin/env python3
"""
Convert USD asset to MJCF XML with meshes (OBJ) and extracted joint definitions.

Workflow:
1. Flatten USD to USDA (via usdcat)
2. Convert meshes to OBJ with UVs
3. Extract joint definitions from USD Physics schema
4. Generate a single MJCF XML that references all meshes and joints
5. Save the MJCF + OBJ files adjacent to each other

Usage:
  python -m rgbd_urdf_mvp.usd_to_mjcf <usd_file> \
    --category refrigerator \
    --output-prefix examples/mujoco_models/refrigerator031 \
    --texture-dir Lightwheel/Refrigerator031/texture \
    --texture-names T_Refrigerator031_BC001.png

Output:
  - <output_prefix>.xml          (MJCF file)
  - <output_prefix>_obj/         (directory with OBJ files)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.categories import SUPPORTED_CATEGORIES, normalize_category
from .usda_to_obj import convert_usda_to_obj
from .usd_joint_parser import USDJointDef, extract_joints_from_usd_file, usd_joint_to_mjcf_string


def _triangular_face_volume(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    ) / 6.0


def _obj_volume(obj_path: Path) -> float:
    vertices: list[tuple[float, float, float]] = []
    volume = 0.0

    for raw_line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            _, x_text, y_text, z_text = line.split(maxsplit=3)
            vertices.append((float(x_text), float(y_text), float(z_text)))
        elif line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) < 3:
                continue
            face_indices: list[int] = []
            for token in tokens:
                vertex_text = token.split("/", 1)[0]
                face_indices.append(int(vertex_text) - 1)
            first_index = face_indices[0]
            for idx in range(1, len(face_indices) - 1):
                a = vertices[first_index]
                b = vertices[face_indices[idx]]
                c = vertices[face_indices[idx + 1]]
                volume += _triangular_face_volume(a, b, c)

    return abs(volume)


def _total_obj_volume(obj_files: list[Path]) -> float:
    return sum(_obj_volume(obj_file) for obj_file in obj_files)


def _obj_bounds(
    obj_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    min_x = math.inf
    min_y = math.inf
    min_z = math.inf
    max_x = -math.inf
    max_y = -math.inf
    max_z = -math.inf
    has_vertex = False

    for raw_line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("v "):
            continue
        _, x_text, y_text, z_text = line.split(maxsplit=3)
        x = float(x_text)
        y = float(y_text)
        z = float(z_text)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
        has_vertex = True

    if not has_vertex:
        return None
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def _total_obj_bounds(
    obj_files: list[Path],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    lower: tuple[float, float, float] | None = None
    upper: tuple[float, float, float] | None = None
    for obj_file in obj_files:
        bounds = _obj_bounds(obj_file)
        if bounds is None:
            continue
        lo, hi = bounds
        if lower is None or upper is None:
            lower = lo
            upper = hi
            continue
        lower = (
            min(lower[0], lo[0]),
            min(lower[1], lo[1]),
            min(lower[2], lo[2]),
        )
        upper = (
            max(upper[0], hi[0]),
            max(upper[1], hi[1]),
            max(upper[2], hi[2]),
        )
    if lower is None or upper is None:
        return None
    return lower, upper


@dataclass
class MJCFConfig:
    """Configuration for MJCF generation."""
    model_name: str
    category: str  # e.g. "door", "drawer", "blender", "refrigerator", "microwave", ...
    object_dir_name: str  # e.g., "refrigerator031_obj"
    texture_dir: Optional[str] = None
    texture_names: Optional[list[str]] = None
    
    # Category-specific defaults
    default_mass: dict[str, float] = None
    
    def __post_init__(self):
        if self.default_mass is None:
            self.default_mass = {
                "door": 3.0,
                "drawer": 2.0,
                "blender": 2.0,
                "refrigerator": 15.0,
                "microwave": 12.0,
                "coffeemachine": 7.0,
                "dishwasher": 35.0,
                "electrickettle": 2.5,
                "oven": 30.0,
                "rangehood": 12.0,
                "sink": 10.0,
                "standmixer": 6.0,
                "stove": 20.0,
                "stovetop": 14.0,
                "toaster": 3.0,
                "toasteroven": 8.0,
            }


def generate_mjcf_xml(
    model_name: str,
    obj_files: list[str],
    joints: list[USDJointDef],
    texture_names: Optional[list[str]] = None,
    texture_dir: Optional[str] = None,
    asset_dir: Optional[str] = None,
    category: str = "door",
    object_dir_name: str = "",
    mass_kg: float = 2.0,
    root_body_pos_z: float = 0.0,
) -> str:
    """
    Generate MJCF XML with meshes, materials, and joints.
    
    Args:
        model_name: model name for MJCF
        obj_files: list of OBJ file basenames (without .obj)
        joints: extracted joint definitions
        texture_names: list of texture PNG filenames
        texture_dir: relative path to texture directory
        asset_dir: relative path to asset directory (where OBJ files are)
        category: object category for naming / fallback logic
        object_dir_name: name of OBJ directory (e.g. "refrigerator031_obj")
        mass_kg: total mass to assign to the main rigid body
        root_body_pos_z: root body z translation used to place model on the ground
    
    Returns:
        MJCF XML string
    """
    if asset_dir is None:
        asset_dir = f"../../examples/mujoco_models/{object_dir_name}"
    
    if texture_dir is None:
        texture_dir = asset_dir  # Fallback to asset dir if no texture dir specified
    
    # Generate material definitions for textures
    materials_xml = ""
    if texture_names:
        for i, tex_name in enumerate(texture_names):
            mat_name = f"material_{i}"
            materials_xml += f'''    <texture name="{Path(tex_name).stem}" type="2d" file="{tex_name}" />
    <material
      name="{mat_name}"
      texture="{Path(tex_name).stem}"
      rgba="1 1 1 1"
      texuniform="false"
      texrepeat="1 1"
      specular="0"
      reflectance="0"
      shininess="0"
    />
'''
    
    # Generate mesh definitions
    meshes_xml = ""
    for obj_file in obj_files:
        mesh_name = obj_file.replace(".obj", "")
        meshes_xml += f'    <mesh name="{mesh_name}" file="{obj_file}.obj" />\n'

    mesh_names = [obj_file.replace(".obj", "") for obj_file in obj_files]

    joint_by_child: dict[str, USDJointDef] = {}
    children_by_parent: dict[str, list[str]] = {}
    known_bodies: set[str] = {model_name}

    for joint in joints:
        child_name = joint.body1 if joint.body1 else f"{joint.name}_link"
        parent_name = joint.body0 if joint.body0 else model_name
        known_bodies.add(parent_name)
        known_bodies.add(child_name)
        joint_by_child[child_name] = joint
        children_by_parent.setdefault(parent_name, []).append(child_name)

    for parent_name in children_by_parent:
        children_by_parent[parent_name] = sorted(set(children_by_parent[parent_name]))

    body_meshes: dict[str, list[str]] = {body_name: [] for body_name in known_bodies}
    unassigned_meshes: list[str] = []
    for mesh_name in mesh_names:
        candidates = [
            body_name
            for body_name in known_bodies
            if mesh_name == body_name or mesh_name.startswith(f"{body_name}_")
        ]
        if candidates:
            best_body = max(candidates, key=len)
            body_meshes.setdefault(best_body, []).append(mesh_name)
        else:
            unassigned_meshes.append(mesh_name)

    def emit_body(body_name: str, indent: int) -> str:
        indent_str = " " * indent
        out = f'{indent_str}<body name="{body_name}" pos="0 0 0">\n'

        joint_def = joint_by_child.get(body_name)
        if joint_def is not None:
            out += usd_joint_to_mjcf_string(joint_def, indent=indent + 2) + "\n"

        for mesh_name in body_meshes.get(body_name, []):
            out += f'{" " * (indent + 2)}<geom type="mesh" mesh="{mesh_name}" material="material_0" />\n'

        for child_name in children_by_parent.get(body_name, []):
            out += emit_body(child_name, indent + 2)

        out += f"{indent_str}</body>\n"
        return out

    root_geoms_xml = ""
    for mesh_name in body_meshes.get(model_name, []) + unassigned_meshes:
        root_geoms_xml += f'      <geom type="mesh" mesh="{mesh_name}" material="material_0" />\n'

    articulated_bodies_xml = ""
    for child_name in children_by_parent.get(model_name, []):
        articulated_bodies_xml += emit_body(child_name, indent=6)
    
    # Build MJCF
    xml = f'''<mujoco model="{model_name}">
  <compiler
    angle="radian"
    assetdir="{asset_dir}"
    texturedir="{texture_dir}"
  />

  <visual>
    <global offwidth="1024" offheight="1024" />
  </visual>

  <asset>
{materials_xml}{meshes_xml}  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1" />
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3" />
    <light pos="2 2 2" dir="-1 -1 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" />

    <!-- Main object body -->
    <body name="{model_name}" pos="0 0 {root_body_pos_z:.6f}">
    <inertial pos="0 0 0" quat="1 0 0 0" mass="{mass_kg:.6f}" diaginertia="1 1 1" />
      
{root_geoms_xml}
{articulated_bodies_xml}    </body>
  </worldbody>
</mujoco>
'''
    return xml


def convert_usd_to_mjcf(
    usd_path: Path,
    output_prefix: Path | None = None,
    category: str = "door",
    texture_dir: Optional[str] = None,
    texture_names: Optional[list[str]] = None,
) -> None:
    """
    Convert USD asset to MJCF + OBJ files.
    
    Args:
        usd_path: path to USD file
        output_prefix: output path prefix (defaults to examples/mujoco_models/<usd_stem>)
                      Will create:
                      - <prefix>.xml
                      - <prefix>_obj/ (directory with OBJs)
        category: object category
        texture_dir: relative path to texture directory
        texture_names: list of texture file names to include in MJCF
    """
    if output_prefix is None:
        output_prefix = Path("examples") / "mujoco_models" / usd_path.stem
    output_prefix = Path(output_prefix)
    obj_dir = output_prefix.parent / f"{output_prefix.name}_obj"
    obj_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {usd_path} to MJCF...")
    print(f"  Output MJCF: {output_prefix}.xml")
    print(f"  Output OBJ dir: {obj_dir}")

    if texture_dir is None:
        texture_source_dir = usd_path.parent / "texture"
        texture_dir = os.path.relpath(texture_source_dir, output_prefix.parent)
        print(f"  Default texture dir: {texture_dir}")
    else:
        texture_source_dir = usd_path.parent / "texture"

    if texture_names is None:
        texture_names = [
            path.name
            for path in sorted(texture_source_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if texture_names:
            print(f"  Default texture names: {', '.join(texture_names)}")
    
    # Step 1: Flatten USD and convert to OBJ (piped, no intermediate USDA saved)
    print("\n[1/3] Flattening USD and converting meshes to OBJ...")
    try:
        result = subprocess.run(
            ["usdcat", "--flatten", str(usd_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        usda_text = result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to flatten USD: {e.stderr}")
    
    # Convert USDA to OBJ via stdin
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        f.write(usda_text)
        usda_path_temp = Path(f.name)
    
    try:
        obj_files = convert_usda_to_obj(usda_path_temp, obj_dir)
        obj_basenames = [f.stem for f in obj_files]
        print(f"  Generated {len(obj_files)} OBJ files")
    finally:
        usda_path_temp.unlink()

    # Estimate mass from category density and OBJ volume.
    density_map = {
        "door": 650.0,
        "drawer": 550.0,
        "blender": 750.0,
        "refrigerator": 280.0,
        "microwave": 420.0,
        "coffeemachine": 680.0,
        "dishwasher": 300.0,
        "electrickettle": 720.0,
        "oven": 340.0,
        "rangehood": 360.0,
        "sink": 780.0,
        "standmixer": 700.0,
        "stove": 420.0,
        "stovetop": 460.0,
        "toaster": 650.0,
        "toasteroven": 520.0,
    }
    density_kg_m3 = density_map.get(category, 500.0)
    total_volume_m3 = _total_obj_volume(obj_files)
    mass_kg = max(0.05, density_kg_m3 * total_volume_m3)
    print(f"  Estimated volume: {total_volume_m3:.6f} m^3")
    print(f"  Density for {category}: {density_kg_m3:.1f} kg/m^3")
    print(f"  Estimated mass: {mass_kg:.6f} kg")

    bounds = _total_obj_bounds(obj_files)
    if bounds is not None:
        lower, upper = bounds
        root_body_pos_z = max(0.0, -lower[2] + 0.001)
        print(
            f"  Mesh bounds z: [{lower[2]:.6f}, {upper[2]:.6f}] -> root body z={root_body_pos_z:.6f}"
        )
    else:
        root_body_pos_z = 0.0
    
    # Step 2: Extract joints from USD
    print("\n[2/3] Extracting joint definitions...")
    joints = extract_joints_from_usd_file(usd_path)
    print(f"  Found {len(joints)} joints")
    
    # Step 3: Generate MJCF XML
    print("\n[3/3] Generating MJCF XML...")
    model_name = output_prefix.name
    object_dir_name = f"{output_prefix.name}_obj"
    
    mjcf_xml = generate_mjcf_xml(
        model_name=model_name,
        obj_files=obj_basenames,
        joints=joints,
        texture_names=texture_names,
        texture_dir=texture_dir,
        asset_dir=f"../../examples/mujoco_models/{object_dir_name}",
        category=category,
        object_dir_name=object_dir_name,
        mass_kg=mass_kg,
        root_body_pos_z=root_body_pos_z,
    )
    
    # Check if this is a known category; if so, add helpful comments
    if category == "refrigerator":
        mjcf_xml = mjcf_xml.replace(
            "<!-- TODO: Add visual geoms here -->",
            "<!-- Refrigerator meshes - customize as needed -->",
        )
    elif category == "door":
        mjcf_xml = mjcf_xml.replace(
            "<!-- TODO: Add visual geoms here -->",
            "<!-- Door meshes -->",
        )
    elif category == "drawer":
        mjcf_xml = mjcf_xml.replace(
            "<!-- TODO: Add visual geoms here -->",
            "<!-- Drawer meshes -->",
        )
    
    # Write MJCF XML
    mjcf_output_path = output_prefix.parent / f"{output_prefix.name}.xml"
    with open(mjcf_output_path, "w") as f:
        f.write(mjcf_xml)
    print(f"  Saved MJCF to {mjcf_output_path}")
    
    print(f"\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. (Optional) Inspect/adjust generated joints and body hierarchy in {mjcf_output_path}")
    print(f"2. Record with:")
    print(f"   PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco {mjcf_output_path} \\")
    print(f"     --category {category} --object-id {model_name} ...")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert USD asset to MJCF XML with meshes and extracted joints."
    )
    p.add_argument("usd", type=Path, help="Path to USD file")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix (defaults to examples/mujoco_models/<usd_stem>)",
    )
    p.add_argument(
        "--category",
        type=normalize_category,
        choices=list(SUPPORTED_CATEGORIES),
        default="door",
        help="Object category (affects density-based mass estimate)",
    )
    p.add_argument(
        "--texture-dir",
        type=str,
        help="Relative path to texture directory (defaults to the USD sibling texture/ folder)",
    )
    p.add_argument(
        "--texture-names",
        nargs="*",
        help="Texture filenames to reference in MJCF (defaults to files found in the USD sibling texture/ folder)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    
    try:
        convert_usd_to_mjcf(
            usd_path=args.usd,
            output_prefix=args.output_prefix,
            category=args.category,
            texture_dir=args.texture_dir,
            texture_names=args.texture_names,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
