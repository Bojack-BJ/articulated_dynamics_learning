from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class USDJointDef:
    """Represents a joint definition parsed from USD."""
    name: str
    joint_type: str  # "revolute" or "prismatic"
    axis: str  # "X", "Y", or "Z"
    local_rot: tuple[float, float, float, float]  # (w, x, y, z)
    body0: str  # first body name
    body1: str  # second body name
    local_pos: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    metadata: dict[str, Any]


def _parse_vector3(text: str) -> tuple[float, float, float] | None:
    """Parse a point3f or float3 vector from USD text."""
    match = re.search(r'\(([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\)', text)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    return None


def _parse_quat4(text: str) -> tuple[float, float, float, float] | None:
    """Parse a quatf(w, x, y, z) vector from USD text."""
    match = re.search(r'\(([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\)', text)
    if match:
        return (
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
    return None


def _quat_rotate(v: tuple[float, float, float], q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    # q = (w, x, y, z)
    w, x, y, z = q
    vx, vy, vz = v

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx

    return (
        vx + w * tx + cx,
        vy + w * ty + cy,
        vz + w * tz + cz,
    )


def _parse_float(text: str) -> float | None:
    """Parse a single float value from USD text."""
    match = re.search(
        r'=\s*([+-]?(?:inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        token = match.group(1).lower()
        if token in {"inf", "+inf"}:
            return math.inf
        if token == "-inf":
            return -math.inf
        return float(token)
    return None


def _parse_rel(text: str) -> str | None:
    """Parse a rel (relationship) path, extracting just the prim name."""
    match = re.search(r'=\s*<([^>]+)>', text)
    if match:
        path = match.group(1)
        # Return just the last component (prim name)
        return path.split('/')[-1]
    return None


def _parse_token(text: str) -> str | None:
    """Parse a token value from USD text."""
    match = re.search(r'=\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def parse_usd_joints(usd_text: str) -> list[USDJointDef]:
    """
    Parse joint definitions from flattened USD text.
    Looks for PhysicsRevoluteJoint and PhysicsPrismaticJoint definitions.
    """
    joints: list[USDJointDef] = []
    lines = usd_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for joint definitions
        joint_type_match = re.search(r'def (PhysicsRevoluteJoint|PhysicsPrismaticJoint)\s+"([^"]+)"', line)
        if joint_type_match:
            joint_class = joint_type_match.group(1)
            joint_name = joint_type_match.group(2)
            joint_type = "revolute" if "Revolute" in joint_class else "prismatic"
            
            # Parse joint properties until the closing brace
            axis = "Z"  # default
            local_rot = (1.0, 0.0, 0.0, 0.0)
            body0 = ""
            body1 = ""
            local_pos = (0.0, 0.0, 0.0)
            lower_limit = 0.0
            upper_limit = 1.57  # default ~90 degrees
            metadata: dict[str, Any] = {}
            
            brace_depth = 0
            i += 1
            while i < len(lines):
                prop_line = lines[i]
                
                # Track braces to find end of joint block
                brace_depth += prop_line.count('{')
                brace_depth -= prop_line.count('}')
                
                # Parse properties
                if 'physics:axis' in prop_line:
                    axis_token = _parse_token(prop_line)
                    if axis_token in {"X", "Y", "Z"}:
                        axis = axis_token
                
                if 'physics:body0' in prop_line:
                    body0 = _parse_rel(prop_line) or ""
                
                if 'physics:body1' in prop_line:
                    body1 = _parse_rel(prop_line) or ""
                
                if 'physics:localPos0' in prop_line or 'physics:localPos1' in prop_line:
                    pos = _parse_vector3(prop_line)
                    if pos:
                        local_pos = pos

                if 'physics:localRot0' in prop_line:
                    rot = _parse_quat4(prop_line)
                    if rot:
                        local_rot = rot
                
                if 'physics:lowerLimit' in prop_line:
                    val = _parse_float(prop_line)
                    if val is not None:
                        lower_limit = val
                
                if 'physics:upperLimit' in prop_line:
                    val = _parse_float(prop_line)
                    if val is not None:
                        upper_limit = val
                
                if 'physics:damping' in prop_line or 'drive:X:physics:damping' in prop_line:
                    val = _parse_float(prop_line)
                    if val is not None:
                        metadata['damping'] = val
                
                # End of joint block
                if brace_depth < 0:
                    break
                
                i += 1
            
            joints.append(
                USDJointDef(
                    name=joint_name,
                    joint_type=joint_type,
                    axis=axis,
                    local_rot=local_rot,
                    body0=body0,
                    body1=body1,
                    local_pos=local_pos,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    metadata=metadata,
                )
            )
        
        i += 1
    
    return joints


def usd_joint_to_mjcf_string(joint: USDJointDef, indent: int = 4) -> str:
    """
    Convert a USD joint definition to MuJoCo MJCF XML joint string.
    Returns a multi-line string representing the joint.
    
    Note: revolute joints in USD use degrees; MuJoCo uses radians.
    """
    indent_str = " " * indent
    
    # Convert USD axis to MuJoCo axis vector
    axis_map = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    axis_local = axis_map.get(joint.axis, (0.0, 0.0, 1.0))
    axis_world = _quat_rotate(axis_local, joint.local_rot)
    axis_norm = math.sqrt(axis_world[0] ** 2 + axis_world[1] ** 2 + axis_world[2] ** 2)
    if axis_norm > 1e-9:
        axis_world = (
            axis_world[0] / axis_norm,
            axis_world[1] / axis_norm,
            axis_world[2] / axis_norm,
        )
    axis_world = tuple(0.0 if abs(v) < 1e-6 else v for v in axis_world)
    axis_vec = f"{axis_world[0]:.6f} {axis_world[1]:.6f} {axis_world[2]:.6f}"
    
    # Convert degree limits to radians if revolute
    lower = joint.lower_limit
    upper = joint.upper_limit
    if joint.joint_type == "revolute":
        lower = math.radians(lower)
        upper = math.radians(upper)

    has_valid_limits = math.isfinite(lower) and math.isfinite(upper) and (upper > lower)
    
    # Extract damping if available
    damping = joint.metadata.get('damping', 0.2)
    
    # Format position (use localPos0 as the hinge point)
    pos_str = f"{joint.local_pos[0]:.6f} {joint.local_pos[1]:.6f} {joint.local_pos[2]:.6f}"
    
    # Build the joint XML
    lines = [
        f'{indent_str}<joint',
        f'{indent_str}  name="{joint.name}"',
        f'{indent_str}  type="{"hinge" if joint.joint_type == "revolute" else "slide"}"',
        f'{indent_str}  axis="{axis_vec}"',
        f'{indent_str}  pos="{pos_str}"',
        f'{indent_str}  damping="{damping}"',
    ]
    if has_valid_limits:
        lines.append(f'{indent_str}  limited="true"')
        lines.append(f'{indent_str}  range="{lower:.6f} {upper:.6f}"')
    else:
        lines.append(f'{indent_str}  limited="false"')
    lines.append(f'{indent_str}/>')
    
    return '\n'.join(lines)


def extract_joints_from_usd_file(usd_path: Path) -> list[USDJointDef]:
    """
    Extract joint definitions from a USD file by first flattening it to text.
    Requires 'usdcat' command-line tool.
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ['usdcat', str(usd_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"usdcat failed: {result.stderr}")
        
        return parse_usd_joints(result.stdout)
    except FileNotFoundError:
        raise RuntimeError(
            "usdcat command not found. Install Pixar USD tools or use the Python pxr module."
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract joints from USD file and generate MJCF XML")
    parser.add_argument("usd", type=Path, help="Path to USD file")
    parser.add_argument("--output", type=Path, default=None, help="Output file for generated joints")
    
    args = parser.parse_args()
    
    joints = extract_joints_from_usd_file(args.usd)
    
    print(f"Found {len(joints)} joints:")
    print()
    
    for joint in joints:
        print(joint.name)
        print(f"  Type: {joint.joint_type}")
        print(f"  Axis: {joint.axis}")
        print(f"  Bodies: {joint.body0} <-> {joint.body1}")
        print(f"  Position: {joint.local_pos}")
        print(f"  Limits: [{joint.lower_limit}, {joint.upper_limit}]")
        print()
        print("  MJCF XML:")
        print(usd_joint_to_mjcf_string(joint, indent=4))
        print()
    
    if args.output:
        with args.output.open('w') as f:
            f.write("<!-- Generated joint definitions from USD -->\n\n")
            for joint in joints:
                f.write(usd_joint_to_mjcf_string(joint) + "\n\n")
        print(f"Wrote joint definitions to {args.output}")
