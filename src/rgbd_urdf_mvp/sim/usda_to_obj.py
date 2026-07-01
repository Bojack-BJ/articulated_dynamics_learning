from __future__ import annotations

import argparse
import io
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_INT_RE = re.compile(r"[-+]?\d+")


@dataclass
class MeshData:
    name: str
    points: list[tuple[float, float, float]]
    face_vertex_counts: list[int]
    face_vertex_indices: list[int]
    st: list[tuple[float, float]]
    st_indices: list[int] | None
    st_interpolation: str | None

    xform_translate: tuple[float, float, float] | None
    xform_scale: tuple[float, float, float] | None
    xform_orient: tuple[float, float, float, float] | None  # (w, x, y, z)
    xform_rotate_xyz_deg: tuple[float, float, float] | None
    xform_rotate_zyx_deg: tuple[float, float, float] | None
    xform_order: list[str] | None

    material_binding: str | None


def _parse_tuple_floats(text: str, n: int) -> tuple[float, ...]:
    # Many USDA lines start with type tokens like `double3` which contain digits.
    # Only parse the numeric tuple payload inside parentheses.
    if "(" in text and ")" in text:
        payload = text.split("(", 1)[1].rsplit(")", 1)[0]
    else:
        payload = text
    nums = [float(x) for x in _FLOAT_RE.findall(payload)]
    if len(nums) != n:
        raise ValueError(f"Expected {n} floats, got {len(nums)} in: {text[:120]}...")
    return tuple(nums)


def _iter_file_lines(path: str | Path) -> Iterator[str]:
    if isinstance(path, str) and path == "-":
        # Read from stdin
        for line in sys.stdin:
            yield line
    else:
        # Read from file
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line


def _collect_bracket_content(first_line: str, it: Iterator[str]) -> str:
    # Collect everything between the first '[' and the matching ']' (non-nested for USDA arrays).
    eq = first_line.find("=")
    search_from = eq if eq >= 0 else 0
    start = first_line.find("[", search_from)
    if start < 0:
        raise ValueError("Expected '['")

    buf_parts: list[str] = []
    line = first_line
    while True:
        if "]" in line:
            # Find the ']' that closes the '[' we selected, not the type token's ']'.
            end = line.find("]", max(0, start))
            buf_parts.append(line[start + 1 : end])
            break
        buf_parts.append(line[start + 1 :])
        start = -1
        try:
            line = next(it)
        except StopIteration as e:
            raise ValueError("Unterminated '[' array") from e

    return "".join(buf_parts)


def _parse_int_array(first_line: str, it: Iterator[str]) -> list[int]:
    content = _collect_bracket_content(first_line, it)
    return [int(x) for x in _INT_RE.findall(content)]


def _parse_float_array(first_line: str, it: Iterator[str]) -> list[float]:
    content = _collect_bracket_content(first_line, it)
    return [float(x) for x in _FLOAT_RE.findall(content)]


def _parse_points(first_line: str, it: Iterator[str]) -> list[tuple[float, float, float]]:
    nums = _parse_float_array(first_line, it)
    if len(nums) % 3 != 0:
        raise ValueError(f"points float count {len(nums)} not multiple of 3")
    return [(nums[i], nums[i + 1], nums[i + 2]) for i in range(0, len(nums), 3)]


def _parse_st(first_line: str, it: Iterator[str]) -> list[tuple[float, float]]:
    nums = _parse_float_array(first_line, it)
    if len(nums) % 2 != 0:
        raise ValueError(f"primvars:st float count {len(nums)} not multiple of 2")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _quat_rotate(v: tuple[float, float, float], q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    # q = (w, x, y, z)
    w, x, y, z = q
    vx, vy, vz = v

    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v' = v + w*t + cross(q_xyz, t)
    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx

    return (
        vx + w * tx + cx,
        vy + w * ty + cy,
        vz + w * tz + cz,
    )


def _rot_xyz(v: tuple[float, float, float], rot_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    rx, ry, rz = (math.radians(rot_deg[0]), math.radians(rot_deg[1]), math.radians(rot_deg[2]))
    x, y, z = v

    # Rotate X
    cy, sy = math.cos(rx), math.sin(rx)
    y, z = (y * cy - z * sy, y * sy + z * cy)

    # Rotate Y
    cx, sx = math.cos(ry), math.sin(ry)
    x, z = (x * cx + z * sx, -x * sx + z * cx)

    # Rotate Z
    cz, sz = math.cos(rz), math.sin(rz)
    x, y = (x * cz - y * sz, x * sz + y * cz)

    return (x, y, z)


def _rot_zyx(v: tuple[float, float, float], rot_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    rz, ry, rx = (math.radians(rot_deg[0]), math.radians(rot_deg[1]), math.radians(rot_deg[2]))
    x, y, z = v

    # Rotate Z
    cz, sz = math.cos(rz), math.sin(rz)
    x, y = (x * cz - y * sz, x * sz + y * cz)

    # Rotate Y
    cy, sy = math.cos(ry), math.sin(ry)
    x, z = (x * cy + z * sy, -x * sy + z * cy)

    # Rotate X
    cx, sx = math.cos(rx), math.sin(rx)
    y, z = (y * cx - z * sx, y * sx + z * cx)

    return (x, y, z)


def _apply_xform(mesh: MeshData) -> None:
    if not mesh.xform_order:
        return

    translate = mesh.xform_translate
    scale = mesh.xform_scale
    orient = mesh.xform_orient
    rot_xyz = mesh.xform_rotate_xyz_deg
    rot_zyx = mesh.xform_rotate_zyx_deg

    ops = mesh.xform_order

    def apply_op(p: tuple[float, float, float], op: str) -> tuple[float, float, float]:
        if op.endswith("xformOp:translate") and translate is not None:
            return (p[0] + translate[0], p[1] + translate[1], p[2] + translate[2])
        if op.endswith("xformOp:scale") and scale is not None:
            return (p[0] * scale[0], p[1] * scale[1], p[2] * scale[2])
        if op.endswith("xformOp:orient") and orient is not None:
            return _quat_rotate(p, orient)
        if op.endswith("xformOp:rotateXYZ") and rot_xyz is not None:
            return _rot_xyz(p, rot_xyz)
        if op.endswith("xformOp:rotateZYX") and rot_zyx is not None:
            return _rot_zyx(p, rot_zyx)
        return p

    new_points: list[tuple[float, float, float]] = []
    for p in mesh.points:
        out = p
        for op in ops:
            # USD xformOpOrder is applied left-to-right.
            out = apply_op(out, op)
        new_points.append(out)

    mesh.points = new_points


def parse_usda_meshes(usda_path: str | Path, only_meshes: set[str] | None = None) -> list[MeshData]:
    meshes: list[MeshData] = []
    it = _iter_file_lines(usda_path)

    in_mesh = False
    brace_depth = 0
    saw_open_brace = False
    current: MeshData | None = None

    for line in it:
        if not in_mesh:
            if "def Mesh \"" in line:
                name = line.split('def Mesh "', 1)[1].split('"', 1)[0]
                if only_meshes is not None and name not in only_meshes:
                    # Still need to track brace depth to skip the block.
                    in_mesh = True
                    current = None
                    brace_depth = 0
                    saw_open_brace = False
                    continue

                in_mesh = True
                brace_depth = 0
                saw_open_brace = False
                current = MeshData(
                    name=name,
                    points=[],
                    face_vertex_counts=[],
                    face_vertex_indices=[],
                    st=[],
                    st_indices=None,
                    st_interpolation=None,
                    xform_translate=None,
                    xform_scale=None,
                    xform_orient=None,
                    xform_rotate_xyz_deg=None,
                    xform_rotate_zyx_deg=None,
                    xform_order=None,
                    material_binding=None,
                )
                continue
            continue

        # in_mesh: track braces
        if "{" in line:
            saw_open_brace = True
        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        if current is not None:
            stripped = line.strip()

            if stripped.startswith("int[] faceVertexCounts"):
                current.face_vertex_counts = _parse_int_array(line, it)
                continue
            if stripped.startswith("int[] faceVertexIndices"):
                current.face_vertex_indices = _parse_int_array(line, it)
                continue
            if stripped.startswith("point3f[] points") or stripped.startswith("double3[] points"):
                current.points = _parse_points(line, it)
                continue
            if stripped.startswith("texCoord2f[] primvars:st"):
                current.st = _parse_st(line, it)
                continue
            if stripped.startswith("int[] primvars:st:indices"):
                current.st_indices = _parse_int_array(line, it)
                continue
            if "primvars:st:interpolation" in stripped:
                # token primvars:st:interpolation = "faceVarying"
                if '="' in stripped:
                    current.st_interpolation = stripped.split('="', 1)[1].split('"', 1)[0]
                elif "= \"" in stripped:
                    current.st_interpolation = stripped.split('= "', 1)[1].split('"', 1)[0]
                continue

            if stripped.startswith("double3 xformOp:translate") or stripped.startswith("float3 xformOp:translate"):
                current.xform_translate = _parse_tuple_floats(stripped, 3)  # type: ignore[assignment]
                continue
            if stripped.startswith("double3 xformOp:scale") or stripped.startswith("float3 xformOp:scale"):
                current.xform_scale = _parse_tuple_floats(stripped, 3)  # type: ignore[assignment]
                continue
            if stripped.startswith("quatd xformOp:orient") or stripped.startswith("quatf xformOp:orient"):
                current.xform_orient = _parse_tuple_floats(stripped, 4)  # type: ignore[assignment]
                continue
            if stripped.startswith("double3 xformOp:rotateXYZ") or stripped.startswith("float3 xformOp:rotateXYZ"):
                current.xform_rotate_xyz_deg = _parse_tuple_floats(stripped, 3)  # type: ignore[assignment]
                continue
            if stripped.startswith("double3 xformOp:rotateZYX") or stripped.startswith("float3 xformOp:rotateZYX"):
                current.xform_rotate_zyx_deg = _parse_tuple_floats(stripped, 3)  # type: ignore[assignment]
                continue
            if stripped.startswith("uniform token[] xformOpOrder"):
                # uniform token[] xformOpOrder = ["xformOp:translate", ...]
                content = _collect_bracket_content(line, it)
                tokens = [t.strip().strip('"') for t in content.split(",") if t.strip()]
                current.xform_order = tokens
                continue

            if stripped.startswith("rel material:binding"):
                # rel material:binding = </root/Looks/material_Body001_material> (
                if "<" in stripped and ">" in stripped:
                    current.material_binding = stripped.split("<", 1)[1].split(">", 1)[0]
                continue

        if saw_open_brace and brace_depth <= 0:
            # End of mesh block
            if current is not None:
                _apply_xform(current)
                meshes.append(current)
            in_mesh = False
            current = None
            brace_depth = 0
            saw_open_brace = False

    return meshes


def _triangulate_face(count: int) -> list[tuple[int, int, int]]:
    if count < 3:
        return []
    if count == 3:
        return [(0, 1, 2)]
    return [(0, i, i + 1) for i in range(1, count - 1)]


def write_obj(mesh: MeshData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    st_interp = mesh.st_interpolation or "faceVarying"

    if not mesh.face_vertex_counts or not mesh.face_vertex_indices or not mesh.points:
        raise ValueError(f"Mesh '{mesh.name}' missing topology/points")

    has_uvs = bool(mesh.st)
    if has_uvs and st_interp == "vertex":
        if len(mesh.st) != len(mesh.points):
            raise ValueError(
                f"Mesh '{mesh.name}': primvars:st vertex interpolation expects {len(mesh.points)} uvs, got {len(mesh.st)}"
            )
    elif has_uvs:
        corner_count = sum(mesh.face_vertex_counts)
        if mesh.st_indices is None:
            if len(mesh.st) != corner_count:
                raise ValueError(
                    f"Mesh '{mesh.name}': primvars:st faceVarying expects {corner_count} uvs, got {len(mesh.st)}"
                )
        else:
            # Indexed faceVarying; st is unique list and indices map corners.
            if len(mesh.st_indices) != corner_count:
                raise ValueError(
                    f"Mesh '{mesh.name}': primvars:st:indices expects {corner_count} entries, got {len(mesh.st_indices)}"
                )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"o {mesh.name}\n")

        for (x, y, z) in mesh.points:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")

        if mesh.st:
            for (u, v) in mesh.st:
                f.write(f"vt {u:.9g} {v:.9g}\n")

        corner = 0
        for count in mesh.face_vertex_counts:
            if corner + count > len(mesh.face_vertex_indices):
                raise ValueError(f"Mesh '{mesh.name}': faceVertexIndices shorter than expected")

            v_idx = [mesh.face_vertex_indices[corner + i] + 1 for i in range(count)]

            if not mesh.st:
                vt_idx = [0 for _ in range(count)]
            else:
                if st_interp == "vertex":
                    # vt list is aligned with points; reuse vertex indices.
                    vt_idx = v_idx
                else:
                    if mesh.st_indices is None:
                        vt_idx = [corner + i + 1 for i in range(count)]
                    else:
                        vt_idx = [mesh.st_indices[corner + i] + 1 for i in range(count)]

            for (a, b, c) in _triangulate_face(count):
                if mesh.st:
                    f.write(
                        "f "
                        + f"{v_idx[a]}/{vt_idx[a]} "
                        + f"{v_idx[b]}/{vt_idx[b]} "
                        + f"{v_idx[c]}/{vt_idx[c]}\n"
                    )
                else:
                    f.write(f"f {v_idx[a]} {v_idx[b]} {v_idx[c]}\n")

            corner += count


def convert_usda_to_obj(usda_path: str | Path, out_dir: Path, only_meshes: set[str] | None = None) -> list[Path]:
    meshes = parse_usda_meshes(usda_path, only_meshes=only_meshes)
    written: list[Path] = []

    for mesh in meshes:
        out_path = out_dir / f"{mesh.name}.obj"
        write_obj(mesh, out_path)
        written.append(out_path)

    return written


def build_argparser() -> argparse.ArgumentParser:
    def usda_path_or_stdin(s: str) -> str | Path:
        """Accept either '-' for stdin or a file path."""
        if s == "-":
            return "-"
        return Path(s)

    p = argparse.ArgumentParser(description="Convert flattened USDA Mesh prims to OBJ files (with UVs).")
    p.add_argument("usda", type=usda_path_or_stdin, help="Path to a flattened .usda file, or '-' for stdin")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "usd_convert" / "obj",
        help="Output directory for OBJ files",
    )
    p.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Only convert these Mesh prim names (space-separated)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    only_meshes = set(args.only) if args.only else None
    written = convert_usda_to_obj(args.usda, args.out_dir, only_meshes=only_meshes)

    print(f"Wrote {len(written)} OBJ(s) to {args.out_dir}")
    for p in written:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
