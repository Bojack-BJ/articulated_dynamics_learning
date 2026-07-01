#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.kinematics.evaluation import _axis_angle_error_deg, _line_distance
from rgbd_urdf_mvp.kinematics.joint_inference import (
    _add,
    _dot,
    _matmul3,
    _matvec3,
    _mean_point,
    _norm,
    _normalize,
    _rotation_angle,
    _rotation_axis,
    _scale,
    _signed_angle_about_axis,
    _solve_3x3,
    _subtract,
    _transpose3,
)


COLORS = [
    "#86e07d",
    "#a06ce8",
    "#75c7ff",
    "#ffb86b",
    "#ff6b9a",
    "#c8d86b",
    "#66dcc7",
    "#d58cff",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export browser and URDF-style ReArt visualizations.")
    parser.add_argument("result_pkl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=6000)
    args = parser.parse_args()

    result_path = args.result_pkl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else result_path.parent / "reart_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    with result_path.open("rb") as f:
        result = pickle.load(f)

    cano_pc = np.asarray(result["cano_pc"], dtype=float)
    complete = np.asarray(result["complete_pc_list"], dtype=float)
    part_labels = np.asarray(result["pred_cano_part"], dtype=int)
    poses = np.asarray(result["pred_pose_list"], dtype=float)
    cano_idx = int(result.get("cano_idx", 0))
    connections = result.get("joint_connection") or []

    evaluation = load_json(args.eval_json) if args.eval_json and args.eval_json.exists() else None
    joints = choose_joints(evaluation, poses, connections)
    joint = joints[0] if joints else None

    write_html(
        output_dir / "viewer_reart.html",
        cano_pc,
        complete,
        part_labels,
        poses,
        cano_idx,
        joints,
        max_points=args.max_points,
    )
    write_urdf_package(output_dir / "urdf_reart", cano_pc, part_labels, joints)
    write_json(
        output_dir / "reart_visualization_manifest.json",
        {
            "source": "reart-visualization-export",
            "result_pkl": str(result_path),
            "cano_idx": cano_idx,
            "viewer_html": str((output_dir / "viewer_reart.html").resolve()),
            "urdf_path": str((output_dir / "urdf_reart" / "model.urdf").resolve()),
            "joint": joint,
            "joints": joints,
            "notes": [
                "The URDF mesh package is for visual inspection only.",
                "Part meshes are convex hulls of ReArt canonical segmented points.",
                "ReArt GIF/HTML files from the wrapper may be placeholders when Kaleido is unavailable.",
            ],
        },
    )
    print(json.dumps({"viewer_html": str(output_dir / "viewer_reart.html"), "urdf_path": str(output_dir / "urdf_reart" / "model.urdf")}, indent=2))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_joints(
    evaluation: dict[str, Any] | None,
    poses: np.ndarray,
    connections: list[Any],
) -> list[dict[str, Any]]:
    if evaluation:
        best = evaluation.get("reart_best_by_gt_score")
        if isinstance(best, dict):
            return [
                {
                    "parent": int(best["parent"]),
                    "child": int(best["child"]),
                    "joint_type": str(best["joint_type"]),
                    "axis": [float(v) for v in best["axis"]],
                    "pivot": [float(v) for v in best["pivot"]],
                    "source": "evaluation_best_by_gt_score",
                }
            ]
    joints = []
    for connection in connections:
        endpoint_a, endpoint_b = [int(v) for v in connection]
        parent, child = select_parent_child(poses, endpoint_a, endpoint_b)
        joints.append(infer_joint_from_poses(poses, parent, child))
    return joints


def select_parent_child(poses: np.ndarray, endpoint_a: int, endpoint_b: int) -> tuple[int, int]:
    score_a = part_motion_score(poses[:, endpoint_a])
    score_b = part_motion_score(poses[:, endpoint_b])
    return (endpoint_a, endpoint_b) if score_a <= score_b else (endpoint_b, endpoint_a)


def part_motion_score(transforms: np.ndarray) -> float:
    translations = transforms[:, :3, 3]
    translation_range = float(np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)))
    angles = [_rotation_angle(transform[:3, :3].tolist()) for transform in transforms]
    return translation_range + max(angles) - min(angles)


def infer_joint_from_poses(poses: np.ndarray, parent: int, child: int) -> dict[str, Any]:
    relative = []
    for frame_idx in range(poses.shape[0]):
        rel = invert_transform(poses[frame_idx, parent]) @ poses[frame_idx, child]
        relative.append({"rotation": rel[:3, :3].tolist(), "translation": rel[:3, 3].tolist()})
    rotation_range, translation_range = motion_ranges(relative)
    joint_type = "revolute" if rotation_range >= 0.2 and rotation_range >= translation_range else "prismatic"
    if joint_type == "revolute":
        axis, pivot, _q, alignment = infer_revolute(relative)
    else:
        axis, pivot, _q = infer_prismatic(relative)
        alignment = None
    return {
        "parent": parent,
        "child": child,
        "joint_type": joint_type,
        "axis": axis,
        "pivot": pivot,
        "rotation_range_rad": rotation_range,
        "translation_range": translation_range,
        "axis_fit_alignment": alignment,
        "source": "pose_heuristic",
    }


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = -(rotation.T @ translation)
    return out


def motion_ranges(relative: list[dict[str, Any]]) -> tuple[float, float]:
    rotations = [
        _rotation_angle(_matmul3(pose["rotation"], _transpose3(relative[0]["rotation"])))
        for pose in relative
    ]
    translations = np.asarray([pose["translation"] for pose in relative], dtype=float)
    return max(rotations) - min(rotations), float(np.linalg.norm(translations.max(axis=0) - translations.min(axis=0)))


def infer_revolute(relative: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], float]:
    # ReArt poses map each canonical part directly into an observed frame. The
    # parent-relative rotations therefore share one hinge axis in the canonical
    # parent frame. Fitting that common axis is substantially more stable than
    # averaging consecutive-frame axes, especially when most increments are
    # only a few milliradians.
    axis_scatter = np.zeros((3, 3), dtype=float)
    valid_rotations: list[tuple[list[list[float]], list[float], float]] = []
    for pose in relative:
        rotation = pose["rotation"]
        angle = _rotation_angle(rotation)
        if angle < 0.01:
            continue
        vote = np.asarray(_rotation_axis(rotation, angle), dtype=float)
        axis_scatter += angle * np.outer(vote, vote)
        valid_rotations.append((rotation, pose["translation"], angle))

    if valid_rotations:
        eigenvalues, eigenvectors = np.linalg.eigh(axis_scatter)
        axis_array = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis = _normalize(axis_array.tolist(), fallback=[0.0, 0.0, 1.0])
        alignment = float(max(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-12))
    else:
        axis = [0.0, 0.0, 1.0]
        alignment = 0.0

    ata = [[0.0, 0.0, 0.0] for _ in range(3)]
    atb = [0.0, 0.0, 0.0]
    for rotation, translation, _angle in valid_rotations:
        block = [
            [1.0 - rotation[row][col] if row == col else -rotation[row][col] for col in range(3)]
            for row in range(3)
        ]
        block_t = _transpose3(block)
        block_ata = _matmul3(block_t, block)
        block_atb = _matvec3(block_t, translation)
        for row in range(3):
            for col in range(3):
                ata[row][col] += block_ata[row][col]
            atb[row] += block_atb[row]
    pivot = _solve_3x3(ata, atb, damping=1e-5)
    # Position along a revolute axis is unobservable. Use the minimum-norm
    # representative so the exported line remains deterministic.
    pivot = _subtract(pivot, _scale(axis, _dot(pivot, axis)))
    q_values = [
        _signed_angle_about_axis(pose["rotation"], axis)
        for pose in relative
    ]
    return axis, pivot, q_values, alignment


def infer_prismatic(relative: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    translations = [pose["translation"] for pose in relative]
    votes = []
    weights = []
    for point_a, point_b in zip(translations[:-1], translations[1:]):
        delta = _subtract(point_b, point_a)
        magnitude = _norm(delta)
        if magnitude < 1e-4:
            continue
        axis = _scale(delta, 1.0 / magnitude)
        if votes and _dot(axis, votes[0]) < 0.0:
            axis = _scale(axis, -1.0)
        votes.append(axis)
        weights.append(magnitude)
    if votes:
        weighted = [0.0, 0.0, 0.0]
        for axis, weight in zip(votes, weights):
            weighted = _add(weighted, _scale(axis, weight))
        axis = _normalize(weighted, fallback=votes[0])
    else:
        axis = [1.0, 0.0, 0.0]
    pivot = _mean_point(translations)
    q_values = [_dot(_subtract(point, translations[0]), axis) for point in translations]
    return axis, pivot, q_values


def sample_indices(count: int, max_count: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count)
    stride = max(1, math.ceil(count / max_count))
    indices = np.arange(0, count, stride)
    return indices[:max_count]


def write_html(
    output_path: Path,
    cano_pc: np.ndarray,
    complete: np.ndarray,
    part_labels: np.ndarray,
    poses: np.ndarray,
    cano_idx: int,
    joints: list[dict[str, Any]],
    *,
    max_points: int,
) -> None:
    indices = sample_indices(len(cano_pc), max_points)
    reconstructed = reconstruct_labelled_frames(cano_pc, complete.shape[0], part_labels, poses, cano_idx)
    frames = []
    for frame_index in range(reconstructed.shape[0]):
        points = reconstructed[frame_index, indices]
        labels = part_labels[indices]
        frames.append([[float(x), float(y), float(z), int(label)] for (x, y, z), label in zip(points, labels)])
    payload = {
        "frames": frames,
        "cano_idx": int(cano_idx),
        "frame_mode": "reconstructed_from_canonical_parts",
        "part_ids": sorted(int(v) for v in np.unique(part_labels)),
        "colors": COLORS,
        "joint": joints[0] if joints else None,
        "joints": joints,
    }
    output_path.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


def reconstruct_labelled_frames(
    cano_pc: np.ndarray,
    frame_count: int,
    part_labels: np.ndarray,
    poses: np.ndarray,
    cano_idx: int,
) -> np.ndarray:
    frames = []
    for frame_index in range(frame_count):
        if frame_index == cano_idx:
            frames.append(cano_pc.copy())
            continue
        pose_index = frame_index if frame_index < cano_idx else frame_index - 1
        frame_points = np.empty_like(cano_pc)
        for part_id in sorted(int(value) for value in np.unique(part_labels)):
            mask = part_labels == part_id
            if part_id < 0 or part_id >= poses.shape[1]:
                frame_points[mask] = cano_pc[mask]
                continue
            transform = poses[pose_index, part_id]
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            frame_points[mask] = cano_pc[mask] @ rotation.T + translation
        frames.append(frame_points)
    return np.stack(frames)


def write_urdf_package(
    output_dir: Path,
    cano_pc: np.ndarray,
    part_labels: np.ndarray,
    joints: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    part_ids = sorted(int(v) for v in np.unique(part_labels))
    mesh_paths = {}
    for part_id in part_ids:
        points = cano_pc[part_labels == part_id]
        mesh_path = mesh_dir / f"part_{part_id}.obj"
        export_part_obj(mesh_path, points)
        mesh_paths[part_id] = mesh_path

    robot = ET.Element("robot", {"name": "reart_visualization"})
    for part_id in part_ids:
        link = ET.SubElement(robot, "link", {"name": f"part_{part_id}"})
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"./meshes/part_{part_id}.obj"})
        material = ET.SubElement(visual, "material", {"name": f"part_{part_id}_mat"})
        rgba = color_rgba(part_id)
        ET.SubElement(material, "color", {"rgba": rgba})

    valid_joints = [
        joint
        for joint in joints
        if joint.get("parent") in part_ids and joint.get("child") in part_ids
    ]
    if valid_joints:
        for edge_index, joint in enumerate(valid_joints):
            joint_type = "revolute" if joint.get("joint_type") == "revolute" else "prismatic"
            parent = int(joint["parent"])
            child = int(joint["child"])
            axis = [float(v) for v in joint["axis"]]
            pivot = [float(v) for v in joint["pivot"]]
            joint_node = ET.SubElement(
                robot,
                "joint",
                {"name": f"joint_{edge_index}_{parent}_{child}", "type": joint_type},
            )
            ET.SubElement(joint_node, "parent", {"link": f"part_{parent}"})
            ET.SubElement(joint_node, "child", {"link": f"part_{child}"})
            ET.SubElement(joint_node, "origin", {"xyz": fmt_vec(pivot), "rpy": "0 0 0"})
            ET.SubElement(joint_node, "axis", {"xyz": fmt_vec(axis)})
            ET.SubElement(
                joint_node,
                "limit",
                {"lower": "-3.14159", "upper": "3.14159", "effort": "1", "velocity": "1"},
            )
    elif len(part_ids) > 1:
        for part_id in part_ids[1:]:
            joint_node = ET.SubElement(robot, "joint", {"name": f"fixed_part_{part_id}", "type": "fixed"})
            ET.SubElement(joint_node, "parent", {"link": f"part_{part_ids[0]}"})
            ET.SubElement(joint_node, "child", {"link": f"part_{part_id}"})
            ET.SubElement(joint_node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    ET.indent(robot)
    ET.ElementTree(robot).write(output_dir / "model.urdf", encoding="utf-8", xml_declaration=True)


def export_part_obj(path: Path, points: np.ndarray) -> None:
    try:
        import trimesh

        if len(points) >= 4:
            mesh = trimesh.PointCloud(points).convex_hull
            mesh.export(path)
            return
    except Exception:
        pass
    with path.open("w", encoding="utf-8") as f:
        for point in points:
            f.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")


def color_rgba(part_id: int) -> str:
    hex_color = COLORS[part_id % len(COLORS)].lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return f"{r:.4f} {g:.4f} {b:.4f} 0.85"


def fmt_vec(vec: list[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in vec)


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ReArt Point Cloud Viewer</title>
  <style>
    body { margin: 0; background: #101418; color: #d8e2ea; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
    aside { padding: 16px; background: #18212a; border-right: 1px solid #2a3845; overflow: auto; }
    main { padding: 16px; display: grid; grid-template-rows: auto 1fr; gap: 12px; }
    canvas { background: #070a0d; border: 1px solid #2a3845; border-radius: 8px; width: 100%; height: 100%; }
    input[type=range] { width: 100%; }
    .row { margin: 12px 0; }
    .legend { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
    .swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    code { color: #9fd0ff; overflow-wrap: anywhere; }
  </style>
</head>
<body>
<div id="app">
  <aside>
    <h2>ReArt Viewer</h2>
    <div class="row">Frame <strong id="frameLabel">0</strong> / <span id="frameMax">0</span></div>
    <input id="frame" type="range" min="0" max="0" value="0" />
    <div class="row">
      <label><input id="showJoint" type="checkbox" checked /> show joint axis</label>
    </div>
    <h3>Parts</h3>
    <div id="legend"></div>
    <h3>Joint</h3>
    <pre id="jointText"></pre>
  </aside>
  <main>
    <div>Drag to rotate, wheel to zoom. Colors are ReArt canonical part labels replayed through predicted part poses. Canonical frame: <code id="canoText"></code>.</div>
    <canvas id="canvas"></canvas>
  </main>
</div>
<script>
const DATA = __PAYLOAD__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('frame');
const label = document.getElementById('frameLabel');
const maxLabel = document.getElementById('frameMax');
const showJoint = document.getElementById('showJoint');
let yaw = -0.8, pitch = 0.55, zoom = 1.15;
let dragging = false, last = null;

slider.max = DATA.frames.length - 1;
slider.value = DATA.cano_idx || 0;
maxLabel.textContent = DATA.frames.length - 1;
document.getElementById('legend').innerHTML = DATA.part_ids.map(id => `<div class="legend"><span class="swatch" style="background:${DATA.colors[id % DATA.colors.length]}"></span>part_${id}</div>`).join('');
document.getElementById('jointText').textContent = DATA.joints.length ? JSON.stringify(DATA.joints, null, 2) : 'No joint.';
document.getElementById('canoText').textContent = DATA.cano_idx || 0;

function resize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * devicePixelRatio);
  canvas.height = Math.floor(rect.height * devicePixelRatio);
  draw();
}
window.addEventListener('resize', resize);
slider.addEventListener('input', draw);
showJoint.addEventListener('change', draw);
canvas.addEventListener('mousedown', e => { dragging = true; last = [e.clientX, e.clientY]; });
window.addEventListener('mouseup', () => { dragging = false; });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  yaw += (e.clientX - last[0]) * 0.008;
  pitch += (e.clientY - last[1]) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  last = [e.clientX, e.clientY];
  draw();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  draw();
}, {passive:false});

function rotate(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = cy*p[0] + sy*p[2];
  const z = -sy*p[0] + cy*p[2];
  const y = cp*p[1] - sp*z;
  const zz = sp*p[1] + cp*z;
  return [x, y, zz];
}
function project(p, scale, cx, cy) {
  const r = rotate(p);
  return [cx + r[0]*scale, cy - r[1]*scale, r[2]];
}
function drawLine3(a, b, color, scale, cx, cy) {
  const pa = project(a, scale, cx, cy);
  const pb = project(b, scale, cx, cy);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3 * devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
}
function draw() {
  const idx = Number(slider.value);
  label.textContent = idx;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pts = DATA.frames[idx];
  let maxAbs = 1e-6;
  for (const p of pts) maxAbs = Math.max(maxAbs, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
  const scale = Math.min(canvas.width, canvas.height) * 0.42 * zoom / maxAbs;
  const cx = canvas.width/2, cy = canvas.height/2;
  const sorted = pts.map(p => ({p, q: rotate(p)[2]})).sort((a,b)=>a.q-b.q);
  for (const item of sorted) {
    const p = item.p;
    const pr = project(p, scale, cx, cy);
    ctx.fillStyle = DATA.colors[p[3] % DATA.colors.length];
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.arc(pr[0], pr[1], 2.0*devicePixelRatio, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (showJoint.checked && DATA.joints.length) {
    const axisColors = ['#ff4444', '#ffcc33', '#3dd6ff', '#ff72c6', '#ffffff'];
    DATA.joints.forEach((joint, index) => {
      const pivot = joint.pivot;
      const axis = joint.axis;
      const len = maxAbs * 0.65;
      drawLine3([pivot[0]-axis[0]*len, pivot[1]-axis[1]*len, pivot[2]-axis[2]*len],
                [pivot[0]+axis[0]*len, pivot[1]+axis[1]*len, pivot[2]+axis[2]*len],
                axisColors[index % axisColors.length], scale, cx, cy);
    });
  }
}
resize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
