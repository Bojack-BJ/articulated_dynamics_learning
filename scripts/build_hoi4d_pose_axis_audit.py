#!/usr/bin/env python3
"""Fit and visualize HOI4D joint axes from official per-part pose trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROLE_NAMES = {
    "C3": ("keyboard", "display", "revolute"),
    "C6": ("box", "door", "revolute"),
    "C14": ("base", "cover", "revolute"),
}


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def pose_item(payload: dict, needle: str) -> dict | None:
    needle = needle.lower()
    return next((x for x in payload.get("dataList", []) if needle in str(x.get("label", "")).lower()), None)


def fit(
    row: dict[str, str],
    annotation_root: Path,
    raw_root: Path,
    *,
    end_frame: int | None = None,
) -> dict:
    category = row["category"]
    if category == "C4":
        parent_name = "body"
        child_name = "drawer" if row["motion"] == "open_close_drawer" else "sldingdoor"
        joint_type = "prismatic" if row["motion"] == "open_close_drawer" else "revolute"
    else:
        parent_name, child_name, joint_type = ROLE_NAMES[category]
    samples = []
    for path in sorted((annotation_root / row["sequence"] / "objpose").glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("isEffective"):
            continue
        if end_frame is not None and int(payload["frameId"]) - 1 >= end_frame:
            continue
        parent, child = pose_item(payload, parent_name), pose_item(payload, child_name)
        if parent is None or child is None:
            continue
        def vector(item: dict, key: str) -> np.ndarray:
            value = item[key]
            return np.array([value[axis] for axis in "xyz"], dtype=float)
        samples.append((int(payload["frameId"]), vector(parent, "center"), vector(parent, "rotation"), vector(child, "center"), vector(child, "rotation")))
    if len(samples) < 3:
        raise ValueError(f"Insufficient pose samples for {row['sequence']}")
    samples.sort(key=lambda sample: sample[0])
    frames = np.array([x[0] for x in samples])
    parent_centers = np.stack([x[1] for x in samples])
    child_centers = np.stack([x[3] for x in samples])
    info_path = raw_root / row["sequence"].replace("/", "_") / "camera/recon/split_0/info.json"
    extrinsics = np.asarray(json.loads(info_path.read_text())["extrinsics"], dtype=float)
    selected_extrinsics = extrinsics[np.clip(frames - 1, 0, len(extrinsics) - 1)]
    design, targets = [], []
    for matrix, center in zip(selected_extrinsics, parent_centers):
        for coordinate in range(3):
            equation = np.zeros(4)
            equation[:3] = matrix[coordinate, :3]
            equation[3] = matrix[coordinate, 3]
            design.append(equation)
            targets.append(center[coordinate])
    slam_scale = float(np.linalg.lstsq(np.stack(design), np.asarray(targets), rcond=None)[0][3])
    metric_extrinsics = selected_extrinsics.copy()
    metric_extrinsics[:, :3, 3] *= slam_scale
    camera_to_world = np.linalg.inv(metric_extrinsics)

    def transform_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        return np.einsum("nij,nj->ni", transforms, homogeneous)[:, :3]

    parent_world = transform_points(camera_to_world, parent_centers)
    child_world = transform_points(camera_to_world, child_centers)

    candidates = []
    for order in ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"):
        parent_rot = Rotation.from_euler(order, np.stack([x[2] for x in samples]))
        child_rot = Rotation.from_euler(order, np.stack([x[4] for x in samples]))
        relative = parent_rot.inv() * child_rot
        # Relative child orientation is expressed in the parent frame. Left
        # composition keeps the recovered hinge axis in that same frame.
        delta = relative * relative[0].inv()
        rotvec = delta.as_rotvec()
        covariance = rotvec.T @ rotvec
        values, vectors = np.linalg.eigh(covariance)
        axis_parent = normalized(vectors[:, -1])
        concentration = float(values[-1] / max(values.sum(), 1e-12))
        motion = float(np.percentile(np.abs(rotvec @ axis_parent), 95))
        candidates.append((concentration * motion, order, parent_rot, axis_parent, concentration, motion))
    _, order, parent_rot, rotation_axis, concentration, angular_motion = max(candidates, key=lambda x: x[0])

    relative_centers = parent_rot.inv().apply(child_centers - parent_centers)
    relative_world_motion = (child_world - child_world[0]) - (parent_world - parent_world[0])
    centered = relative_world_motion - np.median(relative_world_motion, axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    translation_axis = normalized(vectors[:, -1])
    translation_concentration = float(values[-1] / max(values.sum(), 1e-12))
    translation_motion = float(np.percentile(centered @ translation_axis, 95) - np.percentile(centered @ translation_axis, 5))
    axis_parent = rotation_axis
    axis_world = translation_axis if joint_type == "prismatic" else normalized(
        camera_to_world[0, :3, :3] @ parent_rot[0].apply(axis_parent)
    )
    if joint_type == "revolute":
        relative = parent_rot.inv() * Rotation.from_euler(order, np.stack([x[4] for x in samples]))
        delta_matrices = (relative * relative[0].inv()).as_matrix()
        center0 = relative_centers[0]
        design, targets = [], []
        for matrix, center in zip(delta_matrices, relative_centers):
            design.append(np.eye(3) - matrix)
            targets.append(center - matrix @ center0)
        # Fix the unobservable position along the axis to the moving-center
        # projection, yielding the closest canonical point on the hinge line.
        design.append(axis_parent[None, :])
        targets.append(np.array([float(axis_parent @ np.median(relative_centers, axis=0))]))
        pivot_parent = np.linalg.lstsq(np.concatenate(design), np.concatenate(targets), rcond=None)[0]
        pivot_camera = parent_centers[0] + parent_rot[0].apply(pivot_parent)
        pivot = transform_points(camera_to_world[:1], pivot_camera[None, :])[0]
    else:
        pivot = child_world[0]
    observable = (translation_motion >= 0.015 and translation_concentration >= 0.7) if joint_type == "prismatic" else (angular_motion >= np.deg2rad(5) and concentration >= 0.7)
    return {
        "sequence": row["sequence"], "split": row["split"], "category": category,
        "motion": row["motion"], "joint_type": joint_type, "parent_name": parent_name,
        "child_name": child_name, "axis": axis_world.tolist(), "pivot": pivot.tolist(),
        "euler_order": order, "rotation_concentration": concentration,
        "angular_motion_deg": float(np.rad2deg(angular_motion)),
        "translation_concentration": translation_concentration,
        "translation_motion_m": translation_motion, "observable": bool(observable),
        "slam_translation_scale_to_m": slam_scale,
        "frames": frames.tolist(), "parent_centers": parent_centers.tolist(),
        "child_centers": child_centers.tolist(), "source_frame_range": [0, end_frame],
    }


def relation_gt(result: dict) -> dict:
    joint = {
        "name": "hoi4d_pose_joint", "joint_type": result["joint_type"],
        "parent_part_id": 0, "child_part_id": 1, "axis": result["axis"],
        "source": "hoi4d_official_objpose_full_trajectory",
        "observability": "valid" if result["observable"] else "low",
    }
    if result["joint_type"] == "revolute":
        joint["pivot"] = result["pivot"]
    return {"joints": [joint], "metadata": {key: result[key] for key in ("sequence", "euler_order", "rotation_concentration", "angular_motion_deg", "translation_concentration", "translation_motion_m", "observable")}}


def write_html(results: list[dict], output: Path) -> None:
    payload = json.dumps(results, separators=(",", ":"))
    output.write_text(f'''<!doctype html><meta charset="utf-8"><title>HOI4D axis audit</title>
<style>body{{margin:0;background:#091018;color:#edf5f8;font:14px ui-monospace,monospace}}header{{padding:12px 18px;background:#111d27;display:flex;gap:14px;align-items:center}}select,a{{color:#51d6ad;background:#111d27}}#meta{{white-space:pre-wrap;color:#a9bac5}}#plot{{height:calc(100vh - 95px)}}</style>
<header><b>HOI4D priority-15 pose-axis audit</b><select id="pick"></select><a id="mask" target="_blank">mask review</a><span id="meta"></span></header><div id="plot"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script>const D={payload},pick=document.querySelector('#pick');D.forEach((x,i)=>pick.add(new Option(`${{x.observable?'PASS':'LOW'}} | ${{x.sequence}}`,i)));
function draw(){{const x=D[+pick.value],p=x.parent_centers,c=x.child_centers,a=x.axis,q=x.pivot,L=.35;document.querySelector('#meta').textContent=`${{x.joint_type}} | motion=${{x.joint_type==='revolute'?x.angular_motion_deg.toFixed(1)+'deg':x.translation_motion_m.toFixed(3)+'m'}} | concentration=${{(x.joint_type==='revolute'?x.rotation_concentration:x.translation_concentration).toFixed(3)}} | Euler ${{x.euler_order}}`;document.querySelector('#mask').href=x.sequence.replaceAll('/','_')+'.html';Plotly.react('plot',[{{x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),mode:'lines+markers',name:'parent center'}},{{x:c.map(v=>v[0]),y:c.map(v=>v[1]),z:c.map(v=>v[2]),mode:'lines+markers',name:'moving center'}},{{x:[q[0]-L*a[0],q[0]+L*a[0]],y:[q[1]-L*a[1],q[1]+L*a[1]],z:[q[2]-L*a[2],q[2]+L*a[2]],mode:'lines',line:{{width:8,color:'#ff913d'}},name:'fitted GT axis'}}],{{paper_bgcolor:'#091018',plot_bgcolor:'#091018',font:{{color:'#edf5f8'}},scene:{{aspectmode:'data'}},margin:{{l:0,r:0,b:0,t:0}}}})}}pick.onchange=draw;draw();</script>''')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path); parser.add_argument("annotation_root", type=Path)
    parser.add_argument("raw_root", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--end-frame", action="append", default=[], metavar="SEQUENCE=EXCLUSIVE_END")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    end_frames = {}
    for value in args.end_frame:
        name, separator, raw_end = value.rpartition("=")
        if not separator or not name:
            parser.error(f"invalid --end-frame value: {value!r}")
        end_frames[name] = int(raw_end)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [x for x in csv.DictReader(handle) if x["category"] in ROLE_NAMES or x["category"] == "C4"]
    results = [
        fit(
            row,
            args.annotation_root,
            args.raw_root,
            end_frame=end_frames.get(row["sequence"].replace("/", "_")),
        )
        for row in rows
    ]
    for result in results:
        path = args.output_dir / result["sequence"].replace("/", "_") / "relation_gt.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(relation_gt(result), indent=2) + "\n")
    (args.output_dir / "axis_audit.json").write_text(json.dumps({"sequences": results}, indent=2) + "\n")
    write_html(results, args.output_dir / "index.html")
    print(json.dumps({"count": len(results), "observable": sum(x["observable"] for x in results), "index": str(args.output_dir / "index.html")}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
