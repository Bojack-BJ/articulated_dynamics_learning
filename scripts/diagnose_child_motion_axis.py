#!/usr/bin/env python3
"""Diagnose axis observability from GT-selected child track correspondences."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def angular_error(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(np.clip(abs(a @ b), -1.0, 1.0)))


def kabsch(source, target):
    x = source - source.mean(0)
    y = target - target.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    import mujoco

    rows = []
    for joint_path in sorted(args.root.glob("*/yaw_*/*/joints.json")):
        run = joint_path.parent
        episode = json.loads((joint_path.parents[1] / "episode.json").read_text())
        tracks = json.loads((run / "tracks.json").read_text())["tracks"]
        prediction = json.loads(joint_path.read_text())
        model = mujoco.MjModel.from_xml_path(str(joint_path.parents[1] / "model.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        joints = [jid for jid in range(model.njnt) if model.jnt_type[jid] in
                  (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        if len(joints) != 1:
            continue
        jid = joints[0]
        gt_type = "revolute" if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
        gt_axis = np.asarray(data.xaxis[jid])
        child_body = int(model.jnt_bodyid[jid])
        parts = episode["metadata"]["part_segmentation"]["parts"]
        child_part = next(int(p["part_id"]) for p in parts if child_body in [int(p["body_id"])] )
        selected = []
        for track in tracks:
            view = int(track["view_index"])
            frame = int(track["query_source_frame_index"])
            mask_path = joint_path.parents[1] / f"assets/view_{view}/frame_{frame:04d}_part_mask.png"
            mask = np.asarray(Image.open(mask_path))
            u, v = np.rint(track["query_uv"]).astype(int)
            if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and int(mask[v, u]) == child_part:
                selected.append(track)
        candidates = []
        if gt_type == "prismatic":
            displacements = []
            for track in selected:
                visible = [s for s in track["samples"] if s["visible"] and s.get("xyz_world") is not None]
                if len(visible) >= 2:
                    d = np.asarray(visible[-1]["xyz_world"]) - np.asarray(visible[0]["xyz_world"])
                    if np.linalg.norm(d) > 1e-4:
                        displacements.append(d / np.linalg.norm(d))
            if displacements:
                moment = np.asarray(displacements).T @ np.asarray(displacements)
                candidates.append(np.linalg.eigh(moment)[1][:, -1])
        else:
            frame_count = len(selected[0]["samples"]) if selected else 0
            for frame in range(1, frame_count):
                source, target = [], []
                for track in selected:
                    s0, st = track["samples"][0], track["samples"][frame]
                    if s0["visible"] and st["visible"] and s0.get("xyz_world") is not None and st.get("xyz_world") is not None:
                        source.append(s0["xyz_world"]); target.append(st["xyz_world"])
                if len(source) >= 6:
                    rv = Rotation.from_matrix(kabsch(np.asarray(source), np.asarray(target))).as_rotvec()
                    if np.linalg.norm(rv) > math.radians(1):
                        candidates.append(rv / np.linalg.norm(rv))
            if candidates:
                moment = np.asarray(candidates).T @ np.asarray(candidates)
                candidates = [np.linalg.eigh(moment)[1][:, -1]]
        learned = prediction.get("selected_edges", [])
        row = {
            "run": str(joint_path.relative_to(args.root)), "gt_type": gt_type,
            "child_tracks": len(selected), "motion_fit_valid": bool(candidates),
            "motion_fit_axis_error_deg": angular_error(candidates[0], gt_axis) if candidates else None,
            "learned_edge_count": len(learned),
            "learned_type": learned[0]["joint_type"] if len(learned) == 1 else None,
            "learned_axis_error_deg": angular_error(learned[0]["axis_world"], gt_axis) if len(learned) == 1 else None,
        }
        rows.append(row)
        print(json.dumps(row))
    (args.root / "child_motion_axis_diagnostic.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
