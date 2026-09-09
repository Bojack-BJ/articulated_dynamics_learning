#!/usr/bin/env python3
"""Evaluate retained external joints using geometry-only part association.

Output is versioned separately from historical angle-matched metrics. GT mesh
points are sampled deterministically at the native two-state start pose.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

PALETTE = [(78,205,196),(255,107,107),(196,181,253),(250,204,21),
           (251,146,60),(244,114,182),(74,222,128),(96,165,250)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def part_mapping(points, labels, reference, gt_labels, bbox, min_iou=0.0):
    """One-to-one IoU matching, with no joint type or geometry in the cost."""
    distance, nearest = cKDTree(points).query(reference)
    transferred = np.asarray(labels)[nearest].copy()
    transferred[distance > 0.02 * bbox] = -99999
    pred_ids = np.unique(labels)
    gt_ids = np.unique(gt_labels)
    iou = np.zeros((len(pred_ids), len(gt_ids)))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            a, b = transferred == p, gt_labels == g
            iou[i, j] = np.sum(a & b) / max(1, np.sum(a | b))
    rows, cols = linear_sum_assignment(-iou)
    mapping = {int(pred_ids[i]): int(gt_ids[j]) for i, j in zip(rows, cols)
               if iou[i, j] > min_iou}
    return mapping, {"pred_ids": pred_ids.tolist(), "gt_ids": gt_ids.tolist(),
                     "iou": iou.tolist(), "geometry_coverage": float(np.mean(distance <= .02*bbox))}


def unit(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-9:
        return None
    return a / np.linalg.norm(a)


def joint_metrics(pred, gt, mapping, bbox):
    results = []
    for target in gt:
        candidates = [p for p in pred if mapping.get(p["child"]) == target["child"]]
        # Multiple edges on one recovered child are ambiguous; never choose by GT angle.
        p = candidates[0] if len(candidates) == 1 else None
        correct = p is not None and p["type"] == target["type"]
        a = unit(p.get("axis")) if p is not None else None
        b = unit(target["axis"])
        angle = float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))) if correct and a is not None and b is not None else None
        line = None
        if angle is not None and target["type"] == "revolute" and p.get("origin") is not None:
            delta = np.asarray(p["origin"]) - target["origin"]
            cross = np.cross(a, b)
            line = float(abs(delta @ cross) / np.linalg.norm(cross) if np.linalg.norm(cross) > 1e-8 else np.linalg.norm(np.cross(delta, b))) / bbox
        results.append({"gt_joint": target["name"], "gt_child": target["child"],
                        "gt_type": target["type"], "predicted_type": p["type"] if p else None,
                        "matched": p is not None, "ambiguous": len(candidates) > 1,
                        "type_correct": bool(correct), "axis_deg": angle,
                        "line_bbox": line, "axis_penalty_deg": angle if angle is not None else 90.0,
                        "joint_at20": angle is not None and angle <= 20})
    return results


def ground_truth(root, object_id, samples=4000, episode_path=None, joint_positions=None):
    import mujoco
    import trimesh
    obj = root / "outputs/external_baseline_suite_v1/per_object" / object_id
    package = read(obj / "acquisition/two_state/package_manifest.json")
    episode = read(episode_path or package["source_episode"])
    model = mujoco.MjModel.from_xml_path(episode["metadata"]["model_path"])
    data = mujoco.MjData(model)
    for name, value in (joint_positions if joint_positions is not None else package["states"]["start"]["joint_positions"]).items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"Unknown GT joint {name}")
        data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    domain = read(root / "outputs/external_baseline_suite_v1/kinematic_domain_revaluation_v1" / object_id / "kinematic_evaluation_domain.json")
    parts = domain["part_segmentation"]["parts"] if "part_segmentation" in domain else domain["normalized_part_segmentation"]["parts"]
    body_part = {}
    for part in parts:
        for name in part["metadata"].get("member_body_names", [part["body_name"]]):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            body_part[body] = part["part_id"]
    chunks, labels = [], []
    allowed = set(episode["metadata"].get("target_geom_ids", []))
    hidden = set(episode["metadata"].get("hidden_clear_geom_ids", []))
    for g in range(model.ngeom):
        mesh_id = int(model.geom_dataid[g])
        body = int(model.geom_bodyid[g])
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0 or body not in body_part or g in hidden or (allowed and g not in allowed):
            continue
        v0, n = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[v0:v0+n] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
        mesh = trimesh.Trimesh(vertices=vertices, faces=model.mesh_face[f0:f0+nf], process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=20260909+g)
        chunks.append(pts)
        labels.extend([body_part[body]] * len(pts))
    points, labels = np.concatenate(chunks), np.asarray(labels)
    bbox = float(np.linalg.norm(np.ptp(points, axis=0)))
    joints = []
    joint_names = {name for part in parts for name in part.get("joint_names", [])}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        child = body_part.get(int(model.jnt_bodyid[j]))
        if name not in joint_names or child is None or int(model.jnt_type[j]) not in (2, 3):
            continue
        joints.append({"name": name,
                       "child": child, "type": "revolute" if model.jnt_type[j] == 3 else "prismatic",
                       "axis": data.xaxis[j].tolist(), "origin": data.xanchor[j].tolist()})
    return points, labels, bbox, joints


def native_prediction(root, object_id, method):
    from rgbd_urdf_mvp.perception.baseline_viewer import _read_ply, _reart_joint_axes, _AxisRemap
    if method == "aim":
        run = root/"outputs/external_baselines_v1/aim_style_aligned_runs"/object_id
        motions = read(run/"motion.json")
        chunks, labels, joints = [], [], []
        for i, motion in enumerate(motions):
            points, _, _ = _read_ply(run/f"sub_{i}_point_cloud_start.ply")
            chunks.append(points)
            labels.extend([i]*len(points))
            if i:
                joints.append({"child":i, "type":"prismatic" if int(motion["motion_type"]) == 0 else "revolute",
                               "axis":motion["axis"], "origin":motion.get("center")})
        episode = root/"outputs/aim_style_protocol_v1_recordings"/object_id/"episode.json"
        qpos = read(episode.parent/"aim_protocol/static_scan/cameras.json")["joint_positions"]
        return np.concatenate(chunks), np.asarray(labels), joints, episode, qpos
    import pickle
    run = root/"outputs/external_baselines_v1/reart_runs_aligned_4x512"/object_id/object_id/"result.pkl"
    with run.open("rb") as stream:
        payload = pickle.load(stream)
    points = np.asarray(payload["cano_pc"])
    labels = np.asarray(payload["pred_cano_part"])
    poses = np.asarray(payload["pred_pose_list"])
    canonical = int(payload["cano_idx"])
    identity = np.broadcast_to(np.eye(4), (1, poses.shape[1], 4, 4))
    poses = np.concatenate([poses[:canonical], identity, poses[canonical:]], axis=0)
    native = _reart_joint_axes(poses, payload["joint_connection"], points=points,
                               predicted=labels, transform=_AxisRemap.from_raw("x,y,z"))
    joints = []
    for joint in native:
        parent = joint["parent_part_id"]
        # The converter fits in parent coordinates. Bring the line to the
        # canonical observation frame used by the point labels and GT.
        pose = poses[canonical, parent]
        joints.append({"child":joint["child_part_id"], "type":joint["joint_type"],
                       "axis":(pose[:3,:3]@joint["axis"]).tolist(),
                       "origin":(pose[:3,:3]@joint["origin"]+pose[:3,3]).tolist()})
    manifest = read(root/"outputs/external_baselines_v1/reart_sequences_dense20"/object_id/"reart_sequence_manifest.json")
    source = Path(manifest["frames"][canonical]["source_frame_path"])
    episode = source.parents[2]/"episode.json"
    frame = int(source.stem.split("_")[-1])
    qpos = read(episode)["frames"][frame]["action_log"]["joint_positions"]
    return points, labels, joints, episode, qpos


def load_prediction(obj, method, adapter_path=None):
    from rgbd_urdf_mvp.perception.baseline_viewer import _read_ply
    adapter = adapter_path or obj / method / "adapter"
    payload = read(adapter / "predictions.json")
    file = payload["states"][0]["ply"] if method == "artgs" else payload["labeled_point_cloud"]
    points, colors, extra = _read_ply(adapter / Path(file).name)
    # Preserve exporter slot IDs; sorting RGB tuples permutes joint associations.
    palette = PALETTE
    if method == "videoartgs":
        palette = [(78,205,196),(255,107,107),(190,174,255),(255,196,61),
                   (57,211,167),(255,132,51),(235,88,139),(88,166,255)]
    elif method == "paris":
        palette = [(91,192,235),(255,126,95)]
    lookup = {tuple(c): i for i, c in enumerate(palette)}
    labels = np.asarray([lookup[tuple(map(int,c))] for c in colors])
    joints = payload.get("joints", [])
    if method == "dta":
        selected = read(adapter / "replay_model_selection.json")["selections"]
        joints = []
        for s in selected:
            t = s["selected_type"]
            idx = s["joint_index"]
            if t == "ambiguous":
                joints.append({"child": idx+1, "type": "ambiguous"})
            else:
                joints.append({**payload["joint_hypotheses"][t][idx], "child": idx+1, "type":t})
    result = []
    for i, joint in enumerate(joints):
        t = {"r":"revolute", "p":"prismatic"}.get(joint.get("type"), joint.get("type"))
        result.append({"child": int(joint.get("child", i+1)), "type":t,
                       "axis":joint.get("axis_direction"), "origin":joint.get("axis_position")})
    return points, labels, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--objects", nargs="*")
    p.add_argument("--methods", nargs="+", default=["artgs","videoartgs","paris","dta"])
    args = p.parse_args()
    suite = args.root / "outputs/external_baseline_suite_v1"
    objects = args.objects or [r["object_id"] for r in csv.DictReader((suite/"manifest.csv").open())]
    rows, failures = [], []
    for object_id in objects:
        try:
            ref, gt_labels, bbox, gt = ground_truth(args.root, object_id)
        except Exception as exc:
            failures.append({"object":object_id, "method":"gt", "error":str(exc)})
            print(f"{object_id}: GT failed: {exc}", flush=True)
            continue
        for method in args.methods:
            try:
                method_ref, method_labels, method_bbox, method_gt = ref, gt_labels, bbox, gt
                if method in {"aim", "reart"}:
                    points, labels, joints, episode, qpos = native_prediction(args.root, object_id, method)
                    method_ref, method_labels, method_bbox, method_gt = ground_truth(args.root, object_id, episode_path=episode, joint_positions=qpos)
                else:
                    points, labels, joints = load_prediction(suite/"per_object"/object_id, method)
                mapping, overlap = part_mapping(points, labels, method_ref, method_labels, method_bbox)
                metrics = joint_metrics(joints, method_gt, mapping, method_bbox)
                record = {"object":object_id,"method":method,"mapping":mapping,"overlap":overlap,
                          "predicted_joints":joints,"gt_joints":method_gt,"bbox":method_bbox,"metrics":metrics}
                write(args.output/object_id/f"{method}.json",record)
                rows.extend({"object":object_id,"method":method,**m} for m in metrics)
                print(f"{object_id} {method}: {len(mapping)} parts, {sum(m['matched'] for m in metrics)}/{len(gt)} joints",flush=True)
            except Exception as exc:
                failures.append({"object":object_id,"method":method,"error":str(exc)})
    summaries=[]
    for method in args.methods:
        subset=[r for r in rows if r["method"]==method]
        if not subset:
            continue
        angles=[r["axis_deg"] for r in subset if r["axis_deg"] is not None]
        lines=[r["line_bbox"] for r in subset if r["line_bbox"] is not None]
        summaries.append({"method":method,"objects_with_output":len({r['object'] for r in subset}),
                          "gt_joints_on_output_objects":len(subset),"matched_joints":sum(r['matched'] for r in subset),
                          "type_correct_joints":sum(r['type_correct'] for r in subset),"axis_valid_joints":len(angles),
                          "axis_mean":float(np.mean(angles)) if angles else None,
                          "axis_median":float(np.median(angles)) if angles else None,
                          "line_bbox_mean":float(np.mean(lines)) if lines else None,
                          "penalized_axis_output_objects":float(np.mean([r['axis_penalty_deg'] for r in subset])),
                          "joint_at20_output_objects":float(np.mean([r['joint_at20'] for r in subset]))})
    write(args.output/"summary.json",{"protocol":"geometry-only Hungarian; 2% bbox support; native start pose GT mesh; output-object denominator", "summary":summaries,"failures":failures})
    if rows:
        with (args.output/"per_joint.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
