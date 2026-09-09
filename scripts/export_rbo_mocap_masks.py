#!/usr/bin/env python3
"""Project RBO mocap-aligned link meshes into registered RGB images."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.spatial.transform import Rotation


ORIGIN_RE = re.compile(r"xyz='([^']+)' rpy='([^']+)'")


def transform(xyz, quat_xyzw=None, rpy=None):
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    if quat_xyzw is not None:
        out[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    elif rpy is not None:
        out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return out


def parse_config(path: Path):
    text = path.read_text()
    link_ids = ast.literal_eval(re.search(r"^link_ids:\s*(\{.*\})", text, re.M).group(1))
    meshes = ast.literal_eval(re.search(r"^meshes:\s*(\{.*\})", text, re.M).group(1))
    poses_raw = ast.literal_eval(re.search(r"^mesh_poses:\s*(\{.*\})", text, re.M).group(1))
    poses = {}
    for link, value in poses_raw.items():
        match = ORIGIN_RE.search(value)
        poses[link] = transform(
            [float(x) for x in match.group(1).split()],
            rpy=[float(x) for x in match.group(2).split()],
        )
    return {int(k): f"rb{v}" for k, v in link_ids.items()}, meshes, poses


def choose_config(model_dir: Path, timestamp_s: float) -> Path:
    stamp = datetime.fromtimestamp(timestamp_s, tz=timezone.utc).date()
    configs = sorted(model_dir.glob("configuration_*/*_in.yaml"))
    return min(configs, key=lambda p: abs((datetime.strptime(p.parent.name[14:], "%Y-%m-%d").date() - stamp).days))


def read_intrinsics(path: Path):
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    # ROS CameraInfo stores K as field.K0 ... field.K8.
    return tuple(float(row[f"field.K{i}"]) for i in (0, 4, 2, 5))


def read_pose_rows(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            poses = {}
            index = 0
            while f"field.markers{index}.id" in row:
                prefix = f"field.markers{index}."
                if row[prefix + "id"]:
                    body_id = int(row[prefix + "id"])
                    xyz = [float(row[prefix + f"pose.position.{a}"]) for a in "xyz"]
                    quat = [float(row[prefix + f"pose.orientation.{a}"]) for a in "xyzw"]
                    poses[body_id] = transform(xyz, quat_xyzw=quat)
                index += 1
            rows.append((int(row["%time"]) * 1e-9, poses))
    return rows


def read_camera_rows(path: Path):
    dynamic = []
    static = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parent = row["field.transforms0.header.frame_id"].lstrip("/")
            child = row["field.transforms0.child_frame_id"].lstrip("/")
            xyz = [float(row[f"field.transforms0.transform.translation.{a}"]) for a in "xyz"]
            quat = [float(row[f"field.transforms0.transform.rotation.{a}"]) for a in "xyzw"]
            item = (int(row["field.transforms0.header.stamp"]) * 1e-9, transform(xyz, quat_xyzw=quat))
            if (parent, child) == ("map", "AsusXtionCameraFrame"):
                dynamic.append(item)
            elif (parent, child) in {
                ("AsusXtionCameraFrame", "camera_link"),
                ("camera_link", "camera_rgb_frame"),
                ("camera_rgb_frame", "camera_rgb_optical_frame"),
            }:
                static[(parent, child)] = item[1]
    chain = (
        static[("AsusXtionCameraFrame", "camera_link")]
        @ static[("camera_link", "camera_rgb_frame")]
        @ static[("camera_rgb_frame", "camera_rgb_optical_frame")]
    )
    return dynamic, chain


def nearest(rows, timestamp_s):
    return min(rows, key=lambda item: abs(item[0] - timestamp_s))[1]


def nearest_complete_pose(rows, timestamp_s, required_ids):
    required = set(required_ids)
    complete = [item for item in rows if required.issubset(item[1])]
    if not complete:
        raise ValueError(f"No mocap packet contains rigid bodies {sorted(required)}")
    return nearest(complete, timestamp_s)


def timestamp_from_name(path: Path):
    return float(path.stem.split("-", 1)[1])


def load_depth(sequence: Path, timestamp_s: float):
    files = list((sequence / "camera_depth_registered").glob("*.txt"))
    path = min(files, key=lambda p: abs(timestamp_from_name(p) - timestamp_s))
    depth = np.loadtxt(path, dtype=np.float32)
    if np.nanmedian(depth[depth > 0]) > 20:
        depth *= 0.001
    return depth


def surface_points(mesh_path: Path, count: int, seed: int):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, count)
    return np.asarray(points, dtype=np.float64)


def render_labels(points_by_part, camera_from_map, intrinsics, shape, observed_depth, tolerance):
    height, width = shape
    fx, fy, cx, cy = intrinsics
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.uint16)
    for part_id, points_map in points_by_part:
        points = (camera_from_map[:3, :3] @ points_map.T).T + camera_from_map[:3, 3]
        z = points[:, 2]
        valid = z > 0.05
        u = np.rint(fx * points[:, 0] / z + cx).astype(np.int32)
        v = np.rint(fy * points[:, 1] / z + cy).astype(np.int32)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx = np.flatnonzero(valid)
        order = idx[np.argsort(z[idx])[::-1]]
        for i in order:
            if z[i] < zbuf[v[i], u[i]]:
                zbuf[v[i], u[i]] = z[i]
                labels[v[i], u[i]] = part_id
    visible = (observed_depth <= 0) | (zbuf <= observed_depth + tolerance)
    labels[~visible] = 0
    result = np.zeros_like(labels)
    for part_id in np.unique(labels):
        if part_id:
            result[binary_closing(labels == part_id, structure=np.ones((3, 3), dtype=bool))] = part_id
    return result, zbuf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--samples-per-link", type=int, default=300_000)
    parser.add_argument("--depth-tolerance", type=float, default=0.04)
    args = parser.parse_args()

    rgb_files = sorted((args.sequence / "camera_rgb").glob("*.png"))[:: args.frame_stride]
    first_time = timestamp_from_name(rgb_files[0])
    category = re.sub(r"\d+_o$", "", args.sequence.name)
    model_dir = args.model_root / category
    config = choose_config(model_dir, first_time)
    link_ids, meshes, mesh_poses = parse_config(config)
    body_rows = read_pose_rows(args.sequence / "rb_poses_array.csv")
    camera_rows, optical_chain = read_camera_rows(args.sequence / "tf.csv")
    intrinsics = read_intrinsics(args.sequence / "camera_rgb_camera_info.csv")
    sampled = {
        link: surface_points(args.model_root / mesh, args.samples_per_link, seed=i)
        for i, (link, mesh) in enumerate(sorted(meshes.items()))
    }
    mask_dir = args.output / "part_masks"
    overlay_dir = args.output / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    palette = np.asarray([[0, 0, 0], [255, 128, 32], [32, 220, 170], [80, 140, 255]], dtype=np.uint8)
    diagnostics = []
    for frame_index, rgb_path in enumerate(rgb_files):
        timestamp_s = timestamp_from_name(rgb_path)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = load_depth(args.sequence, timestamp_s)
        poses = nearest_complete_pose(body_rows, timestamp_s, link_ids)
        map_from_optical = nearest(camera_rows, timestamp_s) @ optical_chain
        points_by_part = []
        for body_id, link in sorted(link_ids.items(), key=lambda item: item[1]):
            points = sampled[link]
            map_from_mesh = poses[body_id] @ mesh_poses[link]
            points_map = (map_from_mesh[:3, :3] @ points.T).T + map_from_mesh[:3, 3]
            points_by_part.append((int(link[2:]) + 1, points_map))
        labels, rendered_depth = render_labels(
            points_by_part, np.linalg.inv(map_from_optical), intrinsics, rgb.shape[:2], depth, args.depth_tolerance
        )
        stem = f"{frame_index:06d}"
        Image.fromarray(labels).save(mask_dir / f"{stem}.png")
        color = palette[np.minimum(labels, len(palette) - 1)]
        overlay = rgb.copy()
        mask = labels > 0
        overlay[mask] = (0.55 * rgb[mask] + 0.45 * color[mask]).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_dir / f"{stem}.jpg", quality=92)
        valid = mask & (depth > 0) & np.isfinite(rendered_depth)
        residual = np.abs(rendered_depth[valid] - depth[valid])
        diagnostics.append({
            "frame": frame_index,
            "source": str(rgb_path),
            "timestamp_s": timestamp_s,
            "mask_pixels": int(mask.sum()),
            "depth_residual_median_m": float(np.median(residual)) if residual.size else None,
            "depth_residual_p90_m": float(np.percentile(residual, 90)) if residual.size else None,
        })
    (args.output / "metadata.json").write_text(json.dumps({
        "sequence": str(args.sequence), "config": str(config), "frames": diagnostics,
    }, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
