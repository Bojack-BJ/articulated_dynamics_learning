#!/usr/bin/env python3
"""Transform HOI4D CAD mobility axes into each sequence's metric world frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


CATEGORY = {
    "C3": ("Laptop", "keyboard", "screen"),
    "C4": ("StorageFurniture", "body", None),
    "C6": ("Safe", "box", "door"),
    "C14": ("TrashCan", "base", "lid"),
}
EULER_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def find_item(payload: dict, needle: str) -> dict | None:
    needle = needle.lower()
    return next(
        (item for item in payload.get("dataList", []) if needle in str(item.get("label", "")).lower()),
        None,
    )


def vector(item: dict, key: str) -> np.ndarray:
    return np.asarray([item[key][axis] for axis in "xyz"], dtype=float)


def unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def fit(row: dict[str, str], annotation_root: Path, episode_root: Path, cad_root: Path) -> dict:
    category = row["category"]
    cad_category, parent_label, joint_needle = CATEGORY[category]
    if category == "C4":
        joint_needle = "drawer" if row["motion"] == "open_close_drawer" else "door"
    instance = int(row["instance"].removeprefix("N"))
    cad_dir = cad_root / cad_category / f"{instance:03d}"
    mobility = json.loads((cad_dir / "mobility_v2.json").read_text())
    candidates = [
        item for item in mobility
        if item.get("jointData", {}).get("axis") and joint_needle in str(item.get("name", "")).lower()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {joint_needle} CAD joint in {cad_dir}, found {len(candidates)}")
    cad_joint = candidates[0]
    cad_axis = unit(np.asarray(cad_joint["jointData"]["axis"]["direction"], dtype=float))
    cad_origin = np.asarray(cad_joint["jointData"]["axis"]["origin"], dtype=float)

    name = row["sequence"].replace("/", "_")
    episode = json.loads((episode_root / name / "episode.json").read_text())
    camera_by_frame = {
        int(frame["action_log"]["source_frame_index"]): np.asarray(frame["camera_pose"], dtype=float)
        for frame in episode["frames"]
    }
    annotation = annotation_root / row["sequence"] / "objpose"
    samples = []
    for path in sorted(annotation.glob("*.json")):
        payload = json.loads(path.read_text())
        frame = int(payload["frameId"]) - 1
        parent = find_item(payload, parent_label)
        if not payload.get("isEffective") or parent is None or frame not in camera_by_frame:
            continue
        samples.append((frame, vector(parent, "center"), vector(parent, "rotation"), camera_by_frame[frame]))
    if len(samples) < 3:
        raise ValueError(f"Insufficient parent poses for {row['sequence']}")

    order_results = []
    for order in EULER_ORDERS:
        parent_rotations = Rotation.from_euler(order, np.stack([sample[2] for sample in samples])).as_matrix()
        directions, origins = [], []
        for (_, center, _, camera_to_world), parent_rotation in zip(samples, parent_rotations):
            direction_camera = parent_rotation @ cad_axis
            origin_camera = center + parent_rotation @ cad_origin
            directions.append(unit(camera_to_world[:3, :3] @ direction_camera))
            origins.append((camera_to_world @ np.r_[origin_camera, 1.0])[:3])
        directions = np.stack(directions)
        reference = directions[0]
        directions *= np.where(directions @ reference < 0, -1.0, 1.0)[:, None]
        mean_direction = unit(np.mean(directions, axis=0))
        concentration = float(np.mean(np.abs(directions @ mean_direction)))
        origins = np.stack(origins)
        perpendicular = origins - np.outer(origins @ mean_direction, mean_direction)
        origin_spread = float(np.median(np.linalg.norm(perpendicular - np.median(perpendicular, axis=0), axis=1)))
        order_results.append((concentration, -origin_spread, order, mean_direction, origins))
    concentration, negative_spread, order, axis_world, origins = max(order_results, key=lambda item: item[:2])
    pivot = np.median(origins, axis=0)
    joint_type = "prismatic" if "drawer" in joint_needle else "revolute"
    return {
        "sequence": row["sequence"],
        "joint_type": joint_type,
        "axis": axis_world.tolist(),
        "pivot": pivot.tolist(),
        "cad_category": cad_category,
        "cad_instance": f"{instance:03d}",
        "cad_joint_id": cad_joint["id"],
        "cad_joint_name": cad_joint["name"],
        "cad_axis_local": cad_axis.tolist(),
        "cad_origin_local": cad_origin.tolist(),
        "euler_order": order,
        "direction_concentration": concentration,
        "axis_origin_spread_m": -negative_spread,
        "sample_count": len(samples),
    }


def relation(result: dict) -> dict:
    joint = {
        "name": "hoi4d_cad_joint",
        "joint_type": result["joint_type"],
        "parent_part_id": 0,
        "child_part_id": 1,
        "axis": result["axis"],
        "pivot": result["pivot"],
        "annotation_source": "hoi4d_cad_mobility_transformed",
    }
    return {"joints": [joint], "metadata": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("annotation_root", type=Path)
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("cad_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["category"] in CATEGORY]
    results = []
    for row in rows:
        result = fit(row, args.annotation_root, args.episode_root, args.cad_root)
        results.append(result)
        target = args.output_root / row["sequence"].replace("/", "_") / "relation_gt_cad.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(relation(result), indent=2) + "\n")
    (args.output_root / "cad_axis_audit.json").write_text(json.dumps({"sequences": results}, indent=2) + "\n")
    print(json.dumps({"count": len(results), "output": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
