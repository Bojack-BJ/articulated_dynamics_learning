#!/usr/bin/env python3
"""Build and audit the articulation-focused subset of the HOI4D release."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


# Tasks that directly operate an articulated part. Pouring, pick/place, and
# container insertion tasks are deliberately excluded.
ARTICULATION_TASKS = {
    ("C3", "T2"): "open_close_display",
    ("C4", "T1"): "open_close_drawer",
    ("C4", "T2"): "open_close_door",
    ("C6", "T1"): "open_close_safe_door",
    ("C9", "T2"): "cut_with_scissors",
    ("C11", "T4"): "clamp_with_pliers",
    ("C14", "T1"): "open_close_trash_can",
    ("C17", "T2"): "turn_fold_lamp",
    ("C17", "T3"): "toggle_lamp",
    ("C18", "T2"): "staple_paper",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("annotation_zip", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-mask-frames", type=int, default=30)
    parser.add_argument("--min-effective-pose-ratio", type=float, default=0.5)
    return parser.parse_args()


def sequence_from_member(member: str) -> tuple[str, str] | None:
    parts = PurePosixPath(member).parts
    if len(parts) < 10 or parts[0] != "HOI4D_annotations":
        return None
    sequence = "/".join(parts[1:8])
    return sequence, "/".join(parts[8:])


def vector(row: dict, key: str) -> list[float] | None:
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    try:
        return [float(value[axis]) for axis in "xyz"]
    except (KeyError, TypeError, ValueError):
        return None


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> int:
    args = parse_args()
    candidates: dict[str, dict] = {}
    for raw in args.release.read_text(encoding="utf-8").splitlines():
        sequence = raw.strip().strip("/")
        fields = sequence.split("/")
        if len(fields) != 7:
            continue
        category, task = fields[2], fields[6]
        motion = ARTICULATION_TASKS.get((category, task))
        if motion:
            candidates[sequence] = {
                "sequence": sequence,
                "category": category,
                "instance": fields[3],
                "task": task,
                "motion": motion,
            }

    members: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(args.annotation_zip) as archive:
        for name in archive.namelist():
            parsed = sequence_from_member(name)
            if parsed is None or parsed[0] not in candidates:
                continue
            sequence, suffix = parsed
            if suffix.startswith("2Dseg/mask/") and suffix.endswith(".png"):
                members[sequence]["mask"].append(name)
            elif suffix.startswith("objpose/") and suffix.endswith(".json"):
                members[sequence]["pose"].append(name)

        rows = []
        for sequence, base in sorted(candidates.items()):
            entry = members[sequence]
            pose_files = sorted(entry["pose"])
            effective = 0
            labels: set[str] = set()
            centers: dict[str, list[list[float]]] = defaultdict(list)
            rotations: dict[str, list[list[float]]] = defaultdict(list)
            for name in pose_files:
                try:
                    payload = json.loads(archive.read(name))
                except (json.JSONDecodeError, KeyError):
                    continue
                if not payload.get("isEffective"):
                    continue
                effective += 1
                for item in payload.get("dataList", payload.get("objects", [])):
                    label = str(item.get("label", item.get("id", "unknown")))
                    labels.add(label)
                    center = vector(item, "center")
                    rotation = vector(item, "rotation")
                    if center is not None:
                        centers[label].append(center)
                    if rotation is not None:
                        rotations[label].append(rotation)

            pose_ratio = effective / len(pose_files) if pose_files else 0.0
            center_range = max(
                (distance(points[0], point) for points in centers.values() for point in points[1:]),
                default=0.0,
            )
            rotation_range = max(
                (distance(points[0], point) for points in rotations.values() for point in points[1:]),
                default=0.0,
            )
            mask_frames = len(entry["mask"])
            if mask_frames < args.min_mask_frames:
                status, reason = "reject", "insufficient_masks"
            elif pose_ratio >= args.min_effective_pose_ratio and len(labels) >= 2:
                status, reason = "usable_pose_gt", "effective_part_poses"
            else:
                status, reason = "review_mask_only", "pose_missing_or_ineffective"
            rows.append({
                **base,
                "mask_frames": mask_frames,
                "pose_frames": len(pose_files),
                "effective_pose_frames": effective,
                "effective_pose_ratio": round(pose_ratio, 6),
                "pose_labels": "|".join(sorted(labels)),
                "center_range_m": round(center_range, 6),
                "euler_range_rad": round(rotation_range, 6),
                "status": status,
                "reason": reason,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hoi4d_articulation_sequences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "candidate_count": len(rows),
        "by_status": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "by_motion": {
            motion: sum(row["motion"] == motion for row in rows)
            for motion in sorted({row["motion"] for row in rows})
        },
        "csv": str(csv_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
