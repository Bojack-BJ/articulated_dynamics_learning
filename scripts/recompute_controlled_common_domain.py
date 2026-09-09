#!/usr/bin/env python3
"""Re-evaluate controlled slot predictions on one common kinematic GT domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_labeled_points_on_reference,
    read_track_json_labeled_points,
)
from rgbd_urdf_mvp.benchmarks.kinematic_part_domain import (
    build_kinematic_evaluation_domain,
    load_kinematic_evaluation_domain,
)
from rgbd_urdf_mvp.core.serialization import load_episode
from rgbd_urdf_mvp.perception.pointcloud_fusion import (
    _camera_to_world_point,
    _depth_convention,
    _fuse_view_points,
    _load_depth_u16,
    _resolve_view_camera_poses,
    _resolve_view_depth_paths,
    _resolve_view_part_mask_paths,
    _write_frame_ply,
)


METHODS = ("cotracker_only", "tapip_only", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--min-union-points", type=int, default=32)
    parser.add_argument("--min-visible-frames", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _flatten_part_masks(episode_path: Path, pixel_stride: int) -> list[np.ndarray]:
    episode = load_episode(episode_path)
    root = episode_path.parent
    frames: list[np.ndarray] = []
    stride = max(1, int(pixel_stride))
    for frame in episode.frames:
        labels = []
        for mask_path in _resolve_view_part_mask_paths(frame, root):
            mask = np.asarray(_load_depth_u16(mask_path), dtype=np.int64)
            sampled = mask[::stride, ::stride].reshape(-1)
            labels.append(sampled[sampled > 0])
        frames.append(np.concatenate(labels) if labels else np.empty(0, dtype=np.int64))
    return frames


def _build_reference_ply(
    episode_path: Path,
    output_path: Path,
    *,
    pixel_stride: int,
    voxel_size_m: float,
) -> int:
    episode = load_episode(episode_path)
    root = episode_path.parent
    frame = episode.frames[0]
    depth_paths = _resolve_view_depth_paths(frame, root)
    part_mask_paths = _resolve_view_part_mask_paths(frame, root)
    poses, _ = _resolve_view_camera_poses(frame, episode.metadata, len(depth_paths))
    if len(part_mask_paths) != len(depth_paths):
        raise ValueError(
            f"Expected one part mask per depth view, got {len(part_mask_paths)}/{len(depth_paths)}"
        )
    depth_kind = _depth_convention(episode.metadata)
    stride = max(1, int(pixel_stride))
    points: list[tuple[float, float, float, int]] = []
    for depth_path, mask_path, pose in zip(depth_paths, part_mask_paths, poses):
        depth = _load_depth_u16(depth_path)
        mask = _load_depth_u16(mask_path)
        for v_coord in range(0, len(depth), stride):
            for u_coord in range(0, len(depth[v_coord]), stride):
                part_id = int(mask[v_coord][u_coord])
                depth_m = float(depth[v_coord][u_coord]) / 1000.0
                if part_id <= 0 or not 0.05 <= depth_m <= 6.0:
                    continue
                xyz = _camera_to_world_point(
                    u_coord,
                    v_coord,
                    depth_m,
                    episode.camera_intrinsics,
                    pose,
                    depth_convention=depth_kind,
                )
                points.append((*xyz, part_id))
    fused = _fuse_view_points(points, float(voxel_size_m))
    if not fused:
        raise ValueError(f"Source frame produced no GT reference points: {episode_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame_ply(output_path, fused)
    return len(fused)


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    primary = result["primary"]
    covered = primary.get("covered_only_metrics") or {}
    aware = primary.get("coverage_aware_metrics") or {}
    return {
        "point_iou": aware.get("one_to_one_mean_iou"),
        "ari": covered.get("adjusted_rand_index"),
        "rand_index": covered.get("rand_index"),
        "predicted_part_count": result.get("predicted_part_count"),
        "gt_part_count": result.get("gt_part_count"),
        "part_count_exact": result.get("predicted_part_count") == result.get("gt_part_count"),
        "unmatched_gt_part_count": aware.get("unmatched_gt_part_count"),
        "geometry_coverage": primary.get("geometry_coverage"),
    }


def _common_objects(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("common_objects")
    if not isinstance(objects, list):
        raise ValueError(f"Alignment JSON has no common_objects list: {path}")
    return sorted(str(value) for value in objects)


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    object_ids = _common_objects(args.alignment_json.expanduser().resolve())

    for object_index, object_id in enumerate(object_ids, start=1):
        object_dir = output / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        seed_prediction = prediction_root / "hybrid/test" / object_id / "motion_part_tracks_slots.json"
        try:
            seed_payload = json.loads(seed_prediction.read_text(encoding="utf-8"))
            episode_path = Path(seed_payload["input_episode_path"]).expanduser().resolve()
            episode = load_episode(episode_path)
            segmentation = episode.metadata.get("part_segmentation")
            if not isinstance(segmentation, dict):
                raise ValueError("episode has no part_segmentation metadata")
            reference_path = object_dir / "source_frame_reference.ply"
            domain_path = object_dir / "kinematic_evaluation_domain.json"
            if not (args.resume and reference_path.exists() and domain_path.exists()):
                frame_labels = _flatten_part_masks(episode_path, args.pixel_stride)
                domain = build_kinematic_evaluation_domain(
                    segmentation,
                    frame_labels,
                    min_union_points=args.min_union_points,
                    min_visible_frames=args.min_visible_frames,
                )
                reference_count = _build_reference_ply(
                    episode_path,
                    reference_path,
                    pixel_stride=args.pixel_stride,
                    voxel_size_m=args.voxel_size_m,
                )
                domain.update({
                    "object_id": object_id,
                    "episode_json": str(episode_path),
                    "common_reference_ply": str(reference_path),
                    "reference_point_count": reference_count,
                    "reference_source_frame_index": 0,
                })
                domain_path.write_text(json.dumps(domain, indent=2) + "\n", encoding="utf-8")

            for method in METHODS:
                prediction = prediction_root / method / "test" / object_id / "motion_part_tracks_slots.json"
                pending_domains = [
                    domain_name
                    for domain_name in ("all", "observable")
                    if not (args.resume and (object_dir / f"{method}_{domain_name}.json").exists())
                ]
                points = labels = None
                selected_frame = None
                if pending_domains:
                    points, labels, selected_frame = read_track_json_labeled_points(
                        prediction,
                        source_frame_index=0,
                    )
                for domain_name in ("all", "observable"):
                    result_path = object_dir / f"{method}_{domain_name}.json"
                    if args.resume and result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        part_map, selected = load_kinematic_evaluation_domain(domain_path, domain_name)
                        assert points is not None and labels is not None
                        result = evaluate_labeled_points_on_reference(
                            points,
                            labels,
                            reference_path,
                            reference_part_id_map=part_map,
                            reference_part_ids=selected,
                            distance_ratios=(0.02, 0.05, 1.0),
                            primary_distance_ratio=1.0,
                        )
                        result.update({
                            "object_id": object_id,
                            "method": method,
                            "prediction_source": "motion-part-track-json",
                            "predicted_tracks_json": str(prediction.resolve()),
                            "source_frame_index": selected_frame,
                            "kinematic_domain": domain_name,
                            "kinematic_domain_manifest": str(domain_path),
                        })
                        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    rows.append({
                        "object_id": object_id,
                        "category": str(episode.category),
                        "split": "test",
                        "method": method,
                        "domain": domain_name,
                        **_metric_row(result),
                    })
            print(f"[{object_index}/{len(object_ids)}] {object_id}", flush=True)
        except Exception as exc:
            failures.append({
                "object_id": object_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{object_index}/{len(object_ids)}] {object_id}: FAILED {exc}", flush=True)

    fieldnames = [
        "object_id", "category", "split", "method", "domain", "point_iou", "ari",
        "rand_index", "predicted_part_count", "gt_part_count", "part_count_exact",
        "unmatched_gt_part_count", "geometry_coverage",
    ]
    with (output / "per_object_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    controlled_rows = [
        {
            **row,
            "one_to_one_mean_iou": row["point_iou"],
            "adjusted_rand_index": row["ari"],
            "gt_collision_count": row["unmatched_gt_part_count"],
            "undersegmented_gt_part_count": row["unmatched_gt_part_count"],
            "oversegmented_pred_slot_count": max(
                0, int(row["predicted_part_count"]) - int(row["gt_part_count"])
            ),
        }
        for row in rows
        if row["domain"] == "observable"
    ]
    (output / "benchmark_metrics.json").write_text(
        json.dumps(
            {
                "schema": "controlled-common-kinematic-domain-v1",
                "domain": "observable",
                "per_object": controlled_rows,
                "summary": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requested": len(object_ids), "rows": len(rows), "failures": len(failures)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
