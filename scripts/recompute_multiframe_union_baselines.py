#!/usr/bin/env python3
"""Recompute AiM/ReArt segmentation on a time-aligned multi-frame GT union."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_aim_multiframe_union,
    read_ascii_labeled_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("aim", "reart"))
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--aim-root", type=Path)
    parser.add_argument(
        "--aim-python",
        type=Path,
        help="Python interpreter with AiM's CUDA dependencies (defaults to this interpreter).",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reart-frame-indices",
        type=int,
        nargs="+",
        default=(0, 30, 60, 90),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--reuse-trajectories", action="store_true")
    parser.add_argument(
        "--selection-root",
        type=Path,
        help=(
            "Optional root containing <object>/<method>.json files whose "
            "selected_reference_part_ids define the benchmark GT domain."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for object_id in args.objects:
        try:
            result = (
                _evaluate_aim(args, object_id, output_root)
                if args.method == "aim"
                else _evaluate_reart(args, object_id, output_root)
            )
            output_path = output_root / object_id / f"{args.method}_multiframe_union.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            rows.append(_summary_row(args.method, object_id, result, output_path))
            print(f"[{args.method}] {object_id}: success", flush=True)
        except Exception as exc:  # Keep every requested object in the audit.
            failure = {
                "method": args.method,
                "object_id": object_id,
                "exception": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{args.method}] {object_id}: {failure['exception']}", flush=True)

    _write_csv(output_root / f"{args.method}_multiframe_union_summary.csv", rows)
    (output_root / f"{args.method}_multiframe_union_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if not failures else 1


def _evaluate_aim(
    args: argparse.Namespace, object_id: str, output_root: Path
) -> dict[str, Any]:
    if args.aim_root is None:
        raise ValueError("--aim-root is required for AiM")
    run_dir = args.runs_root / object_id
    frames_dir = (
        args.recordings_root
        / object_id
        / "pointcloud_4d_partseg"
        / "frames"
    )
    all_references = sorted(frames_dir.glob("frame_*.ply"))
    references = [
        path
        for path in all_references[:: args.frame_stride]
        if _ply_vertex_count(path) > 0
    ]
    if not references:
        raise FileNotFoundError(f"No GT frames under {frames_dir}")
    trajectory_path = output_root / object_id / "dense_trajectory.npz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if not (args.reuse_trajectories and trajectory_path.is_file()):
        subprocess.run(
            [
                str(args.aim_python or sys.executable),
                str(args.project_root / "scripts" / "export_aim_dense_trajectory.py"),
                str(run_dir),
                "--aim-root",
                str(args.aim_root),
                "--frames",
                str(len(all_references)),
                "--output",
                str(trajectory_path),
            ],
            check=True,
        )
    payload = np.load(trajectory_path)
    full_trajectory = np.asarray(payload["trajectory"], dtype=np.float64)
    source_indices = [int(path.stem.split("_")[-1]) for path in references]
    trajectory = full_trajectory[source_indices]
    if len(trajectory) != len(references):
        raise ValueError(
            f"AiM/GT frame mismatch: prediction={len(trajectory)}, GT={len(references)}"
        )
    segmented = run_dir / "motion_seg_final" / "segmented_point.ply"
    _, labels = read_ascii_labeled_ply(segmented, label_mode="rgb")
    if trajectory.shape[1] != len(labels):
        raise ValueError(
            f"AiM label mismatch: trajectory={trajectory.shape[1]}, labels={len(labels)}"
        )
    with tempfile.TemporaryDirectory(prefix=f"aim_union_{object_id}_") as tmp:
        tmp_dir = Path(tmp)
        predictions = []
        for frame_index, points in enumerate(trajectory):
            path = tmp_dir / f"frame_{frame_index:04d}.ply"
            _write_part_ply(path, points, labels)
            predictions.append(path)
        result = evaluate_aim_multiframe_union(
            list(zip(predictions, references, strict=True)),
            distance_ratios=(0.02, 0.05, 1.0),
            primary_distance_ratio=1.0,
            reference_part_ids=_reference_part_ids(args, object_id),
        )
    result["prediction_source"] = "aim-frozen-dense-trajectory"
    result["trajectory_npz"] = str(trajectory_path)
    result["source_frame_indices"] = source_indices
    result["empty_reference_frames_skipped"] = len(
        all_references[:: args.frame_stride]
    ) - len(references)
    return result


def _evaluate_reart(
    args: argparse.Namespace, object_id: str, output_root: Path
) -> dict[str, Any]:
    result_path = args.runs_root / object_id / object_id / "result.pkl"
    with result_path.open("rb") as stream:
        payload = pickle.load(stream)
    points_by_frame = np.asarray(payload["complete_pc_list"], dtype=np.float64)
    labels = np.asarray(payload["pred_cano_part"], dtype=np.int64)
    frame_indices = list(args.reart_frame_indices)
    if len(points_by_frame) != len(frame_indices):
        raise ValueError(
            f"ReArt frame mismatch: result={len(points_by_frame)}, indices={frame_indices}"
        )
    references = [
        args.recordings_root
        / object_id
        / "pointcloud_4d_partseg"
        / "frames"
        / f"frame_{frame_index:04d}.ply"
        for frame_index in frame_indices
    ]
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ReArt GT frames: {missing}")
    with tempfile.TemporaryDirectory(prefix=f"reart_union_{object_id}_") as tmp:
        tmp_dir = Path(tmp)
        predictions = []
        for local_index, points in enumerate(points_by_frame):
            path = tmp_dir / f"frame_{frame_indices[local_index]:04d}.ply"
            _write_part_ply(path, points, labels)
            predictions.append(path)
        result = evaluate_aim_multiframe_union(
            list(zip(predictions, references, strict=True)),
            distance_ratios=(0.02, 0.05, 1.0),
            primary_distance_ratio=1.0,
            reference_part_ids=_reference_part_ids(args, object_id),
        )
    result["prediction_source"] = "reart-complete-pc-list"
    result["result_pkl"] = str(result_path)
    result["source_frame_indices"] = frame_indices
    return result


def _reference_part_ids(
    args: argparse.Namespace, object_id: str
) -> list[int] | None:
    if args.selection_root is None:
        return None
    selection_path = args.selection_root / object_id / f"{args.method}.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing GT-domain selection: {selection_path}")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_reference_part_ids")
    if not selected:
        raise ValueError(
            f"No selected_reference_part_ids in GT-domain selection: {selection_path}"
        )
    return sorted({int(value) for value in selected})


def _ply_vertex_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                return int(fields[2])
            if fields == ["end_header"]:
                break
    return 0


def _write_part_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    if len(points) != len(labels):
        raise ValueError(f"Point/label mismatch: {len(points)} != {len(labels)}")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property int part_id\n"
            "end_header\n"
        )
        for point, label in zip(points, labels, strict=True):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {int(label)}\n"
            )


def _summary_row(
    method: str, object_id: str, result: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    metrics = result["primary"]["covered_only_metrics"]
    return {
        "method": method,
        "object_id": object_id,
        "status": "success",
        "frame_count": result["frame_count"],
        "gt_part_count": result["gt_part_count"],
        "predicted_part_count": result["predicted_part_count"],
        "point_iou": metrics["one_to_one_mean_iou"],
        "ari": metrics["adjusted_rand_index"],
        "ri": metrics["rand_index"],
        "undersegmented": (
            result["predicted_part_count"] < result["gt_part_count"]
        ),
        "largest_cluster_ratio": metrics["largest_cluster_ratio"],
        "output_json": str(output_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
