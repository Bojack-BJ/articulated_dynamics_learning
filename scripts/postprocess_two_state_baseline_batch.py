#!/usr/bin/env python3
"""Adapt and evaluate completed DTA or ArtGS runs on a shared GT point domain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from rgbd_urdf_mvp.benchmarks.aim_pointcloud_iou import (
    evaluate_aim_pointcloud_iou,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dta", "artgs"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-python")
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def write_reference_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if points.shape != (len(labels), 3):
        raise ValueError("Reference points and semantics do not align")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property int part_id\nend_header\n"
        )
        for point, label in zip(points, labels):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {int(label)}\n"
            )


def ensure_reference(object_root: Path) -> Path:
    reference = object_root / "evaluation_reference" / "two_state_start.ply"
    if reference.exists():
        return reference
    source = object_root / "acquisition" / "two_state" / "gaussianart"
    cloud = trimesh.load(source / "points3d.ply", process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    labels = np.load(source / "semantics.npy")
    write_reference_ply(reference, points, labels)
    return reference


def artgs_repo_for_object(worker_root: Path, object_id: str) -> Path | None:
    for worker in sorted(worker_root.glob("worker_*")):
        native = (
            worker
            / "outputs"
            / "external_suite"
            / "aligned"
            / object_id
            / "artgs"
            / "point_cloud"
            / "iteration_20000"
        )
        if native.exists():
            return worker
    return None


def run_adapter(
    args: argparse.Namespace,
    *,
    object_id: str,
    object_root: Path,
) -> tuple[Path, dict] | None:
    adapter = object_root / args.method / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    if args.method == "dta":
        run_dir = args.repo / "runs" / "external_baseline_suite_v1" / object_id
        if not run_dir.exists():
            return None
        command = [
            str(args.project_root / ".venv" / "bin" / "python"),
            str(args.project_root / "scripts" / "export_dta_predictions.py"),
            str(run_dir),
            "--output-dir",
            str(adapter),
        ]
        subprocess.run(command, cwd=args.project_root, check=True)
        prediction_path = adapter / "start_labeled_parts.ply"
    else:
        if args.worker_root is None or args.official_python is None:
            raise ValueError("ArtGS requires --worker-root and --official-python")
        repo = artgs_repo_for_object(args.worker_root, object_id)
        if repo is None:
            return None
        model_path = repo / "outputs" / "external_suite" / "aligned" / object_id / "artgs"
        command = [
            args.official_python,
            str(args.project_root / "scripts" / "export_artgs_predictions.py"),
            "--dataset",
            "external_suite",
            "--subset",
            "aligned",
            "--scene_name",
            object_id,
            "--model_path",
            str(model_path),
            "--output-dir",
            str(adapter),
            "--eval",
            "--quiet",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo / "submodules" / "diff-gaussian-rasterization"),
                str(repo / "submodules" / "simple-knn"),
                str(repo),
            )
        )
        subprocess.run(command, cwd=repo, env=environment, check=True)
        prediction_path = adapter / "start_labeled_gaussians.ply"
    predictions = json.loads((adapter / "predictions.json").read_text(encoding="utf-8"))
    return prediction_path, predictions


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    if args.worker_root is not None:
        args.worker_root = args.worker_root.expanduser().resolve()
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))[args.shard_index :: args.shard_count]

    summary = []
    for row in rows:
        object_id = row["object_id"]
        object_root = args.output_root / "per_object" / object_id
        metrics_path = object_root / args.method / "metrics.json"
        previous = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        try:
            reference = ensure_reference(object_root)
            adapter_result = run_adapter(
                args, object_id=object_id, object_root=object_root
            )
            if adapter_result is None:
                summary.append(
                    {
                        "object_id": object_id,
                        "status": "skipped_missing_native_output",
                        "exception": None,
                    }
                )
                print(json.dumps(summary[-1]), flush=True)
                continue
            prediction, adapted = adapter_result
            evaluation = evaluate_aim_pointcloud_iou(prediction, reference)
            evaluation_path = object_root / args.method / "segmentation_evaluation.json"
            evaluation_path.write_text(
                json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
            )
            primary = evaluation["primary"]
            covered = primary["covered_only_metrics"]
            aware = primary["coverage_aware_metrics"]
            predicted_count = int(evaluation["predicted_part_count"])
            gt_count = int(evaluation["gt_part_count"])
            metrics = {
                "schema": "external-baseline-result-v1",
                "method": args.method,
                "object_id": object_id,
                "category": row["category"],
                "protocol": "two_state_100view_start_end",
                "status": "success",
                "runtime_s": previous.get("runtime_s"),
                "oracle": {
                    "gt_part_count": int(row["gt_part_count"]),
                    "used_for_inference": True,
                    "inputs": ["gt_part_count"],
                },
                "segmentation": {
                    "point_iou": aware["one_to_one_mean_iou"],
                    "ari": covered["adjusted_rand_index"],
                    "ri": covered["rand_index"],
                    "predicted_part_count": predicted_count,
                    "gt_part_count": gt_count,
                    "undersegmented": predicted_count < gt_count,
                    "unmatched_gt_part_count": aware["unmatched_gt_part_count"],
                    "largest_cluster_ratio": covered["largest_cluster_ratio"],
                    "geometry_coverage": primary["geometry_coverage"],
                    "metric_domain": "two_state_start_points3d_semantics",
                },
                "kinematics": (
                    {
                        "support": "native_predictions_pending_common_frame_evaluation",
                        "joint_types": adapted.get("joint_types"),
                    }
                    if args.method == "artgs"
                    else {
                        "support": "unsupported",
                        "reason": (
                            "Released non-GT DTA exports both motion hypotheses "
                            "without selecting a joint type."
                        ),
                    }
                ),
                "artifacts": {
                    "predictions": str((object_root / args.method / "adapter" / "predictions.json")),
                    "segmentation_evaluation": str(evaluation_path),
                    "reference": str(reference),
                },
            }
        except Exception as error:
            metrics = {
                **previous,
                "schema": "external-baseline-result-v1",
                "method": args.method,
                "object_id": object_id,
                "category": row["category"],
                "status": "failed",
                "failure_stage": "adapter_or_common_evaluation",
                "exception": f"{type(error).__name__}: {error}",
            }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "object_id": object_id,
                "status": metrics["status"],
                "exception": metrics.get("exception"),
            }
        )
        print(json.dumps(summary[-1]), flush=True)
    print(json.dumps({"method": args.method, "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
