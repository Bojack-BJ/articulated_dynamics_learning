#!/usr/bin/env python3
"""Prepare, run, and evaluate official AiM on a stratified PartNet subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_tsv", type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "evaluate", "all"), default="all")
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--aim-root", type=Path, required=True)
    parser.add_argument("--aim-python", type=Path, required=True)
    parser.add_argument("--project-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--boundary-frame-count", type=int, default=4)
    parser.add_argument("--static-iterations", type=int, default=5000)
    parser.add_argument("--motion-iterations", type=int, default=8000)
    parser.add_argument("--source-frame-index", type=int, default=116)
    parser.add_argument("--protocol-label", default="shared_3view")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_objects(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter="\t")]


def run_logged(command: list[str], log: Path, *, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def prepare(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    for row in rows:
        object_id = row["object_id"]
        source_frame_index = _source_frame_index(args, row)
        episode = args.recordings_root / object_id / "episode.json"
        dataset = args.dataset_root / object_id
        reference_root = args.recordings_root / object_id / "pointcloud_4d_partseg"
        reference = _reference_frame_path(reference_root, source_frame_index)
        if not reference.is_file():
            subprocess.run([
                str(args.project_python), "-m", "rgbd_urdf_mvp", "fuse-pointcloud",
                str(episode), "--output-dir", str(reference_root), "--pixel-stride", "8",
                "--voxel-size-m", "0.02", "--no-shared-pose-fallback",
            ], check=True)
        if not (args.resume and (dataset / "export_manifest.json").is_file()):
            subprocess.run([
                str(args.project_python), "scripts/export_episode_to_aim.py", str(episode),
                "--output-dir", str(dataset), "--boundary-frame-count",
                str(args.boundary_frame_count), "--motion-frame-stride", "1",
                "--test-stride", "8", "--overwrite",
            ], check=True)


def aim_environment(args: argparse.Namespace, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    extension_roots = [
        args.aim_root,
        args.aim_root / "submodules/diff-gaussian-rasterization1/build/lib.linux-x86_64-cpython-310",
        args.aim_root / "submodules/simple-knn/build/lib.linux-x86_64-cpython-310",
        args.aim_root / "lib/pointops/build/lib.linux-x86_64-cpython-310",
    ]
    env["PYTHONPATH"] = ":".join(map(str, extension_roots))
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run_one(args: argparse.Namespace, row: dict[str, str], gpu: str) -> dict[str, Any]:
    object_id = row["object_id"]
    output = args.run_root / object_id
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    if args.resume and (output / "motion_seg_final" / "segmented_point.ply").is_file():
        return {"object_id": object_id, "status": "success", "resumed": True}
    env = aim_environment(args, gpu)
    status: dict[str, Any] = {"object_id": object_id, "gpu": gpu, "status": "failed"}
    started = time.perf_counter()
    try:
        run_logged([
            str(args.aim_python), str(args.aim_root / "train_main.py"),
            "--source_path", str(args.dataset_root / object_id),
            "--model_path", str(output), "--is_blender", "--eval", "--random_bg_color",
            "--iterations_start", str(args.static_iterations),
            "--iterations_motion", str(args.motion_iterations),
            "--iterations", str(args.static_iterations + args.motion_iterations),
        ], output / "train.log", env=env)
        status["train_runtime_s"] = time.perf_counter() - started
        seg_started = time.perf_counter()
        run_logged([
            str(args.aim_python), str(args.aim_root / "seg_main.py"),
            "--source_path", str(args.dataset_root / object_id),
            "--model_path", str(output), "--is_blender", "--eval", "--random_bg_color",
        ], output / "segmentation.log", env=env)
        status["segmentation_runtime_s"] = time.perf_counter() - seg_started
        status["status"] = "success"
    except subprocess.CalledProcessError as error:
        status["failure_stage"] = (
            "segmentation" if (output / "segmentation.log").is_file() else "reconstruction"
        )
        status["exception"] = str(error)
    status["runtime_s"] = time.perf_counter() - started
    status.update(parse_official_diagnostics(output))
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def parse_official_diagnostics(run_root: Path) -> dict[str, Any]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (run_root / "train.log", run_root / "segmentation.log")
        if path.is_file()
    )
    moving = re.findall(r"(\d+)\s+Moving Gaussians", text, flags=re.IGNORECASE)
    final_motion_lines = [
        line for line in text.splitlines() if line.startswith("After Merging: [Seq] Motion #")
    ]
    initial_motion_lines = [
        line for line in text.splitlines() if line.startswith("[Seq] Motion #")
    ]
    diagnostic_lines = [
        line.strip() for line in text.splitlines()
        if any(token in line.lower() for token in ("neighbor", "ransac", "[seq]", "no progress"))
    ]
    return {
        "moving_gaussian_count": int(moving[-1]) if moving else None,
        "predicted_component_count": (
            len(final_motion_lines) if final_motion_lines else len(initial_motion_lines) or None
        ),
        "nn_ransac_statistics": diagnostic_lines[-20:],
    }


def run_official(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    queues = [rows[index::len(gpus)] for index in range(len(gpus))]

    def run_queue(gpu: str, queue: list[dict[str, str]]) -> list[dict[str, Any]]:
        results = []
        for row in queue:
            result = run_one(args, row, gpu)
            print(json.dumps(result), flush=True)
            results.append(result)
        return results

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_queue, gpu, queue): gpu
            for gpu, queue in zip(gpus, queues, strict=True)
        }
        for future in as_completed(futures):
            future.result()


def evaluate(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    args.report_root.mkdir(parents=True, exist_ok=True)
    raw_root = args.report_root / "raw"
    raw_root.mkdir(exist_ok=True)
    manifest_rows = []
    for row in rows:
        object_id = row["object_id"]
        source_frame_index = _source_frame_index(args, row)
        hybrid = args.hybrid_root / object_id / "motion_part_tracks_slots.json"
        reference = (
            args.recordings_root / object_id
            / "pointcloud_4d_partseg/frames"
            / f"frame_{source_frame_index:04d}.ply"
        )
        aim_prediction = args.run_root / object_id / "motion_seg_final/segmented_point.ply"
        status_path = args.run_root / object_id / "run_status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else {"status": "failed", "failure_stage": "unknown"}
        )
        for method, prediction, prediction_format in (
            ("hybrid", hybrid, "track-json"),
            ("aim", aim_prediction, "aim-ply"),
        ):
            evaluation_path = raw_root / object_id / f"{method}.json"
            method_status = "success" if prediction.is_file() else "failed"
            if method_status == "success":
                evaluation_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(args.project_python), "scripts/evaluate_aim_pointcloud_iou.py",
                    str(prediction), str(reference), "--prediction-format", prediction_format,
                    "--primary-distance-ratio", "1.0", "--distance-ratios", "0.02", "0.05",
                    "--output-json", str(evaluation_path),
                ]
                if hybrid.is_file():
                    command.extend(["--reference-part-ids-from-tracks", str(hybrid)])
                if method == "hybrid":
                    command.extend(
                        ["--source-frame-index", str(source_frame_index)]
                    )
                subprocess.run(command, check=True)
            manifest_rows.append({
                "object_id": object_id,
                "category": row["category"],
                "method": method,
                "protocol": args.protocol_label,
                "source_frame_index": source_frame_index,
                "status": method_status,
                "evaluation_json": str(evaluation_path),
                "failure_stage": status.get("failure_stage", "") if method == "aim" else "",
                "exception": status.get("exception", "") if method == "aim" else "",
                "moving_gaussian_count": status.get("moving_gaussian_count", ""),
                "predicted_component_count": status.get("predicted_component_count", ""),
                "nn_ransac_statistics": json.dumps(status.get("nn_ransac_statistics", [])),
            })
    manifest_path = args.report_root / "run_manifest.csv"
    fields = list(manifest_rows[0])
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def _source_frame_index(args: argparse.Namespace, row: dict[str, str]) -> int:
    value = row.get("source_frame_index")
    return int(value) if value not in (None, "") else args.source_frame_index


def main() -> int:
    args = parse_args()
    rows = load_objects(args.objects_tsv)
    if args.stage in {"prepare", "all"}:
        prepare(args, rows)
    if args.stage in {"run", "all"}:
        run_official(args, rows)
    if args.stage in {"evaluate", "all"}:
        evaluate(args, rows)
    return 0


def _reference_frame_path(reference_root: Path, frame_index: int) -> Path:
    return reference_root / "frames" / f"frame_{frame_index:04d}.ply"


if __name__ == "__main__":
    raise SystemExit(main())
