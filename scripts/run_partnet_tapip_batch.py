#!/usr/bin/env python3
"""Prepare, infer, import, and merge multiview TAPIP3D PartNet artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rgbd_urdf_mvp.perception.tapip3d_adapter import (
    TAPIP3DImportConfig,
    TAPIP3DInputConfig,
    TAPIP3DInputPreparer,
    TAPIP3DMergeConfig,
    TAPIP3DMultiViewMerger,
    TAPIP3DTrackImporter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument("--tapip-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU IDs")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resolution-factor", type=float, default=2.0)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument(
        "--gpu-worker-retries",
        type=int,
        default=2,
        help="Retry a failed persistent GPU worker without discarding completed view outputs.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _objects(catalog: Path, limit: int | None) -> list[dict[str, str]]:
    payload = json.loads(catalog.resolve().read_text(encoding="utf-8"))
    values = payload.get("objects", payload) if isinstance(payload, dict) else payload
    result = [dict(item) for item in values]
    return result[: max(0, limit)] if limit is not None else result


def _paths(recording_root: Path, output_root: Path, object_id: str, view: int) -> dict[str, Path]:
    artifact = recording_root / object_id / "pointcloud_4d_partseg"
    view_root = output_root / object_id / f"view_{view}"
    return {
        "episode": recording_root / object_id / "episode.json",
        "seeds": artifact / "part_tracks.json",
        "input": view_root / "input.npz",
        "result": view_root / "result.npz",
        "features": view_root / "features.npz",
        "tracks": view_root / "tracks.json",
        "merged_tracks": artifact / "tapip3d_tracks.json",
        "merged_features": artifact / "tapip3d_features.npz",
    }


def _prepare_job(item, view, args):
    object_id = str(item["object_id"])
    paths = _paths(args.recording_root, args.output_root, object_id, view)
    if not paths["episode"].is_file() or not paths["seeds"].is_file():
        raise FileNotFoundError(f"Missing episode/seeds for {object_id}")
    if not (args.resume and paths["input"].is_file()):
        TAPIP3DInputPreparer(TAPIP3DInputConfig(
            episode_path=paths["episode"], output_npz=paths["input"], view_index=view,
            frame_stride=max(1, args.frame_stride), seed_tracks=paths["seeds"], compressed=False,
        )).prepare()
    return item, view, paths


def _write_jobs(path: Path, jobs: list[tuple[dict, int, dict[str, Path]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["input", "output", "features_output"], delimiter="\t")
        writer.writeheader()
        for _, _, paths in jobs:
            writer.writerow({
                "input": paths["input"], "output": paths["result"],
                "features_output": paths["features"],
            })


def _run_gpu_shard(gpu: str, manifest: Path, args: argparse.Namespace, shard_index: int) -> None:
    if sum(1 for _ in manifest.open(encoding="utf-8")) <= 1:
        return
    log_path = args.output_root / "logs" / f"gpu_{gpu}_shard_{shard_index}.log"
    failure_path = args.output_root / "logs" / f"gpu_{gpu}_shard_{shard_index}.failures.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        args.python, "scripts/run_tapip3d_known_rgbd.py",
        "--tapip-root", str(args.tapip_root), "--checkpoint", str(args.checkpoint),
        "--jobs-manifest", str(manifest), "--device", "cuda", "--continue-on-error",
        "--failures-output", str(failure_path), "--uncompressed-output",
        "--resolution-factor", str(max(0.25, float(args.resolution_factor))),
        "--wait-for-input-s", "7200",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = max(1, args.gpu_worker_retries + 1)
    for attempt in range(1, attempts + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== GPU worker attempt {attempt}/{attempts} ===\n")
            log.flush()
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        if attempt < attempts:
            # Model startup can transiently fail while torch.hub initializes its cache.
            time.sleep(min(30, 5 * attempt))
    raise subprocess.CalledProcessError(result.returncode, command)


def _import_object(item: dict, args: argparse.Namespace) -> dict[str, str] | None:
    object_id = str(item["object_id"])
    per_view = [_paths(args.recording_root, args.output_root, object_id, view) for view in range(args.views)]
    if not all(paths["result"].is_file() and paths["features"].is_file() for paths in per_view):
        return None
    for paths in per_view:
        if not (args.resume and paths["tracks"].is_file()):
            TAPIP3DTrackImporter(TAPIP3DImportConfig(
                tapip_input_npz=paths["input"], tapip_result_npz=paths["result"],
                seed_tracks=paths["seeds"], output_json=paths["tracks"],
            )).import_tracks()
    first = per_view[0]
    TAPIP3DMultiViewMerger(TAPIP3DMergeConfig(
        track_paths=[paths["tracks"] for paths in per_view], output_tracks=first["merged_tracks"],
        feature_paths=[paths["features"] for paths in per_view], output_features=first["merged_features"],
    )).merge()
    return {
        "object_id": object_id, "tracks_path": str(first["merged_tracks"]),
        "features_npz": str(first["merged_features"]), "split": str(item["split"]),
    }


def _write_learning_manifests(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    for name, feature_name in (("tapip", "tapip3d_features.npz"), ("hybrid", "cotracker_features.npz")):
        path = args.output_root / f"{name}_learning_manifest.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["object_id", "tracks_path", "features_npz", "split"], delimiter="\t")
            writer.writeheader()
            for row in rows:
                output = dict(row)
                if name == "hybrid":
                    output["features_npz"] = str(
                        args.recording_root / row["object_id"] / "pointcloud_4d_partseg" / feature_name
                    )
                writer.writerow(output)


def main() -> int:
    args = parse_args()
    args.catalog = args.catalog.expanduser().resolve()
    args.recording_root = args.recording_root.expanduser().resolve()
    args.tapip_root = args.tapip_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    objects = _objects(args.catalog, args.limit)
    started = time.perf_counter()
    prepare_started = time.perf_counter()
    prepared, failures = [], []
    planned = []
    for item in objects:
        object_id = str(item["object_id"])
        first = _paths(args.recording_root, args.output_root, object_id, 0)
        if not first["episode"].is_file() or not first["seeds"].is_file():
            failures.append({"stage": "prepare", "object_id": object_id, "error": "missing episode/seeds"})
            continue
        planned.extend((item, view, _paths(args.recording_root, args.output_root, object_id, view)) for view in range(max(1, args.views)))
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID.")
    shards = [[] for _ in gpu_ids]
    pending = [job for job in planned if not (args.resume and job[2]["result"].is_file() and job[2]["features"].is_file())]
    for index, job in enumerate(pending):
        shards[index % len(shards)].append(job)
    manifests = []
    for index, jobs in enumerate(shards):
        manifest = args.output_root / "jobs" / f"shard_{index:02d}.tsv"
        _write_jobs(manifest, jobs)
        manifests.append(manifest)
    gpu_executor = ThreadPoolExecutor(max_workers=len(gpu_ids)) if not args.prepare_only else None
    gpu_futures = (
        [gpu_executor.submit(_run_gpu_shard, gpu, manifest, args, index) for index, (gpu, manifest) in enumerate(zip(gpu_ids, manifests))]
        if gpu_executor is not None else []
    )
    with ThreadPoolExecutor(max_workers=max(1, args.cpu_workers)) as executor:
        futures = {
            executor.submit(_prepare_job, item, view, args): (str(item["object_id"]), view)
            for item, view, _ in pending
        }
        for future in as_completed(futures):
            try:
                prepared.append(future.result())
            except Exception as exc:
                object_id, view = futures[future]
                failures.append({"stage": "prepare", "object_id": object_id, "view": view, "error": repr(exc)})
    prepare_wall_s = time.perf_counter() - prepare_started
    prepared.sort(key=lambda value: (str(value[0]["object_id"]), value[1]))
    if not args.prepare_only:
        gpu_started = time.perf_counter()
        for future in gpu_futures:
            future.result()
        gpu_executor.shutdown()
        gpu_wall_s = time.perf_counter() - gpu_started
        rows = []
        import_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, args.cpu_workers)) as executor:
            futures = {executor.submit(_import_object, item, args): str(item["object_id"]) for item in objects}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    if row is not None:
                        rows.append(row)
                except Exception as exc:
                    failures.append({"stage": "import", "object_id": futures[future], "error": repr(exc)})
        import_wall_s = time.perf_counter() - import_started
        rows.sort(key=lambda row: row["object_id"])
        _write_learning_manifests(rows, args)
    else:
        rows = []
        gpu_wall_s = 0.0
        import_wall_s = 0.0
    summary = {
        "catalog_objects": len(objects), "planned_view_jobs": len(planned),
        "prepared_view_jobs": len(prepared), "pending_gpu_jobs": len(pending),
        "merged_objects": len(rows), "failures": failures, "elapsed_s": time.perf_counter() - started,
        "stage_wall_s": {
            "parallel_cpu_prepare": prepare_wall_s,
            "parallel_gpu_tracking_and_feature_export": gpu_wall_s,
            "parallel_cpu_import_and_multiview_merge": import_wall_s,
        },
        "execution": {
            "gpu_ids": gpu_ids,
            "gpu_worker_count": len(gpu_ids),
            "gpu_parallelism": "one persistent subprocess worker per GPU; view jobs are sharded round-robin",
            "cpu_workers": max(1, int(args.cpu_workers)),
            "cpu_parallelism": "ThreadPoolExecutor for input preparation and output import",
            "views_per_object": max(1, int(args.views)),
        },
        "algorithm_parameters": {
            "frame_stride": max(1, int(args.frame_stride)),
            "resolution_factor": max(0.25, float(args.resolution_factor)),
            "checkpoint": str(args.checkpoint),
        },
    }
    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
