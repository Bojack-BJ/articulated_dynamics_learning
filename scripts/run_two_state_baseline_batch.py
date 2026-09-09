#!/usr/bin/env python3
"""Run sharded DTA or ArtGS jobs with resumable fixed-GPU workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


SUCCESS_STATUSES = {
    "success",
    "completed_unadapted",
    "completed_with_exporter_failure",
}


@dataclass(frozen=True)
class ObjectJob:
    object_id: str
    category: str
    gt_part_count: int


@dataclass(frozen=True)
class Worker:
    index: int
    gpu: int
    repo: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dta", "artgs"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--priority-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-python", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--artgs-worker-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jobs(
    manifest: Path,
    *,
    priority_manifest: Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[ObjectJob]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")

    def read(path: Path) -> list[ObjectJob]:
        with path.open(encoding="utf-8", newline="") as stream:
            return [
                ObjectJob(
                    object_id=row["object_id"],
                    category=row["category"],
                    gt_part_count=int(row["gt_part_count"]),
                )
                for row in csv.DictReader(stream)
            ]

    jobs = read(manifest)
    if priority_manifest is not None:
        priority = {job.object_id: index for index, job in enumerate(read(priority_manifest))}
        jobs.sort(
            key=lambda job: (
                0 if job.object_id in priority else 1,
                priority.get(job.object_id, len(priority)),
            )
        )
    return jobs[shard_index::shard_count]


def prepare_artgs_worker(template: Path, target: Path) -> None:
    """Copy mutable ArtGS code once while excluding scene data and outputs."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        template,
        target,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "data", "outputs", "__pycache__"),
    )
    (target / "data" / "external_suite" / "aligned").mkdir(parents=True)
    (target / "outputs").mkdir()


def native_complete(method: str, repo: Path, job: ObjectJob) -> bool:
    if method == "dta":
        step = (
            repo
            / "runs"
            / "external_baseline_suite_v1"
            / job.object_id
            / "results"
            / "step_0004000"
        )
        expected = [
            *(step / f"init_part_{index}_clustered.obj" for index in range(job.gt_part_count)),
            step / "init_prismatic_motion.json",
            step / "init_revolute_motion.json",
        ]
        return all(path.exists() for path in expected)
    native = repo / "outputs" / "external_suite" / "aligned" / job.object_id / "artgs"
    return (native / "point_cloud" / "iteration_20000").exists()


def run_job(
    *,
    method: str,
    job: ObjectJob,
    worker: Worker,
    project_root: Path,
    output_root: Path,
    official_python: str,
    force: bool,
) -> dict[str, object]:
    output_dir = output_root / "per_object" / job.object_id / method
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if not force and native_complete(method, worker.repo, job):
        return {
            "object_id": job.object_id,
            "status": "skipped_native_complete",
            "gpu": worker.gpu,
            "worker": worker.index,
        }
    if not force and metrics_path.exists():
        status = json.loads(metrics_path.read_text(encoding="utf-8")).get("status")
        if status in SUCCESS_STATUSES:
            return {
                "object_id": job.object_id,
                "status": "skipped_metrics_complete",
                "gpu": worker.gpu,
                "worker": worker.index,
            }

    package = (
        output_root
        / "per_object"
        / job.object_id
        / "acquisition"
        / "two_state"
    )
    if not (package / "package_manifest.json").exists():
        raise FileNotFoundError(f"Missing two-state package: {package}")
    if method == "artgs":
        scene_link = (
            worker.repo / "data" / "external_suite" / "aligned" / job.object_id
        )
        scene_link.parent.mkdir(parents=True, exist_ok=True)
        if not scene_link.exists():
            scene_link.symlink_to(package / "artgs", target_is_directory=True)

    command = [
        str(project_root / ".venv" / "bin" / "python"),
        str(project_root / "scripts" / "run_two_state_external_baseline.py"),
        method,
        str(package),
        "--repo",
        str(worker.repo),
        "--output-dir",
        str(output_dir),
        "--object-id",
        job.object_id,
        "--gt-part-count",
        str(job.gt_part_count),
        "--python",
        official_python,
        "--gpu",
        str(worker.gpu),
        "--run",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(worker.gpu)
    environment["PYTHONPATH"] = str(project_root / "src")
    log_path = output_dir / "formal.log"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== start method={method} object={job.object_id} "
            f"worker={worker.index} gpu={worker.gpu} ===\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "object_id": job.object_id,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "runtime_s": time.monotonic() - started,
        "gpu": worker.gpu,
        "worker": worker.index,
        "log": str(log_path),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = load_jobs(
        args.manifest,
        priority_manifest=args.priority_manifest,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    workers = []
    for index, gpu in enumerate(args.gpus):
        worker_repo = repo
        if args.method == "artgs":
            if args.artgs_worker_root is None:
                raise SystemExit("--artgs-worker-root is required for ArtGS")
            worker_repo = args.artgs_worker_root.expanduser().resolve() / f"worker_{index}"
            prepare_artgs_worker(repo, worker_repo)
        workers.append(Worker(index=index, gpu=gpu, repo=worker_repo))

    print(
        json.dumps(
            {
                "method": args.method,
                "job_count": len(jobs),
                "workers": [
                    {"index": worker.index, "gpu": worker.gpu, "repo": str(worker.repo)}
                    for worker in workers
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    lock = threading.Lock()
    next_job = 0

    def worker_loop(worker: Worker) -> list[dict[str, object]]:
        nonlocal next_job
        results = []
        while True:
            with lock:
                if next_job >= len(jobs):
                    return results
                job = jobs[next_job]
                next_job += 1
            print(
                f"[{args.method}] worker={worker.index} gpu={worker.gpu} "
                f"object={job.object_id} start",
                flush=True,
            )
            try:
                result = run_job(
                    method=args.method,
                    job=job,
                    worker=worker,
                    project_root=project_root,
                    output_root=output_root,
                    official_python=args.official_python,
                    force=args.force,
                )
            except Exception as exc:
                result = {
                    "object_id": job.object_id,
                    "status": "batch_error",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "gpu": worker.gpu,
                    "worker": worker.index,
                }
            results.append(result)
            print(json.dumps(result), flush=True)

    all_results = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(worker_loop, worker) for worker in workers]
        for future in as_completed(futures):
            all_results.extend(future.result())
    summary_path = (
        output_root
        / f"{args.method}_batch_shard_{args.shard_index}_of_{args.shard_count}.json"
    )
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    return 1 if any(result["status"] in {"failed", "batch_error"} for result in all_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
