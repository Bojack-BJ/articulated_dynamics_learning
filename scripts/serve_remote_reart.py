#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class ReArtRequest(BaseModel):
    sequence_name: str = "sequence"
    sequence_zip_base64: str
    reart: dict[str, Any] = Field(default_factory=dict)


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="RGBD URDF Remote ReArt Server")
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def require_token(authorization: str | None) -> None:
        if not args.api_token:
            return
        if authorization != f"Bearer {args.api_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    def update_job(uid: str, **updates: Any) -> None:
        with jobs_lock:
            jobs.setdefault(uid, {"uid": uid}).update(updates)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "reart_root": str(Path(args.reart_root).expanduser().resolve()),
            "reart_python": args.reart_python,
            "output_root": str(Path(args.output_root).expanduser().resolve()),
        }

    @app.post("/send-reart")
    def send_reart(payload: ReArtRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
        require_token(authorization)
        uid = uuid.uuid4().hex
        with jobs_lock:
            jobs[uid] = {"uid": uid, "status": "queued"}
        thread = threading.Thread(target=run_job, args=(uid, payload), daemon=True)
        thread.start()
        return {"uid": uid}

    @app.get("/status-reart/{uid}")
    def status_reart(uid: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_token(authorization)
        with jobs_lock:
            payload = jobs.get(uid)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"Unknown job: {uid}")
            return dict(payload)

    @app.get("/download-reart-zip/{uid}")
    def download_reart_zip(uid: str, authorization: str | None = Header(default=None)) -> FileResponse:
        require_token(authorization)
        with jobs_lock:
            payload = jobs.get(uid)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"Unknown job: {uid}")
            if payload.get("status") != "completed":
                raise HTTPException(status_code=409, detail=f"Job is not complete: {uid}")
            zip_path = payload.get("result_zip_path")
        if not isinstance(zip_path, str) or not Path(zip_path).exists():
            raise HTTPException(status_code=500, detail=f"Completed job has no result zip file: {uid}")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{uid}_remote_reart_result.zip")

    def run_job(uid: str, payload: ReArtRequest) -> None:
        timings: dict[str, float] = {}
        resource_usage: dict[str, Any] = {}
        job_started = time.perf_counter()
        try:
            persistent_root = Path(args.output_root).expanduser().resolve()
            work_root = Path(args.scratch_root).expanduser().resolve() if args.scratch_root else persistent_root
            persistent_root.mkdir(parents=True, exist_ok=True)
            work_root.mkdir(parents=True, exist_ok=True)
            job_dir = work_root / uid
            seq_dir = job_dir / "sequence" / _safe_name(payload.sequence_name)
            result_root = job_dir / "reart"
            seq_dir.mkdir(parents=True, exist_ok=True)
            result_root.mkdir(parents=True, exist_ok=True)
            update_job(uid, status="running", stage="unpack", output_dir=str(job_dir), timings=timings)

            stage_started = time.perf_counter()
            zip_path = job_dir / "sequence.zip"
            zip_path.write_bytes(base64.b64decode(payload.sequence_zip_base64))
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(seq_dir)
            timings["unpack_sequence_s"] = time.perf_counter() - stage_started

            cfg = dict(payload.reart)
            command = [
                args.reart_python,
                "-m",
                "rgbd_urdf_mvp.perception.reart_run_wrapper",
                "--reart-root",
                str(Path(args.reart_root).expanduser().resolve()),
                "--seq-path",
                str(seq_dir),
                "--save-root",
                str(result_root),
                "--cano-idx",
                str(int(cfg.get("cano_idx", args.cano_idx))),
                "--num-points",
                str(int(cfg.get("num_points", args.num_points))),
                "--num-parts",
                str(int(cfg.get("num_parts", args.num_parts))),
                "--stage",
                str(cfg.get("stage", args.stage)),
                "--base-n-iter",
                str(int(cfg.get("base_n_iter", args.base_n_iter))),
                "--kinematic-n-iter",
                str(int(cfg.get("kinematic_n_iter", args.kinematic_n_iter))),
                "--assign-iter",
                str(int(cfg.get("assign_iter", args.assign_iter))),
                "--snapshot-gap",
                str(int(cfg.get("snapshot_gap", args.snapshot_gap))),
            ]
            if bool(cfg.get("use_assign_loss", False)):
                command.append("--use-assign-loss")
            if bool(cfg.get("use_flow_loss", False)):
                command.append("--use-flow-loss")
            if bool(cfg.get("use_nproc", False)):
                command.append("--use-nproc")

            stdout_path = job_dir / "reart_stdout.log"
            stderr_path = job_dir / "reart_stderr.log"
            update_job(uid, status="running", stage="reart", command=command, timings=timings)
            stage_started = time.perf_counter()
            with _GpuStageMonitor("reart", float(args.gpu_monitor_interval_s)) as gpu_monitor:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    subprocess.run(
                        command,
                        cwd=Path(args.project_root).expanduser().resolve(),
                        env=_subprocess_env(args.project_root),
                        stdout=stdout,
                        stderr=stderr,
                        check=True,
                    )
            resource_usage["reart"] = gpu_monitor.summary()
            timings["reart_s"] = time.perf_counter() - stage_started

            wrapper_manifest = result_root / seq_dir.name / "reart_result.json"
            wrapper_payload = _read_json_if_exists(wrapper_manifest)
            if isinstance(wrapper_payload.get("timings"), dict):
                timings["reart_wrapper_total_s"] = float(wrapper_payload["timings"].get("total_s", timings["reart_s"]))

            update_job(uid, status="running", stage="packaging", timings=timings, resource_usage=resource_usage)
            stage_started = time.perf_counter()
            if job_dir.parent != persistent_root:
                persistent_job_dir = persistent_root / uid
                if persistent_job_dir.exists():
                    shutil.rmtree(persistent_job_dir)
                shutil.copytree(job_dir, persistent_job_dir)
            zip_base = persistent_root / f"{uid}_remote_reart_result"
            result_zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=job_dir)
            timings["packaging_s"] = time.perf_counter() - stage_started
            timings["total_s"] = time.perf_counter() - job_started
            timing_path = job_dir / "remote_reart_server_timing.json"
            timing_path.write_text(
                json.dumps(
                    {
                        "uid": uid,
                        "output_dir": str(job_dir),
                        "sequence_dir": str(seq_dir),
                        "result_root": str(result_root),
                        "wrapper_manifest": str(wrapper_manifest),
                        "timings": timings,
                        "resource_usage": resource_usage,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            update_job(
                uid,
                status="completed",
                stage="completed",
                output_dir=str(job_dir),
                sequence_dir=str(seq_dir),
                result_root=str(result_root),
                reart_result=str(wrapper_manifest),
                timings=timings,
                resource_usage=resource_usage,
                timing_path=str(timing_path),
                result_zip_path=str(result_zip_path),
            )
        except Exception as exc:
            timings["total_s"] = time.perf_counter() - job_started
            update_job(
                uid,
                status="failed",
                stage="failed",
                timings=timings,
                resource_usage=resource_usage,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )

    return app


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve ReArt as a remote 4D point cloud articulation backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--api-token", default=os.environ.get("REMOTE_REART_API_TOKEN"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--reart-root", default="ReArt")
    parser.add_argument("--reart-python", default=os.environ.get("REART_PYTHON", "python"))
    parser.add_argument("--output-root", default="outputs/remote_reart_server")
    parser.add_argument("--scratch-root", default=os.environ.get("REMOTE_REART_SCRATCH_ROOT"))
    parser.add_argument("--stage", choices=["base", "kinematic", "both", "evaluate"], default="base")
    parser.add_argument("--cano-idx", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-parts", type=int, default=10)
    parser.add_argument("--base-n-iter", type=int, default=2000)
    parser.add_argument("--kinematic-n-iter", type=int, default=200)
    parser.add_argument("--assign-iter", type=int, default=1000)
    parser.add_argument("--snapshot-gap", type=int, default=100)
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=1.0)
    return parser


def _subprocess_env(project_root: str) -> dict[str, str]:
    env = os.environ.copy()
    project = str(Path(project_root).expanduser().resolve())
    entries = [str(Path(project) / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "sequence"


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


class _GpuStageMonitor:
    def __init__(self, stage: str, interval_s: float) -> None:
        self.stage = stage
        self.interval_s = max(0.2, float(interval_s))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuStageMonitor":
        self._sample("before")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
        self._sample("after")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample("during")

    def _sample(self, phase: str) -> None:
        snapshot = _query_gpu_snapshot()
        snapshot["phase"] = phase
        snapshot["timestamp_s"] = time.time()
        self.samples.append(snapshot)

    def summary(self) -> dict[str, Any]:
        usable = [sample for sample in self.samples if isinstance(sample.get("gpus"), list)]
        if not usable:
            return {"stage": self.stage, "available": False, "sample_count": len(self.samples)}
        baseline = {
            int(gpu["index"]): int(gpu["memory_used_mib"])
            for gpu in usable[0].get("gpus", [])
            if isinstance(gpu, dict)
        }
        peak_used = 0
        peak_delta = 0
        peak_gpu_index = None
        for sample in usable:
            for gpu in sample.get("gpus", []):
                if not isinstance(gpu, dict):
                    continue
                index = int(gpu.get("index", -1))
                used = int(gpu.get("memory_used_mib", 0))
                delta = used - baseline.get(index, used)
                if used > peak_used:
                    peak_used = used
                    peak_gpu_index = index
                peak_delta = max(peak_delta, delta)
        return {
            "stage": self.stage,
            "available": True,
            "sample_count": len(self.samples),
            "peak_gpu_index": peak_gpu_index,
            "peak_memory_used_mib": peak_used,
            "peak_memory_delta_mib": peak_delta,
            "samples": self.samples,
        }


def _query_gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": int(float(fields[2])),
                "memory_total_mib": int(float(fields[3])),
            }
        )
    return {"available": True, "gpus": gpus}


def main() -> int:
    args = build_argparser().parse_args()
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
