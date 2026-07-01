#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rgbd_urdf_mvp.perception.hunyuan3d_client import Hunyuan3DClient
from rgbd_urdf_mvp.perception.particulate_adapter import ParticulateInferenceConfig, ParticulateInferenceRunner


class ArticulateRequest(BaseModel):
    hunyuan: dict[str, Any] = Field(default_factory=dict)
    particulate: dict[str, Any] = Field(default_factory=dict)


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="RGBD URDF Remote Articulation Server")
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def require_token(authorization: str | None) -> None:
        if not args.api_token:
            return
        expected = f"Bearer {args.api_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "hunyuan_url": args.hunyuan_url,
            "particulate_root": str(Path(args.particulate_root).resolve()),
            "output_root": str(Path(args.output_root).resolve()),
        }

    @app.post("/send-articulate")
    def send_articulate(payload: ArticulateRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
        require_token(authorization)
        uid = uuid.uuid4().hex
        with jobs_lock:
            jobs[uid] = {"uid": uid, "status": "queued"}
        thread = threading.Thread(target=run_job, args=(uid, payload), daemon=True)
        thread.start()
        return {"uid": uid}

    @app.get("/status-articulate/{uid}")
    def status_articulate(uid: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_token(authorization)
        with jobs_lock:
            payload = jobs.get(uid)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"Unknown job: {uid}")
            response = dict(payload)
            if isinstance(response.get("result_zip_base64"), str):
                response["result_zip_base64_bytes"] = len(response["result_zip_base64"])
                response.pop("result_zip_base64", None)
            return response

    @app.get("/download-articulate/{uid}")
    def download_articulate(uid: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_token(authorization)
        with jobs_lock:
            payload = jobs.get(uid)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"Unknown job: {uid}")
            if payload.get("status") != "completed":
                raise HTTPException(status_code=409, detail=f"Job is not complete: {uid}")
            result_zip_base64 = payload.get("result_zip_base64")
            if not isinstance(result_zip_base64, str) or not result_zip_base64:
                raise HTTPException(status_code=500, detail=f"Completed job has no result zip: {uid}")
            return {
                "uid": uid,
                "result_zip_base64": result_zip_base64,
                "result_zip_base64_bytes": len(result_zip_base64),
            }

    @app.get("/download-articulate-zip/{uid}")
    def download_articulate_zip(uid: str, authorization: str | None = Header(default=None)) -> FileResponse:
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
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{uid}_remote_articulation_result.zip",
        )

    def update_job(uid: str, **updates: Any) -> None:
        with jobs_lock:
            jobs.setdefault(uid, {"uid": uid}).update(updates)

    def run_job(uid: str, payload: ArticulateRequest) -> None:
        persistent_root = Path(args.output_root).expanduser().resolve()
        work_root = Path(args.scratch_root).expanduser().resolve() if args.scratch_root else persistent_root
        persistent_root.mkdir(parents=True, exist_ok=True)
        work_root.mkdir(parents=True, exist_ok=True)
        job_dir = work_root / uid
        mesh_path = job_dir / "hunyuan" / "generated.glb"
        particulate_dir = job_dir / "particulate"
        timings: dict[str, float] = {}
        resource_usage: dict[str, Any] = {}
        try:
            job_started = time.perf_counter()
            update_job(
                uid,
                status="running",
                stage="hunyuan3d",
                output_dir=str(job_dir),
                timings=timings,
                resource_usage=resource_usage,
            )
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            hunyuan_payload = dict(payload.hunyuan)
            hunyuan_payload.setdefault("type", "glb")
            _log_job(uid, f"hunyuan3d start work_dir={job_dir}")
            stage_started = time.perf_counter()
            with _GpuStageMonitor("hunyuan3d", float(args.gpu_monitor_interval_s)) as gpu_monitor:
                hclient = Hunyuan3DClient(args.hunyuan_url, api_token=args.hunyuan_api_token, timeout_s=120.0)
                hunyuan_uid = hclient.send(hunyuan_payload)
                model_base64 = hclient.wait_for_model_base64(
                    hunyuan_uid,
                    timeout_s=float(args.hunyuan_timeout_s),
                    poll_interval_s=float(args.hunyuan_poll_interval_s),
                )
                mesh_path.write_bytes(base64.b64decode(model_base64))
            resource_usage["hunyuan3d"] = gpu_monitor.summary()
            timings["hunyuan3d_s"] = time.perf_counter() - stage_started
            _log_job(uid, f"hunyuan3d done elapsed_s={timings['hunyuan3d_s']:.2f}")

            update_job(
                uid,
                status="running",
                stage="particulate",
                generated_mesh_path=str(mesh_path),
                timings=timings,
                resource_usage=resource_usage,
            )
            particulate_cfg = dict(payload.particulate)
            _log_job(uid, "particulate start")
            stage_started = time.perf_counter()
            with _GpuStageMonitor("particulate", float(args.gpu_monitor_interval_s)) as gpu_monitor:
                manifest_path = ParticulateInferenceRunner().run(
                    ParticulateInferenceConfig(
                        mesh_path=mesh_path,
                        output_dir=particulate_dir,
                        particulate_root=args.particulate_root,
                        python_bin=args.particulate_python,
                        model_config=particulate_cfg.get("model_config", args.particulate_model_config),
                        ckpt_path=particulate_cfg.get("ckpt_path") or args.particulate_ckpt_path,
                        up_dir=particulate_cfg.get("up_dir", "-Z"),
                        num_points=int(particulate_cfg.get("num_points", args.particulate_num_points)),
                        num_points_global=int(
                            particulate_cfg.get("num_points_global", args.particulate_num_points_global)
                        ),
                        target_faces=_optional_int(particulate_cfg.get("target_faces", args.particulate_target_faces)),
                        min_part_confidence=float(
                            particulate_cfg.get("min_part_confidence", args.particulate_min_part_confidence)
                        ),
                        strict=bool(particulate_cfg.get("strict", True)),
                        animation_frames=int(particulate_cfg.get("animation_frames", args.particulate_animation_frames)),
                        export_urdf=True,
                        export_mjcf=True,
                        eval=True,
                        dry_run=False,
                    )
                )
            resource_usage["particulate"] = gpu_monitor.summary()
            timings["particulate_s"] = time.perf_counter() - stage_started
            particulate_manifest = _read_json_if_exists(manifest_path)
            if isinstance(particulate_manifest.get("timings"), dict):
                timings["particulate_manifest_total_s"] = float(
                    particulate_manifest["timings"].get("total_s", timings["particulate_s"])
                )
            _log_job(uid, f"particulate done elapsed_s={timings['particulate_s']:.2f}")

            update_job(
                uid,
                status="running",
                stage="packaging",
                particulate_result=str(manifest_path),
                timings=timings,
                resource_usage=resource_usage,
            )
            _log_job(uid, "packaging start")
            stage_started = time.perf_counter()
            if job_dir.parent != persistent_root:
                copy_started = time.perf_counter()
                persistent_job_dir = persistent_root / uid
                if persistent_job_dir.exists():
                    shutil.rmtree(persistent_job_dir)
                shutil.copytree(job_dir, persistent_job_dir)
                timings["copy_to_output_root_s"] = time.perf_counter() - copy_started
            zip_base = persistent_root / f"{uid}_remote_articulation_result"
            zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=job_dir)
            timings["zip_create_s"] = time.perf_counter() - stage_started
            zip_read_started = time.perf_counter()
            result_zip_base64 = base64.b64encode(Path(zip_path).read_bytes()).decode("utf-8")
            timings["zip_base64_encode_s"] = time.perf_counter() - zip_read_started
            timings["packaging_s"] = time.perf_counter() - stage_started
            timings["total_s"] = time.perf_counter() - job_started
            timing_path = job_dir / "remote_articulation_server_timing.json"
            timing_path.write_text(
                _json_dumps(
                    {
                        "uid": uid,
                        "output_dir": str(job_dir),
                        "generated_mesh_path": str(mesh_path),
                        "particulate_result": str(manifest_path),
                        "timings": timings,
                        "resource_usage": resource_usage,
                        "particulate_timings": particulate_manifest.get("timings", {}),
                    }
                ),
                encoding="utf-8",
            )
            _log_job(
                uid,
                f"packaging done elapsed_s={timings['packaging_s']:.2f} total_s={timings['total_s']:.2f}",
            )
            update_job(
                uid,
                status="completed",
                stage="completed",
                generated_mesh_path=str(mesh_path),
                particulate_result=str(manifest_path),
                timings=timings,
                resource_usage=resource_usage,
                timing_path=str(timing_path),
                result_zip_path=str(zip_path),
                result_zip_base64=result_zip_base64,
            )
        except Exception as exc:
            timings["total_s"] = time.perf_counter() - job_started if "job_started" in locals() else 0.0
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
    parser = argparse.ArgumentParser(description="Serve Hunyuan3D + PARTICULATE as one remote articulation API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--api-token", default=os.environ.get("REMOTE_ARTICULATION_API_TOKEN"))
    parser.add_argument("--hunyuan-url", default="http://127.0.0.1:8080")
    parser.add_argument("--hunyuan-api-token", default=os.environ.get("HUNYUAN3D_API_TOKEN"))
    parser.add_argument("--hunyuan-timeout-s", type=float, default=1800.0)
    parser.add_argument("--hunyuan-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--output-root", default="outputs/remote_articulation_server")
    parser.add_argument(
        "--scratch-root",
        default=os.environ.get("REMOTE_ARTICULATION_SCRATCH_ROOT"),
        help="Optional node-local working directory; completed jobs are copied back to --output-root",
    )
    parser.add_argument("--particulate-root", default="Particulate")
    parser.add_argument("--particulate-python", default=os.environ.get("PARTICULATE_PYTHON", "python"))
    parser.add_argument("--particulate-model-config", default="configs/particulate-B.yaml")
    parser.add_argument("--particulate-ckpt-path", default=None)
    parser.add_argument("--particulate-num-points", type=int, default=102400)
    parser.add_argument("--particulate-num-points-global", type=int, default=40000)
    parser.add_argument("--particulate-target-faces", type=int, default=None)
    parser.add_argument("--particulate-min-part-confidence", type=float, default=0.0)
    parser.add_argument("--particulate-animation-frames", type=int, default=50)
    parser.add_argument(
        "--gpu-monitor-interval-s",
        type=float,
        default=1.0,
        help="Interval for nvidia-smi GPU memory sampling around Hunyuan3D and PARTICULATE stages.",
    )
    return parser


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _log_job(uid: str, message: str) -> None:
    print(f"[remote-articulation:{uid}] {message}", flush=True)


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
        snapshot["elapsed_s"] = time.perf_counter()
        self.samples.append(snapshot)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"stage": self.stage, "available": False, "sample_count": 0}
        errors = [sample.get("error") for sample in self.samples if sample.get("error")]
        usable = [sample for sample in self.samples if isinstance(sample.get("gpus"), list)]
        if not usable:
            return {
                "stage": self.stage,
                "available": False,
                "sample_count": len(self.samples),
                "error": errors[-1] if errors else "No GPU samples available.",
            }

        before = usable[0]
        after = usable[-1]
        baseline_by_gpu = {
            int(gpu["index"]): int(gpu["memory_used_mib"])
            for gpu in before.get("gpus", [])
            if isinstance(gpu, dict) and "index" in gpu and "memory_used_mib" in gpu
        }
        peak_by_gpu: dict[int, dict[str, Any]] = {}
        for sample in usable:
            for gpu in sample.get("gpus", []):
                if not isinstance(gpu, dict):
                    continue
                index = int(gpu.get("index", -1))
                if index < 0:
                    continue
                used = int(gpu.get("memory_used_mib", 0))
                previous = peak_by_gpu.get(index)
                if previous is None or used > int(previous.get("memory_used_mib", 0)):
                    peak_by_gpu[index] = dict(gpu)

        deltas = []
        for index, peak_gpu in sorted(peak_by_gpu.items()):
            baseline = baseline_by_gpu.get(index)
            if baseline is None:
                continue
            peak_used = int(peak_gpu.get("memory_used_mib", 0))
            deltas.append(
                {
                    "index": index,
                    "baseline_memory_used_mib": baseline,
                    "peak_memory_used_mib": peak_used,
                    "peak_memory_delta_mib": peak_used - baseline,
                }
            )
        peak_gpu = max(peak_by_gpu.values(), key=lambda gpu: int(gpu.get("memory_used_mib", 0)))
        max_delta = max(deltas, key=lambda item: item["peak_memory_delta_mib"]) if deltas else None
        return {
            "stage": self.stage,
            "available": True,
            "sample_count": len(usable),
            "monitor_interval_s": self.interval_s,
            "baseline_gpus": before.get("gpus", []),
            "final_gpus": after.get("gpus", []),
            "peak_gpus": [peak_by_gpu[index] for index in sorted(peak_by_gpu)],
            "peak_gpu_index": int(peak_gpu.get("index", -1)),
            "peak_memory_used_mib": int(peak_gpu.get("memory_used_mib", 0)),
            "max_delta_gpu_index": max_delta["index"] if max_delta else None,
            "peak_memory_delta_mib": max_delta["peak_memory_delta_mib"] if max_delta else None,
            "per_gpu_deltas": deltas,
            "errors": errors,
        }


def _query_gpu_snapshot() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception as exc:
        return {"timestamp_s": time.time(), "query_elapsed_s": time.perf_counter() - started, "error": str(exc)}
    if result.returncode != 0:
        return {
            "timestamp_s": time.time(),
            "query_elapsed_s": time.perf_counter() - started,
            "error": result.stderr.strip() or f"nvidia-smi exited {result.returncode}",
        }
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, used, free = (int(field) for field in fields)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
            }
        )
    return {"timestamp_s": time.time(), "query_elapsed_s": time.perf_counter() - started, "gpus": gpus}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    args = build_argparser().parse_args()
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
