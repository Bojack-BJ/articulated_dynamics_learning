#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
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
            return dict(payload)

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
        try:
            job_started = time.perf_counter()
            update_job(uid, status="running", stage="hunyuan3d", output_dir=str(job_dir))
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            hunyuan_payload = dict(payload.hunyuan)
            hunyuan_payload.setdefault("type", "glb")
            _log_job(uid, f"hunyuan3d start work_dir={job_dir}")
            stage_started = time.perf_counter()
            hclient = Hunyuan3DClient(args.hunyuan_url, api_token=args.hunyuan_api_token, timeout_s=120.0)
            hunyuan_uid = hclient.send(hunyuan_payload)
            model_base64 = hclient.wait_for_model_base64(
                hunyuan_uid,
                timeout_s=float(args.hunyuan_timeout_s),
                poll_interval_s=float(args.hunyuan_poll_interval_s),
            )
            mesh_path.write_bytes(base64.b64decode(model_base64))
            _log_job(uid, f"hunyuan3d done elapsed_s={time.perf_counter() - stage_started:.2f}")

            update_job(uid, status="running", stage="particulate", generated_mesh_path=str(mesh_path))
            particulate_cfg = dict(payload.particulate)
            _log_job(uid, "particulate start")
            stage_started = time.perf_counter()
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
            _log_job(uid, f"particulate done elapsed_s={time.perf_counter() - stage_started:.2f}")

            update_job(uid, status="running", stage="packaging", particulate_result=str(manifest_path))
            _log_job(uid, "packaging start")
            stage_started = time.perf_counter()
            if job_dir.parent != persistent_root:
                persistent_job_dir = persistent_root / uid
                if persistent_job_dir.exists():
                    shutil.rmtree(persistent_job_dir)
                shutil.copytree(job_dir, persistent_job_dir)
            zip_base = persistent_root / f"{uid}_remote_articulation_result"
            zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=job_dir)
            result_zip_base64 = base64.b64encode(Path(zip_path).read_bytes()).decode("utf-8")
            _log_job(
                uid,
                f"packaging done elapsed_s={time.perf_counter() - stage_started:.2f} total_s={time.perf_counter() - job_started:.2f}",
            )
            update_job(
                uid,
                status="completed",
                stage="completed",
                generated_mesh_path=str(mesh_path),
                particulate_result=str(manifest_path),
                result_zip_base64=result_zip_base64,
            )
        except Exception as exc:
            update_job(
                uid,
                status="failed",
                stage="failed",
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
    return parser


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _log_job(uid: str, message: str) -> None:
    print(f"[remote-articulation:{uid}] {message}", flush=True)


def main() -> int:
    args = build_argparser().parse_args()
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
