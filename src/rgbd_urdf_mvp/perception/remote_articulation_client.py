from __future__ import annotations

import base64
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

from .hunyuan3d_client import Hunyuan3DClientError, Hunyuan3DGenerationConfig, build_generation_payload


@dataclass(slots=True)
class RemoteArticulationConfig:
    server_url: str
    output_dir: str | Path
    image_path: str | Path | Sequence[str | Path] | None = None
    image_views: Sequence[str] | None = None
    text: str | None = None
    texture: bool = False
    seed: int = 1234
    output_type: str = "glb"
    octree_resolution: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    face_count: int | None = None
    particulate_up_dir: str = "-Z"
    particulate_num_points: int = 102400
    particulate_global_points: int = 40000
    particulate_target_faces: int | None = None
    particulate_min_part_confidence: float = 0.0
    particulate_strict: bool = True
    timeout_s: float = 3600.0
    poll_interval_s: float = 5.0
    api_token: str | None = None
    download_result: bool = True


class RemoteArticulationClient:
    def __init__(self, server_url: str, api_token: str | None = None, timeout_s: float = 60.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self.timeout_s = timeout_s

    def generate(self, config: RemoteArticulationConfig) -> Path:
        total_started = time.perf_counter()
        timings: dict[str, Any] = {}
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        payload_started = time.perf_counter()
        payload = self._build_payload(config)
        timings["build_payload_s"] = time.perf_counter() - payload_started
        send_started = time.perf_counter()
        uid = self._send(payload)
        timings["send_s"] = time.perf_counter() - send_started
        wait_started = time.perf_counter()
        status = self._wait(uid, timeout_s=float(config.timeout_s), poll_interval_s=float(config.poll_interval_s))
        timings["wait_until_completed_s"] = time.perf_counter() - wait_started
        if not bool(config.download_result):
            timings["total_s"] = time.perf_counter() - total_started
            sanitized_status = self._sanitize_status(status)
            self._write_status_and_timing(output_dir, uid, sanitized_status, timings)
            return output_dir
        zip_path = output_dir / "remote_articulation_result.zip"
        download_started = time.perf_counter()
        downloaded_zip_bytes = self._try_request_bytes("GET", f"/download-articulate-zip/{uid}")
        timings["download_result_s"] = time.perf_counter() - download_started
        if downloaded_zip_bytes is not None:
            write_started = time.perf_counter()
            zip_path.write_bytes(downloaded_zip_bytes)
            timings["zip_write_s"] = time.perf_counter() - write_started
        else:
            result_zip_base64 = status.get("result_zip_base64")
            if not isinstance(result_zip_base64, str) or not result_zip_base64:
                download_started = time.perf_counter()
                download_payload = self._request_json("GET", f"/download-articulate/{uid}")
                timings["download_result_base64_s"] = time.perf_counter() - download_started
                result_zip_base64 = download_payload.get("result_zip_base64")
                status.setdefault("result_zip_base64_bytes", download_payload.get("result_zip_base64_bytes"))
            if not isinstance(result_zip_base64, str) or not result_zip_base64:
                raise Hunyuan3DClientError(f"Completed articulation job has no result zip: {status}")
            decode_started = time.perf_counter()
            zip_path.write_bytes(base64.b64decode(result_zip_base64))
            timings["zip_base64_decode_write_s"] = time.perf_counter() - decode_started
        unpack_started = time.perf_counter()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
        timings["unpack_zip_s"] = time.perf_counter() - unpack_started
        timings["total_s"] = time.perf_counter() - total_started
        sanitized_status = self._sanitize_status(status)
        self._write_status_and_timing(output_dir, uid, sanitized_status, timings)
        return output_dir

    def _sanitize_status(self, status: dict[str, Any]) -> dict[str, Any]:
        sanitized_status = dict(status)
        if isinstance(sanitized_status.get("result_zip_base64"), str):
            sanitized_status["result_zip_base64_bytes"] = len(sanitized_status["result_zip_base64"])
            sanitized_status.pop("result_zip_base64", None)
        return sanitized_status

    def _write_status_and_timing(
        self,
        output_dir: Path,
        uid: str,
        sanitized_status: dict[str, Any],
        timings: dict[str, Any],
    ) -> None:
        (output_dir / "remote_articulation_status.json").write_text(
            json.dumps(sanitized_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "remote_articulation_timing.json").write_text(
            json.dumps(
                {
                    "uid": uid,
                    "output_dir": str(output_dir),
                    "client_timings": timings,
                    "server_timings": sanitized_status.get("timings", {}),
                    "server_resource_usage": sanitized_status.get("resource_usage", {}),
                    "server_stage": sanitized_status.get("stage"),
                    "server_status": sanitized_status.get("status"),
                    "generated_mesh_path": sanitized_status.get("generated_mesh_path"),
                    "particulate_result": sanitized_status.get("particulate_result"),
                    "remote_result_zip_path": sanitized_status.get("result_zip_path"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _build_payload(self, config: RemoteArticulationConfig) -> dict[str, Any]:
        hunyuan_payload = build_generation_payload(
            Hunyuan3DGenerationConfig(
                server_url=config.server_url,
                output_path=Path(config.output_dir) / "generated.glb",
                image_path=config.image_path,
                image_views=config.image_views,
                text=config.text,
                texture=bool(config.texture),
                seed=int(config.seed),
                output_type=config.output_type,
                octree_resolution=config.octree_resolution,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                face_count=config.face_count,
                api_token=config.api_token,
            )
        )
        return {
            "hunyuan": hunyuan_payload,
            "particulate": {
                "up_dir": str(config.particulate_up_dir),
                "num_points": int(config.particulate_num_points),
                "num_points_global": int(config.particulate_global_points),
                "target_faces": config.particulate_target_faces,
                "min_part_confidence": float(config.particulate_min_part_confidence),
                "strict": bool(config.particulate_strict),
            },
        }

    def _send(self, payload: dict[str, Any]) -> str:
        response = self._request_json("POST", "/send-articulate", payload)
        uid = response.get("uid")
        if not isinstance(uid, str) or not uid:
            raise Hunyuan3DClientError(f"Missing uid in /send-articulate response: {response}")
        return uid

    def _wait(self, uid: str, timeout_s: float, poll_interval_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._request_json("GET", f"/status-articulate/{uid}")
            state = status.get("status")
            if state == "completed":
                return status
            if state not in {"queued", "running", "processing"}:
                raise Hunyuan3DClientError(f"Remote articulation failed or returned unknown status: {status}")
            time.sleep(max(0.5, poll_interval_s))
        raise Hunyuan3DClientError(f"Timed out waiting for remote articulation job {uid}")

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        req = request.Request(
            url=f"{self.server_url}{endpoint}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Hunyuan3DClientError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except error.URLError as exc:
            raise Hunyuan3DClientError(f"Failed to reach remote articulation server {self.server_url}: {exc}") from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise Hunyuan3DClientError(f"Expected JSON from {endpoint}, got {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise Hunyuan3DClientError(f"Expected JSON object from {endpoint}, got {parsed!r}")
        return parsed

    def _try_request_bytes(self, method: str, endpoint: str) -> bytes | None:
        headers = {"Accept": "application/zip"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        req = request.Request(
            url=f"{self.server_url}{endpoint}",
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                return response.read()
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise Hunyuan3DClientError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except error.URLError as exc:
            raise Hunyuan3DClientError(f"Failed to reach remote articulation server {self.server_url}: {exc}") from exc
