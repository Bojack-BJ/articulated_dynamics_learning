from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib import error, request

HUNYUAN_VIEW_ORDER = ("front", "left", "right", "back")


@dataclass(slots=True)
class Hunyuan3DGenerationConfig:
    server_url: str
    output_path: str | Path
    image_path: str | Path | Sequence[str | Path] | None = None
    image_views: Sequence[str] | None = None
    text: str | None = None
    mesh_path: str | Path | None = None
    mode: str = "async"
    texture: bool = False
    seed: int = 1234
    output_type: str = "glb"
    octree_resolution: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    face_count: int | None = None
    timeout_s: float = 1800.0
    poll_interval_s: float = 5.0
    api_token: str | None = None


class Hunyuan3DClientError(RuntimeError):
    pass


class Hunyuan3DClient:
    """Small stdlib client for the Hunyuan3D FastAPI sample server."""

    def __init__(self, server_url: str, api_token: str | None = None, timeout_s: float = 60.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def generate(self, config: Hunyuan3DGenerationConfig) -> Path:
        payload = build_generation_payload(config)
        output_path = Path(config.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.mode == "sync":
            model_bytes = self._request_bytes("POST", "/generate", payload)
            output_path.write_bytes(model_bytes)
            return output_path
        if config.mode == "async":
            uid = self.send(payload)
            model_base64 = self.wait_for_model_base64(
                uid,
                timeout_s=float(config.timeout_s),
                poll_interval_s=float(config.poll_interval_s),
            )
            output_path.write_bytes(base64.b64decode(model_base64))
            return output_path
        raise Hunyuan3DClientError(f"Unsupported generation mode: {config.mode}")

    def send(self, payload: dict[str, Any]) -> str:
        response = self._request_json("POST", "/send", payload)
        uid = response.get("uid")
        if not isinstance(uid, str) or not uid:
            raise Hunyuan3DClientError(f"Missing uid in /send response: {response}")
        return uid

    def status(self, uid: str) -> dict[str, Any]:
        return self._request_json("GET", f"/status/{uid}")

    def wait_for_model_base64(
        self,
        uid: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.status(uid)
            state = status.get("status")
            if state == "completed":
                model_base64 = status.get("model_base64")
                if not isinstance(model_base64, str) or not model_base64:
                    raise Hunyuan3DClientError(f"Completed job has no model_base64 field: {status}")
                return model_base64
            if state not in {"processing", "queued", "running"}:
                raise Hunyuan3DClientError(f"Remote generation failed or returned unknown status: {status}")
            time.sleep(max(0.5, poll_interval_s))
        raise Hunyuan3DClientError(f"Timed out waiting for Hunyuan3D job {uid}")

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._request(method, endpoint, payload)
        try:
            parsed = json.loads(response.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise Hunyuan3DClientError(f"Expected JSON from {endpoint}, got {response[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise Hunyuan3DClientError(f"Expected JSON object from {endpoint}, got {parsed!r}")
        return parsed

    def _request_bytes(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> bytes:
        return self._request(method, endpoint, payload)

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> bytes:
        body = None
        headers = {"Accept": "application/json, application/octet-stream"}
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
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Hunyuan3DClientError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except error.URLError as exc:
            raise Hunyuan3DClientError(f"Failed to reach Hunyuan3D server {self.server_url}: {exc}") from exc


def build_generation_payload(config: Hunyuan3DGenerationConfig) -> dict[str, Any]:
    if config.image_path is None and config.text is None and config.mesh_path is None:
        raise Hunyuan3DClientError("Provide at least one of image_path, text, or mesh_path.")

    payload: dict[str, Any] = {
        "texture": bool(config.texture),
        "seed": int(config.seed),
        "type": config.output_type,
    }
    if config.image_path is not None:
        payload["image"] = _encode_image_payload(config.image_path, config.image_views)
    if config.text is not None:
        payload["text"] = config.text
    if config.mesh_path is not None:
        payload["mesh"] = _file_to_base64(config.mesh_path)

    optional_fields = {
        "octree_resolution": config.octree_resolution,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "face_count": config.face_count,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def _encode_image_payload(
    image_path: str | Path | Sequence[str | Path],
    image_views: Sequence[str] | None,
) -> str | dict[str, str]:
    image_paths = _normalize_image_paths(image_path)
    if not image_paths:
        raise Hunyuan3DClientError("image_path must include at least one image.")
    if image_views is not None and len(image_views) != len(image_paths):
        raise Hunyuan3DClientError(
            f"image_views length ({len(image_views)}) must match image_path length ({len(image_paths)})."
        )
    if len(image_paths) == 1:
        return _file_to_base64(image_paths[0])
    if len(image_paths) > len(HUNYUAN_VIEW_ORDER):
        raise Hunyuan3DClientError(
            f"Hunyuan3D multiview generation supports at most {len(HUNYUAN_VIEW_ORDER)} views."
        )

    views = list(image_views) if image_views is not None else list(HUNYUAN_VIEW_ORDER[: len(image_paths)])
    invalid_views = [view for view in views if view not in HUNYUAN_VIEW_ORDER]
    if invalid_views:
        raise Hunyuan3DClientError(
            f"Unsupported Hunyuan3D view names: {invalid_views}. Expected names: {list(HUNYUAN_VIEW_ORDER)}"
        )
    if len(set(views)) != len(views):
        raise Hunyuan3DClientError(f"Duplicate Hunyuan3D view names are not allowed: {views}")

    return {view: _file_to_base64(path) for view, path in zip(views, image_paths)}


def _normalize_image_paths(image_path: str | Path | Sequence[str | Path]) -> list[str | Path]:
    if isinstance(image_path, (str, Path)):
        return [image_path]
    return list(image_path)


def _file_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).expanduser().resolve().read_bytes()).decode("utf-8")
