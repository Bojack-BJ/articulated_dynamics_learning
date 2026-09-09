from __future__ import annotations

import base64
import json
import random
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .hunyuan3d_client import Hunyuan3DClientError


@dataclass(slots=True)
class ReArtSequenceExportConfig:
    fusion_manifest_path: str | Path
    output_dir: str | Path
    frame_stride: int = 1
    max_frames: int | None = None
    frame_indices: tuple[int, ...] | None = None
    protocol_profile: str = "custom"
    max_points_per_frame: int | None = 20000
    foreground_only: bool = True
    random_seed: int = 1234


class ReArtSequenceExporter:
    def export(self, config: ReArtSequenceExportConfig) -> Path:
        manifest_path = Path(config.fusion_manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_frame_dir = _resolve_manifest_path(manifest_path, manifest.get("per_frame_dir"))
        if per_frame_dir is None or not per_frame_dir.exists():
            raise FileNotFoundError(f"fusion_manifest.json does not reference an existing per_frame_dir: {manifest_path}")
        mask_aware_fusion = bool(manifest.get("mask_aware_fusion"))
        labeled_part_ids = {
            int(value)
            for value in manifest.get("part_ids_present", [])
            if int(value) > 0
        }

        output_dir = Path(config.output_dir).expanduser().resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(int(config.random_seed))
        frame_paths = sorted(per_frame_dir.glob("frame_*.ply"))
        source_frames = [(_source_frame_index(path), path) for path in frame_paths]
        if config.frame_indices is not None:
            if int(config.frame_stride) != 1 or config.max_frames is not None:
                raise ValueError("Explicit frame_indices cannot be combined with frame_stride or max_frames.")
            frame_by_index = {index: path for index, path in source_frames}
            missing = [index for index in config.frame_indices if index not in frame_by_index]
            if missing:
                raise ValueError(f"Requested ReArt source frame indices are unavailable: {missing}")
            selected = [(index, frame_by_index[index]) for index in config.frame_indices]
            temporal_sampling_provenance = "explicit_source_frame_indices"
        else:
            selected = source_frames[:: max(1, int(config.frame_stride))]
            if config.max_frames is not None:
                selected = selected[: max(1, int(config.max_frames))]
            temporal_sampling_provenance = "adapter_stride_and_prefix_cap"
        if config.protocol_profile not in {"custom", "sapien_count_matched_4frame"}:
            raise ValueError(f"Unsupported ReArt protocol profile: {config.protocol_profile}")
        if config.protocol_profile == "sapien_count_matched_4frame" and len(selected) != 4:
            raise ValueError("sapien_count_matched_4frame requires exactly four exported point-cloud frames.")
        if len(selected) < 2:
            raise ValueError("ReArt needs at least two frames; lower --frame-stride or increase --max-frames.")

        exported_frames = []
        for local_index, (source_frame_index, frame_path) in enumerate(selected):
            points = _read_frame_ply(frame_path)
            # Object-mask-only fusion uses part_id=0 for valid foreground points.
            # Positive IDs are available only when simulation part masks were fused.
            if bool(config.foreground_only) and labeled_part_ids:
                points = [point for point in points if int(point[3]) > 0]
            elif bool(config.foreground_only) and not mask_aware_fusion:
                raise ValueError(
                    "Foreground-only ReArt export requires an object/part-mask-aware fusion manifest. "
                    "Re-fuse with object masks or pass --include-background explicitly."
                )
            if not points:
                raise ValueError(f"No points left after filtering {frame_path}")
            max_points = config.max_points_per_frame
            if max_points is not None and len(points) > int(max_points):
                points = rng.sample(points, int(max_points))
            out_name = f"frame_{local_index:04d}.ply"
            out_path = output_dir / out_name
            _write_vertex_only_ply(out_path, points)
            exported_frames.append(
                {
                    "local_frame_index": local_index,
                    "source_frame_index": source_frame_index,
                    "source_frame_path": str(frame_path),
                    "output_path": str(out_path),
                    "point_count": len(points),
                }
            )

        export_manifest = {
            "source": "rgbd_urdf_mvp_reart_sequence_export",
            "fusion_manifest_path": str(manifest_path),
            "source_per_frame_dir": str(per_frame_dir),
            "output_dir": str(output_dir),
            "frame_stride": int(config.frame_stride),
            "max_frames": config.max_frames,
            "requested_frame_indices": list(config.frame_indices) if config.frame_indices is not None else None,
            "source_frame_indices": [item["source_frame_index"] for item in exported_frames],
            "protocol_profile": config.protocol_profile,
            "temporal_sampling_provenance": temporal_sampling_provenance,
            "max_points_per_frame": config.max_points_per_frame,
            "foreground_only": bool(config.foreground_only),
            "foreground_filter_mode": (
                "positive-part-id"
                if bool(config.foreground_only) and labeled_part_ids
                else "object-mask-prefiltered"
                if bool(config.foreground_only) and mask_aware_fusion
                else "disabled"
            ),
            "source_mask_aware_fusion": mask_aware_fusion,
            "source_part_ids_present": sorted(labeled_part_ids),
            "frame_count": len(exported_frames),
            "frames": exported_frames,
            "notes": [
                "Files are xyz-only vertex PLY point clouds. Simulation GT part labels are not exported.",
                "ReArt's upstream real loader samples mesh surfaces; the wrapper samples vertices for point-cloud PLY inputs.",
                (
                    "The official ReArt Sapiens benchmark provides four point-cloud frames per object. "
                    "The sapien_count_matched_4frame profile matches only that count; source states, "
                    "timestamps, and camera/global transforms remain adapter-defined."
                ),
            ],
        }
        (output_dir / "reart_sequence_manifest.json").write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_dir


def _source_frame_index(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Expected a frame_<index>.ply filename, got: {path.name}")
    return int(match.group(1))


@dataclass(slots=True)
class RemoteReArtConfig:
    server_url: str
    sequence_dir: str | Path
    output_dir: str | Path
    sequence_name: str | None = None
    cano_idx: int = 0
    num_points: int = 4096
    num_parts: int = 10
    stage: str = "base"
    base_n_iter: int = 2000
    kinematic_n_iter: int = 200
    assign_iter: int = 1000
    snapshot_gap: int = 100
    use_assign_loss: bool = False
    use_flow_loss: bool = False
    use_nproc: bool = False
    timeout_s: float = 7200.0
    poll_interval_s: float = 5.0
    api_token: str | None = None
    download_result: bool = True


class RemoteReArtClient:
    def __init__(self, server_url: str, api_token: str | None = None, timeout_s: float = 60.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self.timeout_s = timeout_s

    def run(self, config: RemoteReArtConfig) -> Path:
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        payload = self._build_payload(config)
        uid = self._send(payload)
        status = self._wait(uid, timeout_s=float(config.timeout_s), poll_interval_s=float(config.poll_interval_s))
        timings = {"wait_until_completed_s": time.perf_counter() - started}
        if bool(config.download_result):
            zip_path = output_dir / "remote_reart_result.zip"
            zip_path.write_bytes(self._request_bytes("GET", f"/download-reart-zip/{uid}"))
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(output_dir)
        timings["total_s"] = time.perf_counter() - started
        sanitized = _sanitize_status(status)
        (output_dir / "remote_reart_status.json").write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "remote_reart_timing.json").write_text(
            json.dumps(
                {
                    "uid": uid,
                    "client_timings": timings,
                    "server_timings": sanitized.get("timings", {}),
                    "server_resource_usage": sanitized.get("resource_usage", {}),
                    "server_status": sanitized.get("status"),
                    "server_stage": sanitized.get("stage"),
                    "remote_result_zip_path": sanitized.get("result_zip_path"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_dir

    def _build_payload(self, config: RemoteReArtConfig) -> dict[str, Any]:
        sequence_dir = Path(config.sequence_dir).expanduser().resolve()
        if not sequence_dir.exists():
            raise FileNotFoundError(f"ReArt sequence directory does not exist: {sequence_dir}")
        zip_bytes = _zip_directory_to_bytes(sequence_dir)
        return {
            "sequence_name": config.sequence_name or sequence_dir.name,
            "sequence_zip_base64": base64.b64encode(zip_bytes).decode("utf-8"),
            "sequence_zip_base64_bytes": len(zip_bytes),
            "reart": {
                "cano_idx": int(config.cano_idx),
                "num_points": int(config.num_points),
                "num_parts": int(config.num_parts),
                "stage": str(config.stage),
                "base_n_iter": int(config.base_n_iter),
                "kinematic_n_iter": int(config.kinematic_n_iter),
                "assign_iter": int(config.assign_iter),
                "snapshot_gap": int(config.snapshot_gap),
                "use_assign_loss": bool(config.use_assign_loss),
                "use_flow_loss": bool(config.use_flow_loss),
                "use_nproc": bool(config.use_nproc),
            },
        }

    def _send(self, payload: dict[str, Any]) -> str:
        response = self._request_json("POST", "/send-reart", payload)
        uid = response.get("uid")
        if not isinstance(uid, str) or not uid:
            raise Hunyuan3DClientError(f"Missing uid in /send-reart response: {response}")
        return uid

    def _wait(self, uid: str, timeout_s: float, poll_interval_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._request_json("GET", f"/status-reart/{uid}")
            state = status.get("status")
            if state == "completed":
                return status
            if state not in {"queued", "running", "processing"}:
                raise Hunyuan3DClientError(f"Remote ReArt failed or returned unknown status: {status}")
            time.sleep(max(0.5, float(poll_interval_s)))
        raise Hunyuan3DClientError(f"Timed out waiting for remote ReArt job {uid}")

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        req = request.Request(f"{self.server_url}{endpoint}", data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Hunyuan3DClientError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except error.URLError as exc:
            raise Hunyuan3DClientError(f"Failed to reach remote ReArt server {self.server_url}: {exc}") from exc
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise Hunyuan3DClientError(f"Expected JSON object from {endpoint}, got {parsed!r}")
        return parsed

    def _request_bytes(self, method: str, endpoint: str) -> bytes:
        headers = {"Accept": "application/zip"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        req = request.Request(f"{self.server_url}{endpoint}", headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Hunyuan3DClientError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except error.URLError as exc:
            raise Hunyuan3DClientError(f"Failed to reach remote ReArt server {self.server_url}: {exc}") from exc


def _read_frame_ply(path: Path) -> list[tuple[float, float, float, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Expected ASCII PLY: {path}")
    properties: list[str] = []
    vertex_count = 0
    data_start = None
    in_vertex = False
    for index, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            properties.append(parts[-1])
        elif parts[0] == "end_header":
            data_start = index + 1
            break
    if data_start is None:
        raise ValueError(f"PLY has no end_header: {path}")
    prop = {name: idx for idx, name in enumerate(properties)}
    required = {"x", "y", "z"}
    if not required.issubset(prop):
        raise ValueError(f"PLY missing xyz properties: {path}")
    part_index = prop.get("part_id")
    points = []
    for line in lines[data_start : data_start + vertex_count]:
        items = line.split()
        if len(items) < len(properties):
            continue
        part_id = int(float(items[part_index])) if part_index is not None else 1
        points.append((float(items[prop["x"]]), float(items[prop["y"]]), float(items[prop["z"]]), part_id))
    return points


def _write_vertex_only_ply(path: Path, points: list[tuple[float, float, float, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for x_coord, y_coord, z_coord, _part_id in points:
        lines.append(f"{x_coord:.6f} {y_coord:.6f} {z_coord:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_manifest_path(manifest_path: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _zip_directory_to_bytes(directory: Path) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return buffer.getvalue()


def _sanitize_status(status: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(status)
    if isinstance(sanitized.get("result_zip_base64"), str):
        sanitized["result_zip_base64_bytes"] = len(sanitized["result_zip_base64"])
        sanitized.pop("result_zip_base64", None)
    return sanitized
