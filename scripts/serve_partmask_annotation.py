from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


DEFAULT_PARTS = [
    {"part_id": 1, "name": "base", "role": "base", "color": "#62d26f"},
    {"part_id": 2, "name": "moving", "role": "moving", "color": "#a56de2"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local RGB-D episode part-mask annotation UI.")
    parser.add_argument("episode", type=Path, help="Path to episode.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-episode", type=Path, default=None)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--propagate-python",
        type=Path,
        default=None,
        help="Python executable for backend propagation. Use .venvs/sam2/bin/python for SAM2 video.",
    )
    parser.add_argument("--sam2-root", type=Path, default=Path("sam2"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--cotracker-repo", type=Path, default=Path("co-tracker"))
    parser.add_argument("--cotracker-checkpoint", type=Path, default=Path("co-tracker/ckpt/scaled_offline.pth"))
    args = parser.parse_args()

    state = AnnotationState(
        episode_path=args.episode,
        output_dir=args.output_dir,
        output_episode=args.output_episode,
        default_view_index=args.view_index,
        default_frame_index=args.frame_index,
        propagate_python=args.propagate_python,
        sam2_root=args.sam2_root,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_device=args.sam2_device,
        cotracker_repo=args.cotracker_repo,
        cotracker_checkpoint=args.cotracker_checkpoint,
    )
    handler = _make_handler(state)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"Part-mask annotation UI: http://{args.host}:{args.port}")
    print(f"Episode: {state.episode_path}")
    print(f"Output episode: {state.output_episode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    return 0


class AnnotationState:
    def __init__(
        self,
        *,
        episode_path: Path,
        output_dir: Path | None,
        output_episode: Path | None,
        default_view_index: int,
        default_frame_index: int,
        propagate_python: Path | None,
        sam2_root: Path,
        sam2_config: str,
        sam2_checkpoint: Path | None,
        sam2_device: str,
        cotracker_repo: Path | None,
        cotracker_checkpoint: Path | None,
    ) -> None:
        self.episode_path = episode_path.expanduser().resolve()
        if not self.episode_path.exists():
            raise FileNotFoundError(f"Episode not found: {self.episode_path}")
        self.episode_root = self.episode_path.parent
        self.episode = _load_json(self.episode_path)
        self.output_dir = (
            output_dir.expanduser().resolve()
            if output_dir is not None
            else self.episode_root / "assets" / "masks" / "manual_partmask"
        )
        self.output_episode = (
            output_episode.expanduser().resolve()
            if output_episode is not None
            else self.episode_root / "episode.partmask-annotated.json"
        )
        if self.output_episode.parent != self.episode_root:
            raise ValueError(
                "--output-episode must be written next to the source episode.json because episode asset paths "
                "are stored relative to that directory."
            )
        self.default_view_index = max(0, int(default_view_index))
        self.default_frame_index = max(0, int(default_frame_index))
        self.prompt_log_path = self.output_dir / "annotation_prompts.json"
        self.prompt_log: dict[str, Any] = (
            _load_json(self.prompt_log_path) if self.prompt_log_path.exists() else {"records": []}
        )
        self.project_root = Path.cwd().resolve()
        self.propagate_python = _resolve_project_path(propagate_python, self.project_root) if propagate_python is not None else None
        self.sam2_root = _resolve_project_path(sam2_root, self.project_root)
        self.sam2_config = str(sam2_config)
        self.sam2_checkpoint = _resolve_project_path(sam2_checkpoint, self.project_root) if sam2_checkpoint is not None else None
        self.sam2_device = str(sam2_device)
        self.cotracker_repo = _resolve_project_path(cotracker_repo, self.project_root) if cotracker_repo is not None else None
        self.cotracker_checkpoint = (
            _resolve_project_path(cotracker_checkpoint, self.project_root) if cotracker_checkpoint is not None else None
        )
        self._process_lock = threading.Lock()
        self._propagation_process: subprocess.Popen[str] | None = None
        self._propagation: dict[str, Any] = {
            "status": "idle",
            "started_at": None,
            "ended_at": None,
            "backend": None,
            "reference_frame": None,
            "view_index": None,
            "output_episode": None,
            "output_dir": None,
            "log_path": None,
            "command": None,
            "returncode": None,
        }

    def frame_count(self) -> int:
        return len(self.episode.get("frames", []))

    def view_count(self, frame_index: int) -> int:
        frame = self.frame(frame_index)
        paths = frame.get("rgb_paths_by_view") or [frame.get("rgb_path")]
        return len([path for path in paths if path])

    def frame(self, frame_index: int) -> dict[str, Any]:
        frames = self.episode.get("frames", [])
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"Frame index out of range: {frame_index}")
        return frames[frame_index]

    def rgb_path(self, frame_index: int, view_index: int) -> Path:
        frame = self.frame(frame_index)
        paths = frame.get("rgb_paths_by_view") or [frame.get("rgb_path")]
        if view_index < 0 or view_index >= len(paths):
            raise ValueError(f"View index out of range: {view_index}")
        return self._resolve_episode_path(paths[view_index])

    def existing_part_mask_path(self, frame_index: int, view_index: int) -> Path | None:
        live_path = self._live_mask_path(frame_index, view_index)
        if live_path is not None and live_path.exists():
            return live_path
        frame = self.frame(frame_index)
        paths = frame.get("part_mask_paths_by_view") or ([frame.get("part_mask_path")] if frame.get("part_mask_path") else [])
        if view_index >= len(paths) or not paths[view_index]:
            return None
        return self._resolve_episode_path(paths[view_index])

    def mask_status(self, view_index: int) -> dict[str, Any]:
        frames = []
        completed_frames = set()
        for frame_index in range(self.frame_count()):
            path = self.existing_part_mask_path(frame_index, view_index)
            exists = path is not None and path.exists() and _mask_has_foreground(path)
            stat = path.stat() if exists and path is not None else None
            if exists:
                completed_frames.add(frame_index)
            frames.append(
                {
                    "frame_index": frame_index,
                    "exists": bool(exists),
                    "mask_path": str(path) if exists and path is not None else None,
                    "mask_mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
                    "mask_size_bytes": int(stat.st_size) if stat is not None else None,
                }
            )
        return {
            "view_index": int(view_index),
            "completed_count": len(completed_frames),
            "completed_frames": sorted(completed_frames),
            "latest_frame": max(completed_frames) if completed_frames else None,
            "frames": frames,
        }

    def _resolve_episode_path(self, raw: str) -> Path:
        path = Path(raw)
        resolved = path if path.is_absolute() else self.episode_root / path
        resolved = resolved.expanduser().resolve()
        try:
            resolved.relative_to(self.episode_root)
        except ValueError as exc:
            raise ValueError(f"Episode asset path escapes episode root: {raw}") from exc
        return resolved

    def save_mask(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._process_lock:
            self._refresh_process_locked()
            if self._propagation_process is not None and self._propagation_process.poll() is None:
                raise ValueError("Pause the backend before saving manual corrections.")

        frame_index = int(payload["frame_index"])
        view_index = int(payload.get("view_index", 0))
        width = int(payload["width"])
        height = int(payload["height"])
        labels = _decode_label_png(payload["mask_data_url"], width=width, height=height)
        parts = _normalize_parts(payload.get("parts") or DEFAULT_PARTS)
        valid_part_ids = {int(part["part_id"]) for part in parts}
        labels = _strip_unknown_labels(labels, valid_part_ids)
        prompts = payload.get("prompts") or []
        if not bool((labels > 0).any()):
            raise ValueError("Current part mask is empty. Brush a part region or run SAM2 prompt prediction before saving.")
        frame = self.frame(frame_index)

        output_path = self.output_dir / f"view_{view_index}" / f"frame_{frame_index:04d}_mask.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels.astype(np.uint16), mode="I;16").save(output_path)
        rel_path = _relative_to_episode(output_path, self.episode_root)

        mask_paths = list(frame.get("part_mask_paths_by_view") or [])
        while len(mask_paths) <= view_index:
            mask_paths.append("")
        mask_paths[view_index] = rel_path
        frame["part_mask_paths_by_view"] = mask_paths
        frame["part_mask_path"] = mask_paths[0] if mask_paths and mask_paths[0] else frame.get("part_mask_path")

        self.episode.setdefault("metadata", {})["part_segmentation"] = {
            "provider": "manual-prompt-annotation",
            "version": 1,
            "mask_encoding": "indexed-mask-u16",
            "background_part_id": 0,
            "parts": [
                {
                    "part_id": int(part["part_id"]),
                    "name": str(part.get("name") or f"part_{part['part_id']}"),
                    "role": str(part.get("role") or "unknown"),
                    "color": str(part.get("color") or ""),
                }
                for part in parts
            ],
            "notes": "Manual part mask annotations with optional text/point/bbox prompt records.",
        }
        self.episode.setdefault("metadata", {})["mask_provider"] = {
            "provider": "manual-prompt-annotation",
            "mask_kind": "part",
            "output_dir": str(self.output_dir),
        }

        record = {
            "frame_index": frame_index,
            "view_index": view_index,
            "mask_path": rel_path,
            "foreground_pixels": int((labels > 0).sum()),
            "part_pixels": {
                str(part_id): int((labels == part_id).sum())
                for part_id in sorted(int(value) for value in np.unique(labels) if int(value) > 0)
            },
            "parts": parts,
            "prompts": prompts,
        }
        self.prompt_log.setdefault("records", []).append(record)
        _save_json(self.prompt_log, self.prompt_log_path)
        _save_json(self.episode, self.output_episode)
        with self._process_lock:
            self._propagation["output_dir"] = None

        return {
            "output_episode": str(self.output_episode),
            "mask_path": str(output_path),
            "mask_path_relative": rel_path,
            "foreground_pixels": record["foreground_pixels"],
            "part_pixels": record["part_pixels"],
            "propagate_command": _propagate_command(self.output_episode, frame_index, view_index),
        }

    def predict_prompt_mask(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_index = int(payload["frame_index"])
        view_index = int(payload.get("view_index", 0))
        part_id = int(payload.get("part_id") or 0)
        if part_id <= 0:
            raise ValueError("part_id must be a positive integer")
        width = int(payload["width"])
        height = int(payload["height"])
        labels = _decode_label_png(payload["mask_data_url"], width=width, height=height)
        prompts = [
            prompt
            for prompt in payload.get("prompts") or []
            if int(prompt.get("part_id") or 0) == part_id and str(prompt.get("type") or "") in {"point_pos", "point_neg", "bbox"}
        ]
        if not prompts:
            raise ValueError("Add at least one point or bbox prompt for the selected part before running SAM2.")
        if self.sam2_checkpoint is None:
            raise ValueError("Start the server with --sam2-checkpoint before using SAM2 prompt prediction.")
        if not self.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")

        prompt_root = self.output_dir.parent / "_sam2_prompt_predictions"
        prompt_root.mkdir(parents=True, exist_ok=True)
        prompt_json = prompt_root / f"frame_{frame_index:04d}_view_{view_index}_part_{part_id}_prompts.json"
        output_mask = prompt_root / f"frame_{frame_index:04d}_view_{view_index}_part_{part_id}_mask.png"
        _save_json({"part_id": part_id, "prompts": prompts}, prompt_json)
        command = [
            str(self.propagate_python or sys.executable),
            str(self.project_root / "scripts" / "predict_sam2_prompt_mask.py"),
            "--image",
            str(self.rgb_path(frame_index, view_index)),
            "--prompts-json",
            str(prompt_json),
            "--output-mask",
            str(output_mask),
            "--sam2-root",
            str(self.sam2_root),
            "--sam2-config",
            self.sam2_config,
            "--sam2-checkpoint",
            str(self.sam2_checkpoint),
            "--sam2-device",
            self.sam2_device,
        ]
        env = dict(**__import__("os").environ)
        env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"), str(self.project_root / "src"))
        env.pop("__PYVENV_LAUNCHER__", None)
        completed = subprocess.run(command, cwd=self.sam2_root, env=env, text=True, capture_output=True)
        if completed.returncode != 0:
            tail = (completed.stdout + "\n" + completed.stderr)[-3000:]
            raise RuntimeError(f"SAM2 prompt prediction failed with rc={completed.returncode}\n{tail}")
        binary = np.asarray(Image.open(output_mask), dtype=np.uint16) > 0
        if binary.shape != labels.shape:
            raise ValueError(f"SAM2 output shape {binary.shape} does not match label shape {labels.shape}")
        labels[labels == part_id] = 0
        labels[binary] = np.uint16(part_id)
        return {
            "part_id": part_id,
            "mask_pixels": int(binary.sum()),
            "labels_data_url": _encode_label_png(labels),
            "stdout": completed.stdout[-3000:],
        }

    def start_propagation(self, payload: dict[str, Any]) -> dict[str, Any]:
        backend = str(payload.get("backend") or "sam2-video")
        reference_frame = int(payload.get("reference_frame", self.default_frame_index))
        view_index = int(payload.get("view_index", self.default_view_index))
        frame_stride = max(1, int(payload.get("frame_stride", 1)))
        sam2_part_mode = str(payload.get("sam2_part_mode") or "independent")
        if backend not in {"sam2-video", "cotracker-sparse"}:
            raise ValueError("backend must be sam2-video or cotracker-sparse")
        if sam2_part_mode not in {"independent", "joint"}:
            raise ValueError("sam2_part_mode must be independent or joint")
        if backend == "sam2-video" and self.sam2_checkpoint is None:
            raise ValueError("Start the server with --sam2-checkpoint before using SAM2 video propagation.")
        if backend == "sam2-video" and self.sam2_checkpoint is not None and not self.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")
        source_episode = self.output_episode if self.output_episode.exists() else self.episode_path
        run_name = f"live_{backend.replace('-', '_')}_f{reference_frame:04d}_v{view_index}_{int(time.time())}"
        output_dir = self.output_dir.parent / run_name
        output_episode = self.episode_root / f"episode.{run_name}.json"
        log_path = output_dir / "propagation.log"
        output_dir.mkdir(parents=True, exist_ok=True)

        python_bin = self.propagate_python or Path(sys.executable)
        command = [
            str(python_bin),
            "-m",
            "rgbd_urdf_mvp",
            "propagate-episode-masks",
            str(source_episode),
            "--backend",
            backend,
            "--mask-kind",
            "part",
            "--reference-frame",
            str(reference_frame),
            "--frame-stride",
            str(frame_stride),
            "--view-indices",
            str(view_index),
            "--output-episode",
            str(output_episode),
            "--output-dir",
            str(output_dir),
            "--force",
        ]
        if backend == "sam2-video":
            command.extend(
                [
                    "--sam2-root",
                    str(self.sam2_root),
                    "--sam2-config",
                    self.sam2_config,
                    "--sam2-checkpoint",
                    str(self.sam2_checkpoint),
                    "--sam2-device",
                    self.sam2_device,
                    "--sam2-part-mode",
                    sam2_part_mode,
                ]
            )
        else:
            if self.cotracker_repo is not None:
                command.extend(["--cotracker-repo", str(self.cotracker_repo)])
            if self.cotracker_checkpoint is not None:
                command.extend(["--cotracker-checkpoint", str(self.cotracker_checkpoint)])

        with self._process_lock:
            self._refresh_process_locked()
            if self._propagation_process is not None and self._propagation_process.poll() is None:
                raise RuntimeError("A propagation job is already running. Pause it before starting another one.")
            env = dict(**__import__("os").environ)
            env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"), str(self.project_root / "src"))
            env.pop("__PYVENV_LAUNCHER__", None)
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=self.sam2_root if backend == "sam2-video" else self.project_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_handle.close()
            self._propagation_process = process
            self._propagation = {
                "status": "running",
                "started_at": time.time(),
                "ended_at": None,
                "backend": backend,
                "sam2_part_mode": sam2_part_mode,
                "reference_frame": reference_frame,
                "view_index": view_index,
                "output_episode": str(output_episode),
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command": command,
                "returncode": None,
            }
        return self.propagation_status(view_index=view_index)

    def pause_propagation(self) -> dict[str, Any]:
        with self._process_lock:
            self._refresh_process_locked()
            process = self._propagation_process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self._propagation["status"] = "paused"
                self._propagation["ended_at"] = time.time()
                self._propagation["returncode"] = process.returncode
            self._propagation_process = None
        return self.propagation_status()

    def propagation_status(self, view_index: int | None = None) -> dict[str, Any]:
        with self._process_lock:
            self._refresh_process_locked()
            propagation = dict(self._propagation)
        selected_view = int(view_index if view_index is not None else propagation.get("view_index") or self.default_view_index)
        status = self.mask_status(selected_view)
        propagation["mask_status"] = status
        log_path = propagation.get("log_path")
        if log_path:
            propagation["log_tail"] = _tail_text(Path(log_path), max_lines=40)
        return propagation

    def _refresh_process_locked(self) -> None:
        process = self._propagation_process
        if process is None:
            return
        returncode = process.poll()
        if returncode is None:
            self._propagation["status"] = "running"
            return
        self._propagation["returncode"] = returncode
        self._propagation["ended_at"] = self._propagation.get("ended_at") or time.time()
        if self._propagation.get("status") == "running":
            self._propagation["status"] = "completed" if returncode == 0 else "failed"
        output_episode = self._propagation.get("output_episode")
        if returncode == 0 and output_episode and Path(output_episode).exists():
            self.output_episode = Path(output_episode).expanduser().resolve()
            self.episode = _load_json(self.output_episode)
        self._propagation_process = None

    def _live_mask_path(self, frame_index: int, view_index: int) -> Path | None:
        output_dir = self._propagation.get("output_dir")
        if not output_dir:
            return None
        return Path(output_dir) / f"view_{view_index}" / f"frame_{frame_index:04d}_mask.png"


def _make_handler(state: AnnotationState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PartMaskAnnotation/0.1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_bytes(INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self._send_json(_state_payload(state))
                    return
                if parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    frame_index = int(query.get("frame", [state.default_frame_index])[0])
                    view_index = int(query.get("view", [state.default_view_index])[0])
                    self._send_file(state.rgb_path(frame_index, view_index))
                    return
                if parsed.path == "/api/mask":
                    query = parse_qs(parsed.query)
                    frame_index = int(query.get("frame", [state.default_frame_index])[0])
                    view_index = int(query.get("view", [state.default_view_index])[0])
                    mask_path = state.existing_part_mask_path(frame_index, view_index)
                    if mask_path is None or not mask_path.exists():
                        self._send_json({"exists": False})
                        return
                    self._send_label_png(mask_path)
                    return
                if parsed.path == "/api/mask-status":
                    query = parse_qs(parsed.query)
                    view_index = int(query.get("view", [state.default_view_index])[0])
                    self._send_json(state.mask_status(view_index))
                    return
                if parsed.path == "/api/propagation/status":
                    query = parse_qs(parsed.query)
                    view_index = int(query.get("view", [state.default_view_index])[0])
                    self._send_json(state.propagation_status(view_index=view_index))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._send_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/save":
                    payload = self._read_json()
                    self._send_json(state.save_mask(payload))
                    return
                if parsed.path == "/api/predict-current-mask":
                    payload = self._read_json()
                    self._send_json(state.predict_prompt_mask(payload))
                    return
                if parsed.path == "/api/propagation/start":
                    payload = self._read_json()
                    self._send_json(state.start_propagation(payload))
                    return
                if parsed.path == "/api/propagation/pause":
                    self._send_json(state.pause_propagation())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._send_error(exc)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[partmask] " + fmt % args + "\n")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send_bytes(
                (json.dumps(payload, indent=2) + "\n").encode("utf-8"),
                status=status,
                content_type="application/json; charset=utf-8",
            )

        def _send_file(self, path: Path) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), content_type=content_type)

        def _send_label_png(self, path: Path) -> None:
            labels = np.asarray(Image.open(path), dtype=np.uint16)
            label8 = np.clip(labels, 0, 255).astype(np.uint8)
            import io

            buffer = io.BytesIO()
            Image.fromarray(label8, mode="L").save(buffer, format="PNG")
            self._send_bytes(buffer.getvalue(), content_type="image/png")

        def _send_error(self, exc: Exception) -> None:
            self._send_json({"error": str(exc)}, status=500)

        def _send_bytes(self, data: bytes, *, status: int = 200, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _state_payload(state: AnnotationState) -> dict[str, Any]:
    frames = state.episode.get("frames", [])
    frame_summaries = []
    for index, frame in enumerate(frames):
        frame_summaries.append(
            {
                "frame_index": index,
                "timestamp_s": frame.get("timestamp_s"),
                "joint_position_hint": frame.get("joint_position_hint"),
                "view_count": len(frame.get("rgb_paths_by_view") or [frame.get("rgb_path")]),
                "has_part_mask": bool(frame.get("part_mask_paths_by_view") or frame.get("part_mask_path")),
            }
        )
    metadata = state.episode.get("metadata", {})
    parts = metadata.get("part_segmentation", {}).get("parts") or DEFAULT_PARTS
    return {
        "object_instance_id": state.episode.get("object_instance_id"),
        "category": state.episode.get("category"),
        "frame_count": len(frames),
        "default_frame_index": min(state.default_frame_index, max(0, len(frames) - 1)),
        "default_view_index": state.default_view_index,
        "output_episode": str(state.output_episode),
        "output_dir": str(state.output_dir),
        "frames": frame_summaries,
        "parts": _normalize_parts(parts),
    }


def _normalize_parts(raw_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts = []
    used_ids: set[int] = set()
    for index, raw in enumerate(raw_parts, start=1):
        part_id = int(raw.get("part_id") or index)
        if part_id <= 0 or part_id in used_ids:
            continue
        used_ids.add(part_id)
        parts.append(
            {
                "part_id": part_id,
                "name": str(raw.get("name") or f"part_{part_id}"),
                "role": str(raw.get("role") or ("base" if part_id == 1 else "moving")),
                "color": str(raw.get("color") or _default_color(part_id)),
            }
        )
    return parts or DEFAULT_PARTS


def _decode_label_png(data_url: str, *, width: int, height: int) -> np.ndarray:
    match = re.match(r"^data:image/png;base64,(.*)$", data_url)
    if not match:
        raise ValueError("mask_data_url must be a PNG data URL")
    raw = base64.b64decode(match.group(1))
    import io

    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    if image.size != (width, height):
        raise ValueError(f"Mask size {image.size} does not match expected {(width, height)}")
    rgba = np.asarray(image, dtype=np.uint8)
    labels = rgba[..., 0].astype(np.uint16)
    labels[rgba[..., 3] == 0] = 0
    return labels


def _strip_unknown_labels(labels: np.ndarray, valid_part_ids: set[int]) -> np.ndarray:
    if not valid_part_ids:
        return np.zeros(labels.shape, dtype=np.uint16)
    valid = np.isin(labels, np.asarray(sorted(valid_part_ids), dtype=np.uint16))
    output = np.asarray(labels, dtype=np.uint16).copy()
    output[~valid] = 0
    return output


def _encode_label_png(labels: np.ndarray) -> str:
    import io

    label8 = np.clip(np.asarray(labels, dtype=np.uint16), 0, 255).astype(np.uint8)
    rgba = np.zeros((label8.shape[0], label8.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = label8
    rgba[..., 3] = (label8 > 0).astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _mask_has_foreground(path: Path) -> bool:
    try:
        return bool((np.asarray(Image.open(path), dtype=np.uint16) > 0).any())
    except Exception:
        return False


def _relative_to_episode(path: Path, episode_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(episode_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Output path must be inside the episode directory: {resolved}") from exc


def _propagate_command(output_episode: Path, frame_index: int, view_index: int) -> str:
    return (
        "PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks "
        f"{output_episode} --backend sam2-video --mask-kind part --reference-frame {frame_index} "
        f"--view-indices {view_index} --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt "
        "--output-episode <episode.propagated.json>"
    )


def _default_color(part_id: int) -> str:
    palette = ["#62d26f", "#a56de2", "#f2c14e", "#4cc9f0", "#ff6b6b", "#2ec4b6", "#f72585", "#b8f35a"]
    return palette[(max(1, part_id) - 1) % len(palette)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.absolute() if expanded.is_absolute() else project_root / expanded


def _prepend_pythonpath(existing: str | None, path: str) -> str:
    if not existing:
        return path
    parts = existing.split(":")
    return existing if path in parts else f"{path}:{existing}"


def _tail_text(path: Path, *, max_lines: int) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)):])


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Part Mask Annotation</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #182028;
      --panel2: #202a34;
      --text: #e8eef5;
      --muted: #9dacba;
      --line: #32404d;
      --accent: #82d4ff;
      --danger: #ff7b7b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      height: 100vh;
      overflow: hidden;
    }
    button, input, select {
      font: inherit;
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
    }
    button { cursor: pointer; }
    button.active { border-color: var(--accent); background: #113143; }
    button.danger { color: var(--danger); }
    input[type="range"] { padding: 0; }
    .app {
      display: grid;
      grid-template-columns: 320px minmax(400px, 1fr) 360px;
      height: 100vh;
    }
    .side, .right {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 14px;
      overflow: auto;
    }
    .right { border-left: 1px solid var(--line); border-right: 0; }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #121922;
    }
    .canvas-wrap {
      overflow: auto;
      display: grid;
      place-items: center;
      background: #070a0d;
      position: relative;
    }
    .stage {
      position: relative;
      margin: 24px;
      box-shadow: 0 0 0 1px #000, 0 18px 60px rgba(0,0,0,.45);
    }
    canvas { display: block; image-rendering: auto; }
    #overlayCanvas, #promptCanvas {
      position: absolute;
      left: 0;
      top: 0;
    }
    #promptCanvas { pointer-events: auto; }
    .row { display: flex; gap: 8px; align-items: center; margin: 8px 0; }
    .row > label { min-width: 84px; color: var(--muted); }
    .stack { display: grid; gap: 8px; }
    .part {
      display: grid;
      grid-template-columns: 22px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 6px 0;
      background: #151d25;
    }
    .part.active { border-color: var(--accent); }
    .swatch { width: 20px; height: 20px; border-radius: 50%; border: 1px solid rgba(255,255,255,.35); }
    h2, h3 { margin: 0 0 12px; font-size: 16px; }
    h3 { margin-top: 18px; color: #cfe6f7; }
    .muted { color: var(--muted); }
    .status {
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      min-height: 42px;
      color: var(--muted);
      background: #121922;
      white-space: pre-wrap;
    }
    .inline-check {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: var(--muted);
      white-space: nowrap;
    }
    .inline-check input { width: auto; }
    .prop-summary {
      color: #cfe6f7;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .prop-log {
      max-height: 128px;
      overflow: auto;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101720;
      color: #9fb0bf;
      font: 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .prompt-list {
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow: auto;
    }
    .prompt {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
      background: #141b23;
      color: #d8e4ed;
    }
    .progress {
      height: 10px;
      border-radius: 999px;
      background: #0e141a;
      border: 1px solid var(--line);
      overflow: hidden;
      margin: 8px 0;
    }
    .progress > div {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #62d26f, #82d4ff);
    }
    .mask-strip {
      display: grid;
      grid-template-columns: repeat(30, 1fr);
      gap: 2px;
      margin-top: 8px;
    }
    .mask-cell {
      height: 8px;
      border-radius: 2px;
      background: #26313c;
      cursor: pointer;
    }
    .mask-cell.done { background: #62d26f; }
    .mask-cell.current { outline: 1px solid #fff; }
    .kbd {
      color: #b8cad8;
      border: 1px solid var(--line);
      padding: 1px 5px;
      border-radius: 4px;
      background: #111820;
    }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      color: var(--text);
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="side">
      <h2>Episode</h2>
      <div id="episodeInfo" class="muted"></div>
      <div class="row"><label>Frame</label><input id="frameSlider" type="range" min="0" max="0" value="0" /></div>
      <div class="row"><label></label><input id="frameInput" type="number" min="0" value="0" style="width: 100px" /><button id="loadFrameBtn">Load</button></div>
      <div class="row"><label>View</label><input id="viewInput" type="number" min="0" value="0" style="width: 100px" /></div>
      <h3>Parts</h3>
      <div id="parts"></div>
      <div class="row"><button id="addPartBtn">Add Part</button><button id="clearPartBtn" class="danger">Clear Part</button></div>
      <h3>Text Prompt</h3>
      <div class="stack">
        <input id="textPrompt" placeholder="e.g. microwave door, box lid, hand" />
        <button id="addTextPromptBtn">Attach Text To Part</button>
      </div>
    </aside>
    <main class="main">
      <div class="toolbar">
        <button data-mode="brush" class="mode active">Brush</button>
        <button data-mode="erase" class="mode">Erase</button>
        <button data-mode="point_pos" class="mode">+ Point</button>
        <button data-mode="point_neg" class="mode">- Point</button>
        <button data-mode="bbox" class="mode">BBox</button>
        <button id="predictPartBtn">SAM2 Current Part</button>
        <label class="muted">Brush</label>
        <input id="brushSize" type="range" min="1" max="80" value="18" />
        <span id="brushSizeLabel" class="muted">18</span>
        <label class="muted">Overlay</label>
        <input id="overlayAlpha" type="range" min="0" max="100" value="48" />
        <span id="overlayAlphaLabel" class="muted">48%</span>
        <label class="inline-check"><input id="currentOnlyToggle" type="checkbox" /> Current only</label>
        <button id="undoBtn">Undo</button>
        <button id="keepLargestBtn">Keep Largest</button>
        <button id="clearAllBtn" class="danger">Clear All</button>
        <button id="saveBtn">Save Mask</button>
        <button id="reviewPlayBtn">Play Review</button>
        <button id="gotoLatestBtn">Latest Mask</button>
      </div>
      <div class="canvas-wrap">
        <div id="stage" class="stage">
          <canvas id="imageCanvas"></canvas>
          <canvas id="overlayCanvas"></canvas>
          <canvas id="promptCanvas"></canvas>
        </div>
      </div>
      <div id="status" class="status">Loading...</div>
    </main>
    <aside class="right">
      <h2>Prompts</h2>
      <p class="muted">Point and bbox prompts can generate the selected part with SAM2. Text prompts are stored with the saved frame for audit/future text backends.</p>
      <div id="promptList" class="prompt-list"></div>
      <h3>Workflow</h3>
      <p class="muted">
        1. Pick a part.<br />
        2. Add point/bbox prompts, run SAM2 Current Part, or brush manually.<br />
        3. Save the indexed mask.<br />
        4. Start backend propagation from that frame.<br />
        5. Watch frames as masks arrive. Pause, correct the current frame, save, and restart from that frame.
      </p>
      <h3>Backend Propagation</h3>
      <div class="stack">
        <select id="backendSelect">
          <option value="sam2-video">SAM2 Video</option>
          <option value="cotracker-sparse">CoTracker Sparse</option>
        </select>
        <select id="sam2PartModeSelect" title="How SAM2 propagates multiple part ids">
          <option value="independent">SAM2 parts: independent</option>
          <option value="joint">SAM2 parts: joint</option>
        </select>
        <div class="row"><label>Stride</label><input id="propFrameStride" type="number" min="1" value="1" style="width: 88px" /></div>
        <div class="row">
          <button id="startPropBtn">Start From Current</button>
          <button id="pausePropBtn" class="danger">Pause Backend</button>
        </div>
        <label class="inline-check"><input id="liveFollowToggle" type="checkbox" checked /> Live follow latest frame</label>
      </div>
      <div id="propStatus" class="prop-summary">Backend idle.</div>
      <div class="progress"><div id="maskProgress"></div></div>
      <div class="row"><label>Log lines</label><input id="logLineLimit" type="range" min="0" max="80" value="12" /><span id="logLineLimitLabel" class="muted">12</span></div>
      <div id="propLog" class="prop-log"></div>
      <div id="maskStrip" class="mask-strip"></div>
      <h3>Last Save</h3>
      <textarea id="saveOutput" readonly></textarea>
      <h3>Shortcuts</h3>
      <p class="muted"><span class="kbd">1-9</span> select part, <span class="kbd">B</span> brush, <span class="kbd">E</span> erase, <span class="kbd">P</span> point, <span class="kbd">X</span> bbox, <span class="kbd">S</span> save.</p>
    </aside>
  </div>
<script>
const palette = ["#000000", "#62d26f", "#a56de2", "#f2c14e", "#4cc9f0", "#ff6b6b", "#2ec4b6", "#f72585", "#b8f35a", "#ff9f1c"];
let state = null;
let parts = [];
let currentPart = 1;
let currentFrame = 0;
let currentView = 0;
let mode = "brush";
let labels = null;
let width = 0;
let height = 0;
let drawing = false;
let bboxStart = null;
let history = [];
let prompts = [];
let reviewPlaying = false;
let reviewTimer = null;
let propagationTimer = null;
let lastPropagation = null;
let loadedMaskKey = null;

const imageCanvas = document.getElementById("imageCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const promptCanvas = document.getElementById("promptCanvas");
const imageCtx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");
const promptCtx = promptCanvas.getContext("2d");
const statusEl = document.getElementById("status");

function setStatus(text) { statusEl.textContent = text; }

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [parseInt(value.slice(0,2),16), parseInt(value.slice(2,4),16), parseInt(value.slice(4,6),16)];
}

function partColor(id) {
  const part = parts.find(p => p.part_id === id);
  return part?.color || palette[id % palette.length];
}

function clonePlain(value) {
  return JSON.parse(JSON.stringify(value));
}

async function init() {
  state = await (await fetch("/api/state")).json();
  parts = state.parts;
  currentFrame = state.default_frame_index;
  currentView = state.default_view_index;
  document.getElementById("episodeInfo").textContent = `${state.object_instance_id} / ${state.category}\nframes=${state.frame_count}\noutput=${state.output_episode}`;
  const slider = document.getElementById("frameSlider");
  slider.max = Math.max(0, state.frame_count - 1);
  slider.value = currentFrame;
  document.getElementById("frameInput").value = currentFrame;
  document.getElementById("viewInput").value = currentView;
  renderParts();
  await loadFrame();
  await pollPropagation();
  propagationTimer = setInterval(pollPropagation, 1200);
}

function renderParts() {
  const root = document.getElementById("parts");
  root.innerHTML = "";
  for (const part of parts) {
    const div = document.createElement("div");
    div.className = "part" + (part.part_id === currentPart ? " active" : "");
    div.innerHTML = `<span class="swatch" style="background:${part.color}"></span><span>${part.part_id}: ${part.name}<br><small class="muted">${part.role}</small></span><button>Select</button>`;
    div.querySelector("button").onclick = () => { currentPart = part.part_id; renderParts(); drawOverlay(); };
    root.appendChild(div);
  }
}

async function loadFrame() {
  currentFrame = Number(document.getElementById("frameInput").value);
  currentView = Number(document.getElementById("viewInput").value);
  document.getElementById("frameSlider").value = currentFrame;
  setStatus(`Loading frame ${currentFrame}, view ${currentView}...`);
  const img = new Image();
  img.onload = async () => {
    width = img.naturalWidth; height = img.naturalHeight;
    for (const canvas of [imageCanvas, overlayCanvas, promptCanvas]) {
      canvas.width = width; canvas.height = height;
    }
    imageCtx.drawImage(img, 0, 0);
    labels = new Uint16Array(width * height);
    loadedMaskKey = null;
    history = [];
    prompts = [];
    await loadExistingMask();
    drawOverlay();
    drawPrompts();
    setStatus(`Frame ${currentFrame}, view ${currentView}.`);
  };
  img.src = `/api/frame?frame=${currentFrame}&view=${currentView}&t=${Date.now()}`;
}

async function loadExistingMask() {
  const response = await fetch(`/api/mask?frame=${currentFrame}&view=${currentView}&t=${Date.now()}`);
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    loadedMaskKey = null;
    return;
  }
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, width, height).data;
  for (let i = 0; i < width * height; i++) labels[i] = data[i * 4];
  discardUnknownLabels();
}

async function refreshCurrentMaskFromStatus(maskStatus, force=false) {
  const entry = maskStatus?.frames?.find(item => Number(item.frame_index) === Number(currentFrame));
  const key = maskEntryKey(entry);
  if (!key) return false;
  if (!force && key === loadedMaskKey) return false;
  await loadExistingMask();
  loadedMaskKey = key;
  drawOverlay();
  setStatus(`Frame ${currentFrame}, view ${currentView}. Mask updated.`);
  return true;
}

function maskEntryKey(entry) {
  if (!entry || !entry.exists || !entry.mask_path) return null;
  return `${entry.mask_path}:${entry.mask_mtime_ns || 0}:${entry.mask_size_bytes || 0}`;
}

async function setFrame(frameIndex) {
  const bounded = Math.max(0, Math.min(state.frame_count - 1, Number(frameIndex)));
  document.getElementById("frameInput").value = bounded;
  document.getElementById("frameSlider").value = bounded;
  await loadFrame();
  renderMaskStrip(lastPropagation?.mask_status || null);
}

async function pollPropagation() {
  const response = await fetch(`/api/propagation/status?view=${currentView}&t=${Date.now()}`);
  const status = await response.json();
  if (status.error) throw new Error(status.error);
  lastPropagation = status;
  renderPropagationStatus(status);
  renderMaskStrip(status.mask_status);
  const latest = status.mask_status?.latest_frame;
  const liveFollow = document.getElementById("liveFollowToggle")?.checked;
  let changedFrame = false;
  if (liveFollow && status.status === "running" && latest !== null && latest !== undefined && latest !== currentFrame) {
    await setFrame(latest);
    changedFrame = true;
  } else if (reviewPlaying) {
    const next = nextReviewFrame(status.mask_status);
    if (next !== null && next !== currentFrame) {
      await setFrame(next);
      changedFrame = true;
    }
  }
  await refreshCurrentMaskFromStatus(status.mask_status, changedFrame);
}

function renderPropagationStatus(status) {
  const maskStatus = status.mask_status || {};
  const completed = maskStatus.completed_count || 0;
  const total = state.frame_count || 1;
  const pct = Math.round(100 * completed / total);
  document.getElementById("maskProgress").style.width = `${pct}%`;
  const backend = status.backend || "none";
  const ref = status.reference_frame ?? "-";
  const latest = maskStatus.latest_frame ?? "-";
  const returncode = status.returncode === null || status.returncode === undefined ? "" : ` rc=${status.returncode}`;
  document.getElementById("propStatus").textContent =
    `status=${status.status} backend=${backend} ref=${ref}\nmasks=${completed}/${total} (${pct}%) latest=${latest}${returncode}`;
  renderPropagationLog(status.log_tail || "");
}

function renderPropagationLog(rawLog) {
  const limit = Number(document.getElementById("logLineLimit")?.value || 0);
  const el = document.getElementById("propLog");
  if (!el) return;
  if (limit <= 0 || !rawLog) {
    el.textContent = "";
    return;
  }
  const lines = rawLog.split(/\r?\n/).filter(line => line.trim().length > 0);
  el.textContent = lines.slice(-limit).join("\n");
  el.scrollTop = el.scrollHeight;
}

function renderMaskStrip(maskStatus) {
  const root = document.getElementById("maskStrip");
  if (!root || !state) return;
  const done = new Set((maskStatus?.completed_frames || []).map(Number));
  root.innerHTML = "";
  const total = state.frame_count;
  for (let i = 0; i < total; i++) {
    const cell = document.createElement("div");
    cell.className = "mask-cell" + (done.has(i) ? " done" : "") + (i === currentFrame ? " current" : "");
    cell.title = `frame ${i}` + (done.has(i) ? " mask exists" : " no mask");
    cell.onclick = () => setFrame(i);
    root.appendChild(cell);
  }
}

function nextReviewFrame(maskStatus) {
  const done = (maskStatus?.completed_frames || []).map(Number).sort((a, b) => a - b);
  if (!done.length) return null;
  const greater = done.find(i => i > currentFrame);
  return greater ?? done[0];
}

async function startPropagation() {
  await saveMask();
  const payload = {
    backend: document.getElementById("backendSelect").value,
    reference_frame: currentFrame,
    view_index: currentView,
    frame_stride: Number(document.getElementById("propFrameStride").value || 1),
    sam2_part_mode: document.getElementById("sam2PartModeSelect").value,
  };
  const response = await fetch("/api/propagation/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const status = await response.json();
  if (!response.ok || status.error) throw new Error(status.error || "Propagation start failed");
  reviewPlaying = false;
  document.getElementById("reviewPlayBtn").textContent = "Play Review";
  document.getElementById("liveFollowToggle").checked = true;
  lastPropagation = status;
  renderPropagationStatus(status);
  renderMaskStrip(status.mask_status);
}

async function pausePropagation() {
  const response = await fetch("/api/propagation/pause", {method: "POST"});
  const status = await response.json();
  if (!response.ok || status.error) throw new Error(status.error || "Pause failed");
  lastPropagation = status;
  renderPropagationStatus(status);
  renderMaskStrip(status.mask_status);
}

async function gotoLatestMask() {
  await pollPropagation();
  const latest = lastPropagation?.mask_status?.latest_frame;
  if (latest !== null && latest !== undefined) await setFrame(latest);
}

function foregroundPixelCount() {
  let count = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] > 0) count++;
  return count;
}

async function loadLabelsDataUrl(dataUrl) {
  const blob = await (await fetch(dataUrl)).blob();
  const bitmap = await createImageBitmap(blob);
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, width, height).data;
  for (let i = 0; i < width * height; i++) labels[i] = data[i * 4];
  discardUnknownLabels();
}

function discardUnknownLabels() {
  if (!labels) return;
  const valid = new Set(parts.map(part => Number(part.part_id)));
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > 0 && !valid.has(Number(labels[i]))) labels[i] = 0;
  }
}

async function predictCurrentPartMask() {
  setStatus(`Running SAM2 prompt prediction for part ${currentPart}...`);
  const payload = {
    frame_index: currentFrame,
    view_index: currentView,
    part_id: currentPart,
    width, height,
    mask_data_url: labelDataUrl(),
    prompts,
  };
  const response = await fetch("/api/predict-current-mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || "SAM2 prompt prediction failed");
  pushHistory();
  await loadLabelsDataUrl(result.labels_data_url);
  drawOverlay();
  setStatus(`SAM2 predicted part ${result.part_id}: ${result.mask_pixels} pixels.`);
}

function pushHistory() {
  if (!labels) return;
  history.push({
    labels: labels.slice(),
    prompts: clonePlain(prompts),
    parts: clonePlain(parts),
    currentPart,
  });
  if (history.length > 20) history.shift();
}

function restoreHistory(snapshot) {
  if (!snapshot) {
    setStatus("Nothing to undo.");
    return;
  }
  labels = snapshot.labels;
  prompts = snapshot.prompts || [];
  parts = snapshot.parts || parts;
  currentPart = snapshot.currentPart || currentPart;
  renderParts();
  drawOverlay();
  drawPrompts();
  renderPromptList();
  setStatus("Undo applied. Click Save Mask to persist it.");
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, width, height);
  if (!labels) return;
  const alphaScale = Number(document.getElementById("overlayAlpha")?.value || 48) / 100;
  const currentOnly = Boolean(document.getElementById("currentOnlyToggle")?.checked);
  const img = overlayCtx.createImageData(width, height);
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i];
    if (!id) continue;
    if (currentOnly && id !== currentPart) continue;
    const [r,g,b] = hexToRgb(partColor(id));
    img.data[i*4] = r;
    img.data[i*4+1] = g;
    img.data[i*4+2] = b;
    const baseAlpha = id === currentPart ? 150 : 95;
    img.data[i*4+3] = Math.round(baseAlpha * alphaScale);
  }
  overlayCtx.putImageData(img, 0, 0);
}

function keepLargestCurrentPart() {
  if (!labels || width <= 0 || height <= 0) return;
  const visited = new Uint8Array(labels.length);
  const queue = new Int32Array(labels.length);
  let best = [];
  const neighbors = [[1,0], [-1,0], [0,1], [0,-1]];
  for (let start = 0; start < labels.length; start++) {
    if (visited[start] || labels[start] !== currentPart) continue;
    let head = 0, tail = 0;
    const component = [];
    visited[start] = 1;
    queue[tail++] = start;
    while (head < tail) {
      const index = queue[head++];
      component.push(index);
      const x = index % width;
      const y = Math.floor(index / width);
      for (const [dx, dy] of neighbors) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const ni = ny * width + nx;
        if (!visited[ni] && labels[ni] === currentPart) {
          visited[ni] = 1;
          queue[tail++] = ni;
        }
      }
    }
    if (component.length > best.length) best = component;
  }
  if (!best.length) {
    setStatus(`Part ${currentPart} has no pixels to clean.`);
    return;
  }
  pushHistory();
  for (let i = 0; i < labels.length; i++) if (labels[i] === currentPart) labels[i] = 0;
  for (const index of best) labels[index] = currentPart;
  drawOverlay();
  setStatus(`Kept largest component for part ${currentPart}: ${best.length} pixels.`);
}

function drawPrompts(tempBox=null) {
  promptCtx.clearRect(0,0,width,height);
  promptCtx.lineWidth = 2;
  for (const prompt of prompts) drawPrompt(prompt);
  if (tempBox) drawPrompt(tempBox, true);
}

function drawPrompt(prompt, temporary=false) {
  const color = prompt.type === "point_neg" ? "#ff5d5d" : partColor(prompt.part_id || currentPart);
  promptCtx.strokeStyle = color;
  promptCtx.fillStyle = color;
  promptCtx.setLineDash(temporary ? [6,4] : []);
  if (prompt.type === "point_pos" || prompt.type === "point_neg") {
    promptCtx.beginPath();
    promptCtx.arc(prompt.x, prompt.y, 6, 0, Math.PI * 2);
    promptCtx.stroke();
    promptCtx.beginPath();
    promptCtx.moveTo(prompt.x - 9, prompt.y);
    promptCtx.lineTo(prompt.x + 9, prompt.y);
    promptCtx.moveTo(prompt.x, prompt.y - 9);
    promptCtx.lineTo(prompt.x, prompt.y + 9);
    promptCtx.stroke();
  } else if (prompt.type === "bbox") {
    promptCtx.strokeRect(prompt.x1, prompt.y1, prompt.x2 - prompt.x1, prompt.y2 - prompt.y1);
  }
  promptCtx.setLineDash([]);
}

function renderPromptList() {
  const root = document.getElementById("promptList");
  root.innerHTML = "";
  prompts.forEach((prompt, index) => {
    const div = document.createElement("div");
    div.className = "prompt";
    div.textContent = `${index + 1}. part ${prompt.part_id} ${prompt.type}: ` + JSON.stringify(prompt);
    root.appendChild(div);
  });
}

function canvasPoint(event) {
  const rect = promptCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(width - 1, Math.round((event.clientX - rect.left) * width / rect.width))),
    y: Math.max(0, Math.min(height - 1, Math.round((event.clientY - rect.top) * height / rect.height))),
  };
}

function paintAt(x, y) {
  const radius = Number(document.getElementById("brushSize").value);
  const value = mode === "erase" ? 0 : currentPart;
  const r2 = radius * radius;
  const minX = Math.max(0, x - radius), maxX = Math.min(width - 1, x + radius);
  const minY = Math.max(0, y - radius), maxY = Math.min(height - 1, y + radius);
  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const dx = xx - x, dy = yy - y;
      if (dx*dx + dy*dy <= r2) labels[yy * width + xx] = value;
    }
  }
}

promptCanvas.addEventListener("mousedown", event => {
  const p = canvasPoint(event);
  if (mode === "brush" || mode === "erase") {
    pushHistory(); drawing = true; paintAt(p.x, p.y); drawOverlay();
  } else if (mode === "bbox") {
    bboxStart = p;
  }
});

promptCanvas.addEventListener("mousemove", event => {
  const p = canvasPoint(event);
  if (drawing) { paintAt(p.x, p.y); drawOverlay(); }
  if (bboxStart) {
    drawPrompts({type: "bbox", part_id: currentPart, x1: bboxStart.x, y1: bboxStart.y, x2: p.x, y2: p.y});
  }
});

window.addEventListener("mouseup", event => {
  if (drawing) drawing = false;
  if (bboxStart) {
    const p = canvasPoint(event);
    const prompt = {
      type: "bbox",
      part_id: currentPart,
      x1: Math.min(bboxStart.x, p.x),
      y1: Math.min(bboxStart.y, p.y),
      x2: Math.max(bboxStart.x, p.x),
      y2: Math.max(bboxStart.y, p.y),
    };
    if (Math.abs(prompt.x2 - prompt.x1) > 3 && Math.abs(prompt.y2 - prompt.y1) > 3) {
      pushHistory();
      prompts.push(prompt);
    }
    bboxStart = null;
    drawPrompts(); renderPromptList();
  }
});

promptCanvas.addEventListener("click", event => {
  if (mode !== "point_pos" && mode !== "point_neg") return;
  const p = canvasPoint(event);
  pushHistory();
  prompts.push({type: mode, part_id: currentPart, x: p.x, y: p.y});
  drawPrompts();
  renderPromptList();
});

function setMode(next) {
  mode = next;
  document.querySelectorAll(".mode").forEach(btn => btn.classList.toggle("active", btn.dataset.mode === mode));
}

document.querySelectorAll(".mode").forEach(btn => btn.onclick = () => setMode(btn.dataset.mode));
document.getElementById("brushSize").oninput = e => document.getElementById("brushSizeLabel").textContent = e.target.value;
document.getElementById("overlayAlpha").oninput = e => {
  document.getElementById("overlayAlphaLabel").textContent = `${e.target.value}%`;
  drawOverlay();
};
document.getElementById("currentOnlyToggle").onchange = drawOverlay;
document.getElementById("logLineLimit").oninput = e => {
  document.getElementById("logLineLimitLabel").textContent = e.target.value;
  renderPropagationLog(lastPropagation?.log_tail || "");
};
document.getElementById("frameSlider").oninput = e => { document.getElementById("frameInput").value = e.target.value; };
document.getElementById("loadFrameBtn").onclick = loadFrame;
document.getElementById("startPropBtn").onclick = () => startPropagation().catch(err => setStatus(`Propagation start failed: ${err.message}`));
document.getElementById("pausePropBtn").onclick = () => pausePropagation().catch(err => setStatus(`Pause failed: ${err.message}`));
document.getElementById("gotoLatestBtn").onclick = () => gotoLatestMask().catch(err => setStatus(`Latest failed: ${err.message}`));
document.getElementById("reviewPlayBtn").onclick = () => {
  reviewPlaying = !reviewPlaying;
  document.getElementById("reviewPlayBtn").textContent = reviewPlaying ? "Pause Review" : "Play Review";
};
document.getElementById("undoBtn").onclick = () => {
  restoreHistory(history.pop());
};
document.getElementById("keepLargestBtn").onclick = keepLargestCurrentPart;
document.getElementById("clearPartBtn").onclick = () => {
  pushHistory();
  for (let i = 0; i < labels.length; i++) if (labels[i] === currentPart) labels[i] = 0;
  drawOverlay();
};
document.getElementById("clearAllBtn").onclick = () => {
  pushHistory();
  labels.fill(0);
  prompts = [];
  drawOverlay(); drawPrompts(); renderPromptList();
};
document.getElementById("addPartBtn").onclick = () => {
  pushHistory();
  const nextId = Math.max(...parts.map(p => p.part_id), 0) + 1;
  parts.push({part_id: nextId, name: `part_${nextId}`, role: "moving", color: palette[nextId % palette.length]});
  currentPart = nextId;
  renderParts();
};
document.getElementById("addTextPromptBtn").onclick = () => {
  const text = document.getElementById("textPrompt").value.trim();
  if (!text) return;
  pushHistory();
  prompts.push({type: "text", part_id: currentPart, text});
  document.getElementById("textPrompt").value = "";
  renderPromptList();
};

function labelDataUrl() {
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < labels.length; i++) {
    img.data[i*4] = Math.min(255, labels[i]);
    img.data[i*4+1] = 0;
    img.data[i*4+2] = 0;
    img.data[i*4+3] = labels[i] > 0 ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

async function saveMask() {
  if (foregroundPixelCount() === 0) {
    throw new Error("Current mask is empty. Brush a part or run SAM2 Current Part from point/bbox prompts first.");
  }
  setStatus("Saving mask...");
  const payload = {
    frame_index: currentFrame,
    view_index: currentView,
    width, height,
    mask_data_url: labelDataUrl(),
    parts,
    prompts,
  };
  const response = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || "Save failed");
  document.getElementById("saveOutput").value = JSON.stringify(result, null, 2);
  setStatus(`Saved ${result.mask_path_relative}\n${result.propagate_command}`);
}
document.getElementById("saveBtn").onclick = () => saveMask().catch(err => setStatus(`Save failed: ${err.message}`));
document.getElementById("predictPartBtn").onclick = () => predictCurrentPartMask().catch(err => setStatus(`SAM2 prompt failed: ${err.message}`));

window.addEventListener("keydown", event => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    restoreHistory(history.pop());
    return;
  }
  if (/^[1-9]$/.test(event.key)) {
    const part = parts.find(p => p.part_id === Number(event.key));
    if (part) { currentPart = part.part_id; renderParts(); drawOverlay(); }
  }
  if (event.key.toLowerCase() === "b") setMode("brush");
  if (event.key.toLowerCase() === "e") setMode("erase");
  if (event.key.toLowerCase() === "p") setMode("point_pos");
  if (event.key.toLowerCase() === "x") setMode("bbox");
  if (event.key.toLowerCase() === "s") saveMask().catch(err => setStatus(`Save failed: ${err.message}`));
});

init().catch(err => setStatus(`Failed to load: ${err.message}`));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
