from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve an object-flow viewer and save GT annotations to a fixed data path."
    )
    parser.add_argument("viewer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    viewer = args.viewer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not viewer.is_file():
        raise FileNotFoundError(f"Viewer not found: {viewer}")
    handler = _handler(viewer, output)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Joint annotation viewer: http://{args.host}:{args.port}/")
    print(f"Annotations save to: {output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Annotation payload must be an object.")
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Expected annotation schema_version=2.")
    if not isinstance(payload.get("track_labels", {}), dict):
        raise ValueError("track_labels must be an object.")
    joints = payload.get("joints", [])
    if not isinstance(joints, list):
        raise ValueError("joints must be a list.")
    for index, joint in enumerate(joints):
        if not isinstance(joint, dict):
            raise ValueError(f"joints[{index}] must be an object.")
        if joint.get("joint_type") not in {"revolute", "prismatic"}:
            raise ValueError(f"joints[{index}] has an invalid joint_type.")
        for name in ("axis", "pivot"):
            vector = joint.get(name)
            if not isinstance(vector, list) or len(vector) != 3:
                raise ValueError(f"joints[{index}].{name} must contain three numbers.")
            joint[name] = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in joint["axis"]))
        if norm < 1e-9:
            raise ValueError(f"joints[{index}].axis cannot be zero.")
        joint["axis"] = [value / norm for value in joint["axis"]]
    return payload


def _atomic_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.copy2(output, output.with_suffix(output.suffix + ".bak"))
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _handler(viewer: Path, output: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/annotation":
                payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
                self._json(HTTPStatus.OK, {"annotation": payload, "path": str(output)})
                return
            if self.path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = viewer.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/save-annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = _validate(json.loads(self.rfile.read(length)))
                _atomic_save(payload, output)
                self._json(
                    HTTPStatus.OK,
                    {
                        "path": str(output),
                        "track_label_count": len(payload.get("track_labels", {})),
                        "joint_count": len(payload.get("joints", [])),
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            content = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            print(format % args)

    return Handler


if __name__ == "__main__":
    raise SystemExit(main())
