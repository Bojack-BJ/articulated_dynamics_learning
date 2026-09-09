#!/usr/bin/env python3
"""Serve a directory of flow viewers with per-sequence annotation saving."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from serve_joint_annotation import _atomic_save, _validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8901)
    args = parser.parse_args()
    viewer_root = args.viewer_root.resolve()
    output_root = args.output_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def sequence(self) -> str | None:
            path = urlparse(self.path).path
            pieces = [piece for piece in path.split("/") if piece]
            if pieces and pieces[-1] == "api":
                pieces.pop()
            if len(pieces) >= 2 and pieces[-2:] in (["api", "annotation"], ["api", "save-annotation"]):
                pieces = pieces[:-2]
            if pieces and pieces[0] != "index.html":
                return pieces[0]
            referer = self.headers.get("Referer", "")
            ref_pieces = [piece for piece in urlparse(referer).path.split("/") if piece]
            return ref_pieces[0] if ref_pieces and ref_pieces[0] != "index.html" else None

        def annotation_path(self) -> Path:
            sequence = self.sequence()
            if not sequence or Path(sequence).name != sequence:
                raise ValueError("Cannot determine a safe sequence from the viewer URL")
            return output_root / sequence / "relation_gt_manual.json"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path.endswith("/api/annotation") or path == "/api/annotation":
                try:
                    output = self.annotation_path()
                    payload = json.loads(output.read_text()) if output.exists() else None
                    self.send_json(HTTPStatus.OK, {"annotation": payload, "path": str(output)})
                except ValueError as error:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            relative = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
            target = (viewer_root / relative).resolve()
            if viewer_root not in target.parents and target != viewer_root:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if target.is_dir():
                target /= "viewer_gt_axis.html"
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not (path.endswith("/api/save-annotation") or path == "/api/save-annotation"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = _validate(json.loads(self.rfile.read(length)))
                output = self.annotation_path()
                _atomic_save(payload, output)
                self.send_json(HTTPStatus.OK, {
                    "path": str(output),
                    "track_label_count": len(payload.get("track_labels", {})),
                    "joint_count": len(payload.get("joints", [])),
                })
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def send_json(self, status: HTTPStatus, payload: dict) -> None:
            content = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Annotation collection: http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
