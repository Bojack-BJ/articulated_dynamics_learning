#!/usr/bin/env python3
"""Serve HOI4D GT review pages with an explicit RGB/mask asset allowlist."""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    args = parser.parse_args()
    viewer_root = args.viewer_root.resolve()
    assets: dict[tuple[str, str, int], Path] = {}
    for episode_path in sorted(args.episodes_root.resolve().glob("*/episode.json")):
        episode = json.loads(episode_path.read_text())
        sequence = episode_path.parent.name
        for index, frame in enumerate(episode["frames"]):
            assets[(sequence, "rgb", index)] = Path(frame["rgb_path"]).resolve()
            assets[(sequence, "mask", index)] = Path(frame["part_mask_path"]).resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(viewer_root), **handler_kwargs)

        def do_GET(self) -> None:  # noqa: N802
            parts = unquote(urlparse(self.path).path).strip("/").split("/")
            if len(parts) == 4 and parts[0] == "assets":
                try:
                    key = (parts[1], parts[2], int(parts[3]))
                except ValueError:
                    self.send_error(404)
                    return
                asset = assets.get(key)
                if asset is None or not asset.is_file():
                    self.send_error(404)
                    return
                if key[1] == "mask":
                    with Image.open(asset) as source:
                        # Canvas expands palette PNGs to RGB. Re-encode the raw
                        # palette indices as grayscale so JS receives label IDs.
                        indexed = Image.new("L", source.size)
                        indexed.putdata(list(source.getdata()))
                        buffer = io.BytesIO()
                        indexed.save(buffer, format="PNG")
                        payload = buffer.getvalue()
                else:
                    payload = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header(
                    "Cache-Control",
                    "no-store" if key[1] == "mask" else "public, max-age=3600",
                )
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/", "asset_count": len(assets)}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
