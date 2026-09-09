#!/usr/bin/env python3
"""Build an index for a directory of baseline comparison viewers."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.viewer_root.expanduser().resolve()
    viewers = sorted(root.glob("*/viewer.html"))
    if not viewers:
        raise FileNotFoundError(f"No */viewer.html files found under {root}")
    output = (args.output or root / "index.html").expanduser().resolve()
    links = "\n".join(
        (
            f'<button data-src="{html.escape(path.relative_to(root).as_posix())}">'
            f"{html.escape(path.parent.name.replace('_', ' '))}</button>"
        )
        for path in viewers
    )
    first = viewers[0].relative_to(root).as_posix()
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Baseline Viewer Gallery</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; height: 100vh; display: grid; grid-template-columns: 220px 1fr;
      background: #081018; color: #e8f0f6; }}
    aside {{ padding: 20px 14px; border-right: 1px solid #253442; overflow: auto; }}
    h1 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ color: #8fa5b5; font-size: 12px; line-height: 1.45; }}
    button {{ width: 100%; margin: 5px 0; padding: 10px 12px; text-align: left;
      border: 1px solid #2b4050; border-radius: 8px; background: #111e28;
      color: #dbe8ef; cursor: pointer; text-transform: capitalize; }}
    button.active {{ border-color: #58d6ad; background: #17382f; }}
    iframe {{ width: 100%; height: 100vh; border: 0; background: #081018; }}
  </style>
</head>
<body>
  <aside>
    <h1>Baseline Viewers</h1>
    <p>Select an object, then use the method selector inside the viewer for
    GT, Ours/Hybrid, AiM, and ReArt when available.</p>
    {links}
  </aside>
  <iframe id="viewer" src="{html.escape(first)}"></iframe>
  <script>
    const frame = document.getElementById("viewer");
    const buttons = Array.from(document.querySelectorAll("button[data-src]"));
    function select(button) {{
      buttons.forEach(item => item.classList.toggle("active", item === button));
      frame.src = button.dataset.src;
    }}
    buttons.forEach(button => button.addEventListener("click", () => select(button)));
    select(buttons[0]);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
