from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_baseline_viewer_index_links_all_object_viewers(tmp_path: Path) -> None:
    for name in ("object_a", "object_b"):
        viewer = tmp_path / name / "viewer.html"
        viewer.parent.mkdir()
        viewer.write_text("<html></html>", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_baseline_viewer_index.py",
            str(tmp_path),
        ],
        check=True,
    )

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'data-src="object_a/viewer.html"' in index
    assert 'data-src="object_b/viewer.html"' in index
