from __future__ import annotations

import runpy
import sys
from pathlib import Path


def test_videoartgs_runner_help() -> None:
    script = Path("scripts/run_videoartgs_external_baseline.py")
    original = sys.argv
    sys.argv = [str(script), "--help"]
    try:
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = original
