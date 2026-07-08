#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _module_cmd(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "rgbd_urdf_mvp", *[str(arg) for arg in args]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_paths(object_dir: Path) -> dict[str, Path]:
    baseline_dir = object_dir / "em_lite" / "baseline"
    return {
        "tracks": baseline_dir / "motion_part_tracks_knn.json",
        "joints": baseline_dir / "joint_inference_knn.json",
        "evaluation": baseline_dir / "object_mask_kinematic_evaluation_knn.json",
    }


def _generate_object_viewers(cwd: Path, object_dir: Path, max_tracks: int, dry_run: bool) -> dict[str, str]:
    paths = _baseline_paths(object_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {"object_id": object_dir.name, "status": "missing_inputs", "missing": ", ".join(missing)}

    debug_dir = object_dir / "em_lite" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    quality_dir = debug_dir / "track_quality"
    quality_tracks = quality_dir / "motion_part_tracks_with_quality.json"
    baseline_viewer = debug_dir / "baseline_flow_viewer.html"
    quality_viewer = debug_dir / "baseline_quality_flow_viewer.html"
    index_html = debug_dir / "index.html"

    if not quality_tracks.exists():
        _run(_module_cmd("compute-track-quality", paths["tracks"], "--output-dir", quality_dir), cwd, dry_run)

    _run(
        _module_cmd(
            "visualize-object-mask-flow-html",
            paths["tracks"],
            "--joint-inference",
            paths["joints"],
            "--evaluation-json",
            paths["evaluation"],
            "--output-html",
            baseline_viewer,
            "--max-tracks",
            str(max_tracks),
            "--frame-stride",
            "2",
            "--trail-length",
            "10",
            "--color-by",
            "pred_cluster",
        ),
        cwd,
        dry_run,
    )
    if quality_tracks.exists() or dry_run:
        _run(
            _module_cmd(
                "visualize-object-mask-flow-html",
                quality_tracks,
                "--joint-inference",
                paths["joints"],
                "--evaluation-json",
                paths["evaluation"],
                "--output-html",
                quality_viewer,
                "--max-tracks",
                str(max_tracks),
                "--frame-stride",
                "2",
                "--trail-length",
                "10",
                "--color-by",
                "timestep_quality",
            ),
            cwd,
            dry_run,
        )
    evaluation = _load_json(paths["evaluation"]) if paths["evaluation"].exists() and not dry_run else {}
    summary = evaluation.get("summary") or {}
    index_html.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'>",
                f"<title>{object_dir.name} EM-lite debug</title>",
                "<style>body{font-family:Arial,sans-serif;background:#10151c;color:#dbe7f3;margin:24px}"
                "a{color:#7dd3fc}code{background:#1e293b;padding:2px 4px;border-radius:3px}</style>",
                "</head><body>",
                f"<h1>{object_dir.name} EM-lite debug viewers</h1>",
                f"<p>mean_cluster_purity: {summary.get('mean_cluster_purity')}</p>",
                f"<p>mean_gt_coverage: {summary.get('mean_gt_coverage')}</p>",
                f"<p>axis_mean_deg: {summary.get('axis_angle_error_deg_mean')}</p>",
                f"<p>pivot_mean_m: {summary.get('pivot_error_m_mean')}</p>",
                "<ul>",
                f"<li><a href='{baseline_viewer.name}'>Baseline flow viewer</a></li>",
                f"<li><a href='{quality_viewer.name}'>Timestep-quality flow viewer</a></li>",
                "</ul>",
                "<p>This viewer reflects the current shared <code>em_lite/baseline</code> artifact for the object.</p>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "object_id": object_dir.name,
        "status": "ok",
        "index": str(index_html.resolve()),
        "baseline_viewer": str(baseline_viewer.resolve()),
        "quality_viewer": str(quality_viewer.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate EM-lite debug viewers from existing object artifacts.")
    parser.add_argument("object_dirs", nargs="+", type=Path)
    parser.add_argument("--max-tracks", type=int, default=1000)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    rows = [
        _generate_object_viewers(cwd, object_dir.expanduser().resolve(), max(1, int(args.max_tracks)), args.dry_run)
        for object_dir in args.object_dirs
    ]
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"viewers": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"viewers": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
