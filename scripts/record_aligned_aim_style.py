#!/usr/bin/env python3
"""Record an aligned PartNet subset with the AiM-style acquisition protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_tsv", type=Path)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--mujoco-gl",
        choices=("egl", "glfw", "osmesa", "cgl"),
        help="Optional MuJoCo rendering backend, for example egl on a headless Linux host.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.objects_tsv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_record_one, args, row) for row in rows]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda row: row["object_id"])
    (args.output_root / "recording_manifest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    failed = [row for row in results if row["status"] != "success"]
    return 1 if failed else 0


def _record_one(args: argparse.Namespace, row: dict[str, str]) -> dict[str, Any]:
    object_id = row["object_id"]
    recording_id = row.get("recording_id") or object_id
    output = args.output_root / recording_id
    episode = output / "episode.json"
    if args.resume and episode.is_file():
        return {
            "object_id": object_id,
            "recording_id": recording_id,
            "status": "success",
            "resumed": True,
        }
    model = args.models_root / f"{object_id}.xml"
    started = time.perf_counter()
    command = _build_command(args, row, model)
    log = args.output_root / "_logs" / f"{recording_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "object_id": object_id,
        "recording_id": recording_id,
        "status": "failed",
        "interaction_frames": int(row.get("interaction_frames") or 120),
        "fps": float(row.get("fps") or 15),
        "recording_protocol": row.get("recording_protocol") or "aim_style",
    }
    try:
        env = os.environ.copy()
        if args.mujoco_gl:
            env["MUJOCO_GL"] = args.mujoco_gl
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
            )
        status["status"] = "success"
        status["episode"] = str(episode.resolve())
    except Exception as error:
        status["exception"] = f"{type(error).__name__}: {error}"
        status["log"] = str(log.resolve())
    status["runtime_s"] = time.perf_counter() - started
    print(json.dumps(status), flush=True)
    return status


def _build_command(
    args: argparse.Namespace,
    row: dict[str, str],
    model: Path,
) -> list[str]:
    interaction_frames = int(row.get("interaction_frames") or 120)
    fps = float(row.get("fps") or 15)
    if interaction_frames <= 0 or fps <= 0:
        raise ValueError("interaction_frames and fps must be positive")
    duration_s = interaction_frames / fps
    object_id = row["object_id"]
    recording_id = row.get("recording_id") or object_id
    return [
        str(args.python),
        "-m",
        "rgbd_urdf_mvp",
        "record-mujoco",
        str(model),
        "--category",
        row["category"],
        "--object-id",
        recording_id,
        "--output-dir",
        str(args.output_root),
        "--duration-s",
        f"{duration_s:.12g}",
        "--fps",
        f"{fps:.12g}",
        "--width",
        "480",
        "--height",
        "352",
        "--control-mode",
        row.get("control_mode") or "staggered",
        "--staggered-max-acceleration",
        "50",
        "--all-joints",
        "--disable-gravity",
        "--disable-target-collision",
        "--perturbation-scale",
        "0",
        "--rgb-format",
        "png",
        "--depth-format",
        "png",
        "--mask-format",
        "png",
        "--camera-distance",
        "3",
        "--auto-camera-fit",
        "--camera-fit-fill-ratio",
        str(float(row.get("camera_fit_fill_ratio") or 0.42)),
        "--camera-mode",
        "orbit",
        "--camera-azimuth-start-deg",
        str(float(row.get("camera_azimuth_start_deg") or 50.0)),
        "--camera-elevation-deg",
        "-15",
        "--camera-fovy-deg",
        "60",
        "--lookat",
        "0",
        "0",
        "0",
        "--segmentation-masks",
        "--part-segmentation-masks",
        "--recording-protocol",
        row.get("recording_protocol") or "aim_style",
        "--aim-static-scan-views",
        str(int(row.get("aim_static_scan_views") or 24)),
        "--aim-static-scan-elevation-deg",
        "15",
        "--aim-end-scan-views",
        str(int(row.get("aim_end_scan_views") or 12)),
        "--aim-interaction-camera-orbits",
        str(float(row.get("aim_interaction_camera_orbits") or 1.0)),
        "--aim-interaction-elevation-amplitude-deg",
        str(float(row.get("aim_interaction_elevation_amplitude_deg") or 10.0)),
        "--interaction-camera-trajectory",
        row.get("interaction_camera_trajectory") or "current_orbit",
        "--aim-interaction-motion-end-fraction",
        str(float(row.get("aim_interaction_motion_end_fraction") or 0.94)),
    ] + _paper_simultaneous_args(row) + _fixed_interaction_view_args(row)


def _paper_simultaneous_args(row: dict[str, str]) -> list[str]:
    """Serialize TSV JSON ranges without changing legacy staggered manifests."""
    raw = (row.get("paper_simultaneous_joints") or "").strip()
    if not raw:
        return []
    targets = json.loads(raw)
    if not isinstance(targets, list):
        raise ValueError("paper_simultaneous_joints must be a JSON list of [name, start, end]")
    command: list[str] = []
    for target in targets:
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("Each paper_simultaneous_joints entry must be [name, start, end]")
        command.extend(["--paper-simultaneous-joint", str(target[0]), str(target[1]), str(target[2])])
    return command


def _fixed_interaction_view_args(row: dict[str, str]) -> list[str]:
    azimuths = [
        value.strip()
        for value in (row.get("interaction_fixed_view_azimuths_deg") or "").split(",")
        if value.strip()
    ]
    elevations = [
        value.strip()
        for value in (row.get("interaction_fixed_view_elevations_deg") or "").split(",")
        if value.strip()
    ]
    if not azimuths:
        return []
    args = ["--aim-interaction-fixed-view-azimuths-deg", *azimuths]
    if elevations:
        args.extend(["--aim-interaction-fixed-view-elevations-deg", *elevations])
    return args


if __name__ == "__main__":
    raise SystemExit(main())
