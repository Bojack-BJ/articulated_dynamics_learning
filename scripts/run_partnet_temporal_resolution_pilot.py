#!/usr/bin/env python3
"""Record and track a small PartNet pilot at higher temporal resolution."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_OBJECT_IDS = ("partnet_102149", "partnet_46084", "partnet_46166", "partnet_47466", "partnet_47954")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--recording-fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--tracking-frame-stride", type=int, default=2)
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Skip auto-camera-fit flags for compatibility with older remote CLI versions.",
    )
    parser.add_argument(
        "--reuse-recording-root", type=Path,
        help="Reuse existing episodes from this root and only regenerate tracks/features.",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--cotracker-repo", type=Path, default=Path("co-tracker"))
    parser.add_argument(
        "--cotracker-checkpoint", type=Path,
        default=Path("co-tracker/ckpt/scaled_offline.pth"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_objects(path: Path, requested: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    selected = [row for row in rows if str(row["object_id"]) in requested]
    missing = requested - {str(row["object_id"]) for row in selected}
    if missing:
        raise ValueError(f"Objects missing from catalog: {sorted(missing)}")
    return sorted(selected, key=lambda row: str(row["object_id"]))


def run(command: list[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def process_object(
    row: dict[str, Any], args: argparse.Namespace, gpu: str,
) -> dict[str, Any]:
    object_id = str(row["object_id"])
    object_root = args.output_root / object_id
    episode = (
        args.reuse_recording_root.expanduser().resolve() / object_id / "episode.json"
        if args.reuse_recording_root is not None
        else object_root / "episode.json"
    )
    artifact_dir = object_root / "pointcloud_4d_partseg"
    tracks = artifact_dir / "part_tracks.json"
    features = artifact_dir / "cotracker_features.npz"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.reuse_recording_root is None and not (args.resume and episode.is_file()):
        record_command = [
            args.python, "-m", "rgbd_urdf_mvp", "record-mujoco", str(row["model_path"]),
            "--category", str(row["category"]), "--object-id", object_id,
            "--output-dir", str(args.output_root), "--duration-s", str(args.duration_s),
            "--fps", str(args.recording_fps), "--width", "480", "--height", "352",
            "--control-mode", "staggered", "--staggered-max-acceleration", "50.0",
            "--all-joints", "--disable-gravity", "--disable-target-collision",
            "--perturbation-scale", "0.0", "--rgb-format", "png", "--depth-format", "png",
            "--mask-format", "png", "--camera-distance", "3.0", "--camera-mode", "triview",
            "--camera-triview-spacing-deg", "60", "--camera-elevation-deg", "-15",
            "--camera-fovy-deg", "60", "--lookat", "0", "0", "0",
            "--segmentation-masks", "--part-segmentation-masks",
        ]
        if not args.fixed_camera:
            record_command.extend(["--auto-camera-fit", "--camera-fit-fill-ratio", "0.42"])
        run(record_command, log_path=args.output_root / "_logs" / f"{object_id}.record.log", env=env)
    if not (args.resume and tracks.is_file() and features.is_file()):
        run([
            args.python, "-m", "rgbd_urdf_mvp", "track-part-pixels", str(episode),
            "--output-json", str(tracks), "--cotracker-repo", str(args.cotracker_repo),
            "--cotracker-checkpoint", str(args.cotracker_checkpoint), "--device", "cuda",
            "--reference-frame", "-1", "--frame-stride", str(args.tracking_frame_stride),
            "--seed-stride-px", "12", "--max-tracks-per-part-view", "192",
            "--visibility-threshold", "0.5", "--export-cotracker-features",
        ], log_path=args.output_root / "_logs" / f"{object_id}.track.log", env=env)
    artifact = json.loads(tracks.read_text(encoding="utf-8"))
    return {
        "object_id": object_id,
        "split": str(row.get("split", "test")),
        "tracks_path": str(tracks.resolve()),
        "features_npz": str(features.resolve()),
        "episode_frame_count": len(json.loads(episode.read_text(encoding="utf-8")).get("frames", [])),
        "tracking_frame_count": int(artifact.get("frame_count", 0)),
        "effective_tracking_fps_hz": float(artifact.get("effective_tracking_fps_hz", 0.0)),
        "gpu": gpu,
    }


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    requested = set(args.object_ids or DEFAULT_OBJECT_IDS)
    objects = load_objects(args.catalog, requested)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    results = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(objects))) as executor:
        futures = {
            executor.submit(process_object, row, args, gpus[index % len(gpus)]): str(row["object_id"])
            for index, row in enumerate(objects)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    results.sort(key=lambda row: str(row["object_id"]))
    manifest = args.output_root / "cotracker_temporal_pilot_manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["object_id", "tracks_path", "features_npz", "split"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in results)
    (args.output_root / "temporal_pilot_summary.json").write_text(
        json.dumps({
            "recording_fps": args.recording_fps,
            "reuse_recording_root": (
                str(args.reuse_recording_root.expanduser().resolve())
                if args.reuse_recording_root is not None else None
            ),
            "tracking_frame_stride": args.tracking_frame_stride,
            "expected_effective_tracking_fps": args.recording_fps / args.tracking_frame_stride,
            "objects": results,
            "manifest": str(manifest),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
