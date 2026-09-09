#!/usr/bin/env python3
"""Prepare and launch the validated Track2Art real-scene profile on a remote GPU host."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--host", default="dev-h")
    parser.add_argument(
        "--repo-root",
        default="/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning",
    )
    parser.add_argument(
        "--dataset-root",
        default="/lumos-vePFS/suzhou/Users/lixiaotong/datasets/track2art_real_20260814/scenes",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--python-bin",
        default="/root/Users/miniconda3/envs/particulate/bin/python",
        help="Remote CUDA environment containing PyTorch, CoTracker, sklearn, and MuJoCo",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--object-masks", action="store_true", help="Use binary mask/<view>/1 masks without part labels.")
    return parser.parse_args()


def remote(host: str, command: str, *, dry_run: bool, attempts: int = 3) -> str:
    print(f"[{host}] {command}")
    if dry_run:
        return ""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(["ssh", host, command], text=True, capture_output=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout.rstrip())
            return result.stdout
        if result.stderr.strip():
            print(result.stderr.rstrip())
        if result.returncode != 255 or attempt == attempts:
            result.check_returncode()
        time.sleep(float(attempt))
    raise RuntimeError("unreachable")


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8"), delimiter="\t"))
    ready = [row for row in rows if row["status"] == "ready"]
    blocked = [row for row in rows if row["status"] != "ready"]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU index")

    launched = []
    for index, row in enumerate(ready):
        scene = row["scene_id"].removeprefix("scene")
        dataset_root = row.get("dataset_root", "").strip() or args.dataset_root
        scene_dir = f"{dataset_root.rstrip('/')}/{row['scene_dir_name']}"
        output_dir = f"{args.repo_root}/outputs/{row['output_name']}"
        pose_order = row["camera_pose_order"]
        prepare = " ".join(
            shlex.quote(value)
            for value in [
                f"{args.repo_root}/.venv/bin/python",
                f"{args.repo_root}/scripts/prepare_phystwin_tracking_episode.py" if args.object_masks else f"{args.repo_root}/scripts/prepare_sam_pseudo_tracking_episode.py",
                scene_dir,
                output_dir,
                "--frame-stride",
                "1",
                *(["--views", "0,1,2", "--object-mask-id", "1", "--hand-mask-ids", "", "--object-erosion-px", "2", "--category", "microwave" if "microwave" in row["scene_id"] else "oven"] if args.object_masks else ["--camera-pose-order", pose_order]),
            ]
        )
        remote(args.host, f"cd {shlex.quote(args.repo_root)} && {prepare}", dry_run=args.dry_run)
        reference_command = (
            f"{args.repo_root}/.venv/bin/python -c "
            + shlex.quote(
                "import json; p=json.load(open(" + repr(f"{output_dir}/episode.json") + ")); "
                "print(p['metadata']['recommended_tracking']['reference_source_frame'])"
            )
        )
        reference_text = remote(args.host, reference_command, dry_run=args.dry_run).strip()
        reference = reference_text.splitlines()[-1] if reference_text else "0"
        gpu = gpus[index % len(gpus)]
        session = f"track2art_{row['scene_id']}_dense_v1"
        log = f"{output_dir}/pipeline.log"
        pipeline = (
            f"cd {shlex.quote(args.repo_root)} && "
            f"TRACK2ART_PYTHON={shlex.quote(args.python_bin)} TRACK_FRAME_STRIDE=1 "
            f"bash scripts/run_real_scene_track2art_remote.sh "
            f"{shlex.quote(scene)} {shlex.quote(gpu)} {shlex.quote(reference)} "
            f"{shlex.quote(row['output_name'])} > {shlex.quote(log)} 2>&1"
        )
        launch = f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(pipeline)}"
        remote(args.host, launch, dry_run=args.dry_run)
        launched.append(
            {
                **row,
                "dataset_root": dataset_root,
                "gpu": gpu,
                "reference_frame": int(reference),
                "tmux": session,
                "log": log,
            }
        )

    print(json.dumps({"launched": launched, "blocked": blocked}, indent=2))


if __name__ == "__main__":
    main()
