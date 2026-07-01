#!/usr/bin/env python3
"""Run optimized and ReArt baselines for SAM2 cardboard-box masks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


OBJECTS = {
    "cardboardbox01_o": "episode.manual_partmask_sam2_keyframe.json",
    "cardboardbox02_o": "episode.manual_partmask_sam2_keyframe.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["optimized", "reart", "both"], default="both")
    parser.add_argument("--objects", nargs="*", default=list(OBJECTS))
    parser.add_argument("--recordings-root", type=Path, default=Path("outputs/real_recordings"))
    parser.add_argument("--optimized-root", type=Path, default=Path("outputs/real_optimized"))
    parser.add_argument("--optimized-suffix", default="sam2_keyframe")
    parser.add_argument("--reart-sequence-root", type=Path, default=Path("outputs/reart_sequences/real_cardboardboxes_sam2"))
    parser.add_argument("--reart-output-root", type=Path, default=Path("outputs/reart/real_cardboardboxes_sam2"))
    parser.add_argument("--reart-server-url", default="http://127.0.0.1:8892")
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--cotracker-repo", type=Path, default=Path("co-tracker"))
    parser.add_argument("--cotracker-checkpoint", type=Path, default=Path("co-tracker/ckpt/scaled_offline.pth"))
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--reart-base-n-iter", type=int, default=2000)
    parser.add_argument("--reart-num-points", type=int, default=2048)
    parser.add_argument("--reart-num-parts", type=int, default=4)
    args = parser.parse_args()

    summary = []
    for object_id in args.objects:
        episode = args.recordings_root / object_id / OBJECTS[object_id]
        out_dir = args.optimized_root / f"{object_id}_{args.optimized_suffix}" / "pointcloud_4d_partseg"
        item: dict[str, object] = {"object_id": object_id, "episode": str(episode), "optimized_dir": str(out_dir)}
        if args.mode in {"optimized", "both"}:
            item["optimized"] = run_optimized(episode, out_dir, args)
        if args.mode in {"reart", "both"}:
            item["reart"] = run_reart(object_id, out_dir, args)
        summary.append(item)

    summary_path = args.reart_output_root / "cardboardbox_sam2_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2))
    return 0


def run_optimized(episode: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _env()
    timings = {}
    timings["fuse-pointcloud"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "fuse-pointcloud",
            str(episode),
            "--output-dir",
            str(out_dir),
            "--voxel-size-m",
            "0.01",
            "--pixel-stride",
            "4",
        ],
        env,
    )
    tracks = out_dir / "part_tracks.json"
    timings["track-part-pixels"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "track-part-pixels",
            str(episode),
            "--output-json",
            str(tracks),
            "--device",
            args.device,
            "--cotracker-repo",
            str(args.cotracker_repo),
            "--cotracker-checkpoint",
            str(args.cotracker_checkpoint),
            "--reference-frame",
            "-1",
            "--frame-stride",
            "4",
            "--seed-stride-px",
            "20",
            "--max-tracks-per-part-view",
            "64",
            "--no-part-mask-consistency",
        ],
        env,
    )
    poses = out_dir / "part_poses.json"
    timings["estimate-part-poses"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "estimate-part-poses",
            str(tracks),
            "--method",
            "tracks",
            "--output-json",
            str(poses),
        ],
        env,
    )
    joints = out_dir / "joint_inference.json"
    timings["infer-joints"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "infer-joints",
            str(poses),
            "--output-json",
            str(joints),
            "--mujoco-prior",
            "off",
        ],
        env,
    )
    timings["export-inferred-articulation"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "export-inferred-articulation",
            str(episode),
            str(poses),
            str(joints),
            "--output-dir",
            str(out_dir / "inferred_articulation"),
        ],
        env,
    )
    timings["visualize-pointcloud"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "visualize-pointcloud",
            str(out_dir / "fusion_manifest.json"),
            "--output-html",
            str(out_dir / "viewer_pose_flow.html"),
            "--part-poses-json",
            str(poses),
            "--part-tracks-json",
            str(tracks),
            "--joint-inference-json",
            str(joints),
        ],
        env,
    )
    (out_dir / "pipeline_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return {"output_dir": str(out_dir), "timings": timings}


def run_reart(object_id: str, optimized_out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    env = _env()
    sequence_dir = args.reart_sequence_root / object_id
    output_dir = args.reart_output_root / object_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    timings["export-reart-sequence"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "export-reart-sequence",
            str(optimized_out_dir / "fusion_manifest.json"),
            "--output-dir",
            str(sequence_dir),
            "--frame-stride",
            "4",
            "--max-frames",
            "30",
            "--max-points-per-frame",
            "20000",
        ],
        env,
    )
    timings["remote-reart-run"] = timed_run(
        [
            sys.executable,
            "-m",
            "rgbd_urdf_mvp",
            "remote-reart-run",
            "--server-url",
            args.reart_server_url,
            "--sequence-dir",
            str(sequence_dir),
            "--output-dir",
            str(output_dir),
            "--sequence-name",
            object_id,
            "--cano-idx",
            "20",
            "--stage",
            "base",
            "--base-n-iter",
            str(args.reart_base_n_iter),
            "--snapshot-gap",
            "100",
            "--num-points",
            str(args.reart_num_points),
            "--num-parts",
            str(args.reart_num_parts),
            "--timeout-s",
            str(args.timeout_s),
        ],
        env,
    )
    result_pkl = output_dir / "reart" / object_id / "result.pkl"
    if result_pkl.exists():
        timings["export-reart-visualization"] = timed_run(
            [
                sys.executable,
                "scripts/export_reart_visualization.py",
                str(result_pkl),
                "--output-dir",
                str(output_dir),
            ],
            env,
        )
    (output_dir / "pipeline_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return {"sequence_dir": str(sequence_dir), "output_dir": str(output_dir), "timings": timings}


def timed_run(cmd: list[str], env: dict[str, str]) -> dict[str, object]:
    print("+", " ".join(cmd), flush=True)
    start = time.perf_counter()
    completed = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return {"elapsed_s": elapsed}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TORCH_HOME"] = str(Path.cwd() / ".cache" / "torch")
    env.pop("__PYVENV_LAUNCHER__", None)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
