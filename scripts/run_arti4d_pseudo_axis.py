#!/usr/bin/env python3
"""Launch robust pseudo-axis fitting for manually part-labeled Arti4D episodes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


DEFAULT_ROOT = "/lumos-vePFS/suzhou/Users/lixiaotong/datasets"
DEFAULT_REPO = "/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning"
DEFAULT_OUTPUT = f"{DEFAULT_REPO}/outputs/arti4d_pseudo_axis_v1"

DATASETS = {
    "rh201_stove_oven": f"{DEFAULT_ROOT}/arti4d_manual_pilot_raw/rh201_stove_oven",
    "rh201_fridge": f"{DEFAULT_ROOT}/arti4d_manual_pilot_raw/rh201_fridge",
    "rh201_top_drawer": f"{DEFAULT_ROOT}/arti4d_manual_pilot_raw/rh201_top_drawer",
    "rh201_cabinet": f"{DEFAULT_ROOT}/arti4d_manual_pilot_raw/rh201_cabinet",
    "rh078_blue_drawer": f"{DEFAULT_ROOT}/arti4d_subsets/arti4d_blue_drawer",
    "rh078_cabinet_right": f"{DEFAULT_ROOT}/arti4d_subsets/arti4d_cabinet_right",
    "rh078_right_drawer_1": f"{DEFAULT_ROOT}/arti4d/scene_2025-04-09-10-38-38",
}


def remote(host: str, command: str) -> str:
    result = subprocess.run(["ssh", host, command], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def latest_episode(host: str, dataset_root: str) -> str:
    annotation_root = f"{dataset_root}/annotations/track2art_manual_v1"
    command = (
        f"ls -t {shlex.quote(annotation_root)}/episode.live_sam2_video_*.json "
        "2>/dev/null | head -1"
    )
    episode = remote(host, command)
    if not episode:
        fallback = f"{annotation_root}/episode.partmask-annotated.json"
        remote(host, f"test -s {shlex.quote(fallback)}")
        episode = fallback
    return episode


def pipeline(repo: str, python: str, pythonpath: str, episode: str, output: str, gpu: int) -> str:
    q = shlex.quote
    tracks = output + "/tracking/part_tracks_cotracker.json"
    features = output + "/tracking/cotracker_features.npz"
    quality = output + "/track_quality/motion_part_tracks_with_quality.json"
    poses = output + "/kinematics/part_poses_robust.json"
    joints = output + "/kinematics/joint_inference_pseudo.json"
    viewer = output + "/viewer_pseudo_axis_persistent_annotation.html"
    env = f"PYTHONPATH={q(pythonpath)}"
    return " && ".join([
        f"cd {q(repo)}",
        f"mkdir -p {q(output + '/tracking')} {q(output + '/track_quality')} {q(output + '/kinematics')}",
        (
            f"(test -s {q(tracks)} && test -s {q(features)}) || "
            f"CUDA_VISIBLE_DEVICES={gpu} {env} {q(python)} -m rgbd_urdf_mvp track-part-pixels "
            f"{q(episode)} --output-json {q(tracks)} "
            "--device cuda --cotracker-repo co-tracker --cotracker-checkpoint co-tracker/ckpt/scaled_offline.pth "
            "--reference-frame -1 --frame-stride 1 --seed-stride-px 6 --max-tracks-per-part-view 768 "
            "--max-queries-per-forward 512 --visibility-threshold 0.5 --dynamic-reseeding "
            "--dynamic-reseed-bidirectional "
            f"--export-cotracker-features --cotracker-features-output {q(features)} "
            "--reseed-interval-frames 8 --reseed-coverage-radius-px 10 "
            "--reseed-max-tracks-per-frame-view 48 --reseed-max-tracks-per-view 384 "
            "--repair-temporal-depth-spikes "
            "--depth-consistency-window-radius-px 2 --depth-consistency-max-delta-m 0.08"
        ),
        (
            f"test -s {q(quality)} || {env} {q(python)} -m rgbd_urdf_mvp compute-track-quality "
            f"{q(tracks)} --output-dir {q(output + '/track_quality')} "
            # Retain every valid temporal segment. Downstream robust fitting uses
            # continuous quality weights; only physically implausible jumps are masked.
            "--mask-bad-timesteps --max-step-m 0.05"
        ),
        (
            f"test -s {q(poses)} || {env} {q(python)} -m rgbd_urdf_mvp estimate-part-poses "
            f"{q(quality)} --method tracks "
            f"--quality-weighted --anchor-part-id 1 --min-tracks-per-part 4 "
            f"--output-json {q(output + '/kinematics/part_poses_robust.json')}"
        ),
        (
            f"test -s {q(joints)} || {env} {q(python)} -m rgbd_urdf_mvp infer-joints "
            f"{q(poses)} --output-json {q(joints)} "
            "--mujoco-prior off --quality-weighted-replay --robust-track-model-trim-ratio 0.15 "
            "--orient-parent-by-motion"
        ),
        (
            f"test -s {q(viewer)} || {env} {q(python)} -m rgbd_urdf_mvp visualize-object-mask-flow-html "
            f"{q(quality)} --joint-inference {q(joints)} "
            f"--background-episode {q(episode)} --background-exclude-object-mask "
            f"--background-persistent --background-voxel-size-m 0.02 "
            f"--background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z "
            f"--output-html {q(viewer)}"
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="dev-h")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default="/root/Users/miniconda3/envs/particulate/bin/python")
    parser.add_argument("--pythonpath", default="/tmp/track2art_arti4d_axis_src_v1")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()

    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    gpus = [int(value) for value in args.gpus.split(",")]
    launched = []
    for index, name in enumerate(names):
        dataset_root = DATASETS[name]
        episode = latest_episode(args.host, dataset_root)
        output = f"{args.output_root}/{name}"
        command = pipeline(args.repo, args.python, args.pythonpath, episode, output, gpus[index % len(gpus)])
        session = f"arti4d_axis_{name}"
        launch = (
            f"tmux kill-session -t {q(session)} 2>/dev/null || true; "
            f"tmux new-session -d -s {q(session)} "
            f"{q(command + ' > ' + output + '/pipeline.log 2>&1')}"
        )
        remote(args.host, f"mkdir -p {q(output)}; {launch}")
        launched.append({"dataset": name, "gpu": gpus[index % len(gpus)], "episode": episode, "output": output, "tmux": session})
    print(json.dumps({"launched": launched}, indent=2))


def q(value: str) -> str:
    return shlex.quote(value)


if __name__ == "__main__":
    main()
