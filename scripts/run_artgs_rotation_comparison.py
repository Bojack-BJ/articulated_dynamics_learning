#!/usr/bin/env python3
"""Rerun released ArtGS after a fixed global rotation of its input geometry."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


def prepare(source, destination, rotation):
    from plyfile import PlyData
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if target.exists():
            continue
        if path.name.startswith("transforms") and path.suffix == ".json":
            payload = json.loads(path.read_text())
            transform = np.eye(4)
            transform[:3, :3] = rotation
            for frame in payload["frames"]:
                frame["transform_matrix"] = (transform @ np.asarray(frame["transform_matrix"])).tolist()
            target.write_text(json.dumps(payload))
        elif path.suffix == ".ply":
            ply = PlyData.read(path)
            vertices = ply["vertex"].data
            xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]) @ rotation.T
            for i, key in enumerate(("x", "y", "z")):
                vertices[key] = xyz[:, i]
            if all(k in vertices.dtype.names for k in ("nx", "ny", "nz")):
                normals = np.column_stack([vertices[k] for k in ("nx", "ny", "nz")]) @ rotation.T
                for i, key in enumerate(("nx", "ny", "nz")):
                    vertices[key] = normals[:, i]
            ply.write(target)
        else:
            target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # Private argument registries and output paths prevent collisions with native runs.
    for path in args.repo.iterdir():
        target = work / path.name
        if path.name in {"data", "outputs", "arguments", "__pycache__"} or target.exists():
            continue
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    if not (work / "arguments").exists():
        shutil.copytree(args.repo / "arguments", work / "arguments")
    rotation = Rotation.random(random_state=20260909).as_matrix()
    source = args.root / "outputs/external_baseline_suite_v1/per_object" / args.object / "acquisition/two_state/artgs"
    destination = work / "data/external_suite/aligned" / args.object
    prepare(source, destination, rotation)
    with (args.root / "outputs/external_baseline_suite_v1/manifest.csv").open() as stream:
        slots = next(int(row["gt_part_count"]) for row in csv.DictReader(stream) if row["object_id"] == args.object)
    for name in ("num_slots.json", "joint_types_cgs.json"):
        path = work / "arguments" / name
        payload = json.loads(path.read_text())
        payload.setdefault("external_suite", {}).setdefault("aligned", {})[args.object] = slots if name == "num_slots.json" else ""
        path.write_text(json.dumps(payload))
    native = work / "outputs/external_suite/aligned" / args.object
    native.mkdir(parents=True, exist_ok=True)
    common = ["--dataset", "external_suite", "--subset", "aligned", "--scene_name", args.object,
              "--source_path", str(work / "data")]
    stages = [
        ("train_coarse.py", "coarse_gs", ["--resolution", "2", "--iterations", "10000", "--opacity_reg_weight", "0.1", "--random_bg_color"]),
        ("train_predict.py", "joint_predict", ["--eval", "--resolution", "8", "--iterations", "5000", "--densify_grad_threshold", "0.001", "--coarse_name", "coarse_gs", "--random_bg_color"]),
        ("train.py", "artgs", ["--eval", "--resolution", "1", "--iterations", "20000", "--coarse_name", "coarse_gs", "--seed", "0", "--use_art_type_prior", "--random_bg_color", "--densify_grad_threshold", "0.001"]),
    ]
    search = [work/"submodules/diff-gaussian-rasterization", work/"submodules/simple-knn", work]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=os.pathsep.join(map(str, search)), PYTHONUNBUFFERED="1")
    python = "/root/Users/miniconda3/envs/artgs/bin/python"
    status = {"object":args.object, "rotation":rotation.tolist(), "protocol":"global coordinate rotation; unchanged images and depth; full optimization rerun", "status":"running"}
    status_path = work / "status.json"
    for script, stage, options in stages:
        command = [python, script, *common, "--model_path", str(native/stage), *options]
        status.update(stage=stage, command=command)
        status_path.write_text(json.dumps(status, indent=2))
        with (work/f"{stage}.log").open("w") as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status="failed", returncode=result.returncode)
            status_path.write_text(json.dumps(status, indent=2))
            raise SystemExit(result.returncode)
    command = [python, str(args.root/"scripts/export_artgs_predictions.py"), "--model_path", str(native/"artgs"), "--output-dir", str(work/"adapter")]
    with (work/"export.log").open("w") as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.update(status="complete" if result.returncode == 0 else "export_failed", returncode=result.returncode)
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
