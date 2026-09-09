#!/usr/bin/env python3
"""Export dense Gaussian trajectories from a frozen AiM run."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def _latest_iteration(run_dir: Path) -> int:
    candidates = [
        int(path.name.removeprefix("iteration_"))
        for path in (run_dir / "deform").glob("iteration_*")
        if path.is_dir() and path.name.removeprefix("iteration_").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No deform checkpoint found under {run_dir}")
    return max(candidates)


def _read_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = 0
    data_start = None
    for index, line in enumerate(lines):
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        if fields == ["end_header"]:
            data_start = index + 1
            break
    if data_start is None or vertex_count <= 0:
        raise ValueError(f"Invalid ASCII trajectory PLY: {path}")
    return np.asarray(
        [
            [float(value) for value in line.split()[:3]]
            for line in lines[data_start : data_start + vertex_count]
        ],
        dtype=np.float32,
    )


def _run_cfg_value(run_dir: Path, key: str, default):
    cfg_path = run_dir / "cfg_args"
    if not cfg_path.is_file():
        return default
    payload = cfg_path.read_text(encoding="utf-8")
    if payload.startswith("Namespace(") and payload.endswith(")"):
        for field in payload[len("Namespace(") : -1].split(","):
            field_key, separator, value = field.strip().partition("=")
            if separator and field_key == key:
                return ast.literal_eval(value)
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--aim-root", required=True)
    parser.add_argument("--frames", type=int, default=31)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int)
    parser.add_argument("--is-blender", action="store_true")
    parser.add_argument("--is-6dof", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    aim_root = Path(args.aim_root).expanduser().resolve()
    sys.path.insert(0, str(aim_root))
    # AiM's CUDA extensions are commonly built in-place without being installed.
    for extension_root in (
        aim_root / "submodules" / "simple-knn" / "build",
        aim_root / "submodules" / "diff-gaussian-rasterization1" / "build",
        aim_root / "lib" / "pointops" / "build",
    ):
        for library_dir in sorted(extension_root.glob("lib*")):
            sys.path.insert(0, str(library_dir))
    from scene import DeformModel, GaussianModel  # noqa: PLC0415

    iteration = args.iteration if args.iteration >= 0 else _latest_iteration(run_dir)
    sh_degree = (
        args.sh_degree
        if args.sh_degree is not None
        else int(_run_cfg_value(run_dir, "sh_degree", 3))
    )
    static = GaussianModel(sh_degree)
    moving = GaussianModel(sh_degree)
    static.load_ply(
        str(
            run_dir
            / "point_cloud_start"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
    )
    moving.load_ply(
        str(
            run_dir
            / "point_cloud_motion"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
    )
    deform = DeformModel(
        is_blender=bool(
            args.is_blender or _run_cfg_value(run_dir, "is_blender", False)
        ),
        is_6dof=bool(args.is_6dof or _run_cfg_value(run_dir, "is_6dof", False)),
    )
    deform.load_weights(str(run_dir), iteration=iteration)
    deform.deform.eval()

    static_xyz = static.get_xyz.detach()
    moving_xyz = moving.get_xyz.detach()
    times = np.linspace(0.0, 1.0, args.frames, dtype=np.float32)
    frames: list[np.ndarray] = []
    with torch.inference_mode():
        for value in times:
            time = torch.full(
                (len(moving_xyz), 1),
                float(value),
                dtype=moving_xyz.dtype,
                device=moving_xyz.device,
            )
            displacement = deform.step(moving_xyz, time)[0]
            xyz = torch.cat((static_xyz, moving_xyz + displacement), dim=0)
            frames.append(xyz.detach().cpu().numpy().astype(np.float32))
    trajectory = np.stack(frames, axis=0)
    alignment_path = run_dir / "motion_seg_final" / "segmented_point.ply"
    alignment_max_error = 0.0
    if alignment_path.is_file():
        target = _read_xyz(alignment_path)
        # AiM stores post-merge labels on start-state Gaussian coordinates.
        distance, indices = cKDTree(trajectory[0]).query(target, k=1)
        alignment_max_error = float(np.max(distance))
        if alignment_max_error > 1e-5:
            raise ValueError(
                "Final segmentation does not align with the frozen checkpoint: "
                f"max error {alignment_max_error:.6g} m"
            )
        trajectory = trajectory[:, np.asarray(indices, dtype=np.int64)]

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "dense_trajectory.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        trajectory=trajectory,
        times=times,
        iteration=np.asarray(iteration, dtype=np.int32),
        static_count=np.asarray(len(static_xyz), dtype=np.int64),
        moving_count=np.asarray(len(moving_xyz), dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "shape": list(trajectory.shape),
                "iteration": iteration,
                "static_count": len(static_xyz),
                "moving_count": len(moving_xyz),
                "aligned_output_count": int(trajectory.shape[1]),
                "alignment_max_error_m": alignment_max_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
