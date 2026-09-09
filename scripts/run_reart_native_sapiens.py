#!/usr/bin/env python3
"""Run the official two-stage ReArt Sapiens protocol on a fixed index subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reart-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--indices", required=True, help="Comma-separated Sapiens test indices.")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated CUDA device ids.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def run_position(position: int) -> dict[str, Any]:
        return _run_index(args, indices[position], gpu_ids[position % len(gpu_ids)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run_position, range(len(indices))))
    manifest = {
        "protocol": "reart_native_sapiens",
        "official_frame_count": 4,
        "relaxation_iterations": 2000,
        "projection_iterations": 200,
        "indices": indices,
        "elapsed_s": time.time() - started,
        "runs": rows,
    }
    (output_root / "native_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if all(row["status"] == "success" for row in rows) else 1


def _run_index(args: argparse.Namespace, index: int, gpu_id: str) -> dict[str, Any]:
    root = args.reart_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    relax_root = output / "relaxation"
    projection_root = output / "projection"
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    final_result = projection_root / f"sapien_{index}" / "result.pkl"
    if args.resume and final_result.is_file():
        return {"index": index, "gpu_id": gpu_id, "status": "skipped", "runtime_s": 0.0}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    started = time.time()
    base = [
        args.python,
        "run_sapien.py",
        f"--sapien_idx={index}",
        f"--sapien_base_folder={args.dataset_root.expanduser().resolve()}",
        "--cano_idx=0",
        "--use_flow_loss",
        "--use_nproc",
        "--use_assign_loss",
    ]
    relaxation = base + [f"--save_root={relax_root}", "--n_iter=2000"]
    projection = base + [
        f"--save_root={projection_root}",
        "--n_iter=200",
        "--model=kinematic",
        "--assign_iter=0",
        "--assign_gap=1",
        "--snapshot_gap=10",
        f"--base_result_path={relax_root / f'sapien_{index}' / 'result.pkl'}",
    ]
    try:
        _run(relaxation, root, env, log_root / f"{index:06d}_relax.log")
        _run(projection, root, env, log_root / f"{index:06d}_projection.log")
    except subprocess.CalledProcessError as exc:
        return {
            "index": index,
            "gpu_id": gpu_id,
            "status": "failed",
            "returncode": exc.returncode,
            "runtime_s": time.time() - started,
        }
    return {
        "index": index,
        "gpu_id": gpu_id,
        "status": "success",
        "runtime_s": time.time() - started,
        "result": str(final_result),
    }


def _run(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
