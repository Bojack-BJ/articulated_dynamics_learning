#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rgbd_urdf_mvp.perception.reart_adapter import (
    ReArtSequenceExportConfig,
    ReArtSequenceExporter,
    RemoteReArtClient,
    RemoteReArtConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged Lightwheel refrigerators through remote ReArt.")
    parser.add_argument("--batch-config", type=Path, default=Path("configs/batch_lightwheel_refrigerators.tsv"))
    parser.add_argument("--recordings-root", type=Path, default=Path("outputs/recordings_refrigerators_staged"))
    parser.add_argument("--sequence-root", type=Path, default=Path("outputs/reart_sequences/lightwheel_refrigerators_open"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reart/lightwheel_refrigerators_open"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8892")
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--open-fraction", type=float, default=0.9)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-parts", type=int, default=12)
    parser.add_argument("--base-n-iter", type=int, default=2000)
    parser.add_argument("--snapshot-gap", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    batch_config = args.batch_config.expanduser().resolve()
    recordings_root = args.recordings_root.expanduser().resolve()
    sequence_root = args.sequence_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sequence_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.object_id)
    object_ids = [value for value in _object_ids(batch_config) if not requested or value in requested]
    batch_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    client = RemoteReArtClient(args.server_url, api_token=args.api_token)
    for index, object_id in enumerate(object_ids, start=1):
        started = time.perf_counter()
        recording_dir = recordings_root / object_id
        sequence_dir = sequence_root / object_id
        output_dir = output_root / object_id
        result_path = output_dir / "reart" / object_id / "result.pkl"
        row: dict[str, Any] = {
            "object_id": object_id,
            "sequence_dir": str(sequence_dir),
            "output_dir": str(output_dir),
        }
        try:
            if result_path.exists() and not args.force:
                row.update(status="skipped_existing", result_path=str(result_path), elapsed_s=0.0)
                print(f"[{index}/{len(object_ids)}] {object_id}: existing result, skip", flush=True)
                rows.append(row)
                _write_manifest(output_root, args, rows, batch_started)
                continue
            fusion_manifest = recording_dir / "pointcloud_4d_partseg" / "fusion_manifest.json"
            dynamics_log = recording_dir / "dynamics_log.jsonl"
            if not fusion_manifest.exists() or not dynamics_log.exists():
                raise FileNotFoundError(f"Missing recording inputs under {recording_dir}")

            print(f"[{index}/{len(object_ids)}] {object_id}: export sequence", flush=True)
            ReArtSequenceExporter().export(
                ReArtSequenceExportConfig(
                    fusion_manifest_path=fusion_manifest,
                    output_dir=sequence_dir,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    max_points_per_frame=20000,
                    foreground_only=True,
                )
            )
            cano_idx, source_frame, progress, moving_joints = _choose_canonical_frame(
                dynamics_log,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                open_fraction=args.open_fraction,
            )
            row.update(
                cano_idx=cano_idx,
                cano_source_frame=source_frame,
                aggregate_motion_progress=progress,
                moving_joints=moving_joints,
            )
            print(
                f"[{index}/{len(object_ids)}] {object_id}: cano={cano_idx} "
                f"(source={source_frame}, progress={progress:.3f}, moving={len(moving_joints)}), run ReArt",
                flush=True,
            )
            client.run(
                RemoteReArtConfig(
                    server_url=args.server_url,
                    sequence_dir=sequence_dir,
                    output_dir=output_dir,
                    sequence_name=object_id,
                    cano_idx=cano_idx,
                    num_points=args.num_points,
                    num_parts=args.num_parts,
                    stage="base",
                    base_n_iter=args.base_n_iter,
                    snapshot_gap=args.snapshot_gap,
                    timeout_s=args.timeout_s,
                    api_token=args.api_token,
                )
            )
            result_path = output_dir / "reart" / object_id / "result.pkl"
            if not result_path.exists():
                raise FileNotFoundError(f"Remote job completed without result.pkl: {result_path}")
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("export_reart_visualization.py")), str(result_path)],
                check=True,
            )
            row.update(status="completed", result_path=str(result_path), elapsed_s=time.perf_counter() - started)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter() - started)
            print(f"[{index}/{len(object_ids)}] {object_id}: FAILED: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)
        _write_manifest(output_root, args, rows, batch_started)

    failed = [row["object_id"] for row in rows if row.get("status") == "failed"]
    print(json.dumps({"output_root": str(output_root), "failed": failed}, indent=2), flush=True)
    return 1 if failed else 0


def _choose_canonical_frame(
    dynamics_log: Path,
    *,
    frame_stride: int,
    max_frames: int,
    open_fraction: float,
) -> tuple[int, int, float, list[str]]:
    samples = [json.loads(line) for line in dynamics_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise ValueError(f"Empty dynamics log: {dynamics_log}")
    joint_names = sorted(samples[0].get("joints", {}))
    trajectories = {
        name: [float(sample["joints"][name]["qpos"]) for sample in samples if name in sample.get("joints", {})]
        for name in joint_names
    }
    moving = {
        name: values
        for name, values in trajectories.items()
        if values and max(values) - min(values) > 0.02
    }
    source_frames = list(range(0, max_frames * frame_stride, frame_stride))
    source_frames = [index for index in source_frames if index * 4 < len(samples)]
    target = max(0.0, min(1.0, float(open_fraction)))
    scored = []
    for source_frame in source_frames:
        log_index = min(len(samples) - 1, source_frame * 4)
        progresses = []
        for values in moving.values():
            start = values[0]
            end = max(values, key=lambda value: abs(value - start))
            denominator = abs(end - start)
            progresses.append(min(1.0, abs(values[log_index] - start) / denominator) if denominator > 1e-9 else 0.0)
        aggregate = sum(progresses) / len(progresses) if progresses else source_frame / max(1, source_frames[-1])
        scored.append((abs(aggregate - target), source_frame, aggregate))
    _, source_frame, progress = min(scored)
    return source_frames.index(source_frame), source_frame, progress, sorted(moving)


def _object_ids(path: Path) -> list[str]:
    return [
        fields[2]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and len(fields := line.split("\t")) >= 3
    ]


def _write_manifest(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], started: float) -> None:
    payload = {
        "source": "lightwheel-refrigerator-reart-batch",
        "server_url": args.server_url,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "open_fraction": args.open_fraction,
        "num_points": args.num_points,
        "num_parts": args.num_parts,
        "base_n_iter": args.base_n_iter,
        "elapsed_s": time.perf_counter() - started,
        "objects": rows,
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
