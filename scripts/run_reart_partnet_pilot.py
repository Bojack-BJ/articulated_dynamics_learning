#!/usr/bin/env python3
"""Export, run, and evaluate official ReArt on shared-3view PartNet episodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects_tsv", type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "evaluate", "all"), default="all")
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--hybrid-root", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--server-url")
    parser.add_argument("--reart-root", type=Path)
    parser.add_argument("--project-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-fusion-subdir", default="pointcloud_4d_partseg")
    parser.add_argument("--rebuild-input-fusion", action="store_true")
    parser.add_argument("--fusion-pixel-stride", type=int, default=8)
    parser.add_argument("--fusion-voxel-size-m", type=float, default=0.02)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument(
        "--protocol-profile",
        choices=("custom", "sapien_count_matched_4frame"),
        default="custom",
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-parts", type=int, default=10)
    parser.add_argument("--base-n-iter", type=int, default=2000)
    parser.add_argument("--kinematic-n-iter", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter="\t")]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = [executor.submit(_prepare_one, args, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _prepare_one(args: argparse.Namespace, row: dict[str, str]) -> None:
    object_id = row["object_id"]
    recording = args.recordings_root / object_id
    fusion_root = recording / args.input_fusion_subdir
    fusion_manifest = fusion_root / "fusion_manifest.json"
    if args.rebuild_input_fusion or not fusion_manifest.is_file():
        run(
            [
                str(args.project_python),
                "-m",
                "rgbd_urdf_mvp",
                "fuse-pointcloud",
                str(recording / "episode.json"),
                "--output-dir",
                str(fusion_root),
                "--pixel-stride",
                str(args.fusion_pixel_stride),
                "--voxel-size-m",
                str(args.fusion_voxel_size_m),
                "--no-shared-pose-fallback",
            ]
        )
    output = args.sequence_root / object_id
    manifest = output / "reart_sequence_manifest.json"
    if args.resume and manifest.is_file():
        return
    command = [
        str(args.project_python),
        "-m",
        "rgbd_urdf_mvp",
        "export-reart-sequence",
        str(fusion_manifest),
        "--output-dir",
        str(output),
        "--frame-stride",
        str(max(1, args.frame_stride)),
        "--protocol-profile",
        str(args.protocol_profile),
        "--max-points-per-frame",
        "20000",
    ]
    if args.frame_indices is not None:
        command.extend(["--frame-indices", *(str(value) for value in args.frame_indices)])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    run(command)


def run_official(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    if not args.server_url and args.reart_root is None:
        raise ValueError("stage run/all requires either --server-url or --reart-root")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    jobs = max(1, int(args.jobs))
    if jobs > 1 and args.reart_root is not None and not gpu_ids:
        raise ValueError("parallel direct ReArt runs require --gpu-ids")
    work = [(row, gpu_ids[index % len(gpu_ids)] if gpu_ids else None) for index, row in enumerate(rows)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_run_one_official, args, row, gpu_id) for row, gpu_id in work]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _run_one_official(args: argparse.Namespace, row: dict[str, str], gpu_id: str | None) -> None:
    object_id = row["object_id"]
    output = args.run_root / object_id
    status_path = output / "pilot_status.json"
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "success":
            return
    started = time.perf_counter()
    status: dict[str, Any] = {
        "object_id": object_id,
        "category": row["category"],
        "status": "failed",
        "protocol": "shared_3view_fused_4d_pointcloud",
        "temporal_sampling_profile": str(args.protocol_profile),
        "requested_frame_indices": args.frame_indices,
        "oracle_part_count": False,
        "num_parts_slot_cap": int(args.num_parts),
        "gpu_id": gpu_id,
        "official_configuration": {
            "relaxation": True,
            "projection": True,
            "use_flow_loss": True,
            "use_assign_loss": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "official_reart.log"
    try:
        command = _official_command(args, object_id, output)
        environment = os.environ.copy()
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[{object_id}] GPU={gpu_id or 'default'}: {' '.join(command)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status["status"] = "success"
    except Exception as error:
        status["failure_stage"] = "official_reart_optimization"
        status["exception"] = f"{type(error).__name__}: {error}"
        status["log_path"] = str(log_path.resolve())
    status["runtime_s"] = time.perf_counter() - started
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[{object_id}] {status['status']} in {status['runtime_s']:.1f}s", flush=True)


def _official_command(args: argparse.Namespace, object_id: str, output: Path) -> list[str]:
    if args.reart_root is not None:
        return [
            str(args.project_python),
            "-m",
            "rgbd_urdf_mvp.perception.reart_run_wrapper",
            "--reart-root",
            str(args.reart_root),
            "--seq-path",
            str(args.sequence_root / object_id),
            "--save-root",
            str(output),
            "--stage",
            "both",
            "--base-n-iter",
            str(args.base_n_iter),
            "--kinematic-n-iter",
            str(args.kinematic_n_iter),
            "--num-points",
            str(args.num_points),
            "--num-parts",
            str(args.num_parts),
            "--use-flow-loss",
            "--use-assign-loss",
            "--use-nproc",
        ]
    return [
        str(args.project_python),
        "-m",
        "rgbd_urdf_mvp",
        "remote-reart-run",
        "--server-url",
        str(args.server_url),
        "--sequence-dir",
        str(args.sequence_root / object_id),
        "--output-dir",
        str(output),
        "--sequence-name",
        object_id,
        "--stage",
        "both",
        "--base-n-iter",
        str(args.base_n_iter),
        "--kinematic-n-iter",
        str(args.kinematic_n_iter),
        "--num-points",
        str(args.num_points),
        "--num-parts",
        str(args.num_parts),
        "--use-flow-loss",
        "--use-assign-loss",
        "--use-nproc",
        "--timeout-s",
        str(args.timeout_s),
    ]


def _find_result(root: Path) -> Path | None:
    candidates = sorted(root.glob("**/result.pkl"))
    return candidates[-1] if candidates else None


def evaluate(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    raw_root = args.report_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        object_id = row["object_id"]
        result_pkl = _find_result(args.run_root / object_id)
        status_path = args.run_root / object_id / "pilot_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if result_pkl is None or status.get("status") != "success":
            failures.append(_failure_record(args, row, status, result_pkl))
            continue
        sequence_manifest = json.loads(
            (args.sequence_root / object_id / "reart_sequence_manifest.json").read_text(encoding="utf-8")
        )
        cano_idx = 0
        source_frame_path = Path(sequence_manifest["frames"][cano_idx]["source_frame_path"])
        source_frame_index = int(source_frame_path.stem.split("_")[-1])
        reference = (
            args.recordings_root
            / object_id
            / "pointcloud_4d_partseg"
            / "frames"
            / f"frame_{source_frame_index:04d}.ply"
        )
        hybrid = args.hybrid_root / object_id / "motion_part_tracks_slots.json"
        output_json = raw_root / object_id / "reart.json"
        command = [
            str(args.project_python),
            "scripts/evaluate_reart_baseline.py",
            str(result_pkl),
            str(reference),
            "--output-json",
            str(output_json),
            "--primary-distance-ratio",
            "1.0",
        ]
        if hybrid.is_file():
            command.extend(["--reference-part-ids-from-tracks", str(hybrid)])
        run(command)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        metrics = payload["primary"]["covered_only_metrics"]
        summary_rows.append(
            {
                "object_id": object_id,
                "category": row["category"],
                "status": "success",
                "point_iou": metrics["one_to_one_mean_iou"],
                "ari": metrics["adjusted_rand_index"],
                "gt_part_count": metrics["gt_part_count"],
                "predicted_part_count": metrics["predicted_part_count"],
                "part_count_error": metrics["predicted_part_count"] - metrics["gt_part_count"],
                "undersegmented_gt_part_count": metrics["undersegmented_gt_part_count"],
                "unmatched_gt_part_count": metrics["unmatched_gt_part_count"],
                "largest_cluster_ratio": metrics["largest_cluster_ratio"],
                "runtime_s": status.get("runtime_s"),
                "frame_count": sequence_manifest["frame_count"],
                "input_points_per_frame": args.num_points,
                "joint_output": "unsupported_native",
            }
        )
    _write_csv(args.report_root / "reart_per_object.csv", summary_rows)
    (args.report_root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    _write_summary(args.report_root / "reart_summary.md", rows, summary_rows, failures)


def _sequence_frame_count(path: Path) -> int | None:
    manifest = path / "reart_sequence_manifest.json"
    if not manifest.is_file():
        return None
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("frame_count", 0))


def _failure_record(
    args: argparse.Namespace,
    row: dict[str, str],
    status: dict[str, Any],
    result_pkl: Path | None,
) -> dict[str, Any]:
    log_path = Path(status["log_path"]) if status.get("log_path") else None
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.is_file() else ""
    official_exception = None
    failure_stage = status.get(
        "failure_stage",
        "missing_result" if result_pkl is None else "incomplete_official_pipeline",
    )
    if "networkx.exception.NodeNotFound: Source 0 not in G" in log_text:
        failure_stage = "official_kinematic_tree_projection"
        official_exception = "networkx.exception.NodeNotFound: Source 0 not in G"

    predicted_part_count = None
    if result_pkl is not None:
        with result_pkl.open("rb") as stream:
            payload = pickle.load(stream)
        labels = payload.get("pred_cano_part")
        if labels is not None:
            predicted_part_count = len(set(int(value) for value in labels))
    return {
        "object_id": row["object_id"],
        "category": row["category"],
        "failure_stage": failure_stage,
        "exception": status.get(
            "exception",
            "No result.pkl found" if result_pkl is None else "Official projection did not complete",
        ),
        "official_exception": official_exception,
        "input_point_count": args.num_points,
        "frame_count": _sequence_frame_count(args.sequence_root / row["object_id"]),
        "predicted_part_count_relaxation": predicted_part_count,
        "runtime_s": status.get("runtime_s"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    requested: list[dict[str, str]],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    lines = [
        "# ReArt PartNet Shared-3view Pilot",
        "",
        "This report uses the official ReArt relaxation and kinematic projection stages.",
        "Each timestep is a world-frame point cloud fused from the three RGB-D views.",
        "No GT part labels or joint parameters are present in the exported ReArt files.",
        "",
        "## Aggregate",
        "",
        f"- Success: {len(rows)}/{len(requested)}",
        f"- Mean Point IoU: {_fmt(mean('point_iou'))}",
        f"- Mean ARI: {_fmt(mean('ari'))}",
        f"- Mean largest-cluster ratio: {_fmt(mean('largest_cluster_ratio'))}",
        "",
        "## Per Object",
        "",
        "| Object | Category | Point IoU | ARI | Pred / GT parts | Runtime (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_id']} | {row['category']} | {row['point_iou']:.3f} | "
            f"{row['ari']:.3f} | {row['predicted_part_count']} / {row['gt_part_count']} | "
            f"{_fmt(row.get('runtime_s'), digits=1)} |"
        )
    lines.extend(
        [
            "",
            "## Metric Provenance",
            "",
            "- Segmentation and undirected connectivity are native ReArt outputs.",
            "- Point IoU and ARI are converted with the same observed-reference-point evaluator used by AiM and Hybrid.",
            "- Joint type, axis, and pivot are not native `result.pkl` fields and are not reported in this pilot.",
            "- Directed Edge F1 is unsupported without adding a non-native parent orientation rule.",
            "",
            f"Failures: {len(failures)}. See `failures.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, *, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    rows = load_rows(args.objects_tsv)
    if args.stage in {"prepare", "all"}:
        prepare(args, rows)
    if args.stage in {"run", "all"}:
        run_official(args, rows)
    if args.stage in {"evaluate", "all"}:
        evaluate(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
