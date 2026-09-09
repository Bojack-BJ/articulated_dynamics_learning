#!/usr/bin/env python3
"""Run and summarize CoTracker, TAPIP3D, and hybrid slot-head inference."""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments


METHODS = ("cotracker_only", "tapip_only", "hybrid")
METRICS = (
    "mean_cluster_purity",
    "weighted_cluster_purity",
    "mean_gt_coverage",
    "one_to_one_mean_iou",
    "one_to_one_mean_f1",
    "pairwise_same_part_f1",
    "rand_index",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "largest_cluster_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cotracker-manifest", type=Path, required=True)
    parser.add_argument("--tapip-manifest", type=Path, required=True)
    parser.add_argument("--cotracker-model", type=Path, required=True)
    parser.add_argument("--tapip-model", type=Path, required=True)
    parser.add_argument(
        "--hybrid-model",
        type=Path,
        default=None,
        help="Hybrid slot checkpoint. Defaults to the CoTracker checkpoint for backward compatibility.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=None, help="Optional object catalog with category labels")
    parser.add_argument(
        "--recording-root", type=Path, default=None,
        help="When set, generate enhanced viewers with RGB-D geometry and simulation GT meshes",
    )
    parser.add_argument("--background-max-points", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Restrict inference to one manifest split. Defaults to all rows.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--slot-existence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--tapip-slot-existence-threshold",
        type=float,
        default=0.0,
        help="TAPIP-native existence logits are currently uncalibrated; keep all queries by default.",
    )
    parser.add_argument("--post-ransac-refine", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-viewers", action="store_true")
    parser.add_argument("--viewer-max-tracks", type=int, default=1200)
    parser.add_argument(
        "--viewer-frame-stride",
        type=int,
        default=1,
        help="Embed every Nth tracked frame. Defaults to all frames so articulated arcs are not visually aliased.",
    )
    parser.add_argument(
        "--no-viewer-joint-axes",
        action="store_false",
        dest="viewer_joint_axes",
        help="Skip pose/joint fitting used only to overlay predicted axes in generated viewers.",
    )
    parser.set_defaults(viewer_joint_axes=True)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Objects evaluated concurrently; each worker owns a separate inference subprocess.",
    )
    parser.add_argument(
        "--relation-model",
        type=Path,
        default=None,
        help="Optional learned relation-head checkpoint. Hybrid learned kinematics is timed separately.",
    )
    parser.add_argument("--relation-edge-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        item = dict(row)
        for key in ("tracks_path", "features_npz"):
            item[key] = str((path.resolve().parent / item[key]).resolve())
        result[item["object_id"]] = item
    return result


def method_inputs(
    method: str,
    object_id: str,
    cotracker: dict[str, dict[str, str]],
    tapip: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, str]:
    cot = cotracker[object_id]
    tap = tapip[object_id]
    if cot["split"] != tap["split"]:
        raise ValueError(f"Split mismatch for {object_id}: {cot['split']} vs {tap['split']}")
    if method == "cotracker_only":
        return Path(cot["tracks_path"]), Path(cot["features_npz"]), args.cotracker_model, cot["split"]
    if method == "tapip_only":
        return Path(tap["tracks_path"]), Path(tap["features_npz"]), args.tapip_model, tap["split"]
    hybrid_model = args.hybrid_model or args.cotracker_model
    return Path(tap["tracks_path"]), Path(cot["features_npz"]), hybrid_model, cot["split"]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def timed_command(command: list[str]) -> float:
    started = time.perf_counter()
    run_command(command)
    return time.perf_counter() - started


def evaluate_prediction(
    path: Path, method: str, split: str, runtime_s: float, category: str | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", [])
    predicted = np.asarray([int(track["part_id"]) for track in tracks], dtype=np.int64)
    labels = np.asarray([int(track["original_part_id"]) for track in tracks], dtype=np.int64)
    metrics = evaluate_slot_assignments(predicted, labels)
    segmentation = payload.get("motion_segmentation", {})
    active_slots = list(segmentation.get("active_slots", []))
    existence = list(segmentation.get("slot_existence_probabilities", []))
    object_id = str(payload.get("object_instance_id", path.parent.name))
    if category is None:
        if object_id.startswith("partnet_"):
            category = "partnet"
        elif object_id.startswith("refrigerator"):
            category = "refrigerator"
        elif object_id.startswith("microwave"):
            category = "microwave"
        else:
            category = "unknown"
    return {
        "object_id": object_id,
        "category": category,
        "split": split,
        "method": method,
        "track_count": len(tracks),
        "runtime_s": runtime_s,
        **{key: value for key, value in metrics.items() if not isinstance(value, list)},
        "active_slot_count": len(active_slots),
        "max_existence_probability": max(existence, default=0.0),
        "min_active_existence_probability": min((existence[index] for index in active_slots), default=0.0),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "object_count": len(rows),
        **{metric: mean(float(row[metric]) for row in rows) for metric in METRICS},
        "part_count_exact_rate": mean(float(row["part_count_exact"]) for row in rows),
        "undersegmented_object_rate": mean(float(row["undersegmented_gt_part_count"] > 0) for row in rows),
        "oversegmented_object_rate": mean(float(row["oversegmented_pred_slot_count"] > 0) for row in rows),
        "runtime_s": sum(float(row["runtime_s"]) for row in rows),
    }


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['method']}/all"].append(row)
        groups[f"{row['method']}/{row['split']}/all"].append(row)
        groups[f"{row['method']}/{row['split']}/{row['category']}"].append(row)
    summary = {name: aggregate(values) for name, values in sorted(groups.items())}
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps({"per_object": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "benchmark_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = [{"group": name, **values} for name, values in summary.items()]
    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Three-Tracker Motion-Part Slot Benchmark",
        "",
        "GT part labels are used only for evaluation.",
        "",
        "| Group | N | Purity | GT coverage | 1:1 IoU | Pair F1 | RI | ARI | Exact K | Underseg | Overseg | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append(
            f"| {name} | {values['object_count']} | {values['mean_cluster_purity']:.3f} | "
            f"{values['mean_gt_coverage']:.3f} | {values['one_to_one_mean_iou']:.3f} | "
            f"{values['pairwise_same_part_f1']:.3f} | {values['rand_index']:.3f} | "
            f"{values['adjusted_rand_index']:.3f} | "
            f"{values['part_count_exact_rate']:.3f} | {values['undersegmented_object_rate']:.3f} | "
            f"{values['oversegmented_object_rate']:.3f} | {values['runtime_s']:.1f} |"
        )
    lines.extend(["", "## Viewers", ""])
    for row in rows:
        viewer = Path(row["viewer_html"])
        lines.append(f"- `{row['method']}` / `{row['object_id']}`: [{viewer.name}]({viewer.relative_to(output_dir)})")
    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_object = {
        object_id: {row["method"]: row for row in rows if row["object_id"] == object_id}
        for object_id in sorted({row["object_id"] for row in rows})
    }
    html_rows = []
    for object_id, method_rows in by_object.items():
        cells = [f"<td><code>{html.escape(object_id)}</code></td>"]
        for method in METHODS:
            row = method_rows.get(method)
            if row is None:
                cells.append("<td>not run</td>")
                continue
            viewer = Path(row["viewer_html"])
            link = viewer.relative_to(output_dir).as_posix()
            cells.append(
                "<td>"
                f"<a href='{html.escape(link)}'>viewer</a><br>"
                f"IoU {row['one_to_one_mean_iou']:.3f} / purity {row['mean_cluster_purity']:.3f}<br>"
                f"coverage {row['mean_gt_coverage']:.3f} / K {row['predicted_part_count']}/{row['gt_part_count']}"
                "</td>"
            )
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    index_html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Three-Tracker Benchmark</title>
<style>
body{font:15px ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1118;color:#dce8ef;padding:28px}
table{border-collapse:collapse;width:100%;max-width:1400px}th,td{border:1px solid #2b3b48;padding:10px;vertical-align:top}
th{background:#15232e;position:sticky;top:0}tr:nth-child(even){background:#101b24}a{color:#62d9b0}
</style></head><body><h1>Three-Tracker Motion-Part Slot Benchmark</h1>
<p>CoTracker-only, TAPIP-only, and hybrid use the same object split. GT labels are evaluation-only.</p>
<table><thead><tr><th>Object</th><th>CoTracker only</th><th>TAPIP only</th><th>Hybrid</th></tr></thead><tbody>
""" + "\n".join(html_rows) + "\n</tbody></table></body></html>\n"
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    cotracker = load_manifest(args.cotracker_manifest)
    tapip = load_manifest(args.tapip_manifest)
    object_ids = sorted(set(cotracker) & set(tapip))
    if args.split != "all":
        object_ids = [
            object_id
            for object_id in object_ids
            if cotracker[object_id].get("split") == args.split
            and tapip[object_id].get("split") == args.split
        ]
    if not object_ids:
        raise ValueError("The CoTracker and TAPIP manifests have no common objects.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    python = sys.executable
    categories: dict[str, str] = {}
    if args.catalog is not None:
        catalog_payload = json.loads(args.catalog.expanduser().resolve().read_text(encoding="utf-8"))
        catalog_rows = catalog_payload.get("objects", catalog_payload) if isinstance(catalog_payload, dict) else catalog_payload
        categories = {
            str(item["object_id"]): str(item.get("category", item.get("source_category", "unknown")))
            for item in catalog_rows
        }
    recording_root = args.recording_root.expanduser().resolve() if args.recording_root is not None else None

    def process_object(method: str, index: int, object_id: str) -> dict[str, Any]:
        tracks, features, model, split = method_inputs(method, object_id, cotracker, tapip, args)
        for path in (tracks, features, model):
            if not path.resolve().exists():
                raise FileNotFoundError(path.resolve())
        item_dir = output_dir / method / split / object_id
        item_dir.mkdir(parents=True, exist_ok=True)
        prediction = item_dir / "motion_part_tracks_slots.json"
        part_poses = item_dir / "part_poses.json"
        joint_inference = item_dir / "joint_inference.json"
        relation_inference = item_dir / "joint_inference_relation.json"
        viewer = item_dir / ("viewer_enhanced.html" if recording_root is not None else "viewer.html")
        runtime_path = item_dir / "runtime_s.txt"
        timing_path = item_dir / "runtime_profile.json"
        phase_times: dict[str, float | None] = {
            "slot_subprocess_wall_s": None,
            "learned_slot_relation_subprocess_wall_s": None,
            "analytic_part_pose_subprocess_wall_s": None,
            "analytic_joint_inference_subprocess_wall_s": None,
            "viewer_generation_s": None,
        }
        print(f"[{method} {index}/{len(object_ids)}] {object_id}", flush=True)
        if not (args.resume and prediction.exists()):
            existence_threshold = (
                args.tapip_slot_existence_threshold
                if method == "tapip_only"
                else args.slot_existence_threshold
            )
            command = [
                python, "-m", "rgbd_urdf_mvp", "infer-motion-part-slots",
                str(tracks.resolve()), str(features.resolve()), str(model.resolve()),
                "--output-json", str(prediction), "--device", args.device,
                "--slot-existence-threshold", str(existence_threshold),
            ]
            if args.post_ransac_refine:
                command.append("--post-ransac-refine")
            phase_times["slot_subprocess_wall_s"] = timed_command(command)
            runtime_path.write_text(
                f"{phase_times['slot_subprocess_wall_s']:.9f}\n", encoding="utf-8"
            )
        if method == "hybrid" and args.relation_model is not None:
            if not (args.resume and relation_inference.exists()):
                phase_times["learned_slot_relation_subprocess_wall_s"] = timed_command([
                    python, "-m", "rgbd_urdf_mvp", "infer-slot-relation-head",
                    str(tracks.resolve()), str(features.resolve()), str(model.resolve()),
                    str(args.relation_model.expanduser().resolve()),
                    "--output-json", str(relation_inference),
                    "--device", args.device,
                    "--slot-existence-threshold", str(args.slot_existence_threshold),
                    "--edge-threshold", str(args.relation_edge_threshold),
                ])
        if args.viewer_joint_axes:
            if not (args.resume and part_poses.exists()):
                phase_times["analytic_part_pose_subprocess_wall_s"] = timed_command([
                    python, "-m", "rgbd_urdf_mvp", "estimate-part-poses",
                    str(prediction), "--method", "tracks", "--output-json", str(part_poses),
                ])
            if not (args.resume and joint_inference.exists()):
                phase_times["analytic_joint_inference_subprocess_wall_s"] = timed_command([
                    python, "-m", "rgbd_urdf_mvp", "infer-joints",
                    str(part_poses), "--output-json", str(joint_inference), "--mujoco-prior", "off",
                ])
        viewer_is_current = viewer.exists() and (
            not args.viewer_joint_axes
            or not joint_inference.exists()
            or viewer.stat().st_mtime >= joint_inference.stat().st_mtime
        )
        if not args.no_viewers and not (args.resume and viewer_is_current):
            viewer_command = [
                python, "-m", "rgbd_urdf_mvp", "visualize-object-mask-flow-html",
                str(prediction), "--output-html", str(viewer),
                "--max-tracks", str(max(1, args.viewer_max_tracks)), "--axis-remap", "x,y,z",
                "--frame-stride", str(max(1, int(args.viewer_frame_stride))),
            ]
            if args.viewer_joint_axes and joint_inference.is_file():
                viewer_command.extend(["--joint-inference", str(joint_inference)])
            if recording_root is not None:
                object_root = recording_root / object_id
                fusion_manifest = object_root / "pointcloud_4d_partseg" / "fusion_manifest.json"
                episode = object_root / "episode.json"
                if fusion_manifest.is_file():
                    viewer_command.extend([
                        "--background-fusion-manifest", str(fusion_manifest),
                        "--background-max-points", str(max(1, int(args.background_max_points))),
                    ])
                if episode.is_file():
                    viewer_command.extend(["--mjcf-replay-episode", str(episode)])
            phase_times["viewer_generation_s"] = timed_command(viewer_command)
        previous_timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.exists()
            else {}
        )
        for key, value in phase_times.items():
            if value is None and key in previous_timing.get("phases", {}):
                phase_times[key] = previous_timing["phases"][key]
        timing_payload = {
            "object_id": object_id,
            "method": method,
            "phases": phase_times,
            "timing_semantics": {
                "slot_subprocess_wall_s": "slot segmentation only",
                "learned_slot_relation_subprocess_wall_s": (
                    "slot backbone plus learned topology/type/axis/pivot; do not add slot time"
                ),
                "analytic_part_pose_subprocess_wall_s": "track-based robust SE(3) pose fitting",
                "analytic_joint_inference_subprocess_wall_s": "analytic joint model fitting",
                "viewer_generation_s": "reporting only; excluded from inference",
            },
            "execution": {
                "inter_object_workers": max(1, int(args.workers)),
                "parallelism": "ThreadPoolExecutor launches one subprocess per object",
                "device": args.device,
            },
        }
        timing_path.write_text(json.dumps(timing_payload, indent=2) + "\n", encoding="utf-8")
        runtime_s = float(runtime_path.read_text()) if runtime_path.exists() else 0.0
        row = evaluate_prediction(prediction, method, split, runtime_s, categories.get(object_id))
        row["slot_inference_runtime_s"] = runtime_s
        row["learned_slot_relation_runtime_s"] = (
            phase_times["learned_slot_relation_subprocess_wall_s"]
        )
        row["analytic_part_pose_runtime_s"] = phase_times["analytic_part_pose_subprocess_wall_s"]
        row["analytic_joint_inference_runtime_s"] = (
            phase_times["analytic_joint_inference_subprocess_wall_s"]
        )
        row["timing_profile"] = str(timing_path)
        row["viewer_html"] = str(viewer)
        return row

    for method in args.methods:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(process_object, method, index, object_id): object_id
                for index, object_id in enumerate(object_ids, start=1)
            }
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: (str(row["method"]), str(row["object_id"])))
                write_reports(output_dir, rows)
    print(json.dumps({"output_dir": str(output_dir), "object_count": len(object_ids), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
