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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
    return Path(tap["tracks_path"]), Path(cot["features_npz"]), args.cotracker_model, cot["split"]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def evaluate_prediction(path: Path, method: str, split: str, runtime_s: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", [])
    predicted = np.asarray([int(track["part_id"]) for track in tracks], dtype=np.int64)
    labels = np.asarray([int(track["original_part_id"]) for track in tracks], dtype=np.int64)
    metrics = evaluate_slot_assignments(predicted, labels)
    segmentation = payload.get("motion_segmentation", {})
    active_slots = list(segmentation.get("active_slots", []))
    existence = list(segmentation.get("slot_existence_probabilities", []))
    object_id = str(payload.get("object_instance_id", path.parent.name))
    return {
        "object_id": object_id,
        "category": "refrigerator" if object_id.startswith("refrigerator") else "microwave",
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

    lines = [
        "# Three-Tracker Motion-Part Slot Benchmark",
        "",
        "GT part labels are used only for evaluation.",
        "",
        "| Group | N | Purity | GT coverage | 1:1 IoU | Pair F1 | ARI | Exact K | Underseg | Overseg | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append(
            f"| {name} | {values['object_count']} | {values['mean_cluster_purity']:.3f} | "
            f"{values['mean_gt_coverage']:.3f} | {values['one_to_one_mean_iou']:.3f} | "
            f"{values['pairwise_same_part_f1']:.3f} | {values['adjusted_rand_index']:.3f} | "
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
    if not object_ids:
        raise ValueError("The CoTracker and TAPIP manifests have no common objects.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    python = sys.executable
    for method in args.methods:
        for index, object_id in enumerate(object_ids, start=1):
            tracks, features, model, split = method_inputs(method, object_id, cotracker, tapip, args)
            for path in (tracks, features, model):
                if not path.resolve().exists():
                    raise FileNotFoundError(path.resolve())
            item_dir = output_dir / method / split / object_id
            item_dir.mkdir(parents=True, exist_ok=True)
            prediction = item_dir / "motion_part_tracks_slots.json"
            viewer = item_dir / "viewer.html"
            runtime_path = item_dir / "runtime_s.txt"
            print(f"[{method} {index}/{len(object_ids)}] {object_id}", flush=True)
            if not (args.resume and prediction.exists()):
                started = time.perf_counter()
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
                run_command(command)
                runtime_path.write_text(f"{time.perf_counter() - started:.9f}\n", encoding="utf-8")
            if not args.no_viewers and not (args.resume and viewer.exists()):
                run_command([
                    python, "-m", "rgbd_urdf_mvp", "visualize-object-mask-flow-html",
                    str(prediction), "--output-html", str(viewer),
                    "--max-tracks", str(max(1, args.viewer_max_tracks)), "--axis-remap", "x,y,z",
                ])
            runtime_s = float(runtime_path.read_text()) if runtime_path.exists() else 0.0
            row = evaluate_prediction(prediction, method, split, runtime_s)
            row["viewer_html"] = str(viewer)
            rows.append(row)
            write_reports(output_dir, rows)
    print(json.dumps({"output_dir": str(output_dir), "object_count": len(object_ids), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
