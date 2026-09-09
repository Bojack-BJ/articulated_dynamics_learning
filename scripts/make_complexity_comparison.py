#!/usr/bin/env python3
"""Plot complexity-aware external segmentation comparisons for Track2Art."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/track2art-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BUCKETS = ("2", "3-4", ">=5")
BUCKET_LABELS = ("2 parts", "3--4 parts", r"$\geq$5 parts")
METHODS = {
    "ours_cotracker": {
        "label": "Track2Art",
        "color": "#2457A6",
        "marker": "o",
        "protocol": "continuous_interaction",
        "oracle_requirements": [],
    },
    "artgs": {
        "label": "ArtGS",
        "color": "#D28C28",
        "marker": "s",
        "protocol": "two_state_multiview_rgbd_released_path",
        "oracle_requirements": ["gt_part_count"],
    },
    "dta": {
        "label": "DTA",
        "color": "#4C8C6B",
        "marker": "^",
        "protocol": "two_state_multiview_rgbd",
        "oracle_requirements": ["gt_part_count"],
    },
    "videoartgs": {
        "label": "VideoArtGS",
        "color": "#8C6BB1",
        "marker": "D",
        "protocol": "continuous_monocular_interaction",
        "oracle_requirements": ["joint_count", "joint_types", "parent_topology"],
    },
    "aim_aligned": {
        "label": "AiM",
        "color": "#777777",
        "marker": "P",
        "protocol": "aim_style_cross_protocol_generalization",
        "oracle_requirements": [],
    },
    "reart": {
        "label": "ReArt",
        "color": "#B06A5B",
        "marker": "v",
        "protocol": "native_4d_point_cloud",
        "oracle_requirements": [],
    },
    "paris": {
        "label": "PARIS",
        "color": "#B05A87",
        "marker": "X",
        "protocol": "two_state_multiview_rgb_2part_only",
        "oracle_requirements": ["two_part_object"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complexity-csv",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/complexity_summary.csv"),
    )
    parser.add_argument(
        "--segmentation-csv",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/segmentation_summary.csv"),
    )
    parser.add_argument(
        "--failures-json",
        type=Path,
        default=Path("outputs/external_baseline_suite_v1/failures.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_assets/experiments")
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def collect_values(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in METHODS}
    for row in rows:
        method = row["method"]
        bucket = row["complexity_bucket"]
        if method not in METHODS or bucket not in BUCKETS:
            continue
        point_iou = row.get("point_iou", "")
        ari = row.get("ari", "")
        if not point_iou or not ari:
            continue
        selected[method][bucket] = {
            "point_iou": float(point_iou),
            "ari": float(ari),
            "success_n": int(row["success_n"]),
            "metric_n": int(row["metric_n"]),
            "source_row": row,
        }
    missing = []
    for method, by_bucket in selected.items():
        required = ("2",) if method == "paris" else BUCKETS
        missing.extend(f"{method}/{bucket}" for bucket in required if bucket not in by_bucket)
    if missing:
        raise ValueError(f"Missing supported complexity rows: {', '.join(missing)}")
    return selected


def plot_panel(
    ax: mpl.axes.Axes,
    values: dict[str, dict[str, dict[str, Any]]],
    metric: str,
    panel_label: str,
) -> None:
    x = np.arange(len(BUCKETS))
    for method, spec in METHODS.items():
        available = [(index, values[method][bucket][metric]) for index, bucket in enumerate(BUCKETS) if bucket in values[method]]
        indices = np.asarray([item[0] for item in available])
        y = np.asarray([item[1] for item in available])
        is_ours = method == "ours_cotracker"
        ax.plot(
            indices,
            y,
            color=spec["color"],
            linewidth=2.55 if is_ours else 1.35,
            linestyle="-" if is_ours else "--",
            marker=spec["marker"],
            markersize=5.2 if is_ours else 4.0,
            markerfacecolor=spec["color"] if is_ours else "white",
            markeredgecolor=spec["color"],
            markeredgewidth=1.0,
            label=spec["label"],
            zorder=5 if is_ours else 3,
        )
        if is_ours:
            for local_index, score in enumerate(y):
                index = int(indices[local_index])
                is_final = index == len(BUCKETS) - 1
                ax.annotate(
                    f"Track2Art {score:.2f}" if is_final else f"{score:.2f}",
                    (x[index], score),
                    xytext=(0, 19) if is_final else (0, -13 if score > 0.92 else 8),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color=spec["color"],
                )

    # Label the strongest non-Track2Art result in the hardest bucket.
    strongest_baseline = max(
        (method for method in METHODS if method != "ours_cotracker" and BUCKETS[-1] in values[method]),
        key=lambda method: values[method][BUCKETS[-1]][metric],
    )
    baseline_score = values[strongest_baseline][BUCKETS[-1]][metric]
    baseline_spec = METHODS[strongest_baseline]
    ax.annotate(
        f"{baseline_spec['label']} {baseline_score:.2f}",
        (x[-1], baseline_score),
        xytext=(5, -11),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=baseline_spec["color"],
    )

    ax.set_xlim(-0.12, 2.26)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(x, BUCKET_LABELS)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Score")
    ax.grid(axis="y", color="#E8EBEF", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.text(-0.13, 1.055, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def build_metadata(
    args: argparse.Namespace,
    values: dict[str, dict[str, dict[str, Any]]],
    segmentation_rows: list[dict[str, str]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    segmentation_by_method = {row["method"]: row for row in segmentation_rows}
    methods: dict[str, Any] = {}
    for method, spec in METHODS.items():
        by_bucket = values[method]
        methods[method] = {
            "display_name": spec["label"],
            "color": spec["color"],
            "marker": spec["marker"],
            "protocol": segmentation_by_method.get(method, {}).get("protocol", spec["protocol"]),
            "oracle_requirements": spec["oracle_requirements"],
            "values": {
                bucket: {
                    "point_iou": by_bucket[bucket]["point_iou"],
                    "ari": by_bucket[bucket]["ari"],
                    "valid_n": by_bucket[bucket]["metric_n"],
                    "success_n": by_bucket[bucket]["success_n"],
                }
                for bucket in BUCKETS if bucket in by_bucket
            },
        }
    return {
        "schema": "track2art-complexity-comparison-v1",
        "comparison_contract": "Object-aligned, metric-aligned, protocol-specific",
        "source_paths": {
            "complexity_summary": str(args.complexity_csv),
            "segmentation_summary": str(args.segmentation_csv),
            "failures": str(args.failures_json),
        },
        "complexity_buckets": list(BUCKETS),
        "plotting_order": list(METHODS),
        "methods": methods,
        "failure_record_count": len(failures),
        "bootstrap": {
            "computed": False,
            "reason": (
                "The required source complexity CSV contains aggregate means rather than "
                "per-object samples; confidence intervals are omitted rather than inferred."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    complexity_rows = read_csv(args.complexity_csv)
    segmentation_rows = read_csv(args.segmentation_csv)
    failures = json.loads(args.failures_json.read_text())
    if not isinstance(failures, list):
        raise ValueError("failures.json must contain a list")
    values = collect_values(complexity_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.82), sharey=True)
    plot_panel(axes[0], values, "point_iou", "(a) Point IoU")
    plot_panel(axes[1], values, "ari", "(b) ARI")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.075, right=0.982, top=0.76, bottom=0.17, wspace=0.12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "complexity_comparison"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    plt.close(fig)

    metadata = build_metadata(args, values, segmentation_rows, failures)
    stem.with_name(f"{stem.name}_data").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("Source rows used:")
    for method, spec in METHODS.items():
        counts = ", ".join(
            f"{bucket}:N={values[method][bucket]['metric_n']}" for bucket in BUCKETS if bucket in values[method]
        )
        print(f"  {spec['label']}: {counts}")
    print("Suggested caption:")
    print(
        "Complexity-aware external comparison on object-aligned PartNet instances. "
        "Point IoU falls from two-part to at-least-five-part objects for every method, "
        "while ARI trends are less monotonic. Track2Art maintains the strongest common "
        "point-domain decomposition across all buckets and a clear advantage on "
        "multi-part objects, although objects with at least five parts remain challenging. "
        "PARIS is shown only for two-part objects because this is its native scope; "
        "missing higher-complexity points are not failures or zero scores. "
        "Evaluation metrics are aligned, but acquisition protocols and oracle assumptions "
        "differ across methods."
    )
    print(f"Generated: {stem}.pdf, {stem}.svg, {stem}.png")


if __name__ == "__main__":
    main()
