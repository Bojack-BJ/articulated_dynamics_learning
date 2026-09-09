#!/usr/bin/env python3
"""Plot per-object Relation Head axis errors from easiest to hardest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotate-top", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    payload = json.loads(args.input_json.expanduser().resolve().read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("per_object", [])
        if int(row.get("axis_error_count", 0)) > 0
        and math.isfinite(float(row.get("axis_error_deg", math.inf)))
    ]
    rows.sort(key=lambda row: float(row["axis_error_deg"]))
    categories = sorted({str(row["category"]) for row in rows})
    palette = plt.get_cmap("tab10")
    colors = {category: palette(index % 10) for index, category in enumerate(categories)}

    figure, axis = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    x = np.arange(len(rows))
    errors = np.asarray([float(row["axis_error_deg"]) for row in rows])
    axis.plot(x, errors, color="#73808c", linewidth=1.0, alpha=0.55, zorder=1)
    for category in categories:
        indices = [index for index, row in enumerate(rows) if row["category"] == category]
        joint_counts = [int(rows[index]["joint_count"]) for index in indices]
        axis.scatter(
            indices,
            errors[indices],
            s=[30 + 13 * math.sqrt(max(1, count)) for count in joint_counts],
            color=colors[category],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.9,
            label=category,
            zorder=3,
        )

    median = float(np.median(errors))
    p75 = float(np.percentile(errors, 75))
    axis.axhline(median, color="#159b82", linestyle="--", linewidth=1.4, label=f"object median {median:.1f}°")
    axis.axhline(p75, color="#d7922e", linestyle=":", linewidth=1.4, label=f"object P75 {p75:.1f}°")
    axis.axhspan(45, 90, color="#d94b4b", alpha=0.07)
    axis.text(1, 86, "catastrophic axis-direction failures", color="#b53a3a", fontsize=10)

    annotate_count = min(max(0, int(args.annotate_top)), len(rows))
    for index in range(len(rows) - annotate_count, len(rows)):
        row = rows[index]
        axis.annotate(
            str(row["object_id"]).removeprefix("partnet_"),
            (index, errors[index]),
            xytext=(0, 8 + 12 * ((index - (len(rows) - annotate_count)) % 3)),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#27323a",
            rotation=35,
        )

    axis.set_title(
        "Relation Head Axis Error by Object\n"
        "CoTracker full-slot; sorted from low to high error; marker size = GT joint count",
        fontsize=17,
        weight="bold",
    )
    axis.set_xlabel("Test objects sorted by type-correct mean axis error (easy → hard)")
    axis.set_ylabel("Mean axis angular error per object (degrees)")
    axis.set_xlim(-2, len(rows) + 2)
    axis.set_ylim(0, 96)
    axis.grid(axis="y", color="#d8dee3", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=5, loc="upper left", frameon=False, fontsize=9)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
