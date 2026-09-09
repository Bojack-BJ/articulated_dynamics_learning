#!/usr/bin/env python3
"""Summarize measured external-baseline runtimes without inventing missing values."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "ours_hybrid",
    "aim_aligned",
    "reart",
    "dta",
    "artgs",
    "gaussianart",
    "videoartgs",
    "paris",
    "ditto",
)


def _positive(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _collect_suite(root: Path) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for path in (root / "per_object").glob(f"*/{method}/metrics.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "object_id": path.parents[1].name,
                    "method": method,
                    "status": data.get("status"),
                    "runtime_s": _positive(data.get("runtime_s")),
                    "timing_scope": "official_end_to_end_run",
                    "source": str(path),
                }
            )
    return rows


def _collect_ours(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "object_id": row["object_id"],
            "method": "ours_hybrid",
            "status": "success",
            "runtime_s": _positive(row.get("runtime_s")),
            "timing_scope": "feedforward_slot_segmentation_only_excludes_tracking_and_relation",
            "source": str(path),
        }
        for row in data.get("per_object", [])
        if row.get("method") == "hybrid"
    ]


def _collect_reart(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {
                "object_id": row["object_id"],
                "method": "reart",
                "status": row["status"],
                "runtime_s": _positive(float(row["runtime_s"])) if row["runtime_s"] else None,
                "timing_scope": "official_native_inference",
                "source": str(path),
            }
            for row in csv.DictReader(stream)
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root", type=Path, default=Path("outputs/external_baseline_suite_v1")
    )
    parser.add_argument(
        "--ours-benchmark",
        type=Path,
        default=Path(
            "outputs/partnet_core_v1_training/slot_benchmark_three_methods_test/"
            "benchmark_metrics.json"
        ),
    )
    parser.add_argument(
        "--reart-csv",
        type=Path,
        default=Path(
            "outputs/external_baselines_v1/reart_reports_aligned_4x512/"
            "reart_per_object.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.suite_root.expanduser().resolve()
    output = (args.output_dir or root / "runtime_comparison_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = _collect_suite(root)
    # Prefer dedicated measured sources when suite metrics do not contain timing.
    rows.extend(_collect_ours(args.ours_benchmark.expanduser().resolve()))
    rows.extend(_collect_reart(args.reart_csv.expanduser().resolve()))
    deduplicated = {}
    for row in rows:
        key = (row["method"], row["object_id"])
        if key not in deduplicated or (
            deduplicated[key]["runtime_s"] is None and row["runtime_s"] is not None
        ):
            deduplicated[key] = row
    rows = sorted(deduplicated.values(), key=lambda row: (row["method"], row["object_id"]))

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summary = []
    for method in METHODS:
        method_rows = by_method[method]
        successful_rows = [
            row
            for row in method_rows
            if str(row["status"]).startswith(("success", "completed"))
        ]
        failed_rows = [row for row in method_rows if row not in successful_rows]
        measured = [
            row["runtime_s"]
            for row in successful_rows
            if row["runtime_s"] is not None
        ]
        failed_measured = [
            row["runtime_s"] for row in failed_rows if row["runtime_s"] is not None
        ]
        scopes = sorted(
            {
                row["timing_scope"]
                for row in successful_rows
                if row["runtime_s"] is not None
            }
        )
        summary.append(
            {
                "method": method,
                "requested_count": len(method_rows),
                "successful_count": len(successful_rows),
                "measured_count": len(measured),
                "failed_runtime_count": len(failed_measured),
                "mean_runtime_s": statistics.fmean(measured) if measured else None,
                "median_runtime_s": statistics.median(measured) if measured else None,
                "min_runtime_s": min(measured) if measured else None,
                "max_runtime_s": max(measured) if measured else None,
                "timing_scope": ";".join(scopes) if scopes else "unavailable",
                "comparable_end_to_end": len(scopes) == 1 and scopes[0] == "official_end_to_end_run",
            }
        )

    _write_csv(output / "runtime_per_object.csv", rows)
    _write_csv(output / "runtime_summary.csv", summary)
    (output / "runtime_summary.json").write_text(
        json.dumps({"per_object": rows, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# External Baseline Runtime",
        "",
        "| Method | Success | Measured | Mean (s) | Median (s) | Timing scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        mean = "n/a" if row["mean_runtime_s"] is None else f"{row['mean_runtime_s']:.2f}"
        median = "n/a" if row["median_runtime_s"] is None else f"{row['median_runtime_s']:.2f}"
        lines.append(
            f"| {row['method']} | {row['successful_count']}/{row['requested_count']} | "
            f"{row['measured_count']} | "
            f"{mean} | {median} | {row['timing_scope']} |"
        )
    lines.extend(
        [
            "",
            "Runtime scopes are intentionally not collapsed into one speedup number. "
            "The Ours measurement excludes tracker feature extraction, whereas DTA/ArtGS "
            "measure their official optimization runs. Missing values are not inferred "
            "from file timestamps.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
