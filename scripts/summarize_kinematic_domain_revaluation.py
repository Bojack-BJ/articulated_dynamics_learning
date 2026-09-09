#!/usr/bin/env python3
"""Summarize common all/observable kinematic-domain segmentation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.expanduser().resolve()
    rows = list(csv.DictReader((root / "per_object_metrics.csv").open(encoding="utf-8")))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["domain"])].append(row)
    summary_rows = []
    for (method, domain), values in sorted(grouped.items()):
        summary_rows.append({
            "method": method,
            "domain": domain,
            "metric_n": len(values),
            "point_iou": _mean(values, "point_iou"),
            "ari": _mean(values, "ari"),
            "ri": _mean(values, "ri"),
            "geometry_coverage_2pct": _mean(values, "geometry_coverage_2pct"),
        })
    fieldnames = list(summary_rows[0])
    with (root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))
    failure_counts = Counter(row["method"] for row in failures)
    lines = [
        "# Kinematic-domain segmentation re-evaluation",
        "",
        "All methods use the same fixed-connected kinematic ontology. `all` retains every",
        "kinematic part and assigns zero IoU to parts absent from the reference acquisition;",
        "`observable` applies the fixed GT-acquisition support rule. Segmentation labels are",
        "transferred to all reference points; geometry coverage at 2% bbox is reported separately.",
        "",
        "| Method | Domain | N | Point IoU | ARI | RI | Coverage @2% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['domain']} | {row['metric_n']} | "
            f"{row['point_iou']:.3f} | {row['ari']:.3f} | {row['ri']:.3f} | "
            f"{row['geometry_coverage_2pct']:.3f} |"
        )
    lines.extend(["", "## Missing outputs", ""])
    for method, count in sorted(failure_counts.items()):
        lines.append(f"- `{method}`: {count}")
    lines.extend([
        "",
        "The observable domain is the appropriate common segmentation table. The all-domain",
        "result is a strict topology-complete diagnostic and is strongly affected by annotated",
        "movable links that never receive sufficient image/point support.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate_summary": str(root / "aggregate_summary.csv"), "summary": str(root / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
