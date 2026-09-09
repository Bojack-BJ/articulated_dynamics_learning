#!/usr/bin/env python3
"""Create a non-destructive storage cleanup plan for flow-tracking experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupItem:
    path: str
    size_bytes: int
    size_human: str
    recommendation: str
    rationale: str
    suggested_action: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("outputs/flow_tracking_eval"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _recommend(path: Path) -> tuple[str, str, str]:
    name = path.name
    if name.startswith("slot_relation_head_") or name == "slot_cross_object_summary.md":
        return (
            "keep",
            "Compact relation-head checkpoints and reports remain useful for model comparisons.",
            "Keep in place.",
        )
    if name in {"three_tracker_full_benchmark", "tapip3d_slot_benchmark", "slot_benchmark_refrigerator_microwave"}:
        return (
            "archive",
            "Historical tracker benchmark; preserve its summaries and one representative viewer, but it is not the current PartNet training source.",
            "Archive the directory before deleting duplicate viewers or raw inputs.",
        )
    if name.startswith("refrigerator"):
        return (
            "review-before-delete",
            "Per-object object-mask diagnostic output. Preserve only selected reference objects and their final artifacts after checking active reports no longer link here.",
            "Keep one final artifact/viewer per representative object; archive or delete earlier per-object copies.",
        )
    if any(token in name for token in ("post_spectral_merge", "sequential_ransac", "articulation_affinity")):
        return (
            "archive-or-delete",
            "Versioned intermediate diagnostic run superseded by later PartNet experiments; summaries at the parent root are small and retained separately.",
            "Archive the final chosen version only; delete or archive the remaining full artifacts after confirmation.",
        )
    if name in {"refrigerators_dense_mps_object_mask_debug", "refrigerators_dense_mps_object_mask_grid", "dynamic_reseed_smoke"}:
        return (
            "archive-or-delete",
            "Exploratory debug/smoke output retained only for provenance.",
            "Keep the report or screenshot, then remove the generated artifacts.",
        )
    return (
        "keep",
        "No high-confidence cleanup rule applies; retain until an experiment owner confirms its status.",
        "Keep in place.",
    )


def _write_markdown(items: list[CleanupItem], output: Path) -> None:
    totals: dict[str, int] = {}
    for item in items:
        totals[item.recommendation] = totals.get(item.recommendation, 0) + item.size_bytes
    lines = [
        "# Flow-Tracking Cleanup Dry Run",
        "",
        "This report does not move or delete any files.",
        "",
        "## Summary",
        "",
        "| Recommendation | Size |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {_human_size(value)} |" for key, value in sorted(totals.items()))
    lines.extend(["", "## Items", "", "| Path | Size | Recommendation | Suggested action |", "| --- | ---: | --- | --- |"])
    for item in sorted(items, key=lambda value: value.size_bytes, reverse=True):
        lines.append(
            f"| `{item.path}` | {item.size_human} | {item.recommendation} | {item.suggested_action} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root / "_cleanup_dry_run").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child == output_dir:
            continue
        # This parent is a container for independently generated experiments.
        # List its children so the plan identifies the actual large artifacts.
        if child.name == "refrigerators_dense_mps_object_mask":
            candidates.extend(grandchild for grandchild in sorted(child.iterdir()) if grandchild.is_dir())
        else:
            candidates.append(child)
    for path in candidates:
        recommendation, rationale, action = _recommend(path)
        size = _size_bytes(path)
        items.append(CleanupItem(str(path), size, _human_size(size), recommendation, rationale, action))

    (output_dir / "cleanup_manifest.json").write_text(
        json.dumps([asdict(item) for item in items], indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "cleanup_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CleanupItem.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(item) for item in items)
    _write_markdown(items, output_dir / "cleanup_report.md")
    print(json.dumps({"root": str(root), "items": len(items), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
