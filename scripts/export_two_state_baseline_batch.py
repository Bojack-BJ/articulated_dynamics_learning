#!/usr/bin/env python3
"""Batch-export aligned objects to the shared DTA/ArtGS acquisition package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.two_state_export import export_two_state_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument(
        "--episode-template",
        required=True,
        help="Episode path template containing {object_id}.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-views-per-state", type=int, default=100)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest_csv.open(encoding="utf-8")))
    failures = []
    for index, row in enumerate(rows, start=1):
        object_id = row["object_id"]
        episode = Path(args.episode_template.format(object_id=object_id))
        output = args.output_root / "per_object" / object_id / "acquisition/two_state"
        print(f"[{index}/{len(rows)}] {object_id}: {episode}", flush=True)
        try:
            export_two_state_package(
                episode,
                output,
                object_id=object_id,
                category=row.get("category"),
                gt_part_count=_optional_int(row.get("gt_part_count")),
                required_views_per_state=args.required_views_per_state,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append(
                {
                    "object_id": object_id,
                    "stage": "two_state_export",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{object_id}] FAILED: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "two_state_export_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


def _optional_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
