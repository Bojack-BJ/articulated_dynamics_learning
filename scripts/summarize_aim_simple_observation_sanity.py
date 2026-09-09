#!/usr/bin/env python3
"""Summarize the Storage-47024 favorable-observation AiM sanity check."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from audit_aim_protocol import _read_ply_part_ids, _read_ply_xyz_rgb


GAUSSIAN_PATTERN = re.compile(
    r"(?P<static>\d+) Static Gaussians and (?P<moving>\d+) Moving Gaussians"
)
SEGMENTATION_COUNT_PATTERN = re.compile(
    r"^\s*(?P<static>\d+)\s+(?P<moving>\d+)(?:\s+\[[^\n]+\])?\s*$",
    re.MULTILINE,
)
PROTOCOLS = (
    "paper_like_current",
    "close_monocular",
    "four_view_upper_bound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--base-gt-part-id", type=int, default=2)
    parser.add_argument("--moving-gt-part-id", type=int, default=4)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold(payload: dict[str, Any], ratio: float = 0.02) -> dict[str, Any]:
    return next(
        row
        for row in payload["thresholds"]
        if abs(float(row["distance_ratio_bbox"]) - ratio) <= 1e-9
    )


def _metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload["primary"]["covered_only_metrics"]
    strict = _threshold(payload)
    return {
        "predicted_part_count": int(primary["predicted_part_count"]),
        "gt_part_count": int(primary["gt_part_count"]),
        "point_iou": float(primary["one_to_one_mean_iou"]),
        "ari": float(primary["adjusted_rand_index"]),
        "rand_index": float(primary["rand_index"]),
        "geometry_coverage_at_2pct": float(strict["geometry_coverage"]),
    }


def _parse_gaussian_counts(run_dir: Path) -> tuple[int, int]:
    segmentation_text = (run_dir / "segmentation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    segmentation_matches = list(SEGMENTATION_COUNT_PATTERN.finditer(segmentation_text))
    if segmentation_matches:
        match = segmentation_matches[-1]
        return int(match.group("static")), int(match.group("moving"))

    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    matches = list(GAUSSIAN_PATTERN.finditer(text))
    if not matches:
        raise RuntimeError(f"Could not parse Gaussian counts from {run_dir / 'train.log'}")
    match = matches[-1]
    return int(match.group("static")), int(match.group("moving"))


def _initial_dynamic_support(
    run_dir: Path,
    reference_ply: Path,
    *,
    base_gt_part_id: int,
    moving_gt_part_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_xyz, reference_part = _read_ply_part_ids(reference_ply)
    gaussian_xyz, _ = _read_ply_xyz_rgb(run_dir / "seg_init_dual" / "segmented_point.ply")
    static_count, moving_count = _parse_gaussian_counts(run_dir)
    if static_count + moving_count != len(gaussian_xyz):
        raise RuntimeError(
            "Parsed Gaussian counts do not match seg_init_dual point count: "
            f"{static_count}+{moving_count}!={len(gaussian_xyz)}"
        )

    bbox_diagonal = float(np.linalg.norm(np.ptp(reference_xyz, axis=0)))
    strict_distance = max(0.02 * bbox_diagonal, 1e-9)
    distance, nearest = cKDTree(reference_xyz).query(gaussian_xyz, k=1)
    nearest_part = reference_part[nearest]
    dynamic = np.arange(len(gaussian_xyz)) >= static_count
    strict = distance <= strict_distance

    rows = []
    for part_id in sorted(int(value) for value in np.unique(reference_part)):
        nearest_dynamic = dynamic & (nearest_part == part_id)
        strict_dynamic = nearest_dynamic & strict
        role = (
            "base"
            if part_id == base_gt_part_id
            else "moving"
            if part_id == moving_gt_part_id
            else "other"
        )
        rows.append(
            {
                "gt_part_id": part_id,
                "role": role,
                "nearest_assigned_dynamic_gaussian_count": int(np.sum(nearest_dynamic)),
                "strictly_mapped_dynamic_gaussian_count": int(np.sum(strict_dynamic)),
                "fraction_of_initial_dynamic_set": float(
                    np.sum(nearest_dynamic) / max(moving_count, 1)
                ),
                "strict_fraction_of_initial_dynamic_set": float(
                    np.sum(strict_dynamic) / max(moving_count, 1)
                ),
                "below_official_global_10pct_threshold": bool(
                    np.sum(nearest_dynamic) < 0.1 * moving_count
                ),
                "nearest_distance_median_m": (
                    float(np.median(distance[nearest_dynamic]))
                    if np.any(nearest_dynamic)
                    else None
                ),
            }
        )
    return rows, {
        "static_gaussian_count": static_count,
        "initial_dynamic_gaussian_count": moving_count,
        "official_min_inliers_count": int(math.ceil(0.1 * moving_count)),
        "strict_mapping_distance_m": strict_distance,
        "strict_dynamic_mapping_ratio": float(np.mean(strict[dynamic])),
    }


def _component_sizes(status: dict[str, Any], *, merged: bool) -> list[int]:
    prefix = "After Merging:" if merged else "[Seq] Motion #"
    sizes = []
    for line in status.get("nn_ransac_statistics", []):
        if merged != line.startswith("After Merging:"):
            continue
        if prefix not in line:
            continue
        match = re.search(r"\bsize=(\d+)", line)
        if match:
            sizes.append(int(match.group(1)))
    return sizes


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "failure"
    return f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    observability_by_episode = {
        Path(row["episode"]).parent.name: row for row in _read_csv(root / "observability.csv")
    }
    definitions = {
        "paper_like_current": {
            "episode_id": "partnet_47024",
            "run": args.control_run.resolve(),
            "metrics": args.control_metrics.resolve(),
        },
        "close_monocular": {
            "episode_id": "partnet_47024_close_monocular_v3",
            "run": root / "runs" / "partnet_47024_close_monocular_v3",
            "metrics": root / "raw" / "close_monocular.json",
        },
        "four_view_upper_bound": {
            "episode_id": "partnet_47024_four_view_upper_bound",
            "run": root / "runs" / "partnet_47024_four_view_upper_bound",
            "metrics": None,
        },
    }

    protocol_rows: list[dict[str, Any]] = []
    ransac_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    support_by_protocol: dict[str, dict[str, Any]] = {}
    for protocol in PROTOCOLS:
        definition = definitions[protocol]
        status = _load_json(definition["run"] / "run_status.json")
        observation = observability_by_episode[definition["episode_id"]]
        metrics = (
            _metric_summary(_load_json(definition["metrics"]))
            if definition["metrics"] is not None
            else {}
        )
        support, support_meta = _initial_dynamic_support(
            definition["run"],
            args.reference_ply,
            base_gt_part_id=args.base_gt_part_id,
            moving_gt_part_id=args.moving_gt_part_id,
        )
        support_by_protocol[protocol] = {
            row["role"]: row for row in support if row["role"] in {"base", "moving"}
        }
        mapped_base = support_by_protocol[protocol].get("base", {}).get(
            "strictly_mapped_dynamic_gaussian_count", 0
        )
        mapped_moving = support_by_protocol[protocol].get("moving", {}).get(
            "strictly_mapped_dynamic_gaussian_count", 0
        )
        mapped_total = mapped_base + mapped_moving
        for row in support:
            support_rows.append({"protocol": protocol, **row, **support_meta})

        pre_sizes = _component_sizes(status, merged=False)
        post_sizes = _component_sizes(status, merged=True)
        moving = support_meta["initial_dynamic_gaussian_count"]
        protocol_rows.append(
            {
                "protocol": protocol,
                "status": status.get("status"),
                "predicted_part_count": metrics.get("predicted_part_count"),
                "gt_part_count": metrics.get("gt_part_count", 2),
                "point_iou": metrics.get("point_iou"),
                "ari": metrics.get("ari"),
                "rand_index": metrics.get("rand_index"),
                "geometry_coverage_at_2pct": metrics.get("geometry_coverage_at_2pct"),
                "object_bbox_occupancy_mean": observation["object_bbox_occupancy_mean"],
                "object_bbox_occupancy_min": observation["object_bbox_occupancy_min"],
                "object_bbox_occupancy_max": observation["object_bbox_occupancy_max"],
                "moving_bbox_occupancy_mean": observation["moving_bbox_occupancy_mean"],
                "moving_visible_ratio": observation["moving_visible_ratio"],
                "moving_visible_pixels_mean": observation["moving_visible_pixels_mean"],
                "view_count_per_frame": observation["view_count_per_frame"],
                "initial_dynamic_gaussian_count": moving,
                "moving_part_dynamic_support": support_by_protocol[protocol]
                .get("moving", {})
                .get("fraction_of_initial_dynamic_set"),
                "base_dynamic_support": support_by_protocol[protocol]
                .get("base", {})
                .get("fraction_of_initial_dynamic_set"),
                "strict_gt_mapped_dynamic_count": mapped_total,
                "moving_share_among_strict_gt_mapped_dynamic": (
                    mapped_moving / mapped_total if mapped_total else None
                ),
                "runtime_s": status.get("runtime_s"),
                "failure_stage": status.get("failure_stage"),
                "exception": status.get("exception"),
            }
        )
        ransac_rows.append(
            {
                "protocol": protocol,
                "status": status.get("status"),
                "initial_dynamic_gaussian_count": moving,
                "official_min_inliers_count": support_meta["official_min_inliers_count"],
                "accepted_component_count_pre_merge": len(pre_sizes),
                "accepted_component_sizes_pre_merge": json.dumps(pre_sizes),
                "component_count_post_merge": len(post_sizes),
                "component_sizes_post_merge": json.dumps(post_sizes),
                "failure_stage": status.get("failure_stage"),
                "exception": status.get("exception"),
            }
        )

    four_upstream = _load_json(root / "raw" / "four_view_upstream_dynamic_split.json")
    four_upstream_summary = _metric_summary(four_upstream)
    _write_csv(root / "protocol_comparison.csv", protocol_rows)
    _write_csv(root / "ransac_diagnostics.csv", ransac_rows)
    _write_csv(root / "dynamic_gaussian_support.csv", support_rows)

    trajectory_rows = _read_csv(root / "trajectory_validation.csv")
    max_trajectory_difference = max(
        float(row["max_abs_joint_trajectory_difference"]) for row in trajectory_rows
    )
    control, close, four = protocol_rows
    markdown = f"""# AiM Simple-Object Favorable-Observation Sanity Check

This diagnostic evaluates Storage-47024 (`GT parts = 2`) without changing AiM
reconstruction, dynamic/static segmentation, sequential RANSAC, merge thresholds, or
the joint trajectory. The maximum absolute joint-trajectory difference across
protocols is `{max_trajectory_difference:.3g}`.

## Main Result

| Protocol | Status | Pred./GT | Point IoU | ARI | RI | Coverage @2% bbox | Mean object bbox occupancy | Moving share among GT-mapped dynamic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Paper-like current | {control['status']} | {control['predicted_part_count']}/{control['gt_part_count']} | {_fmt(control['point_iou'])} | {_fmt(control['ari'])} | {_fmt(control['rand_index'])} | {_fmt(control['geometry_coverage_at_2pct'])} | {_fmt(control['object_bbox_occupancy_mean'])} | {_fmt(control['moving_share_among_strict_gt_mapped_dynamic'])} |
| Close monocular | {close['status']} | {close['predicted_part_count']}/{close['gt_part_count']} | {_fmt(close['point_iou'])} | {_fmt(close['ari'])} | {_fmt(close['rand_index'])} | {_fmt(close['geometry_coverage_at_2pct'])} | {_fmt(close['object_bbox_occupancy_mean'])} | {_fmt(close['moving_share_among_strict_gt_mapped_dynamic'])} |
| Four-view upper bound | {four['status']} | failure/2 | failure | failure | failure | failure | {_fmt(four['object_bbox_occupancy_mean'])} | {_fmt(four['moving_share_among_strict_gt_mapped_dynamic'])} |

Point IoU, ARI, and RI use the full common reference domain, matching the existing
AiM baseline reports. Geometry coverage is reported separately at 2% of the reference
bbox diagonal.

The dynamic-support column only uses Gaussians within 2% bbox distance of the sparse
observed GT reference. It is a label-composition diagnostic, not an estimate of the
fraction of all AiM Gaussians supported by a complete GT surface. The mapped counts are
`{control['strict_gt_mapped_dynamic_count']}`,
`{close['strict_gt_mapped_dynamic_count']}`, and
`{four['strict_gt_mapped_dynamic_count']}`, respectively.

## Controlled Observability

- Close monocular increased mean object linear bbox occupancy from
  `{float(control['object_bbox_occupancy_mean']):.3f}` to
  `{float(close['object_bbox_occupancy_mean']):.3f}` and mean moving visible pixels
  from `{float(control['moving_visible_pixels_mean']):.0f}` to
  `{float(close['moving_visible_pixels_mean']):.0f}` without clipping.
- Four-view used four fixed, synchronized cameras at every simulation timestep.
  Mean object occupancy was `{float(four['object_bbox_occupancy_mean']):.3f}`, mean
  moving visible pixels was `{float(four['moving_visible_pixels_mean']):.0f}`, and
  moving visibility ratio was `{float(four['moving_visible_ratio']):.3f}`.
- All views sharing a source timestep were exported with the same normalized
  deformation time; they were not encoded as different articulation states.

## Sequential-RANSAC Diagnostic

The control and close-monocular runs both accepted three moving components and retained
all three after official merging. Including the static component, both therefore
over-segmented the two-part object as `4/2`.

The four-view reconstruction completed, but official `seg_main.py` failed before
producing any RANSAC proposal:

```text
{four['exception']}
```

The exception was triggered by `torch.isfinite(errors)` receiving `errors=None`.
The official implementation was not patched. Its initial dynamic/static split has
`predicted_part_count={four_upstream_summary['predicted_part_count']}` but is not a
final part segmentation. On the common domain it has
`IoU={four_upstream_summary['point_iou']:.3f}`,
`ARI={four_upstream_summary['ari']:.3f}`, and
`RI={four_upstream_summary['rand_index']:.3f}`, indicating a mixed upstream motion
representation despite the nominal two labels.

## Hypothesis Assessment

**H1, camera distance/resolution bottleneck: not supported.** Close monocular produced
larger image occupancy, more moving pixels, more dynamic Gaussians, and higher geometry
coverage, but remained `4/2`; its main-domain IoU and RI decreased rather than
recovering the correct segmentation.

**H2, single-view temporal visibility bottleneck: inconclusive in the four-view
ablation.** The released implementation learned exactly zero deformation for the
grouped same-time multi-camera input. This exposed a protocol/implementation
incompatibility rather than a valid favorable-input result, so the four-view failure
must not be interpreted as direct evidence against the single-view method. The
close-monocular result still shows that image occupancy alone is not the main issue.

**H3, upstream AiM optimization/motion-representation bottleneck: supported by this
sanity check.** Even on a simple two-part object, favorable close-range monocular input
did not recover `2/2`, while synchronized four-view input caused an invalid official
motion-fit result. The next diagnosis should focus on deformation optimization,
renderer/domain mismatch, motion-error construction, and sequential model selection,
not additional camera-distance tuning.

## Scope

This is a diagnosis/upper-bound experiment. It does not replace the paper-like AiM
baseline, use GT-informed camera tracking, alter the motion schedule, tune official
RANSAC thresholds, or select a favorable seed.
"""
    (root / "summary.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(root / "summary.md"),
                "protocols": protocol_rows,
                "four_view_upstream_only": four_upstream_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
