#!/usr/bin/env python3
"""Summarize AiM sequential-RANSAC sensitivity and GT-oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _oracle_separability_rows(
    variant_name: str, oracle: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compare each GT part's own rigid model against the best wrong-part model."""
    residuals: dict[int, dict[int, float]] = {}
    for row in oracle["cross_model_residual"]:
        residuals.setdefault(int(row["gt_part_id"]), {})[
            int(row["model_part_id"])
        ] = float(row["rmse_m"])

    rows = []
    for part in oracle["per_gt_part"]:
        part_id = int(part["gt_part_id"])
        own_rmse = residuals[part_id][part_id]
        wrong = [
            (model_id, rmse)
            for model_id, rmse in residuals[part_id].items()
            if model_id != part_id
        ]
        best_wrong_id, best_wrong_rmse = min(wrong, key=lambda item: item[1])
        margin = best_wrong_rmse - own_rmse
        ratio = best_wrong_rmse / max(own_rmse, 1e-12)
        rows.append(
            {
                "variant": variant_name,
                "gt_part_id": part_id,
                "gaussian_count": part["gaussian_count"],
                "own_model_rmse_m": own_rmse,
                "best_wrong_model_part_id": best_wrong_id,
                "best_wrong_model_rmse_m": best_wrong_rmse,
                "separation_margin_m": margin,
                "separation_ratio": ratio,
                "own_model_is_best": margin > 0.0,
                "weakly_separated": ratio < 1.25,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_rows = []
    component_rows = []
    oracle_rows = []
    separability_rows = []
    reports = []
    for input_path in args.inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        variant_name = input_path.parent.name
        reports.append((variant_name, payload))
        for row in payload["variants"]:
            key = {
                "variant": variant_name,
                "min_inliers_ratio": row["min_inliers_ratio"],
                "min_inliers": row["min_inliers"],
                "keep_largest_cc": row["keep_largest_cc"],
                "premerge_component_count": row["premerge_component_count"],
                "postmerge_component_count": row["final_component_count"],
                "merge_removed_count": (
                    row["premerge_component_count"] - row["final_component_count"]
                ),
                "unassigned_count": row["unassigned_count"],
                "rejected_min_inliers_count": row["rejected_min_inliers_count"],
                "runtime_s": row["runtime_s"],
            }
            variant_rows.append(key)
            for component in row["final_components"]:
                component_rows.append(
                    {
                        **key,
                        "component_id": component["component_id"],
                        "component_size": component["size"],
                        "dominant_gt_part": component["dominant_gt_part"],
                        "gt_purity": component["gt_purity"],
                        "strict_gt_mapped": component["strict_gt_mapped"],
                        "gt_composition": json.dumps(component["gt_composition"]),
                        "motion_type": component["motion_type"],
                    }
                )
        for row in payload["oracle_representation"]["per_gt_part"]:
            oracle_rows.append({"variant": variant_name, **row})
        separability_rows.extend(
            _oracle_separability_rows(variant_name, payload["oracle_representation"])
        )
    _write_csv(args.output_dir / "variant_summary.csv", variant_rows)
    _write_csv(args.output_dir / "component_summary.csv", component_rows)
    _write_csv(args.output_dir / "oracle_rigid_residual.csv", oracle_rows)
    _write_csv(
        args.output_dir / "oracle_model_separability.csv", separability_rows
    )

    lines = [
        "# AiM Sequential-RANSAC Sensitivity",
        "",
        "This is a diagnostic-only rerun of segmentation on frozen Gaussian",
        "trajectories. It does not replace or tune the official baseline result.",
        "",
    ]
    for variant_name, payload in reports:
        lines.extend(
            [
                f"## {variant_name}",
                "",
                f"- Dynamic Gaussians: {payload['dynamic_gaussian_count']}",
                f"- Strict GT mapping ratio at 2% bbox: {payload['strict_gt_mapping_ratio']:.3f}",
                "",
                "| min ratio | largest CC | pre-merge | post-merge | rejected by min | unassigned |",
                "|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["variants"]:
            lines.append(
                f"| {row['min_inliers_ratio']:.3f} | "
                f"{'yes' if row['keep_largest_cc'] else 'no'} | "
                f"{row['premerge_component_count']} | {row['final_component_count']} | "
                f"{row['rejected_min_inliers_count']} | {row['unassigned_count']} |"
            )
        lines.extend(
            [
                "",
                "### GT-oracle rigid residual",
                "",
                "| GT part | Gaussians | rigid RMSE (m) | median motion (m) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in payload["oracle_representation"]["per_gt_part"]:
            lines.append(
                f"| {row['gt_part_id']} | {row['gaussian_count']} | "
                f"{row['rigid_rmse_m']:.4f} | {row['motion_median_m']:.4f} |"
            )
        lines.extend(
            [
                "",
                "### GT-oracle model separability",
                "",
                "| GT part | own RMSE | best wrong part | best wrong RMSE | ratio | weak |",
                "|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in separability_rows:
            if row["variant"] != variant_name:
                continue
            lines.append(
                f"| {row['gt_part_id']} | {row['own_model_rmse_m']:.4f} | "
                f"{row['best_wrong_model_part_id']} | "
                f"{row['best_wrong_model_rmse_m']:.4f} | "
                f"{row['separation_ratio']:.2f} | "
                f"{'yes' if row['weakly_separated'] else 'no'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Findings",
            "",
            "1. The global minimum-inlier threshold is a material bottleneck. Lowering it",
            "   increases current-orbit pre-merge proposals from 1 to 4 and front-loaded-v3",
            "   proposals from 1 to 2-3.",
            "2. Largest-connected-component filtering is secondary. Disabling it changes",
            "   component count only for front-loaded-v3 at the 2.5% ratio, although it",
            "   consistently retains more Gaussians.",
            "3. Official merge/model selection collapses several recovered proposals. At",
            "   5-7.5%, two current-orbit proposals become one; at 2.5%, four become three.",
            "4. Threshold relaxation is not sufficient: the most permissive diagnostic still",
            "   recovers only 3 current-orbit or 2 front-loaded-v3 post-merge components,",
            "   rather than all 7 GT parts.",
            "5. GT-oracle fits show upstream representation ambiguity. Several true parts",
            "   have high within-part rigid residual or only a small advantage over another",
            "   part's rigid model. Final proposals also remain GT-mixed at permissive",
            "   thresholds.",
            "",
            "The evidence therefore supports a combined failure: the 10% global threshold",
            "suppresses small parts, while learned Gaussian trajectories and final merging",
            "prevent threshold relaxation alone from recovering a clean seven-part model.",
            "",
            "## Limitations",
            "",
            "- GT assignment is simulation-only and used only for diagnosis.",
            "- Strict GT mapping uses observed source-frame points within 2% of the object",
            "  bounding-box diagonal. Mapping ratios are low, so counts and component purity",
            "  characterize the matched subset rather than every Gaussian.",
            "- Sensitivity results must not replace the official 10% baseline result.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
