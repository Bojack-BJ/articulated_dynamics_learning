#!/usr/bin/env python3
"""Create the frozen paper-facing canonical baseline tables with provenance."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "outputs" / "external_baseline_suite_v1"
NATIVE = ROOT / "outputs" / "external_baselines_v1" / "native_protocol_comparison_v1" / "final"
OUT = ROOT / "outputs" / "paper_experiments" / "external_canonical_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    segmentation = read_csv(SUITE / "segmentation_summary.csv")
    suite_axis = {row["method"]: row for row in read_csv(SUITE / "kinematics_summary.csv")}
    native_axis = read_csv(NATIVE / "kinematics_axis_comparison.csv")

    seg_rows = []
    for row in segmentation:
        seg_rows.append({
            "method": row["method"],
            "protocol": row["protocol"],
            "success/requested": f'{row["success_n"]}/{row["requested_n"]}',
            "metric_n": row["metric_n"],
            "point_iou": row["point_iou"],
            "ari": row["ari"],
            "ri": row["ri"],
            "status": "frozen canonical prediction",
        })
    write_csv(OUT / "canonical_segmentation.csv", seg_rows)

    support = []
    native_by_name = {row["method"]: row for row in native_axis}
    mappings = [
        ("ours_hybrid", "Ours/hybrid/relation_head", "native PartNet held-out"),
        ("aim_aligned", "AiM/native_screw", "three-object AiM-style pilot only"),
        ("reart", "ReArt/converted_SE3", "official Sapiens; converted diagnostic"),
        ("dta", None, "adapter does not expose comparable joint axis"),
        ("artgs", None, "released-path adapter does not expose comparable joint axis"),
        ("videoartgs", None, "aligned suite native output"),
        ("gaussianart", None, "aligned suite native output; oracle parts"),
        ("paris", None, "aligned suite native output; two-part subset"),
        ("ditto", None, "run failed: no usable checkpoint/output"),
    ]
    for method, native_name, note in mappings:
        suite = suite_axis.get(method, {})
        native = native_by_name.get(native_name, {}) if native_name else {}
        angle = native.get("axis_error_mean_deg") or suite.get("axis_angle_deg_type_correct") or suite.get("axis_angle_error_deg")
        line = native.get("axis_line_error") or suite.get("revolute_axis_line_bbox") or suite.get("axis_position_error")
        type_acc = native.get("joint_type_accuracy") or suite.get("joint_type_accuracy")
        supported = bool(angle or line or type_acc)
        support.append({
            "method": method,
            "axis_supported": "yes" if supported else "no",
            "joint_type_accuracy": type_acc or "",
            "axis_error_mean_deg": angle or "",
            "axis_line_error": line or "",
            "protocol_note": note,
            "rotation_audit_status": "requires full method rerun" if supported else "unsupported by current adapter/output",
        })
    write_csv(OUT / "canonical_axis_support.csv", support)

    lines = [
        "# Frozen external canonical results\n",
        "Generated from existing predictions; no method was silently assigned an unsupported metric.\n",
        "## Status\n",
        "- Canonical part segmentation is frozen for all successful adapters.\n",
        "- Comparable axis output is not available for every external method. Empty cells are unsupported, not unfinished jobs.\n",
        "- A rotation robustness number requires rotating each method's input and rerunning the full method. Rotating a saved axis is not a valid audit.\n",
        "- ReArt and AiM axis rows use different native/pilot protocols and belong in a qualified appendix table, not a paired main-table claim.\n",
        "## Paper recommendation\n",
        "Use `canonical_segmentation.csv` for the frozen external segmentation comparison. Use `canonical_axis_support.csv` with explicit protocol labels; restrict the main SO(3) ablation to old Track2Art head versus the proposed VN head on identical inputs and objects.\n",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
