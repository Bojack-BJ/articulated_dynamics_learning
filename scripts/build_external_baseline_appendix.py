#!/usr/bin/env python3
"""Build a result, failure, and visualization audit for external baselines."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
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
SUCCESS = {"success", "success_native_metrics", "completed_with_incorrect_prediction"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root", type=Path, default=Path("outputs/external_baseline_suite_v1")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.suite_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "appendix_visualization_v1"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = _collect(root)
    _write_csv(output / "visualization_manifest.csv", rows)
    (output / "raw_artifact_retention.json").write_text(
        json.dumps(_retention_manifest(rows), indent=2) + "\n", encoding="utf-8"
    )
    (output / "failure_analysis.json").write_text(
        json.dumps(_failure_analysis(rows), indent=2) + "\n", encoding="utf-8"
    )
    _write_object_pages(output, rows)
    (output / "index.html").write_text(_index_html(rows), encoding="utf-8")
    (output / "summary.md").write_text(_summary_md(rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "rows": len(rows)}, indent=2))
    return 0


def _collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_dir in sorted((root / "per_object").glob("partnet_*")):
        for method in METHODS:
            metrics_path = object_dir / method / "metrics.json"
            if not metrics_path.exists():
                rows.append(_missing_row(object_dir.name, method, metrics_path))
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(_row_from_metrics(metrics_path, data))
        patched = object_dir / "paris_patched" / "metrics.json"
        if patched.exists():
            rows.append(
                _row_from_metrics(
                    patched, json.loads(patched.read_text(encoding="utf-8"))
                )
            )
    return rows


def _missing_row(object_id: str, method: str, path: Path) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "category": "",
        "method": method,
        "status": "missing",
        "gt_part_count": "",
        "predicted_part_count": "",
        "point_iou": "",
        "ari": "",
        "ri": "",
        "gt_domain_valid": "",
        "joint_type_accuracy": "",
        "axis_angle_deg": "",
        "axis_line_bbox": "",
        "part_seg_visualization": "missing_metrics",
        "axis_visualization": "missing_metrics",
        "viewer_status": "unavailable",
        "viewer_path": "",
        "failure_class": "missing_result",
        "failure_reason": f"Missing {path}",
        "metrics_path": str(path),
        "raw_artifacts": "[]",
    }


def _row_from_metrics(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    segmentation = data.get("segmentation") or {}
    kinematics = data.get("kinematics") or {}
    artifacts = data.get("artifacts") or {}
    provenance = data.get("provenance") or {}
    status = str(data.get("status", "unknown"))
    method = str(data.get("method", path.parent.name))
    artifact_paths = sorted(set(_extract_paths(artifacts)))
    provenance_paths = sorted(set(_extract_paths(provenance)))
    raw_paths = sorted(set(artifact_paths + provenance_paths))
    seg_state = _segmentation_state(status, segmentation, artifact_paths)
    axis_state = _axis_state(status, kinematics)
    viewer_path = str(artifacts.get("viewer_html", ""))
    failure_class, failure_reason = _classify_failure(data, status)
    return {
        "object_id": str(data.get("object_id", path.parents[1].name)),
        "category": str(data.get("category", "")),
        "method": method,
        "status": status,
        "gt_part_count": _first(segmentation, "gt_part_count", "observed_gt_part_count"),
        "predicted_part_count": segmentation.get("predicted_part_count", ""),
        "point_iou": segmentation.get("point_iou", ""),
        "ari": segmentation.get("ari", ""),
        "ri": _first(segmentation, "rand_index", "ri"),
        "gt_domain_valid": segmentation.get("gt_domain_valid", ""),
        "joint_type_accuracy": kinematics.get("joint_type_accuracy", ""),
        "axis_angle_deg": _first(
            kinematics,
            "axis_angle_deg_type_correct",
            "axis_angle_error_deg",
            "axis_angle_error_deg_native_all",
        ),
        "axis_line_bbox": _first(
            kinematics,
            "revolute_axis_line_bbox",
            "axis_position_error",
            "axis_position_error_native_x10",
        ),
        "part_seg_visualization": seg_state,
        "axis_visualization": axis_state,
        "viewer_status": _viewer_status(seg_state, axis_state),
        "viewer_path": viewer_path,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "metrics_path": str(path),
        "raw_artifacts": json.dumps(raw_paths),
    }


def _segmentation_state(
    status: str, segmentation: dict[str, Any], raw_paths: list[str]
) -> str:
    if status not in SUCCESS:
        return "method_failure" if status in {"failed", "failure"} else "not_available"
    if not segmentation:
        return "unsupported_or_not_exported"
    if any(path.endswith(".html") for path in raw_paths):
        return "viewer_available"
    if any(path.endswith((".ply", ".obj", ".json", ".pkl")) for path in raw_paths):
        return "raw_prediction_available"
    return "metrics_only_raw_sync_required"


def _axis_state(status: str, kinematics: dict[str, Any]) -> str:
    if status not in SUCCESS:
        return "method_failure" if status in {"failed", "failure"} else "not_available"
    support = str(kinematics.get("support", "")).lower()
    if "unsupported" in support:
        return "unsupported_by_method"
    if "joint_hypotheses" in kinematics:
        return "dual_hypothesis_available"
    visualization = str(kinematics.get("axis_visualization", ""))
    if visualization == "native_motion_json":
        return "native_axis_viewer_available"
    if visualization.startswith("converted_from_native_part_poses"):
        return "converted_axis_viewer_available"
    if visualization in {
        "native_motion_json_no_dynamic_axis",
        "conversion_unavailable_no_observable_relative_motion",
    }:
        return "axis_unavailable_for_prediction"
    if "joints" in kinematics or "native_joints" in kinematics:
        return "axis_values_available"
    if any(
        key in kinematics
        for key in (
            "matches",
            "axis_angle_deg_type_correct",
            "axis_angle_error_deg",
            "axis_angle_error_deg_native_all",
            "axis_position_error",
        )
    ):
        return "axis_values_available"
    return "axis_raw_adapter_required"


def _classify_failure(data: dict[str, Any], status: str) -> tuple[str, str]:
    if status in SUCCESS:
        if status == "completed_with_incorrect_prediction":
            return "valid_method_failure", "Run completed but the predicted model is wrong."
        return "", ""
    if status == "not_applicable":
        return "protocol_not_applicable", str(data.get("failure") or "")
    failure = data.get("failure") or {}
    if isinstance(failure, dict):
        reason = str(failure.get("exception") or failure.get("reason") or failure)
        stage = str(failure.get("stage") or failure.get("failure_stage") or "")
    else:
        reason, stage = str(failure), str(data.get("failure_stage", ""))
    text = f"{stage} {reason}".lower()
    if "space left" in text:
        kind = "infrastructure_storage"
    elif "candidate" in text or "cluster" in text:
        kind = "method_no_valid_motion_candidate"
    elif "export" in text or "palette" in text:
        kind = "official_output_or_export_failure"
    elif status in {"pending", "missing"}:
        kind = "missing_or_pending_artifact"
    else:
        kind = "official_inference_failure"
    return kind, reason


def _extract_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if "/" in value else []
    if isinstance(value, list):
        return [path for item in value for path in _extract_paths(item)]
    if isinstance(value, dict):
        return [path for item in value.values() for path in _extract_paths(item)]
    return []


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return ""


def _viewer_status(segmentation: str, axis: str) -> str:
    if segmentation == "viewer_available":
        return "viewer_available"
    if segmentation == "raw_prediction_available":
        return "ready_for_generic_adapter"
    if axis == "axis_values_available":
        return "axis_only_until_geometry_sync"
    return "unavailable"


def _retention_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep = []
    for row in rows:
        paths = json.loads(row["raw_artifacts"])
        keep.append(
            {
                "object_id": row["object_id"],
                "method": row["method"],
                "status": row["status"],
                "keep": [
                    {
                        "path": path,
                        "reason": _retention_reason(path),
                    }
                    for path in paths
                ],
                "policy": (
                    "Keep minimal labeled geometry, joint parameters, and metrics. "
                    "Full checkpoints may be archived after the viewer bundle is verified."
                ),
            }
        )
    return {"schema": "external-baseline-raw-retention-v1", "entries": keep}


def _retention_reason(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".csv"}:
        return "Parameters, labels, metrics, or provenance required for evaluation."
    if suffix in {".ply", ".obj", ".glb"}:
        return "Prediction geometry required for appendix visualization."
    if suffix in {".pkl", ".npz", ".npy"}:
        return "Native prediction required to regenerate labels or motion."
    return "Referenced native artifact; retain until the appendix bundle is verified."


def _failure_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row["failure_class"]]
    return {
        "schema": "external-baseline-failure-analysis-v1",
        "counts": dict(Counter(row["failure_class"] for row in failures)),
        "failures": [
            {
                key: row[key]
                for key in (
                    "object_id",
                    "method",
                    "status",
                    "failure_class",
                    "failure_reason",
                )
            }
            for row in failures
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_object_pages(output: Path, rows: list[dict[str, Any]]) -> None:
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(row["object_id"], []).append(row)
    objects = output / "objects"
    objects.mkdir(exist_ok=True)
    for object_id, object_rows in by_object.items():
        (objects / f"{object_id}.html").write_text(
            _object_html(object_id, object_rows), encoding="utf-8"
        )


def _index_html(rows: list[dict[str, Any]]) -> str:
    objects = sorted(set(row["object_id"] for row in rows))
    items = "\n".join(
        f'<a class="card" href="objects/{html.escape(object_id)}.html">'
        f"<strong>{html.escape(object_id)}</strong><span>Open comparison</span></a>"
        for object_id in objects
    )
    return _page(
        "External Baseline Appendix",
        "<h1>External Baseline Appendix</h1>"
        "<p>Part segmentation, joint-axis availability, failures, and raw-artifact "
        "retention status for every aligned object.</p>"
        f'<div class="grid">{items}</div>',
    )


def _object_html(object_id: str, rows: list[dict[str, Any]]) -> str:
    body_rows = []
    for row in sorted(rows, key=lambda item: item["method"]):
        viewer = _viewer_link(row)
        body_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_display(row[key]))}</td>"
                for key in (
                    "method",
                    "status",
                    "predicted_part_count",
                    "gt_part_count",
                    "point_iou",
                    "ari",
                    "ri",
                    "joint_type_accuracy",
                    "axis_angle_deg",
                    "axis_line_bbox",
                    "part_seg_visualization",
                    "axis_visualization",
                    "failure_class",
                )
            )
            + f"<td>{viewer}</td>"
            + "</tr>"
        )
    headers = (
        "Method",
        "Status",
        "Pred.",
        "GT",
        "IoU",
        "ARI",
        "RI",
        "Type Acc.",
        "Axis deg",
        "Axis-line",
        "Part viewer",
        "Axis viewer",
        "Failure class",
        "Viewer",
    )
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{value}</th>" for value in headers)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    return _page(
        object_id,
        f'<a href="../index.html">Back</a><h1>{html.escape(object_id)}</h1>{table}'
        "<p class='note'>A method is visualized only from its own native prediction. "
        "Unsupported outputs are not inferred from another method.</p>",
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1014; --panel:#142028; --line:#2c3b45; --text:#eaf1f4; --muted:#93a6b0; --accent:#79d7b4; }}
body {{ margin:0; padding:36px; background:radial-gradient(circle at top right,#17313a,var(--bg) 45%); color:var(--text); font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
a {{ color:var(--accent); }} h1 {{ font-size:28px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.card {{ display:flex; flex-direction:column; gap:8px; padding:18px; background:var(--panel); border:1px solid var(--line); border-radius:12px; text-decoration:none; }}
.card span,.note {{ color:var(--muted); }} table {{ width:100%; border-collapse:collapse; background:rgba(20,32,40,.9); }}
th,td {{ padding:10px; border:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ color:var(--accent); position:sticky; top:0; background:#122029; }}
</style></head><body>{body}</body></html>"""


def _summary_md(rows: list[dict[str, Any]]) -> str:
    status = Counter(row["status"] for row in rows)
    seg = Counter(row["part_seg_visualization"] for row in rows)
    axis = Counter(row["axis_visualization"] for row in rows)
    failures = Counter(row["failure_class"] for row in rows if row["failure_class"])
    aggregates = _aggregate(rows)
    aggregate_table = [
        "| Method | Requested | Successful | IoU N | Point IoU | ARI | RI | "
        "Type Acc. | Axis Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        aggregate_table.append(
            f"| {row['method']} | {row['requested']} | {row['successful']} | "
            f"{row['iou_n']} | {_display(row['point_iou'])} | "
            f"{_display(row['ari'])} | {_display(row['ri'])} | "
            f"{_display(row['joint_type_accuracy'])} | "
            f"{_display(row['axis_angle_deg'])} |"
        )
    lines = [
        "# External Baseline Appendix Audit",
        "",
        "This appendix inventory covers every aligned object and does not fabricate",
        "unsupported outputs. A 3D viewer requires the method's native labeled",
        "geometry; an axis viewer additionally requires native joint parameters.",
        "",
        "## Status",
        "",
        f"- Objects: `{len(set(row['object_id'] for row in rows))}`",
        f"- Method-object rows: `{len(rows)}`",
        f"- Run status: `{dict(status)}`",
        f"- Part visualization: `{dict(seg)}`",
        f"- Axis visualization: `{dict(axis)}`",
        f"- Failure classes: `{dict(failures)}`",
        "",
        "## Aggregate Results",
        "",
        *aggregate_table,
        "",
        "Each mean uses only rows where that method natively or validly converted",
        "the corresponding metric. Unsupported metrics remain `n/a`.",
        "",
        "## Retention policy",
        "",
        "Retain metrics/provenance, native labels or trajectories, joint parameters,",
        "and downsampled prediction geometry. Full checkpoints and dense meshes may",
        "be archived only after the corresponding viewer bundle has been generated",
        "and validated.",
        "",
        "Open `index.html` for the per-object appendix table.",
    ]
    return "\n".join(lines) + "\n"


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted(set(row["method"] for row in rows)):
        selected = [row for row in rows if row["method"] == method]
        successful = [row for row in selected if row["status"] in SUCCESS]
        iou_rows = [
            row
            for row in successful
            if _number(row["point_iou"]) is not None
            and row["gt_domain_valid"] is not False
        ]
        output.append(
            {
                "method": method,
                "requested": len(selected),
                "successful": len(successful),
                "iou_n": len(iou_rows),
                "point_iou": _mean(iou_rows, "point_iou"),
                "ari": _mean(iou_rows, "ari"),
                "ri": _mean(iou_rows, "ri"),
                "joint_type_accuracy": _mean(successful, "joint_type_accuracy"),
                "axis_angle_deg": _mean(successful, "axis_angle_deg"),
            }
        )
    return output


def _mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [_number(row[key]) for row in rows]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else ""


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _viewer_link(row: dict[str, Any]) -> str:
    value = str(row.get("viewer_path", ""))
    if not value:
        return "n/a"
    path = Path(value)
    metrics_path = Path(str(row["metrics_path"]))
    page_root = metrics_path.parents[3] / "appendix_visualization_v1" / "objects"
    if path.exists():
        try:
            target = path.resolve().relative_to(page_root.resolve())
            href = target.as_posix()
        except ValueError:
            href = path.resolve().as_uri()
    else:
        href = value
    return f'<a href="{html.escape(href)}">Open 3D</a>'


if __name__ == "__main__":
    raise SystemExit(main())
