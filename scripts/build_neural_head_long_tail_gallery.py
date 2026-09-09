#!/usr/bin/env python3
"""Build a representative per-joint gallery for neural-head tail analysis."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


ROOT = Path("outputs/neural_slot_so3_pilot_v1/neural_head_final_matrix_v1")
SOURCE = ROOT / "no_rotation_all/seed_20260831/rotation_audit/axis_direction_samples.csv"
OUTPUT = ROOT / "long_tail_gallery"

CASE_NOTES = {
    "partnet_47466": (
        "segmentation-induced",
        "Mixed/incorrect slots contaminate the parent-child relative motion; audit with oracle slots.",
    ),
    "partnet_41086": (
        "trajectory-noise",
        "Early noisy motion conflicts with the later recovered arc; robust full-path weighting is needed.",
    ),
    "partnet_45638": (
        "trajectory-noise",
        "A partial/noisy arc is present; current global fit has no timestep-quality weighting.",
    ),
    "partnet_32052": (
        "short-travel",
        "Very short prismatic evidence; exclude from the primary axis metric under a fixed eligibility rule.",
    ),
    "partnet_48497": (
        "short-travel",
        "Very short prismatic evidence; retain only as a low-observability diagnostic.",
    ),
    "partnet_103069": (
        "short-travel",
        "Tiny appliance subpart with weak evidence; do not use as a primary geometry case.",
    ),
    "partnet_102149": (
        "short-travel",
        "Tiny appliance subpart with weak evidence; do not use as a primary geometry case.",
    ),
}


def _read_rows() -> list[dict]:
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    gt = {
        (row["object_id"], row["joint_id"]): row
        for row in rows if row["source"] == "raw_gt"
    }
    result = []
    for row in rows:
        if row["source"] != "prediction":
            continue
        key = (row["object_id"], row["joint_id"])
        item = dict(row)
        item["gt_axis"] = [float(gt[key][f"axis_{axis}"]) for axis in "xyz"]
        item["pred_axis"] = [float(row[f"axis_{axis}"]) for axis in "xyz"]
        item["motion_magnitude"] = float(row["motion_magnitude"])
        item["observability"] = float(row["observability"])
        item["axis_error_deg"] = float(row["axis_error_deg"] or "nan")
        item["type_correct"] = row["type_correct"] == "True"
        item["edge_detected"] = row["edge_detected"] == "True"
        result.append(item)
    return result


def _take(rows: list[dict], predicate, count: int, score) -> list[dict]:
    candidates = sorted((row for row in rows if predicate(row)), key=score, reverse=True)
    selected = []
    used_objects = set()
    for row in candidates:
        if row["object_id"] in used_objects:
            continue
        selected.append(row)
        used_objects.add(row["object_id"])
        if len(selected) == count:
            break
    return selected


def _select(rows: list[dict]) -> list[dict]:
    groups = [
        (
            "high_motion_success",
            "Strong excitation, correct type and accurate axis",
            lambda r: r["motion_bin"] == "high" and r["type_correct"] and r["axis_error_deg"] < 15,
            lambda r: -r["axis_error_deg"],
        ),
        (
            "very_low_motion_axis_tail",
            "Very-low observed motion, correct type but axis failure",
            lambda r: r["motion_bin"] == "very_low" and r["type_correct"] and r["axis_error_deg"] > 60,
            lambda r: r["axis_error_deg"],
        ),
        (
            "low_observability_type_error",
            "Low-observability joint with incorrect predicted type",
            lambda r: r["observability_bin"] in {"low", "very_low"} and not r["type_correct"],
            lambda r: 1.0 - r["observability"],
        ),
        (
            "non_tail_regression",
            "Non-tail counterexample: usable evidence but large axis error",
            lambda r: r["motion_bin"] in {"medium", "high"} and r["observability_bin"] in {"medium", "high"}
            and r["type_correct"] and r["axis_error_deg"] > 30,
            lambda r: r["axis_error_deg"],
        ),
    ]
    selected = []
    for group, description, predicate, score in groups:
        for row in _take(rows, predicate, 3, score):
            row = dict(row)
            row["gallery_group"] = group
            row["group_description"] = description
            selected.append(row)
    return selected


def _projection_svg(gt: list[float], pred: list[float]) -> str:
    panels = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    chunks = ['<svg viewBox="0 0 420 142" role="img" aria-label="GT and predicted axis projections">']
    for panel, (a, b, label) in enumerate(panels):
        cx, cy, radius = 70 + panel * 140, 70, 49
        chunks.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#f8fafc" stroke="#cbd5e1"/>')
        chunks.append(f'<line x1="{cx-radius}" y1="{cy}" x2="{cx+radius}" y2="{cy}" stroke="#e2e8f0"/>')
        chunks.append(f'<line x1="{cx}" y1="{cy-radius}" x2="{cx}" y2="{cy+radius}" stroke="#e2e8f0"/>')
        for vector, color, width in ((gt, "#111827", 5), (pred, "#f97316", 4)):
            norm = max(math.hypot(vector[a], vector[b]), 1e-8)
            dx, dy = radius * vector[a] / norm, -radius * vector[b] / norm
            chunks.append(
                f'<line x1="{cx-dx}" y1="{cy-dy}" x2="{cx+dx}" y2="{cy+dy}" '
                f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
            )
        chunks.append(f'<text x="{cx}" y="137" text-anchor="middle" class="plane">{label}</text>')
    chunks.append('</svg>')
    return "".join(chunks)


def _viewer_for(object_id: str) -> str | None:
    candidates = [
        OUTPUT / "viewers" / object_id / "viewer_track2art_vn.html",
        Path("outputs/external_baseline_suite_v1/appendix_visualization_v1/objects") / f"{object_id}.html",
        Path("outputs/external_baseline_suite_v1/per_object") / object_id / "artgs/viewer.html",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def main() -> None:
    rows = _read_rows()
    selected = _select(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "gallery_group", "object_id", "category", "joint_id", "joint_type",
        "predicted_type", "type_correct", "edge_detected", "axis_error_deg",
        "motion_magnitude", "motion_bin", "observability", "observability_bin",
        "nearest_axis", "nearest_canonical_angle_deg", "failure_layer", "review_note",
        "viewer",
    ]
    manifest_rows = []
    for row in selected:
        failure_layer, review_note = CASE_NOTES.get(
            row["object_id"],
            ("successful-control" if row["gallery_group"] == "high_motion_success" else "intrinsic-head", ""),
        )
        manifest_rows.append({
            **{key: row.get(key, "") for key in fields},
            "failure_layer": failure_layer,
            "review_note": review_note,
            "viewer": _viewer_for(row["object_id"]) or "",
        })
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    cards = []
    current_group = None
    for row in selected:
        if row["gallery_group"] != current_group:
            current_group = row["gallery_group"]
            cards.append(f'<h2>{html.escape(row["group_description"])}</h2><div class="grid">')
        viewer = _viewer_for(row["object_id"])
        viewer_link = f'<a href="file://{html.escape(viewer)}">Open Track2Art part + axis viewer</a>' if viewer else '<span class="muted">No local geometry viewer</span>'
        failure_layer, review_note = CASE_NOTES.get(
            row["object_id"],
            ("successful-control" if row["gallery_group"] == "high_motion_success" else "intrinsic-head", ""),
        )
        cards.append(f'''<article class="card">
          <header><strong>{html.escape(row["object_id"])}</strong><span>{html.escape(row["category"])}</span></header>
          <div class="joint">{html.escape(row["joint_id"])} · GT {html.escape(row["joint_type"])} · Pred {html.escape(row["predicted_type"])}</div>
          {_projection_svg(row["gt_axis"], row["pred_axis"])}
          <div class="legend"><span class="gt">GT axis</span><span class="pred">Predicted axis</span></div>
          <dl><dt>Axis error</dt><dd>{row["axis_error_deg"]:.2f}°</dd>
          <dt>Observed motion</dt><dd>{row["motion_magnitude"]:.4f} ({row["motion_bin"]})</dd>
          <dt>Observability</dt><dd>{row["observability"]:.3f} ({row["observability_bin"]})</dd>
          <dt>Canonical proximity</dt><dd>{float(row["nearest_canonical_angle_deg"]):.1f}° to {row["nearest_axis"]}</dd></dl>
          <p class="note"><strong>{html.escape(failure_layer)}</strong> · {html.escape(review_note)}</p>
          <footer>{viewer_link}</footer>
        </article>''')
        next_index = selected.index(row) + 1
        if next_index == len(selected) or selected[next_index]["gallery_group"] != current_group:
            cards.append('</div>')

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Neural relation-head long tail</title>
    <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, sans-serif; color:#172033; background:#eef2f6; }}
    body {{ margin:0; }} main {{ max-width:1440px; margin:auto; padding:36px; }}
    h1 {{ margin:0 0 8px; font-size:32px; }} .intro {{ color:#526076; max-width:940px; line-height:1.55; }}
    h2 {{ margin:34px 0 14px; font-size:20px; }} .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ background:white; border:1px solid #d9e0e8; border-radius:14px; padding:16px; box-shadow:0 5px 18px #1f29370d; }}
    header {{ display:flex; justify-content:space-between; gap:12px; }} header span,.muted {{ color:#718096; }}
    .joint {{ margin:7px 0 2px; color:#475569; font-size:14px; }} svg {{ width:100%; height:auto; }} .plane {{ font:12px sans-serif; fill:#64748b; }}
    .legend {{ display:flex; gap:18px; font-size:12px; margin:-2px 0 10px; }} .gt:before,.pred:before {{ content:""; display:inline-block; width:18px; height:4px; margin:0 6px 2px 0; background:#111827; }} .pred:before {{ background:#f97316; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; margin:0; font-size:13px; gap:6px 12px; }} dt {{ color:#64748b; }} dd {{ margin:0; text-align:right; font-variant-numeric:tabular-nums; }}
    .note {{ min-height:38px; color:#475569; font-size:12px; line-height:1.4; background:#f8fafc; border-radius:8px; padding:8px; }}
    footer {{ border-top:1px solid #edf0f4; margin-top:12px; padding-top:10px; font-size:13px; }} a {{ color:#0f67c5; }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:20px; }} }}
    </style></head><body><main><h1>Neural relation-head long-tail examples</h1>
    <p class="intro">Black is the GT undirected axis; orange is the VN prediction. Motion magnitude and observability come from the relation-head input evidence, not from the simulator joint range. Examples are selected by fixed rules with at most one joint per object in each group.</p>
    {''.join(cards)}</main></body></html>'''
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    summary = {
        "source": str(SOURCE.resolve()),
        "joint_count": len(rows),
        "selected_count": len(selected),
        "groups": {group: sum(row["gallery_group"] == group for row in selected) for group in sorted({row["gallery_group"] for row in selected})},
        "selection_is_prediction_independent_for_exclusions": False,
        "note": "This is a diagnostic gallery, not an evaluation-domain filter.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
