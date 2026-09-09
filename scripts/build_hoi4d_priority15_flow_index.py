#!/usr/bin/env python3
"""Build a collection entry for HOI4D RGB/mask and 3D flow review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("viewer_root", type=Path)
    parser.add_argument("--stability-audit", type=Path)
    args = parser.parse_args()
    stability = {}
    if args.stability_audit and args.stability_audit.is_file():
        payload = json.loads(args.stability_audit.read_text())
        stability = {row["sequence"]: row for row in payload.get("sequences", [])}
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["category"] in {"C3", "C4", "C6", "C14"}]
    cards = []
    for row in rows:
        name = row["sequence"].replace("/", "_")
        ready = (args.viewer_root / name / "viewer_gt_axis.html").is_file()
        cad_ready = (args.viewer_root / name / "viewer_cad_axis.html").is_file()
        quality = stability.get(name, {})
        status = quality.get("training_status", "not audited")
        quality_text = ""
        if quality:
            quality_text = (
                f" · coverage {quality['coverage_ratio']:.0%}"
                f" · stable segment {quality['longest_stable_segment_ratio']:.0%}"
                f" · median track life {quality['median_contiguous_lifetime_ratio']:.0%}"
            )
        cards.append(f'''<article class="{'ready' if ready else 'pending'}"><h2>{name}</h2>
<p><b class="{status}">{status}</b> · {row['split']} · {row['motion']} · {'3D ready' if ready else 'processing'}{quality_text}</p>
<a href="/{name}/viewer_gt_axis.html">3D flow + axis</a>
<a href="/{name}/viewer_cad_axis.html">{'CAD axis' if cad_ready else 'CAD pending'}</a>
<a href="http://127.0.0.1:8899/{name}.html" target="_blank">RGB + official mask</a></article>''')
    page = f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>HOI4D priority-15 review</title>
<style>body{{margin:0;padding:28px;background:#091018;color:#edf5f8;font:14px ui-monospace,monospace}}h1{{font-size:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}}article{{padding:16px;background:#111d27;border:1px solid #2b4050;border-radius:10px}}article.pending{{opacity:.55}}h2{{font-size:14px;overflow-wrap:anywhere}}p{{color:#91a7b5}}a{{display:inline-block;margin-right:16px;color:#51d6ad}}b.accept{{color:#51d6ad}}b.reject_or_trim{{color:#ff806d}}</style>
<h1>HOI4D priority-15 quality review</h1><p>Inspect RGB/mask first, then 3D registration and axis. Pending cards refresh every 30 seconds.</p><div class="grid">{''.join(cards)}</div>'''
    (args.viewer_root / "index.html").write_text(page)
    print(f"wrote {args.viewer_root / 'index.html'} ready={sum('3D ready' in card for card in cards)}/{len(cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
