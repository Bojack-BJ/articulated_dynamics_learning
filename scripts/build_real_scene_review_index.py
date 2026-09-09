#!/usr/bin/env python3
"""Build a compact review page for the real-scene Track2Art pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "real_scene29_34_track2art_review_v1"


def scene_root(scene: int) -> Path:
    if scene in {29, 31}:
        return ROOT / "outputs" / "real_depth_quality_pilot_v1" / f"scene{scene}"
    if scene in {32, 34}:
        return ROOT / "outputs" / f"real_scene{scene}_track2art_v2_camera_order"
    return ROOT / "outputs" / f"real_scene{scene}_track2art_v1"


def relative_link(path: Path) -> str:
    return Path("..").joinpath(path.relative_to(ROOT / "outputs")).as_posix()


def load_row(scene: int) -> dict[str, object]:
    root = scene_root(scene)
    if scene == 33:
        return {
            "scene": scene,
            "category": "small box",
            "status": "not run",
            "reason": "Only 21 frames; SAM object/part masks are missing and articulation is not visually resolved.",
        }

    episode = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "pseudo_part_evaluation.json").read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    neural_viewer_path = root / "viewer_with_pcd_neural.html"
    analytic_viewer_path = root / "viewer_with_pcd_analytic.html"
    neural_joint_path = root / "kinematics" / "joint_inference_neural.json"
    neural_joint_count = 0
    if neural_joint_path.exists():
        neural_joint_count = len(
            json.loads(neural_joint_path.read_text(encoding="utf-8")).get("selected_edges", [])
        )
    return {
        "scene": scene,
        "category": episode.get("category", "unknown"),
        "status": "complete",
        "frames": len(episode.get("frames", [])),
        "predicted_parts": metrics["predicted_part_count"],
        "pseudo_parts": metrics["gt_part_count"],
        "iou": metrics["one_to_one_mean_iou"],
        "ari": metrics["adjusted_rand_index"],
        "ri": metrics["rand_index"],
        "largest_cluster_ratio": metrics["largest_cluster_ratio"],
        "evaluated_track_ratio": evaluation["evaluated_track_ratio"],
        "neural_viewer": relative_link(neural_viewer_path),
        "analytic_viewer": relative_link(analytic_viewer_path),
        "neural_joint_count": neural_joint_count,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [load_row(scene) for scene in range(29, 35)]
    (OUTPUT / "summary.json").write_text(json.dumps({"scenes": rows}, indent=2) + "\n", encoding="utf-8")

    cards = []
    for row in rows:
        if row["status"] != "complete":
            body = f'<p class="missing">{row["reason"]}</p>'
        else:
            body = f"""
              <div class="metrics">
                <span>parts <b>{row['predicted_parts']} / {row['pseudo_parts']}</b></span>
                <span>IoU <b>{row['iou']:.3f}</b></span>
                <span>ARI <b>{row['ari']:.3f}</b></span>
                <span>RI <b>{row['ri']:.3f}</b></span>
                <span>largest <b>{row['largest_cluster_ratio']:.1%}</b></span>
              </div>
              <div class="actions">
                <a class="open" href="{row['neural_viewer']}">Neural axis</a>
                <a class="open secondary" href="{row['analytic_viewer']}">Analytic axis</a>
              </div>
              <p class="note">Neural graph: {row['neural_joint_count']} predicted joint(s).</p>
              <p class="note">SAM propagated masks are diagnostic pseudo labels, not manual ground truth.</p>
            """
        cards.append(f"""
          <article>
            <header><span>Scene {row['scene']}</span><strong>{row['category']}</strong></header>
            {body}
          </article>
        """)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real-scene Track2Art review</title>
<style>
:root{{--bg:#f3f0e8;--ink:#18211d;--muted:#66716b;--card:#fffdf8;--line:#d8d2c5;--accent:#0f766e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 Georgia,serif}}
main{{max-width:1080px;margin:0 auto;padding:48px 24px 72px}} h1{{font:700 42px/1.05 Georgia,serif;margin:0 0 12px}}
.lede{{max-width:800px;color:var(--muted);margin:0 0 32px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 8px 30px #463e2d12}}
header{{display:flex;justify-content:space-between;gap:16px;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:16px}}
header span{{font:700 20px/1.2 ui-monospace,monospace}} header strong{{color:var(--muted);font-weight:600;text-transform:capitalize}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-bottom:18px}} .metrics span{{background:#f2f6f3;border-radius:8px;padding:9px;font:13px/1.2 ui-monospace,monospace}}
.metrics b{{display:block;font-size:17px;margin-top:3px}} .open{{display:inline-block;background:var(--accent);color:white;text-decoration:none;padding:10px 14px;border-radius:8px;font-weight:700}}
.actions{{display:flex;gap:9px;flex-wrap:wrap}} .secondary{{background:#52635d}}
.note,.missing{{color:var(--muted);font-size:13px;margin:10px 0 0}} .missing{{font-size:15px}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}} h1{{font-size:34px}}}}
</style></head><body><main>
<h1>Real-scene Track2Art review</h1>
<p class="lede">Scenes 29–34, compared using propagated SAM part masks. New viewers embed calibrated three-view RGB-D point clouds so the predicted motion slots and joint axes can be interpreted in scene context.</p>
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
