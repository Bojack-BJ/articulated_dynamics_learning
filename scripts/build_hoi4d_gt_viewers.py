#!/usr/bin/env python3
"""Build read-only RGB/official-mask review pages for adapted HOI4D episodes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = {
    0: [0, 0, 0],
    1: [255, 145, 61],
    2: [180, 180, 180],
    3: [49, 209, 166],
    4: [255, 99, 132],
    5: [102, 153, 255],
    6: [242, 205, 92],
}


def _sequence_page(episode_path: Path, output_dir: Path) -> tuple[str, dict]:
    episode = json.loads(episode_path.read_text())
    segmentation = episode["metadata"]["part_segmentation"]
    part_by_id = {int(part["part_id"]): part for part in segmentation["parts"]}
    ignored = {int(value) for value in segmentation.get("ignored_part_ids", [])}
    sequence_name = episode_path.parent.name
    frames = [
        {
            "rgb": f"/assets/{sequence_name}/rgb/{index}",
            "mask": f"/assets/{sequence_name}/mask/{index}",
            "source_frame": int(frame.get("action_log", {}).get("source_frame_index", index)),
        }
        for index, frame in enumerate(episode["frames"])
    ]
    labels = []
    for label in sorted(set(part_by_id) | ignored):
        part = part_by_id.get(label)
        labels.append({
            "id": label,
            "name": part.get("name", "ignored") if part else "ignored",
            "role": part.get("role", "ignored") if part else "ignored",
            "ignored": label in ignored,
            "color": COLORS.get(label, [255, 255, 255]),
        })
    payload = {"frames": frames, "labels": labels, "sequence": sequence_name}
    legend = "".join(
        f'<span class="legend-item"><i style="background:rgb({",".join(map(str, row["color"]))})"></i>'
        f'{row["id"]}: {html.escape(row["name"])} ({html.escape(row["role"])})</span>'
        for row in labels
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(episode_path.parent.name)} | HOI4D GT</title>
<style>
:root{{--bg:#091018;--panel:#111d27;--line:#263744;--text:#edf5f8;--muted:#91a7b5;--accent:#51d6ad}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}
header{{padding:16px 20px;border-bottom:1px solid var(--line);background:#0d1720;position:sticky;top:0;z-index:2}}
h1{{font-size:17px;margin:0 0 12px}} .controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
button,select,input{{accent-color:var(--accent)}} button,select{{background:var(--panel);border:1px solid var(--line);color:var(--text);padding:7px 10px;border-radius:6px}}
#timeline{{width:min(620px,60vw)}} #stage{{display:grid;place-items:center;height:calc(100vh - 154px);padding:14px}}
.canvas-wrap{{position:relative;max-width:100%;max-height:100%}} canvas{{display:block;max-width:100%;max-height:calc(100vh - 184px);background:#000}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);margin-top:10px}} .legend-item{{display:flex;align-items:center;gap:5px}}
.legend-item i{{width:11px;height:11px;border-radius:3px}} #status{{color:var(--muted)}} a{{color:var(--accent)}}
</style></head><body>
<header><h1><a href="index.html">HOI4D GT</a> / {html.escape(episode_path.parent.name)}</h1>
<div class="controls"><button id="play">Play</button><input id="timeline" type="range" min="0" max="{len(frames)-1}" value="0">
<span id="status"></span><select id="mode"><option value="overlay">Overlay</option><option value="rgb">RGB</option><option value="mask">Mask</option></select>
<label>Alpha <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.55"></label></div>
<div class="legend">{legend}</div></header><main id="stage"><div class="canvas-wrap"><canvas id="canvas"></canvas></div></main>
<script>
const DATA={json.dumps(payload, separators=(',', ':'))};
const colors={json.dumps(COLORS, separators=(',', ':'))};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),timeline=document.getElementById('timeline');
const mode=document.getElementById('mode'),alpha=document.getElementById('alpha'),status=document.getElementById('status');
let timer=null,token=0;
function load(src){{return new Promise((resolve,reject)=>{{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=src}})}}
async function render(){{const own=++token,index=Number(timeline.value),frame=DATA.frames[index];status.textContent=`loading ${{index+1}}/${{DATA.frames.length}}`;
 const [rgb,mask]=await Promise.all([load(frame.rgb),load(frame.mask)]);if(own!==token)return;canvas.width=rgb.naturalWidth;canvas.height=rgb.naturalHeight;
 ctx.drawImage(rgb,0,0);const base=ctx.getImageData(0,0,canvas.width,canvas.height);const temp=document.createElement('canvas');temp.width=canvas.width;temp.height=canvas.height;
 const tc=temp.getContext('2d');tc.drawImage(mask,0,0,canvas.width,canvas.height);const md=tc.getImageData(0,0,canvas.width,canvas.height).data,out=ctx.createImageData(canvas.width,canvas.height),a=Number(alpha.value);
 for(let p=0;p<md.length;p+=4){{const id=md[p],c=colors[id]||[255,255,255],visible=id!==0;if(mode.value==='rgb'||!visible){{out.data[p]=base.data[p];out.data[p+1]=base.data[p+1];out.data[p+2]=base.data[p+2]}}else if(mode.value==='mask'){{out.data[p]=c[0];out.data[p+1]=c[1];out.data[p+2]=c[2]}}else{{out.data[p]=base.data[p]*(1-a)+c[0]*a;out.data[p+1]=base.data[p+1]*(1-a)+c[1]*a;out.data[p+2]=base.data[p+2]*(1-a)+c[2]*a}}out.data[p+3]=255}}ctx.putImageData(out,0,0);status.textContent=`${{index+1}}/${{DATA.frames.length}} | source frame ${{frame.source_frame}}`}}
function toggle(){{if(timer){{clearInterval(timer);timer=null;play.textContent='Play';return}}play.textContent='Pause';timer=setInterval(()=>{{timeline.value=(Number(timeline.value)+1)%DATA.frames.length;render()}},120)}}
timeline.oninput=render;mode.onchange=render;alpha.oninput=render;document.getElementById('play').onclick=toggle;render();
</script></body></html>"""
    return page, {"sequence": episode_path.parent.name, "frames": len(frames), "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_path in sorted(args.episodes_root.resolve().glob("*/episode.json")):
        page, row = _sequence_page(episode_path, output_dir)
        filename = f"{episode_path.parent.name}.html"
        (output_dir / filename).write_text(page, encoding="utf-8")
        rows.append({**row, "href": filename})
    cards = ""
    for row in rows:
        label_summary = ", ".join(
            f"{item['id']}:{html.escape(item['name'])}" for item in row["labels"]
        )
        cards += (
            f'<a class="card" href="{row["href"]}"><strong>'
            f'{html.escape(row["sequence"])}</strong><span>{row["frames"]} frames</span>'
            f'<span>{label_summary}</span></a>'
        )
    index = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HOI4D GT review</title>
<style>body{{margin:0;padding:28px;background:#091018;color:#edf5f8;font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}h1{{font-size:24px}}p{{color:#91a7b5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}}.card{{display:flex;flex-direction:column;gap:8px;padding:16px;border:1px solid #263744;border-radius:10px;background:#111d27;color:#edf5f8;text-decoration:none}}.card:hover{{border-color:#51d6ad}}.card span{{color:#91a7b5}}</style></head><body>
<h1>HOI4D Official GT Review</h1><p>Read-only RGB + category-specific official 2D motion segmentation. Inspect propagation noise, occlusion and part suitability before training.</p><div class="grid">{cards}</div></body></html>"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"sequences": rows}, indent=2) + "\n")
    print(json.dumps({"index": str(output_dir / "index.html"), "sequence_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
