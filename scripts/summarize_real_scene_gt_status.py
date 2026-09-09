from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENES = {
    "scene29": Path("data/track2art_real_20260814/scenes/scene29_case"),
    "scene30": Path("data/track2art_real_20260814/scenes/scene30_small_box"),
    "scene31": Path("data/track2art_real_20260814/scenes/scene31_small_fridge"),
    "scene32": Path("data/track2art_real_20260814/scenes/scene32"),
    "scene33": Path("data/track2art_real_20260814/scenes/scene33"),
    "scene34": Path("data/track2art_real_20260814/scenes/scene34"),
    "drawer_hand": Path("data/track2art_real_new_20260830/extracted/drawer_hand"),
    "drawer_hand2": Path("data/track2art_real_new_20260830/extracted/drawer_hand2"),
}

ANNOTATION_PORTS = {
    "scene29": 8766,
    "scene30": 8767,
    "scene31": 8768,
    "scene32": 8769,
    "scene34": 8770,
    "drawer_hand": 8771,
    "drawer_hand2": 8772,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manual and pseudo GT coverage.")
    parser.add_argument("--output", type=Path, default=Path("outputs/real_scene_gt_status.md"))
    args = parser.parse_args()

    rows = []
    for scene, root in SCENES.items():
        annotation = root / "annotations" / "real_scene_gt_annotations.json"
        pseudo = root / "pseudo_parts_sam2" / "episode.pseudo_parts.json"
        payload = _load(annotation) if annotation.exists() else {}
        track_labels = payload.get("track_labels") if isinstance(payload.get("track_labels"), dict) else {}
        joints = payload.get("joints") if isinstance(payload.get("joints"), list) else []
        rows.append(
            {
                "scene": scene,
                "joint_count": len(joints),
                "manual_track_label_count": len(track_labels),
                "pseudo_part": pseudo.exists(),
                "annotation_path": annotation if annotation.exists() else None,
            }
        )

    lines = [
        "# Real-scene GT status",
        "",
        "Manual joint GT and manual part GT are independent. SAM2 pseudo labels are not counted as manual GT.",
        "",
        "| Scene | Joint GT | Manual part GT | SAM2 pseudo part | Annotate | Saved file |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        annotation = row["annotation_path"]
        link = f"[{annotation.name}]({annotation.resolve().as_uri()})" if annotation else "-"
        port = ANNOTATION_PORTS.get(row["scene"])
        annotate = f"[open](http://127.0.0.1:{port}/)" if port else "unavailable"
        lines.append(
            f"| {row['scene']} | {row['joint_count']} joints | "
            f"{row['manual_track_label_count']} tracks | "
            f"{'yes' if row['pseudo_part'] else 'no'} | {annotate} | {link} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
