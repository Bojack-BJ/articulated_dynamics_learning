"""Build multi-view SAM2 pseudo part-label seeds for real RGB-D scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    return parser.parse_args()


def sam_inputs(prompts: list[dict]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    points, labels, box = [], [], None
    for prompt in prompts:
        kind = prompt["type"]
        if kind in {"point_pos", "point_neg"}:
            points.append([prompt["x"], prompt["y"]])
            labels.append(1 if kind == "point_pos" else 0)
        elif kind == "bbox":
            box = np.asarray([prompt["x1"], prompt["y1"], prompt["x2"], prompt["y2"]], np.float32)
    return (
        np.asarray(points, np.float32) if points else None,
        np.asarray(labels, np.int32) if labels else None,
        box,
    )


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text())
    scene_root = Path(spec["scene_root"]).expanduser().resolve()
    output_root = scene_root / "pseudo_parts_sam2"
    seed_root = output_root / "seed"
    seed_root.mkdir(parents=True, exist_ok=True)

    sam2_root = args.sam2_root.expanduser().resolve()
    sys.path.insert(0, str(sam2_root))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        args.sam2_config,
        str(args.sam2_checkpoint.expanduser().resolve()),
        device=args.device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    anchor = int(spec["anchor_frame"])
    part_names = {int(item["id"]): item["name"] for item in spec["parts"]}

    seed_paths: dict[int, Path] = {}
    qa = []
    palette = np.asarray([[0, 0, 0], [180, 188, 196], [242, 142, 43], [48, 176, 220], [88, 196, 120]], np.uint8)
    for view_spec in spec["views"]:
        view = int(view_spec["view"])
        rgb_path = scene_root / "color" / str(view) / f"{anchor}.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), np.uint8).copy()
        labels = np.zeros(rgb.shape[:2], np.uint16)
        scores = {}
        with torch.inference_mode():
            predictor.set_image(rgb)
            for part in view_spec["parts"]:
                points, point_labels, box = sam_inputs(part["prompts"])
                masks, ious, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
                index = int(np.argmax(ious))
                mask = np.asarray(masks[index]) > 0.5
                if box is not None and part.get("clip_to_bbox", True):
                    x1, y1, x2, y2 = np.rint(box).astype(int)
                    clipped = np.zeros_like(mask)
                    clipped[max(y1, 0) : min(y2 + 1, mask.shape[0]), max(x1, 0) : min(x2 + 1, mask.shape[1])] = True
                    mask &= clipped
                labels[mask] = int(part["id"])
                scores[part_names[int(part["id"])]] = float(ious[index])
        seed_dir = seed_root / f"view_{view}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_path = seed_dir / f"frame_{anchor:04d}_mask.png"
        Image.fromarray(labels, mode="I;16").save(seed_path)
        seed_paths[view] = seed_path
        colors = palette[np.minimum(labels, len(palette) - 1)]
        overlay = (0.55 * rgb + 0.45 * colors).astype(np.uint8)
        Image.fromarray(overlay).save(seed_dir / f"frame_{anchor:04d}_overlay.jpg", quality=92)
        qa.append({"view": view, "scores": scores, "mask_path": str(seed_path)})

    frame_count = min(len(list((scene_root / "color" / str(view)).glob("*.png"))) for view in seed_paths)
    frames = []
    for frame_index in range(frame_count):
        views = sorted(seed_paths)
        rgb_by_view = [str(scene_root / "color" / str(v) / f"{frame_index}.png") for v in views]
        depth_by_view = [str(scene_root / "depth" / str(v) / f"{frame_index}.png") for v in views]
        seed_by_view = [str(seed_paths[v]) for v in views] if frame_index == anchor else []
        frames.append(
            {
                "timestamp_s": frame_index / 30.0,
                "rgb_path": rgb_by_view[0],
                "depth_path": depth_by_view[0],
                "camera_pose": np.eye(4).tolist(),
                "part_mask_path": seed_by_view[0] if seed_by_view else None,
                "rgb_paths_by_view": rgb_by_view,
                "depth_paths_by_view": depth_by_view,
                "part_mask_paths_by_view": seed_by_view,
                "camera_poses_by_view": [np.eye(4).tolist() for _ in views],
            }
        )
    episode = {
        "object_instance_id": spec["scene_id"],
        "category": spec["category"],
        "camera_intrinsics": {},
        "frames": frames,
        "metadata": {
            "mask_provenance": "SAM2 pseudo part labels; independent of Track2Art predictions",
            "part_names": {str(k): v for k, v in part_names.items()},
            "anchor_frame": anchor,
        },
    }
    (output_root / "episode.seed.json").write_text(json.dumps(episode, indent=2))
    (output_root / "seed_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({"episode": str(output_root / "episode.seed.json"), "views": qa}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
