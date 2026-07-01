from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one binary mask from SAM2 image prompts.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--mask-selection", choices=["best-iou", "largest", "smallest"], default="best-iou")
    args = parser.parse_args()

    sam2_root = args.sam2_root.expanduser().resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    module = sys.modules.get("sam2")
    if module is not None:
        module_path = getattr(module, "__file__", None)
        if module_path is None or not str(module_path).startswith(str(sam2_root)):
            del sys.modules["sam2"]

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _resolve_device(torch, args.sam2_device)
    payload = json.loads(args.prompts_json.read_text())
    prompts = payload.get("prompts") or []
    if not prompts:
        raise ValueError("No SAM2 point/bbox prompts were provided for the current part.")

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    point_coords, point_labels, box = _prompts_to_sam2_inputs(prompts)
    model = build_sam2(
        args.sam2_config,
        str(args.sam2_checkpoint.expanduser().resolve()),
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, iou_predictions, _low_res = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    if len(masks) == 0:
        raise RuntimeError("SAM2 returned no masks.")
    areas = [int(mask.sum()) for mask in masks]
    index = _select_mask(args.mask_selection, iou_predictions, areas)
    output = masks[index].astype(np.uint16)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="I;16").save(args.output_mask)
    print(
        json.dumps(
            {
                "output_mask": str(args.output_mask),
                "mask_pixels": int(output.sum()),
                "mask_choice": int(index),
                "predicted_iou": float(iou_predictions[index]) if len(iou_predictions) else None,
                "areas": areas,
            },
            indent=2,
        )
    )
    return 0


def _resolve_device(torch, device_name: str) -> str:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("SAM2 device requested as mps, but torch.backends.mps.is_available() is false.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM2 device requested as cuda, but torch.cuda.is_available() is false.")
    return device_name


def _prompts_to_sam2_inputs(prompts: list[dict]):
    points: list[list[float]] = []
    labels: list[int] = []
    box = None
    for prompt in prompts:
        prompt_type = str(prompt.get("type") or "")
        if prompt_type == "point_pos":
            points.append([float(prompt["x"]), float(prompt["y"])])
            labels.append(1)
        elif prompt_type == "point_neg":
            points.append([float(prompt["x"]), float(prompt["y"])])
            labels.append(0)
        elif prompt_type == "bbox":
            box = np.asarray(
                [
                    float(prompt["x1"]),
                    float(prompt["y1"]),
                    float(prompt["x2"]),
                    float(prompt["y2"]),
                ],
                dtype=np.float32,
            )
    point_coords = np.asarray(points, dtype=np.float32) if points else None
    point_labels = np.asarray(labels, dtype=np.int32) if labels else None
    if point_coords is None and box is None:
        raise ValueError("SAM2 needs at least one positive/negative point or bbox prompt.")
    return point_coords, point_labels, box


def _select_mask(mask_selection: str, iou_predictions, areas: list[int]) -> int:
    if not len(areas):
        return 0
    if mask_selection == "largest":
        return int(np.argmax(np.asarray(areas)))
    if mask_selection == "smallest":
        return int(np.argmin(np.asarray(areas)))
    if len(iou_predictions):
        return int(np.argmax(iou_predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
