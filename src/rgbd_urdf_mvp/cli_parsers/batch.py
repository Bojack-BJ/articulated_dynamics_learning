from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    batch_parser = subparsers.add_parser(
        "run-articulation-batch",
        help="Run the record -> fuse -> track -> pose -> joint -> viewer pipeline over a batch manifest",
    )
    batch_parser.add_argument(
        "manifest",
        type=Path,
        help="Tab-separated manifest: category, model_path, object_id, joint_name",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any stage whose expected output already exists",
    )
    batch_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a whole object when its final batch artifact already exists",
    )
    batch_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of objects to process concurrently",
    )
    batch_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "recordings",
        help="Root directory for per-object recording outputs",
    )
    batch_parser.add_argument(
        "--batch-log-dir",
        type=Path,
        default=Path("outputs") / "recordings" / "_batch_logs",
        help="Per-object batch logs when --jobs > 1",
    )
    batch_parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(os.environ.get("TORCH_HOME", ".cache/torch")),
        help="Torch cache directory propagated to subprocess stages",
    )
    batch_parser.add_argument(
        "--record-config",
        type=Path,
        default=None,
        help="Optional YAML/JSON record-mujoco template used for every object in the batch",
    )
    batch_parser.add_argument(
        "--track-config",
        type=Path,
        default=None,
        help="Optional YAML/JSON track-part-pixels template used for every object in the batch",
    )
    batch_parser.add_argument(
        "--cotracker-repo",
        type=Path,
        default=None,
        help="Optional override for the CoTracker repo path used by the tracking stage",
    )
    batch_parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=None,
        help="Optional override for the CoTracker checkpoint path used by the tracking stage",
    )
    batch_parser.add_argument(
        "--track-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default=None,
        help="Optional override for the device used by track-part-pixels",
    )
    batch_parser.add_argument(
        "--tracking-jobs",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent track-part-pixels stages. "
            "Defaults to 1 for auto/mps/cuda and to --jobs for cpu."
        ),
    )
    batch_parser.add_argument(
        "--fuse-pixel-stride",
        type=int,
        default=8,
        help="Depth sampling stride for fuse-pointcloud",
    )
    batch_parser.add_argument(
        "--fuse-voxel-size-m",
        type=float,
        default=0.02,
        help="Fusion voxel size for fuse-pointcloud",
    )
    batch_parser.add_argument(
        "--min-tracks-per-part",
        type=int,
        default=4,
        help="Minimum visible 3D tracks required for track-based part pose estimation",
    )
    batch_parser.add_argument(
        "--mujoco-prior",
        choices=["auto", "off", "required"],
        default="off",
        help="MJCF prior mode passed through to infer-joints",
    )
    batch_parser.add_argument(
        "--no-generate-viewer",
        action="store_true",
        help="Skip the final visualize-pointcloud stage",
    )
    batch_parser.add_argument(
        "--dynamics-backend",
        choices=["off", "mujoco", "mjx"],
        default="off",
        help="Optional Stage 4 dynamics identification backend run after joint inference",
    )
    batch_parser.add_argument(
        "--dynamics-config",
        type=Path,
        default=None,
        help="Optional YAML/JSON identify-dynamics template used when --dynamics-backend is enabled",
    )
    batch_parser.add_argument(
        "--dynamics-jobs",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent dynamics stages. "
            "Defaults to 1 for mjx and to --jobs for mujoco."
        ),
    )
    batch_parser.add_argument(
        "--dynamics-jax-platform",
        choices=["cpu", "metal", "auto"],
        default=None,
        help="Optional override passed to identify-dynamics-mjx",
    )
    batch_parser.add_argument(
        "--dynamics-enable-pjrt-compatibility",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override passed to identify-dynamics-mjx",
    )
    batch_parser.add_argument(
        "--dynamics-render-gl-backend",
        choices=["auto", "cgl", "glfw", "osmesa", "egl", "none"],
        default=None,
        help="Optional render backend override passed to the dynamics stage",
    )
    batch_parser.add_argument(
        "--plot-dynamics",
        action="store_true",
        help="Write optimization_history.svg after each completed dynamics identification artifact",
    )

    convert_batch_parser = subparsers.add_parser(
        "convert-usd-mjcf-batch",
        help="Convert USD assets in a batch manifest to MJCF XML before running articulation stages",
    )
    convert_batch_parser.add_argument(
        "manifest",
        type=Path,
        help="Tab-separated manifest: category, model_path, object_id, joint_name",
    )
    convert_batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples") / "mujoco_models",
        help="Directory for generated MJCF XML and OBJ folders",
    )
    convert_batch_parser.add_argument(
        "--converted-manifest",
        type=Path,
        default=None,
        help="Optional TSV to write with USD paths replaced by generated MJCF XML paths",
    )
    convert_batch_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of USD assets to convert concurrently",
    )
    convert_batch_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even when the target MJCF XML already exists",
    )
