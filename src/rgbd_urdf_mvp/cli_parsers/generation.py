from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    generation_image_parser = subparsers.add_parser(
        "prepare-generation-images",
        help="Use existing prior masks to create clean object-centric images for Hunyuan3D",
    )
    generation_image_parser.add_argument("episode", type=Path, help="Path to episode.json")
    generation_image_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for clean PNGs and generation_images.json",
    )
    generation_image_parser.add_argument("--frame-index", type=int, default=0, help="Episode frame used for generation images")
    generation_image_parser.add_argument(
        "--view-indices",
        type=int,
        nargs="+",
        default=None,
        help="View indices to export. Defaults to all RGB/mask views in the selected frame.",
    )
    generation_image_parser.add_argument(
        "--image-views",
        nargs="+",
        choices=["front", "left", "right", "back"],
        default=None,
        help="Hunyuan3D view names matching --view-indices order. Defaults to front left right back.",
    )
    generation_image_parser.add_argument(
        "--mask-source",
        choices=["auto", "object", "part"],
        default="auto",
        help="Use object mask, part-mask union, or auto-prefer object mask.",
    )
    generation_image_parser.add_argument(
        "--background",
        choices=["transparent", "white", "black", "original"],
        default="transparent",
        help="Background policy outside the mask.",
    )
    generation_image_parser.add_argument("--padding-ratio", type=float, default=0.15, help="Square crop padding around mask bbox")
    generation_image_parser.add_argument("--min-mask-pixels", type=int, default=16, help="Minimum foreground mask pixels per view")

    hunyuan_parser = subparsers.add_parser(
        "hunyuan3d-generate",
        help="Request a remote Hunyuan3D API server and save the generated 3D asset",
    )
    hunyuan_parser.add_argument("--server-url", required=True, help="Base URL of the Hunyuan3D API server")
    hunyuan_parser.add_argument("--output", type=Path, required=True, help="Output model path, usually .glb")
    hunyuan_parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more input image paths. A single path uses Hunyuan3D single-view mode; "
            "multiple paths are sent as a multiview payload."
        ),
    )
    hunyuan_parser.add_argument(
        "--image-views",
        nargs="+",
        choices=["front", "left", "right", "back"],
        default=None,
        help=(
            "View names matching --image order for multiview generation. "
            "Defaults to front left right back truncated to the number of images."
        ),
    )
    hunyuan_parser.add_argument(
        "--image-manifest",
        type=Path,
        default=None,
        help="generation_images.json from prepare-generation-images. Overrides --image/--image-views.",
    )
    hunyuan_parser.add_argument("--text", type=str, default=None, help="Optional text prompt")
    hunyuan_parser.add_argument("--mesh", type=Path, default=None, help="Optional input mesh for texture generation")
    hunyuan_parser.add_argument("--mode", choices=["async", "sync"], default="async", help="Use /send polling or /generate")
    hunyuan_parser.add_argument("--texture", action="store_true", help="Request texture generation when supported")
    hunyuan_parser.add_argument("--seed", type=int, default=1234, help="Generation seed")
    hunyuan_parser.add_argument("--type", default="glb", help="Requested output type, e.g. glb or obj")
    hunyuan_parser.add_argument("--octree-resolution", type=int, default=None, help="Optional Hunyuan3D octree resolution")
    hunyuan_parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional diffusion step count")
    hunyuan_parser.add_argument("--guidance-scale", type=float, default=None, help="Optional guidance scale")
    hunyuan_parser.add_argument("--face-count", type=int, default=None, help="Optional target face count")
    hunyuan_parser.add_argument("--timeout-s", type=float, default=1800.0, help="Async polling timeout")
    hunyuan_parser.add_argument("--poll-interval-s", type=float, default=5.0, help="Async polling interval")
    hunyuan_parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Optional bearer token for a reverse proxy, tunnel, or API gateway",
    )

    particulate_parser = subparsers.add_parser(
        "particulate-infer",
        help="Run the Particulate submodule on a generated/reconstructed mesh and save a result manifest",
    )
    particulate_input = particulate_parser.add_mutually_exclusive_group(required=True)
    particulate_input.add_argument("--mesh", type=Path, help="Input mesh path (.glb, .obj, or .ply)")
    particulate_input.add_argument(
        "--reconstruction-artifact",
        type=Path,
        help="Project reconstruction_artifact.json; uses its canonical_mesh_path",
    )
    particulate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for Particulate outputs and particulate_result.json",
    )
    particulate_parser.add_argument(
        "--particulate-root",
        type=Path,
        default=Path("Particulate"),
        help="Path to the Particulate submodule checkout",
    )
    particulate_parser.add_argument(
        "--python-bin",
        default=os.environ.get("PARTICULATE_PYTHON", "python"),
        help="Python executable for the Particulate environment, usually a CUDA GPU env",
    )
    particulate_parser.add_argument(
        "--model-config",
        default="configs/particulate-B.yaml",
        help="Path passed to Particulate infer.py --model_config, relative to the submodule by default",
    )
    particulate_parser.add_argument("--ckpt-path", type=Path, default=None, help="Optional local Particulate checkpoint")
    particulate_parser.add_argument(
        "--up-dir",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        default="-Z",
        help="Input mesh up direction passed to Particulate",
    )
    particulate_parser.add_argument("--num-points", type=int, default=102400, help="Sample count for Particulate")
    particulate_parser.add_argument(
        "--num-points-global",
        type=int,
        default=40000,
        help="PartField encoder point count used before Particulate decode sampling",
    )
    particulate_parser.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Optionally decimate the input mesh before passing it to Particulate",
    )
    particulate_parser.add_argument(
        "--min-part-confidence",
        type=float,
        default=0.0,
        help="Minimum predicted part confidence passed to Particulate",
    )
    particulate_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable Particulate strict connected-component refinement",
    )
    particulate_parser.add_argument("--animation-frames", type=int, default=50, help="Animated GLB frame count")
    particulate_parser.add_argument("--no-export-urdf", action="store_true", help="Do not request URDF export")
    particulate_parser.add_argument("--no-export-mjcf", action="store_true", help="Do not request MJCF export")
    particulate_parser.add_argument("--no-eval", action="store_true", help="Do not request pred.npz evaluation artifact")
    particulate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the project manifest without launching Particulate; useful for checking paths",
    )

    remote_articulation_parser = subparsers.add_parser(
        "remote-articulate-generate",
        help="Send observations to a remote Hunyuan3D+PARTICULATE server and unpack articulated outputs",
    )
    remote_articulation_parser.add_argument("--server-url", required=True, help="Remote articulation server URL")
    remote_articulation_parser.add_argument("--output-dir", type=Path, required=True, help="Directory to unpack results into")
    remote_articulation_parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=None,
        help="One or more observation image paths",
    )
    remote_articulation_parser.add_argument(
        "--image-views",
        nargs="+",
        choices=["front", "left", "right", "back"],
        default=None,
        help="View names matching --image order",
    )
    remote_articulation_parser.add_argument(
        "--image-manifest",
        type=Path,
        default=None,
        help="generation_images.json from prepare-generation-images. Overrides --image/--image-views.",
    )
    remote_articulation_parser.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="Prepare masked generation images from an existing episode.json before sending. Overrides --image.",
    )
    remote_articulation_parser.add_argument(
        "--generation-image-output-dir",
        type=Path,
        default=None,
        help="Where to write prepared generation images when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Episode frame used for generation images when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--view-indices",
        type=int,
        nargs="+",
        default=None,
        help="Episode view indices to export when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--mask-source",
        choices=["auto", "object", "part"],
        default="auto",
        help="Mask source used when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--background",
        choices=["transparent", "white", "black", "original"],
        default="transparent",
        help="Background policy for prepared generation images when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Square crop padding around mask bbox when --episode is used.",
    )
    remote_articulation_parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=16,
        help="Minimum foreground mask pixels per prepared view when --episode is used.",
    )
    remote_articulation_parser.add_argument("--text", type=str, default=None, help="Optional text prompt")
    remote_articulation_parser.add_argument("--texture", action="store_true", help="Request Hunyuan texture generation")
    remote_articulation_parser.add_argument("--seed", type=int, default=1234, help="Hunyuan generation seed")
    remote_articulation_parser.add_argument("--type", default="glb", help="Requested Hunyuan output type")
    remote_articulation_parser.add_argument("--octree-resolution", type=int, default=None)
    remote_articulation_parser.add_argument("--num-inference-steps", type=int, default=None)
    remote_articulation_parser.add_argument("--guidance-scale", type=float, default=None)
    remote_articulation_parser.add_argument("--face-count", type=int, default=None)
    remote_articulation_parser.add_argument(
        "--particulate-up-dir",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        default="-Z",
        help="Input mesh up direction passed to PARTICULATE on the server",
    )
    remote_articulation_parser.add_argument("--particulate-num-points", type=int, default=102400)
    remote_articulation_parser.add_argument(
        "--particulate-global-points",
        type=int,
        default=40000,
        help="PartField encoder point count used by PARTICULATE on the server",
    )
    remote_articulation_parser.add_argument(
        "--particulate-target-faces",
        type=int,
        default=None,
        help="Optionally decimate the generated mesh before PARTICULATE runs",
    )
    remote_articulation_parser.add_argument("--particulate-min-part-confidence", type=float, default=0.0)
    remote_articulation_parser.add_argument(
        "--particulate-no-strict",
        action="store_true",
        help="Disable PARTICULATE strict connected-component refinement",
    )
    remote_articulation_parser.add_argument("--timeout-s", type=float, default=3600.0)
    remote_articulation_parser.add_argument("--poll-interval-s", type=float, default=5.0)
    remote_articulation_parser.add_argument("--api-token", type=str, default=None)

    compare_parser = subparsers.add_parser(
        "compare-articulation-backends",
        help="Summarize tracking joint inference and Particulate mesh articulation outputs side by side",
    )
    compare_parser.add_argument("tracking_joint_inference", type=Path, help="Path to joint_inference.json")
    compare_parser.add_argument("particulate_result", type=Path, help="Path to particulate_result.json")
    compare_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Comparison output path (defaults next to particulate_result.json)",
    )

    kinematic_eval_parser = subparsers.add_parser(
        "evaluate-kinematic-model",
        help="Evaluate inferred joints against MuJoCo GT joints when the recording model is available",
    )
    kinematic_eval_parser.add_argument("joint_inference", type=Path, help="Path to joint_inference.json")
    kinematic_eval_parser.add_argument(
        "--part-poses",
        type=Path,
        default=None,
        help="Optional path to part_poses.json. Defaults to joint_inference.input_path.",
    )
    kinematic_eval_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Evaluation output path (defaults next to joint_inference.json)",
    )
