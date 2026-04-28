from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core.categories import SUPPORTED_CATEGORIES, normalize_category
from .core.cli_config import YAMLSubsetError, expand_config_argv
from .pipeline import RGBDToURDFPipeline
from .core.serialization import load_episode, load_json, validate_episode


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI.

    The parser stays grouped by workflow so the file remains readable even
    though the project exposes recording, pointcloud, articulation, and export
    commands from one entry point.
    """
    parser = argparse.ArgumentParser(description="RGB-D to URDF MVP scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Episode-level schema checks.
    validate_parser = subparsers.add_parser("validate-episode", help="Validate an episode manifest")
    validate_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")

    probe_mps_parser = subparsers.add_parser(
        "probe-torch-mps",
        help="Print local PyTorch MPS capability diagnostics",
    )
    probe_mps_parser.add_argument(
        "--no-tensor-test",
        action="store_true",
        help="Skip the real tensor allocation smoke test on the selected device",
    )
    probe_mps_parser.add_argument(
        "--unsafe-force-mps",
        action="store_true",
        help="Attempt an actual MPS tensor even when torch.backends.mps.is_available() is false",
    )

    probe_mjx_parser = subparsers.add_parser(
        "probe-mjx",
        help="Print local JAX + MuJoCo MJX capability diagnostics",
    )
    probe_mjx_parser.add_argument(
        "--jax-platform",
        choices=["cpu", "metal", "auto"],
        default="cpu",
        help="JAX backend selection before import. Default stays on CPU for predictable local/CI behavior.",
    )
    probe_mjx_parser.add_argument(
        "--enable-pjrt-compatibility",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set ENABLE_PJRT_COMPATIBILITY=1 before importing JAX. Defaults to on when --jax-platform metal.",
    )
    probe_mjx_parser.add_argument(
        "--no-rollout-test",
        action="store_true",
        help="Skip the tiny JIT + mjx.step smoke test",
    )

    # End-to-end scaffold pipeline.
    run_parser = subparsers.add_parser("run", help="Run the end-to-end MVP pipeline")
    run_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "run",
        help="Directory for pipeline outputs",
    )
    run_parser.add_argument(
        "--path",
        choices=["feedforward", "optimization"],
        default="optimization",
        help=(
            "Pipeline path. 'feedforward' stops after articulation init; "
            "'optimization' runs the temporal refinement stage before export."
        ),
    )

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

    # 4D pointcloud fusion and inspection.
    fuse_parser = subparsers.add_parser(
        "fuse-pointcloud",
        help="Fuse multi-view depth observations into a time-indexed 4D point cloud",
    )
    fuse_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    fuse_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for fused point cloud outputs (defaults to <episode_dir>/pointcloud_4d)",
    )
    fuse_parser.add_argument("--pixel-stride", type=int, default=4, help="Depth sampling stride")
    fuse_parser.add_argument("--voxel-size-m", type=float, default=0.015, help="Voxel size for view fusion")
    fuse_parser.add_argument(
        "--bbox-margin-m",
        type=float,
        default=0.20,
        help="Margin added to the estimated object bounds",
    )
    fuse_parser.add_argument(
        "--near-depth-band-m",
        type=float,
        default=0.35,
        help="Near-depth band used during object bootstrap",
    )
    fuse_parser.add_argument(
        "--no-shared-pose-fallback",
        action="store_true",
        help="Fail instead of falling back to the central pose when per-view poses are missing",
    )

    viewer_parser = subparsers.add_parser(
        "visualize-pointcloud",
        help="Build a self-contained HTML viewer for a 4D point cloud manifest or PLY",
    )
    viewer_parser.add_argument(
        "input",
        type=Path,
        help="Path to fusion_manifest.json or pointcloud_4d.ply",
    )
    viewer_parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Where to write the viewer HTML",
    )
    viewer_parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=4000,
        help="Maximum rendered points per frame after uniform downsampling",
    )
    viewer_parser.add_argument(
        "--point-radius-px",
        type=float,
        default=2.0,
        help="Default rendered point radius in pixels",
    )
    viewer_parser.add_argument(
        "--part-poses-json",
        type=Path,
        default=None,
        help="Optional part_poses.json override used for joint overlays",
    )
    viewer_parser.add_argument(
        "--part-tracks-json",
        type=Path,
        default=None,
        help="Optional part_tracks.json override used for CoTracker flow overlays",
    )
    viewer_parser.add_argument(
        "--joint-inference-json",
        type=Path,
        default=None,
        help="Optional joint_inference.json override used for joint overlays",
    )

    # Part-level perception and kinematics from pointclouds.
    part_tracker_parser = subparsers.add_parser(
        "track-part-pixels",
        help="Track part-mask seed pixels with CoTracker and backproject them into 3D tracks",
    )
    part_tracker_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    part_tracker_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the 3D track artifact (defaults to <episode_dir>/part_tracks.json)",
    )
    part_tracker_parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="PyTorch device. On Mac, 'auto' uses MPS when available and otherwise CPU.",
    )
    part_tracker_parser.add_argument(
        "--cotracker-repo",
        type=Path,
        default=None,
        help="Optional local co-tracker checkout for torch.hub source=local",
    )
    part_tracker_parser.add_argument(
        "--cotracker-checkpoint",
        type=Path,
        default=None,
        help="Optional local CoTracker checkpoint path used instead of torch.hub downloading weights",
    )
    part_tracker_parser.add_argument(
        "--cotracker-model",
        default="cotracker3_offline",
        help="torch.hub CoTracker model entry point",
    )
    part_tracker_parser.add_argument(
        "--unsafe-force-mps",
        action="store_true",
        help="When combined with --device mps, bypass the availability guard and try MPS anyway.",
    )
    part_tracker_parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Frame used for seed pixels and canonical 3D references; pass -1 to auto-pick per part",
    )
    part_tracker_parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Temporal subsampling stride for tracking. For a 60 Hz episode, 4 tracks at roughly 15 Hz.",
    )
    part_tracker_parser.add_argument("--seed-stride-px", type=int, default=16, help="Pixel stride for mask seed sampling")
    part_tracker_parser.add_argument(
        "--max-tracks-per-part-view",
        type=int,
        default=128,
        help="Maximum seed tracks per part per view",
    )
    part_tracker_parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Minimum CoTracker visibility score required for a valid 3D sample",
    )
    part_tracker_parser.add_argument(
        "--no-part-mask-consistency",
        action="store_true",
        help="Do not require tracked pixels to remain inside the same part mask before backprojection",
    )
    part_tracker_parser.add_argument(
        "--no-backward-tracking",
        action="store_true",
        help="Disable CoTracker backward tracking from the reference frame",
    )
    part_tracker_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stderr progress updates during CoTracker tracking",
    )

    part_pose_parser = subparsers.add_parser(
        "estimate-part-poses",
        help="Estimate per-part per-frame 6D poses from a part-labeled fused pointcloud",
    )
    part_pose_parser.add_argument(
        "input",
        type=Path,
        help="Path to fusion_manifest.json / pointcloud_4d.ply for PCA, or part_tracks.json for track-based poses",
    )
    part_pose_parser.add_argument(
        "--method",
        choices=["auto", "pca", "tracks"],
        default="auto",
        help="Pose estimator. 'pca' uses fused pointcloud PCA; 'tracks' uses 3D CoTracker correspondences.",
    )
    part_pose_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the estimated part pose artifact",
    )
    part_pose_parser.add_argument(
        "--min-points-per-part",
        type=int,
        default=24,
        help="Minimum per-frame point count required to emit a valid pose sample",
    )
    part_pose_parser.add_argument(
        "--anchor-part-id",
        type=int,
        default=None,
        help="Optional part id to use as the relative-pose anchor instead of auto-selecting the base/static part",
    )
    part_pose_parser.add_argument(
        "--min-tracks-per-part",
        type=int,
        default=4,
        help="Minimum visible 3D tracks required per frame when --method tracks is used",
    )

    joint_parser = subparsers.add_parser(
        "infer-joints",
        help="Infer joint type, axis, and pivot from per-part per-frame pose tracks",
    )
    joint_parser.add_argument(
        "input",
        type=Path,
        help="Path to part_poses.json",
    )
    joint_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the inferred joint artifact",
    )
    joint_parser.add_argument(
        "--rotation-threshold-rad",
        type=float,
        default=0.20,
        help="Minimum rotation range required to classify a moving part as revolute",
    )
    joint_parser.add_argument(
        "--translation-threshold-m",
        type=float,
        default=0.02,
        help="Minimum translation range required to classify a moving part as prismatic when rotation is small",
    )
    joint_parser.add_argument(
        "--mujoco-prior",
        choices=["auto", "off", "required"],
        default="auto",
        help=(
            "Use MJCF joint metadata when it is available through the recording manifest. "
            "'auto' uses it as a simulation/debug prior, 'off' keeps pure geometry inference, "
            "and 'required' fails if no prior can be found."
        ),
    )

    inferred_export_parser = subparsers.add_parser(
        "export-inferred-articulation",
        help="Build articulation_artifact + URDF/MJCF from episode, part poses, and inferred joints",
    )
    inferred_export_parser.add_argument("episode", type=Path, help="Path to episode.json")
    inferred_export_parser.add_argument("part_poses", type=Path, help="Path to part_poses.json")
    inferred_export_parser.add_argument("joint_inference", type=Path, help="Path to joint_inference.json")
    inferred_export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for articulation and URDF outputs (defaults to <joint_inference_dir>/inferred_articulation)",
    )

    dynamics_parser = subparsers.add_parser(
        "identify-dynamics",
        help="Fit part masses and joint damping/friction against the observed joint trajectory in MuJoCo",
    )
    dynamics_parser.add_argument("episode", type=Path, help="Path to episode.json")
    dynamics_parser.add_argument("articulation_artifact", type=Path, help="Path to articulation_artifact.json")
    dynamics_parser.add_argument("mjcf", type=Path, help="Path to the MJCF model to optimize")
    dynamics_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for dynamics identification outputs (defaults to <mjcf_dir>/dynamics_identification)",
    )
    dynamics_parser.add_argument("--max-iterations", type=int, default=10, help="Maximum optimization iterations")
    dynamics_parser.add_argument("--learning-rate", type=float, default=0.25, help="Finite-difference gradient step size")
    dynamics_parser.add_argument("--q-weight", type=float, default=1.0, help="Weight for q(t) trajectory matching")
    dynamics_parser.add_argument("--qdot-weight", type=float, default=0.05, help="Weight for qdot(t) trajectory matching")
    dynamics_parser.add_argument("--prior-weight", type=float, default=0.02, help="Regularization weight toward the initial MJCF parameters")
    dynamics_parser.add_argument(
        "--optimize-static-parts",
        action="store_true",
        help="Also include static/root part masses in the optimization vector",
    )
    dynamics_parser.add_argument(
        "--no-optimize-mass",
        action="store_true",
        help="Do not optimize body masses/inertias",
    )
    dynamics_parser.add_argument(
        "--no-optimize-damping",
        action="store_true",
        help="Do not optimize joint damping",
    )
    dynamics_parser.add_argument(
        "--no-optimize-friction",
        action="store_true",
        help="Do not optimize joint frictionloss",
    )
    dynamics_parser.add_argument(
        "--enable-contact",
        action="store_true",
        help=(
            "Enable MJCF geom contacts during dynamics rollout. "
            "By default contacts are disabled because inferred part proxies can overlap."
        ),
    )
    dynamics_parser.add_argument(
        "--render-gl-backend",
        choices=["auto", "cgl", "glfw", "osmesa", "egl", "none"],
        default="auto",
        help="MuJoCo GL backend for rollout comparison rendering. Use 'none' to disable video rendering.",
    )
    dynamics_parser.add_argument(
        "--gravity-mode",
        choices=["model", "zero"],
        default="model",
        help=(
            "Gravity used during dynamics rollout. Use 'zero' to isolate joint inertia/damping "
            "when inferred geometry produces spurious gravity torque."
        ),
    )

    dynamics_mjx_parser = subparsers.add_parser(
        "identify-dynamics-mjx",
        help="Fit part masses and joint damping/friction with JAX autodiff through MuJoCo MJX rollouts",
    )
    dynamics_mjx_parser.add_argument("episode", type=Path, help="Path to episode.json")
    dynamics_mjx_parser.add_argument("articulation_artifact", type=Path, help="Path to articulation_artifact.json")
    dynamics_mjx_parser.add_argument("mjcf", type=Path, help="Path to the MJCF model to optimize")
    dynamics_mjx_parser.add_argument(
        "--jax-platform",
        choices=["cpu", "metal", "auto"],
        default="cpu",
        help="JAX backend selection before importing JAX. Use 'metal' to request Apple GPU execution.",
    )
    dynamics_mjx_parser.add_argument(
        "--enable-pjrt-compatibility",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set ENABLE_PJRT_COMPATIBILITY=1 before importing JAX. Defaults to on when --jax-platform metal.",
    )
    dynamics_mjx_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MJX dynamics identification outputs (defaults to <mjcf_dir>/dynamics_identification_mjx)",
    )
    dynamics_mjx_parser.add_argument("--max-iterations", type=int, default=25, help="Maximum optimization iterations")
    dynamics_mjx_parser.add_argument("--learning-rate", type=float, default=0.05, help="Adam learning rate")
    dynamics_mjx_parser.add_argument("--q-weight", type=float, default=1.0, help="Weight for q(t) trajectory matching")
    dynamics_mjx_parser.add_argument("--qdot-weight", type=float, default=0.05, help="Weight for qdot(t) trajectory matching")
    dynamics_mjx_parser.add_argument("--prior-weight", type=float, default=0.02, help="Regularization weight toward the initial MJCF parameters")
    dynamics_mjx_parser.add_argument(
        "--optimize-static-parts",
        action="store_true",
        help="Also include static/root part masses in the optimization vector",
    )
    dynamics_mjx_parser.add_argument(
        "--no-optimize-mass",
        action="store_true",
        help="Do not optimize body masses/inertias",
    )
    dynamics_mjx_parser.add_argument(
        "--no-optimize-damping",
        action="store_true",
        help="Do not optimize joint damping",
    )
    dynamics_mjx_parser.add_argument(
        "--no-optimize-friction",
        action="store_true",
        help="Do not optimize joint frictionloss",
    )
    dynamics_mjx_parser.add_argument(
        "--enable-contact",
        action="store_true",
        help=(
            "Enable MJCF geom contacts during MJX rollout. "
            "By default contacts are disabled because inferred part proxies can overlap."
        ),
    )
    dynamics_mjx_parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JAX JIT compilation for easier debugging",
    )
    dynamics_mjx_parser.add_argument(
        "--render-gl-backend",
        choices=["auto", "cgl", "glfw", "osmesa", "egl", "none"],
        default="auto",
        help="MuJoCo GL backend for rollout comparison rendering. Use 'none' to disable video rendering.",
    )

    dynamics_plot_parser = subparsers.add_parser(
        "plot-dynamics-identification",
        help="Plot loss and parameter history from a dynamics_identification.json artifact",
    )
    dynamics_plot_parser.add_argument("input", type=Path, help="Path to dynamics_identification.json")
    dynamics_plot_parser.add_argument(
        "--output-svg",
        type=Path,
        default=None,
        help="Where to write the SVG plot (defaults to optimization_history.svg next to the input)",
    )
    dynamics_plot_parser.add_argument(
        "--linear-loss",
        action="store_true",
        help="Use a linear y-axis for the loss panel instead of log10 scale",
    )
    dynamics_plot_parser.add_argument("--width", type=int, default=1280, help="SVG width in pixels")
    dynamics_plot_parser.add_argument("--height", type=int, default=760, help="SVG height in pixels")

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

    # MuJoCo recording and mask generation.
    render_masks_parser = subparsers.add_parser(
        "render-mujoco-masks",
        help="Render target-object segmentation masks for an existing MuJoCo episode",
    )
    render_masks_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    render_masks_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional MJCF model path override",
    )
    render_masks_parser.add_argument(
        "--mask-format",
        choices=["pgm", "png"],
        default="pgm",
        help="Mask output format",
    )
    render_masks_parser.add_argument(
        "--part-segmentation-masks",
        action="store_true",
        help="Also render indexed per-part masks using MuJoCo body/geom priors",
    )
    render_masks_parser.add_argument(
        "--output-episode",
        type=Path,
        default=None,
        help="Optional output episode JSON path (defaults to in-place update)",
    )

    compact_recording_parser = subparsers.add_parser(
        "compact-mujoco-recording",
        help="Rewrite a triview episode to views-only assets and optionally delete assets/concat",
    )
    compact_recording_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    compact_recording_parser.add_argument(
        "--keep-concat-dir",
        action="store_true",
        help="Do not delete assets/concat after rewriting frame paths",
    )
    compact_recording_parser.add_argument(
        "--remove-concat-video",
        action="store_true",
        help="Also delete episode_concat.mp4 when present",
    )
    compact_recording_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be compacted without modifying the episode or deleting files",
    )

    repack_recording_parser = subparsers.add_parser(
        "repack-mujoco-recording",
        help="Convert a recorded MuJoCo episode's referenced ppm/pgm assets to png and update episode.json",
    )
    repack_recording_parser.add_argument("episode", type=Path, help="Path to the episode JSON file")
    repack_recording_parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep the original ppm/pgm files after writing the png copies",
    )
    repack_recording_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without modifying the episode or writing pngs",
    )

    record_parser = subparsers.add_parser(
        "record-mujoco",
        help="Record a sim RGB-D episode from a MuJoCo articulated object",
    )
    record_parser.add_argument("model", type=Path, help="Path to MJCF XML or URDF model")
    record_parser.add_argument(
        "--category",
        type=normalize_category,
        choices=list(SUPPORTED_CATEGORIES),
        required=True,
    )
    record_parser.add_argument("--object-id", required=True, help="Object instance id")
    record_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "recordings",
        help="Directory for recorded episode outputs",
    )
    record_parser.add_argument("--frames", type=int, default=48, help="Number of frames to record")
    record_parser.add_argument("--sim-dt", type=float, default=1.0 / 240.0, help="MuJoCo sim timestep")
    record_parser.add_argument(
        "--frame-dt",
        type=float,
        default=0.1,
        help="Wall-clock interval represented by consecutive frames",
    )
    record_parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Total recording duration in seconds (overrides --frames when combined with --fps)",
    )
    record_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Recorded frame rate in Hz (used with --duration-s; sets frame_dt=1/fps)",
    )
    record_parser.add_argument("--width", type=int, default=640, help="RGB-D frame width")
    record_parser.add_argument("--height", type=int, default=480, help="RGB-D frame height")
    record_parser.add_argument(
        "--rgb-format",
        choices=["ppm", "png"],
        default="ppm",
        help="RGB output format",
    )
    record_parser.add_argument(
        "--depth-format",
        choices=["pgm", "png"],
        default="pgm",
        help="Depth output format (pgm is 16-bit PGM; png is 16-bit PNG)",
    )
    record_parser.add_argument(
        "--write-concat-assets",
        action="store_true",
        help=(
            "In triview mode, also write duplicated horizontally concatenated RGB/depth/mask assets "
            "under assets/concat. Disabled by default to reduce disk usage."
        ),
    )
    record_parser.add_argument(
        "--camera-distance",
        type=float,
        default=1,
        help="Orbit camera distance from lookat",
    )
    record_parser.add_argument(
        "--camera-elevation-deg",
        type=float,
        default=-28.0,
        help="Orbit camera elevation in degrees",
    )
    record_parser.add_argument(
        "--camera-azimuth-start-deg",
        type=float,
        default=50.0,
        help="Orbit camera start azimuth in degrees",
    )
    record_parser.add_argument(
        "--camera-azimuth-span-deg",
        type=float,
        default=90.0,
        help="Orbit camera azimuth sweep span in degrees",
    )
    record_parser.add_argument(
        "--camera-fovy-deg",
        type=float,
        default=90.0,
        help="Vertical field of view in degrees",
    )
    record_parser.add_argument(
        "--camera-mode",
        choices=["orbit", "triview"],
        default="orbit",
        help="Camera mode: orbit sweep or fixed tri-view capture",
    )
    record_parser.add_argument(
        "--camera-triview-spacing-deg",
        type=float,
        default=45.0,
        help="Azimuth spacing (deg) between neighboring views in triview mode",
    )
    record_parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.3),
        help="Orbit camera lookat center in world frame",
    )
    record_parser.add_argument(
        "--perturbation-scale",
        type=float,
        default=1,
        help="Random perturbation force magnitude",
    )

    record_parser.add_argument(
        "--control-mode",
        choices=["track", "free"],
        default="track",
        help="Control mode: 'track' uses PD tracking; 'free' only applies initial conditions then runs with no external force",
    )
    record_parser.add_argument(
        "--random-initial-qpos",
        action="store_true",
        help="In free mode, initialize each controlled hinge/slide joint position randomly within its limits",
    )
    record_parser.add_argument(
        "--auto-initial-qvel-from-limits",
        action="store_true",
        help=(
            "In free mode, infer each controlled joint's initial qvel direction/magnitude from its limits and initial qpos"
        ),
    )
    record_parser.add_argument(
        "--auto-initial-qvel-min-abs",
        type=float,
        default=1.0,
        help="Minimum absolute initial qvel used by --auto-initial-qvel-from-limits when --initial-qvel is 0",
    )
    record_parser.add_argument(
        "--auto-initial-qvel-max-abs",
        type=float,
        default=8.0,
        help="Maximum absolute initial qvel allowed in --auto-initial-qvel-from-limits",
    )
    record_parser.add_argument(
        "--auto-initial-qvel-direction-mode",
        choices=["away-from-qpos0", "toward-lower", "toward-upper"],
        default="away-from-qpos0",
        help="Direction mode for --auto-initial-qvel-from-limits",
    )
    record_parser.add_argument(
        "--initial-qvel",
        type=float,
        default=0.0,
        help="Initial joint velocity for the selected hinge/slide DOF",
    )
    record_parser.add_argument(
        "--kick-force",
        type=float,
        default=0.0,
        help="External force/torque applied along the selected DOF during the kick window",
    )
    record_parser.add_argument(
        "--kick-start-s",
        type=float,
        default=0.0,
        help="Kick window start time in seconds (simulation time)",
    )
    record_parser.add_argument(
        "--kick-duration-s",
        type=float,
        default=0.0,
        help="Kick window duration in seconds (0 disables kick)",
    )
    record_parser.add_argument(
        "--excitation-mode",
        choices=["none", "pulse", "prbs", "sine"],
        default="none",
        help=(
            "Known generalized-force excitation for dynamics ID. "
            "Use with --control-mode free to record a forced-response episode."
        ),
    )
    record_parser.add_argument(
        "--excitation-force",
        type=float,
        default=0.0,
        help="Excitation generalized force amplitude, torque for hinge joints and force for slide joints",
    )
    record_parser.add_argument(
        "--excitation-start-s",
        type=float,
        default=0.0,
        help="Excitation window start time in seconds",
    )
    record_parser.add_argument(
        "--excitation-duration-s",
        type=float,
        default=0.0,
        help="Excitation window duration in seconds",
    )
    record_parser.add_argument(
        "--excitation-period-s",
        type=float,
        default=0.25,
        help="PRBS segment duration in seconds",
    )
    record_parser.add_argument(
        "--excitation-frequency-hz",
        type=float,
        default=1.0,
        help="Sine excitation frequency in Hz",
    )
    record_parser.add_argument(
        "--no-dynamics-log",
        action="store_true",
        help="Do not write sim-rate dynamics_log.jsonl with q, qvel, and applied generalized force",
    )
    record_parser.add_argument("--control-kp", type=float, default=30.0, help="Joint tracking P gain")
    record_parser.add_argument("--control-kd", type=float, default=3.0, help="Joint tracking D gain")
    record_parser.add_argument("--seed", type=int, default=0, help="Random seed")

    record_parser.add_argument(
        "--joint-name",
        type=str,
        default=None,
        help="Name of the hinge/slide joint to drive/record (defaults to first hinge/slide)",
    )
    record_parser.add_argument(
        "--joint-id",
        type=int,
        default=None,
        help="MuJoCo joint id of the hinge/slide joint to drive/record (defaults to first hinge/slide)",
    )
    record_parser.add_argument(
        "--all-joints",
        action="store_true",
        help="Apply initial velocity/forces to all hinge+slide joints (requires --control-mode free)",
    )

    record_parser.add_argument(
        "--video",
        action="store_true",
        help="Also write an mp4 RGB video into the recorded episode directory",
    )
    record_parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Video frame rate override (defaults to 1/frame_dt)",
    )
    record_parser.add_argument(
        "--segmentation-masks",
        action="store_true",
        help="Also render binary target-object masks for each frame/view",
    )
    record_parser.add_argument(
        "--part-segmentation-masks",
        action="store_true",
        help="Also render indexed per-part masks using MuJoCo body/geom priors",
    )
    record_parser.add_argument(
        "--mask-format",
        choices=["pgm", "png"],
        default="pgm",
        help="Mask output format when --segmentation-masks is enabled",
    )
    record_parser.add_argument(
        "--disable-target-mesh-collision",
        action="store_true",
        help="Disable collision on the target object's mesh geoms while keeping them rendered",
    )
    record_parser.add_argument(
        "--hide-clear-meshes",
        action="store_true",
        help="Hide target-object mesh geoms whose names contain 'Clear' from RGB/depth/mask rendering",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    In addition to normal subcommands, the CLI accepts a single YAML/JSON config
    file path or `--config config.yaml`, which expands into ordinary argv before
    argparse handles the command.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        expanded_argv = expand_config_argv(raw_argv, parser)
    except YAMLSubsetError as exc:
        parser.error(str(exc))
    args = parser.parse_args(expanded_argv)

    if args.command == "validate-episode":
        episode = load_episode(args.episode)
        errors = validate_episode(episode)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"Episode '{episode.object_instance_id}' is valid for category '{episode.category}'.")
        return 0

    if args.command == "probe-torch-mps":
        from .core.torch_probe import probe_torch_mps

        print(
            json.dumps(
                probe_torch_mps(
                    run_tensor_test=not bool(args.no_tensor_test),
                    unsafe_force_mps=bool(args.unsafe_force_mps),
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "probe-mjx":
        from .core.mjx_probe import probe_mjx

        print(
            json.dumps(
                probe_mjx(
                    run_rollout_test=not bool(args.no_rollout_test),
                    jax_platform=str(args.jax_platform),
                    enable_pjrt_compatibility=args.enable_pjrt_compatibility,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "run":
        episode = load_episode(args.episode)
        errors = validate_episode(episode)
        if errors:
            for error in errors:
                print(error)
            return 1
        result = RGBDToURDFPipeline().run(
            episode,
            args.output_dir,
            path_mode=str(args.path),
        )
        print(json.dumps({"path_mode": str(args.path), **result.to_dict()}, indent=2))
        return 0

    if args.command == "run-articulation-batch":
        from .batch.articulation_pipeline import ArticulationBatchConfig, ArticulationBatchRunner

        if args.jobs < 1:
            parser.error("--jobs must be a positive integer")
        result = ArticulationBatchRunner(
            ArticulationBatchConfig(
                manifest_path=args.manifest,
                resume=bool(args.resume),
                skip_existing=bool(args.skip_existing),
                jobs=int(args.jobs),
                output_root=args.output_root,
                batch_log_dir=args.batch_log_dir,
                torch_home=args.torch_home,
                record_config=args.record_config,
                track_config=args.track_config,
                cotracker_repo=args.cotracker_repo,
                cotracker_checkpoint=args.cotracker_checkpoint,
                track_device=args.track_device,
                tracking_jobs=args.tracking_jobs,
                fuse_pixel_stride=max(1, int(args.fuse_pixel_stride)),
                fuse_voxel_size_m=float(args.fuse_voxel_size_m),
                min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                mujoco_prior_mode=str(args.mujoco_prior),
                generate_viewer=not bool(args.no_generate_viewer),
                dynamics_backend=str(args.dynamics_backend),
                dynamics_config=args.dynamics_config,
                dynamics_jobs=args.dynamics_jobs,
                dynamics_jax_platform=args.dynamics_jax_platform,
                dynamics_enable_pjrt_compatibility=args.dynamics_enable_pjrt_compatibility,
            )
        ).run()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "fuse-pointcloud":
        from .perception.pointcloud_fusion import EpisodePointCloudFuser, PointCloudFusionConfig

        manifest_path = EpisodePointCloudFuser(
            PointCloudFusionConfig(
                episode_path=args.episode,
                output_dir=args.output_dir,
                pixel_stride=max(1, int(args.pixel_stride)),
                voxel_size_m=float(args.voxel_size_m),
                bbox_margin_m=float(args.bbox_margin_m),
                near_depth_band_m=float(args.near_depth_band_m),
                allow_shared_pose_fallback=not bool(args.no_shared_pose_fallback),
            )
        ).fuse()
        print(json.dumps({"manifest_path": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "visualize-pointcloud":
        from .perception.pointcloud_viz import PointCloudViewerBuilder, PointCloudVisualizationConfig

        html_path = PointCloudViewerBuilder(
            PointCloudVisualizationConfig(
                input_path=args.input,
                output_html=args.output_html,
                max_points_per_frame=max(1, int(args.max_points_per_frame)),
                point_radius_px=float(args.point_radius_px),
                part_pose_path=args.part_poses_json,
                part_track_path=args.part_tracks_json,
                joint_inference_path=args.joint_inference_json,
            )
        ).build()
        print(json.dumps({"viewer_html": str(html_path.resolve())}, indent=2))
        return 0

    if args.command == "track-part-pixels":
        from .perception.part_tracking import PartPixelTracker, PartPixelTrackingConfig

        output_json = PartPixelTracker(
            PartPixelTrackingConfig(
                episode_path=args.episode,
                output_json=args.output_json,
                device=str(args.device),
                cotracker_repo=args.cotracker_repo,
                cotracker_checkpoint=args.cotracker_checkpoint,
                cotracker_model=str(args.cotracker_model),
                unsafe_force_mps=bool(args.unsafe_force_mps),
                reference_frame=int(args.reference_frame),
                frame_stride=max(1, int(args.frame_stride)),
                seed_stride_px=max(1, int(args.seed_stride_px)),
                max_tracks_per_part_view=max(1, int(args.max_tracks_per_part_view)),
                visibility_threshold=float(args.visibility_threshold),
                require_part_mask_consistency=not bool(args.no_part_mask_consistency),
                allow_backward_tracking=not bool(args.no_backward_tracking),
                show_progress=not bool(args.no_progress),
            )
        ).track()
        print(json.dumps({"part_tracks_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "estimate-part-poses":
        method = str(args.method)
        if method == "auto" and args.input.suffix.lower() == ".json":
            payload = load_json(args.input)
            method = "tracks" if isinstance(payload.get("tracks"), list) else "pca"
        elif method == "auto":
            method = "pca"

        if method == "tracks":
            from .perception.part_tracking import TrackPartPoseEstimationConfig, TrackPartPoseEstimator

            output_json = TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(
                    input_path=args.input,
                    output_json=args.output_json,
                    min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                    anchor_part_id=args.anchor_part_id,
                )
            ).estimate()
        else:
            from .perception.part_pose import PartPoseEstimationConfig, PartPoseEstimator

            output_json = PartPoseEstimator(
                PartPoseEstimationConfig(
                    input_path=args.input,
                    output_json=args.output_json,
                    min_points_per_part=max(1, int(args.min_points_per_part)),
                    anchor_part_id=args.anchor_part_id,
                )
            ).estimate()
        print(json.dumps({"part_pose_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "infer-joints":
        from .kinematics.joint_inference import JointInferenceConfig, JointInferencer

        output_json = JointInferencer(
            JointInferenceConfig(
                input_path=args.input,
                output_json=args.output_json,
                rotation_threshold_rad=float(args.rotation_threshold_rad),
                translation_threshold_m=float(args.translation_threshold_m),
                mujoco_prior=args.mujoco_prior,
            )
        ).infer()
        print(json.dumps({"joint_inference_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "export-inferred-articulation":
        from .kinematics.inferred_articulation import InferredArticulationPipeline, InferredArticulationPipelineConfig

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.joint_inference.resolve().parent / "inferred_articulation"
        )
        result = InferredArticulationPipeline().run(
            InferredArticulationPipelineConfig(
                episode_path=args.episode,
                part_pose_path=args.part_poses,
                joint_inference_path=args.joint_inference,
                output_dir=output_dir,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "identify-dynamics":
        if args.render_gl_backend not in {"auto", "none"}:
            import os

            os.environ["MUJOCO_GL"] = str(args.render_gl_backend)
        from .dynamics.system_id import DynamicsIdentificationConfig, DynamicsIdentifier

        artifact_path = DynamicsIdentifier().run(
            DynamicsIdentificationConfig(
                episode_path=args.episode,
                articulation_artifact_path=args.articulation_artifact,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                max_iterations=max(1, int(args.max_iterations)),
                learning_rate=float(args.learning_rate),
                q_weight=float(args.q_weight),
                qdot_weight=float(args.qdot_weight),
                prior_weight=float(args.prior_weight),
                optimize_static_parts=bool(args.optimize_static_parts),
                optimize_mass=not bool(args.no_optimize_mass),
                optimize_damping=not bool(args.no_optimize_damping),
                optimize_friction=not bool(args.no_optimize_friction),
                enable_contact=bool(args.enable_contact),
                render_gl_backend=str(args.render_gl_backend),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"dynamics_identification_artifact": str(artifact_path.resolve())}, indent=2))
        return 0

    if args.command == "identify-dynamics-mjx":
        if args.render_gl_backend not in {"auto", "none"}:
            import os

            os.environ["MUJOCO_GL"] = str(args.render_gl_backend)
        from .core.jax_runtime import configure_jax_runtime
        from .dynamics.system_id_mjx import MJXDynamicsIdentificationConfig, MJXDynamicsIdentifier

        configure_jax_runtime(
            platform=str(args.jax_platform),
            enable_pjrt_compatibility=args.enable_pjrt_compatibility,
        )
        artifact_path = MJXDynamicsIdentifier().run(
            MJXDynamicsIdentificationConfig(
                episode_path=args.episode,
                articulation_artifact_path=args.articulation_artifact,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                max_iterations=max(1, int(args.max_iterations)),
                learning_rate=float(args.learning_rate),
                q_weight=float(args.q_weight),
                qdot_weight=float(args.qdot_weight),
                prior_weight=float(args.prior_weight),
                optimize_static_parts=bool(args.optimize_static_parts),
                optimize_mass=not bool(args.no_optimize_mass),
                optimize_damping=not bool(args.no_optimize_damping),
                optimize_friction=not bool(args.no_optimize_friction),
                enable_contact=bool(args.enable_contact),
                jit=not bool(args.no_jit),
                jax_platform=str(args.jax_platform),
                enable_pjrt_compatibility=args.enable_pjrt_compatibility,
                render_gl_backend=str(args.render_gl_backend),
            )
        )
        print(json.dumps({"dynamics_identification_artifact": str(artifact_path.resolve())}, indent=2))
        return 0

    if args.command == "plot-dynamics-identification":
        from .dynamics.plotting import DynamicsOptimizationPlotConfig, DynamicsOptimizationPlotter

        output_svg = DynamicsOptimizationPlotter().plot(
            DynamicsOptimizationPlotConfig(
                input_path=args.input,
                output_svg=args.output_svg,
                log_loss=not bool(args.linear_loss),
                width=max(720, int(args.width)),
                height=max(480, int(args.height)),
            )
        )
        print(json.dumps({"optimization_history_svg": str(output_svg.resolve())}, indent=2))
        return 0

    if args.command == "hunyuan3d-generate":
        from .perception.hunyuan3d_client import Hunyuan3DClient, Hunyuan3DGenerationConfig

        output_path = Hunyuan3DClient(
            server_url=args.server_url,
            api_token=args.api_token,
            timeout_s=min(float(args.timeout_s), 120.0),
        ).generate(
            Hunyuan3DGenerationConfig(
                server_url=args.server_url,
                output_path=args.output,
                image_path=args.image,
                image_views=args.image_views,
                text=args.text,
                mesh_path=args.mesh,
                mode=args.mode,
                texture=bool(args.texture),
                seed=int(args.seed),
                output_type=args.type,
                octree_resolution=args.octree_resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                face_count=args.face_count,
                timeout_s=float(args.timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                api_token=args.api_token,
            )
        )
        print(json.dumps({"generated_model_path": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "render-mujoco-masks":
        from .sim.mujoco_recorder import MuJoCoMaskRenderConfig, MuJoCoMaskRenderer

        episode_path = MuJoCoMaskRenderer(
            MuJoCoMaskRenderConfig(
                episode_path=args.episode,
                model_path=args.model,
                mask_format=args.mask_format,
                part_segmentation_masks=bool(args.part_segmentation_masks),
                output_episode_path=args.output_episode,
            )
        ).render()
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    if args.command == "compact-mujoco-recording":
        from .sim.mujoco_recorder import MuJoCoEpisodeCompactConfig, MuJoCoEpisodeCompactor

        result = MuJoCoEpisodeCompactor(
            MuJoCoEpisodeCompactConfig(
                episode_path=args.episode,
                remove_concat_dir=not bool(args.keep_concat_dir),
                remove_concat_video=bool(args.remove_concat_video),
                dry_run=bool(args.dry_run),
            )
        ).compact()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "repack-mujoco-recording":
        from .sim.mujoco_recorder import MuJoCoEpisodeRepackConfig, MuJoCoEpisodeRepacker

        result = MuJoCoEpisodeRepacker(
            MuJoCoEpisodeRepackConfig(
                episode_path=args.episode,
                keep_originals=bool(args.keep_originals),
                dry_run=bool(args.dry_run),
            )
        ).repack()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "record-mujoco":
        from .sim.mujoco_recorder import MuJoCoEpisodeRecorder, MuJoCoRecordConfig

        frame_count = int(args.frames)
        frame_dt = float(args.frame_dt)
        if args.duration_s is not None or args.fps is not None:
            if args.duration_s is None or args.fps is None:
                parser.error("--duration-s and --fps must be provided together")
            if args.duration_s <= 0.0:
                parser.error("--duration-s must be positive")
            if args.fps <= 0.0:
                parser.error("--fps must be positive")
            frame_dt = 1.0 / float(args.fps)
            frame_count = max(1, int(round(float(args.duration_s) * float(args.fps))))

        if args.kick_start_s < 0.0:
            parser.error("--kick-start-s must be >= 0")
        if args.kick_duration_s < 0.0:
            parser.error("--kick-duration-s must be >= 0")
        if args.kick_duration_s == 0.0 and args.kick_force != 0.0:
            parser.error("--kick-force requires --kick-duration-s > 0")
        if args.excitation_start_s < 0.0:
            parser.error("--excitation-start-s must be >= 0")
        if args.excitation_duration_s < 0.0:
            parser.error("--excitation-duration-s must be >= 0")
        if args.excitation_mode != "none" and args.excitation_duration_s <= 0.0:
            parser.error("--excitation-duration-s must be > 0 when --excitation-mode is not none")
        if args.excitation_period_s <= 0.0:
            parser.error("--excitation-period-s must be positive")
        if args.excitation_frequency_hz <= 0.0:
            parser.error("--excitation-frequency-hz must be positive")

        if args.all_joints and (args.joint_name is not None or args.joint_id is not None):
            parser.error("--all-joints cannot be combined with --joint-name/--joint-id")
        if args.joint_name is not None and args.joint_id is not None:
            parser.error("Provide only one of --joint-name or --joint-id")

        config = MuJoCoRecordConfig(
            model_path=args.model,
            output_dir=args.output_dir,
            object_instance_id=args.object_id,
            category=args.category,
            frame_count=frame_count,
            sim_dt=args.sim_dt,
            frame_dt=frame_dt,
            width=args.width,
            height=args.height,
            rgb_format=args.rgb_format,
            depth_format=args.depth_format,
            write_concat_assets=bool(args.write_concat_assets),
            camera_distance=args.camera_distance,
            camera_elevation_deg=args.camera_elevation_deg,
            camera_azimuth_start_deg=args.camera_azimuth_start_deg,
            camera_azimuth_span_deg=args.camera_azimuth_span_deg,
            camera_fovy_deg=args.camera_fovy_deg,
            camera_mode=args.camera_mode,
            camera_triview_spacing_deg=args.camera_triview_spacing_deg,
            lookat=tuple(args.lookat),
            perturbation_scale=args.perturbation_scale,
            control_kp=args.control_kp,
            control_kd=args.control_kd,
            seed=args.seed,
            control_mode=args.control_mode,
            random_initial_qpos=bool(args.random_initial_qpos),
            auto_initial_qvel_from_limits=bool(args.auto_initial_qvel_from_limits),
            auto_initial_qvel_direction_mode=args.auto_initial_qvel_direction_mode,
            auto_initial_qvel_min_abs=float(args.auto_initial_qvel_min_abs),
            auto_initial_qvel_max_abs=float(args.auto_initial_qvel_max_abs),
            initial_joint_qvel=args.initial_qvel,
            kick_force=args.kick_force,
            kick_start_s=args.kick_start_s,
            kick_duration_s=args.kick_duration_s,
            excitation_mode=args.excitation_mode,
            excitation_force=float(args.excitation_force),
            excitation_start_s=float(args.excitation_start_s),
            excitation_duration_s=float(args.excitation_duration_s),
            excitation_period_s=float(args.excitation_period_s),
            excitation_frequency_hz=float(args.excitation_frequency_hz),
            write_dynamics_log=not bool(args.no_dynamics_log),
            make_video=bool(args.video),
            video_fps=args.video_fps,
            segmentation_masks=bool(args.segmentation_masks),
            part_segmentation_masks=bool(args.part_segmentation_masks),
            mask_format=args.mask_format,
            disable_target_mesh_collision=bool(args.disable_target_mesh_collision),
            hide_clear_meshes=bool(args.hide_clear_meshes),
            joint_name=args.joint_name,
            joint_id=args.joint_id,
            all_joints=bool(args.all_joints),
        )
        episode_path = MuJoCoEpisodeRecorder(config).record()
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
