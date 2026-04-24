from __future__ import annotations

import argparse
import json
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
        "--reference-frame",
        type=int,
        default=0,
        help="Frame used for seed pixels and canonical 3D references; pass -1 to auto-pick per part",
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
                reference_frame=int(args.reference_frame),
                seed_stride_px=max(1, int(args.seed_stride_px)),
                max_tracks_per_part_view=max(1, int(args.max_tracks_per_part_view)),
                visibility_threshold=float(args.visibility_threshold),
                require_part_mask_consistency=not bool(args.no_part_mask_consistency),
                allow_backward_tracking=not bool(args.no_backward_tracking),
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
