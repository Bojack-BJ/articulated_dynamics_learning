from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
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
