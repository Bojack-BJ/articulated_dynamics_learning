from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    segment_masks_parser = subparsers.add_parser(
        "segment-episode-masks",
        help="Run a mask provider over an episode and write object/part masks back into episode.json",
    )
    segment_masks_parser.add_argument("episode", type=Path, help="Path to episode.json")
    segment_masks_parser.add_argument(
        "--provider",
        choices=["mask-dir", "external-command", "sam2", "sam3"],
        default="mask-dir",
        help="Mask provider backend. Use external-command for SAM2/SAM3/PartSAM wrappers.",
    )
    segment_masks_parser.add_argument(
        "--mask-kind",
        choices=["object", "part"],
        default="object",
        help="Write binary object masks or indexed part masks",
    )
    segment_masks_parser.add_argument("--mask-dir", type=Path, default=None, help="Directory of precomputed masks")
    segment_masks_parser.add_argument(
        "--command",
        dest="provider_command",
        default=None,
        help=(
            "External command template. Available fields: {rgb_path}, {output_mask_path}, "
            "{frame_index}, {view_index}."
        ),
    )
    segment_masks_parser.add_argument(
        "--sam2-root",
        type=Path,
        default=Path("sam2"),
        help="Path to the SAM2 repository checkout used for native --provider sam2",
    )
    segment_masks_parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 Hydra model config name",
    )
    segment_masks_parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=None,
        help="Path to a SAM2 checkpoint .pt file",
    )
    segment_masks_parser.add_argument(
        "--sam2-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="Device for native SAM2. auto prefers MPS, then CUDA, then CPU.",
    )
    segment_masks_parser.add_argument(
        "--sam2-prompt-mode",
        choices=["full-image-box", "center-box", "box", "auto-masks"],
        default="full-image-box",
        help="Prompt strategy for native SAM2 object masks",
    )
    segment_masks_parser.add_argument(
        "--sam2-box-xyxy",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Explicit SAM2 box prompt when --sam2-prompt-mode box",
    )
    segment_masks_parser.add_argument(
        "--sam2-center-box-scale",
        type=float,
        default=0.75,
        help="Image-relative box size for --sam2-prompt-mode center-box",
    )
    segment_masks_parser.add_argument(
        "--sam2-single-mask",
        action="store_true",
        help="Disable SAM2 multimask output for prompted modes",
    )
    segment_masks_parser.add_argument(
        "--sam2-mask-selection",
        choices=["best-iou", "smallest", "largest", "index"],
        default="best-iou",
        help="How to select one SAM2 mask when a prompt returns multiple candidates",
    )
    segment_masks_parser.add_argument(
        "--sam2-mask-index",
        type=int,
        default=None,
        help="Explicit candidate index when --sam2-mask-selection index",
    )
    segment_masks_parser.add_argument("--sam3-checkpoint", type=Path, default=None, help="Path to a SAM3 checkpoint .pt file")
    segment_masks_parser.add_argument(
        "--sam3-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="Device for native SAM3. auto prefers MPS, then CUDA, then CPU.",
    )
    segment_masks_parser.add_argument(
        "--sam3-prompt-mode",
        choices=["full-image-box", "center-box", "box"],
        default="full-image-box",
        help="Prompt strategy for native SAM3 object masks",
    )
    segment_masks_parser.add_argument(
        "--sam3-box-xyxy",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Explicit SAM3 box prompt when --sam3-prompt-mode box",
    )
    segment_masks_parser.add_argument(
        "--sam3-center-box-scale",
        type=float,
        default=0.75,
        help="Image-relative box size for --sam3-prompt-mode center-box",
    )
    segment_masks_parser.add_argument(
        "--sam3-mask-selection",
        choices=["best-score", "smallest", "largest", "index"],
        default="largest",
        help="How to select one SAM3 mask when a prompt returns multiple candidates",
    )
    segment_masks_parser.add_argument("--sam3-mask-index", type=int, default=None)
    segment_masks_parser.add_argument("--sam3-conf", type=float, default=0.05, help="SAM3 confidence threshold")
    segment_masks_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Mask artifact directory (defaults to <episode_dir>/assets/masks/<provider>)",
    )
    segment_masks_parser.add_argument(
        "--output-episode",
        type=Path,
        default=None,
        help="Where to write the updated episode JSON (defaults to in-place)",
    )
    segment_masks_parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    segment_masks_parser.add_argument("--start-frame", type=int, default=0, help="First frame index to process")
    segment_masks_parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames")
    segment_masks_parser.add_argument("--view-indices", type=int, nargs="+", default=None, help="Views to process")
    segment_masks_parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Optional threshold to binarize provider masks before writing/evaluating",
    )
    segment_masks_parser.add_argument("--force", action="store_true", help="Regenerate masks that already exist")

    segment_masks_batch_parser = subparsers.add_parser(
        "segment-episode-masks-batch",
        help="Batch run a mask provider over episode.json recordings",
    )
    segment_masks_batch_parser.add_argument(
        "manifest",
        type=Path,
        help="TSV with episode_path and optional object_id/output_episode/output_dir/provider override columns",
    )
    segment_masks_batch_parser.add_argument("--jobs", type=int, default=1, help="Parallel segmentation jobs")
    segment_masks_batch_parser.add_argument(
        "--provider",
        choices=["mask-dir", "external-command", "sam2", "sam3"],
        default="sam2",
        help="Default mask provider backend",
    )
    segment_masks_batch_parser.add_argument("--output-root", type=Path, default=None)
    segment_masks_batch_parser.add_argument("--sam2-root", type=Path, default=Path("sam2"))
    segment_masks_batch_parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    segment_masks_batch_parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    segment_masks_batch_parser.add_argument(
        "--sam2-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    segment_masks_batch_parser.add_argument(
        "--sam2-prompt-mode",
        choices=["full-image-box", "center-box", "box", "auto-masks"],
        default="full-image-box",
    )
    segment_masks_batch_parser.add_argument("--sam2-box-xyxy", type=float, nargs=4, default=None)
    segment_masks_batch_parser.add_argument("--sam2-center-box-scale", type=float, default=0.75)
    segment_masks_batch_parser.add_argument("--sam2-single-mask", action="store_true")
    segment_masks_batch_parser.add_argument(
        "--sam2-mask-selection",
        choices=["best-iou", "smallest", "largest", "index"],
        default="best-iou",
    )
    segment_masks_batch_parser.add_argument("--sam2-mask-index", type=int, default=None)
    segment_masks_batch_parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    segment_masks_batch_parser.add_argument(
        "--sam3-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    segment_masks_batch_parser.add_argument(
        "--sam3-prompt-mode",
        choices=["full-image-box", "center-box", "box"],
        default="full-image-box",
    )
    segment_masks_batch_parser.add_argument("--sam3-box-xyxy", type=float, nargs=4, default=None)
    segment_masks_batch_parser.add_argument("--sam3-center-box-scale", type=float, default=0.75)
    segment_masks_batch_parser.add_argument(
        "--sam3-mask-selection",
        choices=["best-score", "smallest", "largest", "index"],
        default="largest",
    )
    segment_masks_batch_parser.add_argument("--sam3-mask-index", type=int, default=None)
    segment_masks_batch_parser.add_argument("--sam3-conf", type=float, default=0.05)
    segment_masks_batch_parser.add_argument("--mask-kind", choices=["object", "part"], default="object")
    segment_masks_batch_parser.add_argument("--frame-stride", type=int, default=1)
    segment_masks_batch_parser.add_argument("--start-frame", type=int, default=0)
    segment_masks_batch_parser.add_argument("--max-frames", type=int, default=None)
    segment_masks_batch_parser.add_argument("--view-indices", type=int, nargs="+", default=None)
    segment_masks_batch_parser.add_argument("--threshold", type=int, default=0)
    segment_masks_batch_parser.add_argument("--force", action="store_true")

    propagate_masks_parser = subparsers.add_parser(
        "propagate-episode-masks",
        help="Propagate an object or indexed part mask from one reference frame through an episode using CoTracker",
    )
    propagate_masks_parser.add_argument("episode", type=Path, help="Episode containing a reference object/part mask")
    propagate_masks_parser.add_argument("--output-episode", type=Path, default=None)
    propagate_masks_parser.add_argument("--output-dir", type=Path, default=None)
    propagate_masks_parser.add_argument(
        "--backend",
        choices=["sam2-video", "cotracker-sparse"],
        default="sam2-video",
        help="Dense SAM2 video propagation is the default; CoTracker sparse propagation is a debug fallback.",
    )
    propagate_masks_parser.add_argument("--mask-kind", choices=["object", "part"], default="object")
    propagate_masks_parser.add_argument("--sam2-root", type=Path, default=Path("sam2"))
    propagate_masks_parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    propagate_masks_parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    propagate_masks_parser.add_argument(
        "--sam2-device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    propagate_masks_parser.add_argument(
        "--sam2-part-mode",
        choices=["independent", "joint"],
        default="independent",
        help=(
            "For indexed part masks, propagate each part independently before merging logits. "
            "Use joint to keep SAM2's native multi-object competition."
        ),
    )
    propagate_masks_parser.add_argument(
        "--sam2-offload-video-to-cpu",
        action="store_true",
        help="Save accelerator memory by keeping loaded video frames on CPU during SAM2 propagation",
    )
    propagate_masks_parser.add_argument(
        "--sam2-offload-state-to-cpu",
        action="store_true",
        help="Save accelerator memory by keeping SAM2 state on CPU; slower but useful for long videos",
    )
    propagate_masks_parser.add_argument("--reference-frame", type=int, default=0)
    propagate_masks_parser.add_argument("--frame-stride", type=int, default=1)
    propagate_masks_parser.add_argument("--view-indices", type=int, nargs="+", default=None)
    propagate_masks_parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    propagate_masks_parser.add_argument("--cotracker-repo", type=Path, default=None)
    propagate_masks_parser.add_argument("--cotracker-checkpoint", type=Path, default=None)
    propagate_masks_parser.add_argument("--cotracker-model", default="cotracker3_offline")
    propagate_masks_parser.add_argument("--seed-stride-px", type=int, default=12)
    propagate_masks_parser.add_argument("--max-tracks-per-view", type=int, default=512)
    propagate_masks_parser.add_argument("--visibility-threshold", type=float, default=0.5)
    propagate_masks_parser.add_argument("--mask-radius-px", type=int, default=8)
    propagate_masks_parser.add_argument("--threshold", type=int, default=0)
    propagate_masks_parser.add_argument("--force", action="store_true")

    evaluate_masks_parser = subparsers.add_parser(
        "evaluate-episode-masks",
        help="Compare predicted episode masks against reference masks with IoU/precision/recall",
    )
    evaluate_masks_parser.add_argument("predicted_episode", type=Path, help="Episode containing predicted masks")
    evaluate_masks_parser.add_argument("reference_episode", type=Path, help="Episode containing reference masks")
    evaluate_masks_parser.add_argument(
        "--mask-kind",
        choices=["object", "part"],
        default="object",
        help="Evaluate object masks or indexed part masks as foreground",
    )
    evaluate_masks_parser.add_argument("--output-json", type=Path, default=None, help="Where to write metrics")
    evaluate_masks_parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold")

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
        "--no-track-residual-type",
        action="store_true",
        help="Disable 3D track replay residual comparison for revolute/prismatic type selection",
    )
    joint_parser.add_argument(
        "--track-residual-requires-pose-candidate",
        action="store_true",
        help=(
            "Require the pose-derived rotation/translation threshold candidate before applying a track-residual "
            "type decision. This reproduces the older pose-gated behavior."
        ),
    )
    joint_parser.add_argument(
        "--no-track-translation-axis",
        action="store_true",
        help="Disable prismatic axis estimation from 3D track endpoint displacement and use SE(3) pose translations only",
    )
    joint_parser.add_argument(
        "--track-residual-decision-ratio",
        type=float,
        default=0.85,
        help="Apply track-residual type decision when best RMSE / other RMSE is at or below this ratio",
    )
    joint_parser.add_argument(
        "--min-track-residual-samples",
        type=int,
        default=20,
        help="Minimum 3D track samples required before track residual can override threshold-priority type selection",
    )
    joint_parser.add_argument(
        "--min-track-residual-tracks",
        type=int,
        default=12,
        help="Minimum distinct tracks required before track residual can override threshold-priority type selection",
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
