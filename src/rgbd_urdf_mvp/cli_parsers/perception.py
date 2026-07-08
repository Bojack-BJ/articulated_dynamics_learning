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

    object_mask_viz_parser = subparsers.add_parser(
        "visualize-object-mask-diagnostics",
        help="Write diagnostic PLY files for object-mask motion clusters, joints, and local split candidates",
    )
    object_mask_viz_parser.add_argument("motion_tracks", type=Path, help="Path to motion_part_tracks.json")
    object_mask_viz_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostic PLY files")
    object_mask_viz_parser.add_argument("--joint-inference", type=Path, default=None, help="Optional joint_inference.json")
    object_mask_viz_parser.add_argument("--evaluation-json", type=Path, default=None, help="Optional object_mask_kinematic_evaluation.json")
    object_mask_viz_parser.add_argument("--local-split-summary", type=Path, default=None, help="Optional local_split_summary.json")
    object_mask_viz_parser.add_argument("--candidate-eval", type=Path, default=None, help="Optional local split candidate evaluation JSON")
    object_mask_viz_parser.add_argument(
        "--frame",
        choices=["first", "median", "last"],
        default="first",
        help="Track sample used for static PLY point locations",
    )
    object_mask_viz_parser.add_argument("--top-n-candidates", type=int, default=8)
    object_mask_viz_parser.add_argument("--axis-length-m", type=float, default=0.5)
    object_mask_viz_parser.add_argument("--viz-frame", choices=["camera", "world", "mujoco", "auto"], default="world")
    object_mask_viz_parser.add_argument(
        "--axis-remap",
        default="x,z,-y",
        help="Visualization-only axis remap, e.g. x,z,-y to convert image-style y-down coordinates to z-up",
    )
    object_mask_viz_parser.add_argument("--flow-min-motion", type=float, default=0.005)
    object_mask_viz_parser.add_argument("--flow-max-tracks", type=int, default=2000)
    object_mask_viz_parser.add_argument("--flow-subsample", type=int, default=1)
    object_mask_viz_parser.add_argument("--flow-scale", type=float, default=1.0)
    object_mask_viz_parser.add_argument("--make-matplotlib", action="store_true")
    object_mask_viz_parser.add_argument(
        "--plot-projections",
        default="xz,xy",
        help="Comma-separated projection list from xy,xz,yz, or all",
    )
    object_mask_viz_parser.add_argument("--make-animation", action="store_true")
    object_mask_viz_parser.add_argument("--animation-fps", type=int, default=12)
    object_mask_viz_parser.add_argument("--animation-max-tracks", type=int, default=1000)
    object_mask_viz_parser.add_argument(
        "--animation-color-by",
        choices=["pred_cluster", "gt_part", "motion_magnitude"],
        default="pred_cluster",
        help="Color temporal track replay by predicted cluster, GT part, or current motion magnitude",
    )

    object_mask_flow_html_parser = subparsers.add_parser(
        "visualize-object-mask-flow-html",
        help="Write an interactive Plotly HTML viewer for object-mask 3D tracks with time and track-count sliders",
    )
    object_mask_flow_html_parser.add_argument("motion_tracks", type=Path, help="Path to motion_part_tracks.json")
    object_mask_flow_html_parser.add_argument("--output-html", type=Path, default=None, help="Output HTML path")
    object_mask_flow_html_parser.add_argument("--joint-inference", type=Path, default=None, help="Optional joint_inference.json")
    object_mask_flow_html_parser.add_argument("--evaluation-json", type=Path, default=None, help="Optional object_mask_kinematic_evaluation.json")
    object_mask_flow_html_parser.add_argument("--max-tracks", type=int, default=1000, help="Maximum tracks embedded in the HTML")
    object_mask_flow_html_parser.add_argument("--frame-stride", type=int, default=2, help="Embed every Nth frame in the time slider")
    object_mask_flow_html_parser.add_argument("--trail-length", type=int, default=10, help="Default temporal trail length in frames")
    object_mask_flow_html_parser.add_argument(
        "--axis-remap",
        default="x,z,-y",
        help="Visualization-only axis remap, e.g. x,z,-y to convert image-style y-down coordinates to z-up",
    )
    object_mask_flow_html_parser.add_argument(
        "--color-by",
        choices=[
            "pred_cluster",
            "gt_part",
            "motion_magnitude",
            "track_quality",
            "timestep_quality",
            "step_length",
            "acceleration",
            "smooth_residual",
            "step_outlier_score",
            "acceleration_outlier_score",
            "direction_change_deg",
            "rigid_residual",
            "rigid_model_residual",
            "articulation_residual",
            "best_motion_type",
            "cluster_articulation_score",
        ],
        default="pred_cluster",
        help="Initial HTML color mode",
    )

    track_quality_parser = subparsers.add_parser(
        "compute-track-quality",
        help="Compute diagnostic per-track and per-timestep quality scores for lifted 3D tracks",
    )
    track_quality_parser.add_argument("motion_tracks", type=Path, help="Path to object_tracks.json or motion_part_tracks.json")
    track_quality_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for quality artifacts")
    track_quality_parser.add_argument("--bad-track-threshold", type=float, default=0.35)
    track_quality_parser.add_argument("--bad-timestep-threshold", type=float, default=0.35)

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
    part_pose_parser.add_argument("--quality-weighted", action="store_true", help="Use diagnostic track/timestep quality as SE(3) fitting weights")
    part_pose_parser.add_argument("--quality-weight-field", default="timestep_quality_score")
    part_pose_parser.add_argument("--track-quality-field", default="track_quality_score")
    part_pose_parser.add_argument("--min-timestep-weight", type=float, default=0.2)

    motion_seg_parser = subparsers.add_parser(
        "segment-motion-parts",
        help="Relabel object-level 3D CoTracker tracks into rigid motion-based parts",
    )
    motion_seg_parser.add_argument("input_tracks", type=Path, help="Path to object-level part_tracks.json")
    motion_seg_parser.add_argument("--output-json", type=Path, default=None, help="Output relabeled part_tracks.json")
    motion_seg_parser.add_argument(
        "--mode",
        choices=["connected", "knn-spectral"],
        default="connected",
        help="Motion segmentation backend. 'connected' preserves the legacy rigidity connected-components baseline.",
    )
    motion_seg_parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        help="Optional diagnostics JSON with edge composition, cluster residuals, and quality-filter stats",
    )
    motion_seg_parser.add_argument(
        "--sweep-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-ablation/per-K relabeled part_tracks artifacts in knn-spectral mode",
    )
    motion_seg_parser.add_argument(
        "--rigidity-threshold-m",
        type=float,
        default=0.015,
        help="Maximum pairwise distance-variation RMSE for connecting two tracks as one rigid part",
    )
    motion_seg_parser.add_argument(
        "--max-neighbor-distance-m",
        type=float,
        default=0.18,
        help="Soft locality cutoff used when linking tracks with similar rigidity scores",
    )
    motion_seg_parser.add_argument(
        "--min-common-frames",
        type=int,
        default=3,
        help="Minimum overlapping visible frames required to compare two variable-start tracks",
    )
    motion_seg_parser.add_argument(
        "--min-tracks-per-part",
        type=int,
        default=4,
        help="Small motion clusters are merged into nearby rigid clusters below this size",
    )
    motion_seg_parser.add_argument(
        "--min-motion-m",
        type=float,
        default=0.01,
        help="Reserved motion scale for downstream filtering and artifact metadata",
    )
    motion_seg_parser.add_argument(
        "--static-motion-threshold-m",
        type=float,
        default=0.01,
        help="Mean endpoint motion below this threshold is treated as static/base for part ordering",
    )
    motion_seg_parser.add_argument(
        "--moving-motion-threshold-m",
        type=float,
        default=0.03,
        help="Endpoint motion above this threshold is moving-confident in knn-spectral mode",
    )
    motion_seg_parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable visible-frame, depth-jump, and trajectory-discontinuity track filtering before clustering",
    )
    motion_seg_parser.add_argument(
        "--min-visible-frames",
        type=int,
        default=2,
        help="Minimum valid 3D samples required when --quality-filter is enabled",
    )
    motion_seg_parser.add_argument(
        "--max-depth-jump-m",
        type=float,
        default=0.0,
        help="Drop tracks with a consecutive depth_m jump above this threshold when positive",
    )
    motion_seg_parser.add_argument(
        "--max-trajectory-jump-m",
        type=float,
        default=0.0,
        help="Drop tracks with a consecutive 3D jump above this threshold when positive",
    )
    motion_seg_parser.add_argument(
        "--knn-k",
        type=int,
        default=12,
        help="Number of spatial neighbors used for the weighted kNN graph in knn-spectral mode",
    )
    motion_seg_parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum fixed K evaluated by spectral clustering",
    )
    motion_seg_parser.add_argument(
        "--k-max",
        type=int,
        default=8,
        help="Maximum fixed K evaluated by spectral clustering",
    )
    motion_seg_parser.add_argument(
        "--spectral-k",
        type=int,
        default=None,
        help="Fixed K assignment to export as the main output. Defaults to --k-min.",
    )
    motion_seg_parser.add_argument(
        "--edge-ablation",
        choices=["A", "B", "C", "all"],
        default="B",
        help="Edge feature ablation for knn-spectral mode, or 'all' to sweep A/B/C.",
    )
    motion_seg_parser.add_argument(
        "--quality-weighted-affinity",
        action="store_true",
        help=(
            "Use track/timestep quality scores to weight pairwise rigidity affinity. "
            "Disabled by default so legacy clustering is unchanged."
        ),
    )
    motion_seg_parser.add_argument(
        "--quality-weighted-affinity-time-only",
        action="store_true",
        help="Only use timestep quality when aggregating pairwise temporal rigidity errors.",
    )
    motion_seg_parser.add_argument(
        "--quality-weighted-affinity-edge-prior",
        action="store_true",
        help="Multiply final edge strength by pair-level track quality prior.",
    )
    motion_seg_parser.add_argument(
        "--quality-affinity-min-pair-weight",
        type=float,
        default=0.0,
        help="Clamp pair-level quality prior to avoid disconnecting useful low-quality edges.",
    )
    motion_seg_parser.add_argument(
        "--articulation-compatible-affinity",
        action="store_true",
        help=(
            "Experimental: multiply motion affinity by static/prismatic/revolute trajectory compatibility. "
            "Disabled by default."
        ),
    )
    motion_seg_parser.add_argument(
        "--skip-base-bridge-checks",
        action="store_true",
        help="Skip expensive O(N^2) base-bridge diagnostics; useful for quick dense-track ablations.",
    )

    local_split_parser = subparsers.add_parser(
        "local-split-motion-cluster",
        help="Diagnostic local kNN-spectral split for bad object-mask motion clusters",
    )
    local_split_parser.add_argument("input_tracks", type=Path, help="Path to motion_part_tracks.json")
    local_split_parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=None,
        help="Object-mask kinematic evaluation JSON. Recommended child clusters are read from failure_reason_guess.",
    )
    local_split_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for candidate split artifacts and local_split_summary.json",
    )
    local_split_parser.add_argument(
        "--split-cluster-id",
        type=int,
        action="append",
        default=None,
        help="Cluster id to split. May be repeated. If omitted, read recommended ids from --evaluation-json.",
    )
    local_split_parser.add_argument("--local-k-min", type=int, default=2, help="Minimum local spectral K")
    local_split_parser.add_argument("--local-k-max", type=int, default=3, help="Maximum local spectral K")
    local_split_parser.add_argument("--knn-k", type=int, default=8, help="Local spatial kNN graph degree")
    local_split_parser.add_argument(
        "--edge-ablation",
        choices=["A", "B", "C", "all"],
        default="B",
        help="Local edge feature ablation, or 'all' to sweep A/B/C",
    )
    local_split_parser.add_argument(
        "--rigidity-threshold-m",
        type=float,
        default=0.015,
        help="Rigidity scale used by local edge weights",
    )
    local_split_parser.add_argument(
        "--max-neighbor-distance-m",
        type=float,
        default=0.18,
        help="Spatial locality scale used by local edge weights",
    )
    local_split_parser.add_argument(
        "--min-common-frames",
        type=int,
        default=3,
        help="Minimum overlapping visible frames required to compare two tracks",
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
        "--robust-track-model-trim-ratio",
        type=float,
        default=0.0,
        help=(
            "Trim this fraction of largest 3D track replay residuals before comparing revolute/prismatic models. "
            "This is an object-mask diagnostic cleanup knob; 0 disables trimming."
        ),
    )
    joint_parser.add_argument(
        "--quality-weighted-replay",
        action="store_true",
        help=(
            "Use track/timestep quality scores to weight 3D track replay residuals when comparing "
            "revolute and prismatic models. Disabled by default for reproducibility."
        ),
    )
    joint_parser.add_argument(
        "--min-replay-weight",
        type=float,
        default=0.2,
        help="Minimum per-observation weight used by --quality-weighted-replay",
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
    joint_parser.add_argument(
        "--orient-parent-by-motion",
        action="store_true",
        help="Experimental object-mask heuristic: orient a joint so the lower-motion cluster is the parent",
    )
    joint_parser.add_argument(
        "--parent-orientation-motion-margin-m",
        type=float,
        default=0.005,
        help="Minimum median endpoint-motion gap required before --orient-parent-by-motion flips parent/child",
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
