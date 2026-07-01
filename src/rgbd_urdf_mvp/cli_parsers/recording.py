from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.categories import SUPPORTED_CATEGORIES, normalize_category


def register(subparsers: Any) -> None:
    import_rbo_parser = subparsers.add_parser(
        "import-rbo-recording",
        help="Convert an RBO/ROSbag-like RGB-D folder export into an episode.json recording",
    )
    import_rbo_parser.add_argument("input_dir", type=Path, help="RBO sequence directory")
    import_rbo_parser.add_argument("--output-dir", type=Path, required=True, help="Output episode directory")
    import_rbo_parser.add_argument("--object-id", default=None, help="Object instance id (defaults to input folder name)")
    import_rbo_parser.add_argument(
        "--category",
        type=normalize_category,
        choices=list(SUPPORTED_CATEGORIES),
        default=None,
        help="Object category override. Unknown RBO object names fall back to door.",
    )
    import_rbo_parser.add_argument("--frame-stride", type=int, default=1, help="Import every Nth RGB frame")
    import_rbo_parser.add_argument("--start-frame", type=int, default=0, help="First RGB frame index to import")
    import_rbo_parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to import")
    import_rbo_parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Resample imported RGB-D frames onto this nominal frame rate before writing episode timestamps",
    )
    import_rbo_parser.add_argument(
        "--max-sync-delta-s",
        type=float,
        default=0.05,
        help="Maximum allowed nearest RGB/depth timestamp delta",
    )
    import_rbo_parser.add_argument(
        "--mask-mode",
        choices=["none", "depth-near"],
        default="none",
        help="Optional heuristic object-mask generation mode",
    )
    import_rbo_parser.add_argument(
        "--mask-depth-percentile",
        type=float,
        default=35.0,
        help="Depth percentile used by --mask-mode depth-near",
    )
    import_rbo_parser.add_argument(
        "--mask-depth-margin-m",
        type=float,
        default=0.15,
        help="Additional depth band in meters used by --mask-mode depth-near",
    )
    import_rbo_parser.add_argument("--min-depth-m", type=float, default=0.05, help="Minimum valid depth in meters")
    import_rbo_parser.add_argument("--max-depth-m", type=float, default=10.0, help="Maximum valid depth in meters")
    import_rbo_parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory")

    import_rbo_batch_parser = subparsers.add_parser(
        "import-rbo-recordings-batch",
        help="Batch convert RBO/ROSbag-like RGB-D folder exports into episode.json recordings",
    )
    import_rbo_batch_parser.add_argument(
        "manifest",
        type=Path,
        help="TSV with input_dir and optional object_id/category/output_dir/import override columns",
    )
    import_rbo_batch_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "real_recordings",
        help="Default root for converted recordings",
    )
    import_rbo_batch_parser.add_argument("--jobs", type=int, default=1, help="Parallel import jobs")
    import_rbo_batch_parser.add_argument("--frame-stride", type=int, default=1, help="Default import stride")
    import_rbo_batch_parser.add_argument("--start-frame", type=int, default=0, help="Default first RGB frame index")
    import_rbo_batch_parser.add_argument("--max-frames", type=int, default=None, help="Default maximum imported frames")
    import_rbo_batch_parser.add_argument("--target-fps", type=float, default=None, help="Default target episode fps")
    import_rbo_batch_parser.add_argument("--max-sync-delta-s", type=float, default=0.05)
    import_rbo_batch_parser.add_argument(
        "--mask-mode",
        choices=["none", "depth-near"],
        default="none",
        help="Default optional heuristic object-mask generation mode",
    )
    import_rbo_batch_parser.add_argument("--mask-depth-percentile", type=float, default=35.0)
    import_rbo_batch_parser.add_argument("--mask-depth-margin-m", type=float, default=0.15)
    import_rbo_batch_parser.add_argument("--min-depth-m", type=float, default=0.05)
    import_rbo_batch_parser.add_argument("--max-depth-m", type=float, default=10.0)
    import_rbo_batch_parser.add_argument("--force", action="store_true")

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
        "--staged-initial-qvel",
        action="store_true",
        help=(
            "In free mode, drive multiple joints in stages: primary token-matched joints at t=0, "
            "then a sampled subset of secondary token-matched joints later"
        ),
    )
    record_parser.add_argument(
        "--staged-primary-tokens",
        nargs="+",
        default=("door",),
        help="Joint/body name tokens for joints opened at t=0 in --staged-initial-qvel mode",
    )
    record_parser.add_argument(
        "--staged-secondary-tokens",
        nargs="+",
        default=("drawer", "slide"),
        help="Joint/body name tokens for delayed joints in --staged-initial-qvel mode",
    )
    record_parser.add_argument(
        "--staged-secondary-count-min",
        type=int,
        default=1,
        help="Minimum number of delayed secondary joints to sample",
    )
    record_parser.add_argument(
        "--staged-secondary-count-max",
        type=int,
        default=2,
        help="Maximum number of delayed secondary joints to sample",
    )
    record_parser.add_argument(
        "--staged-secondary-start-s",
        type=float,
        default=1.2,
        help="Simulation time when delayed secondary joints receive their initial qvel",
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
