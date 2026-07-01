from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    manipulation_parser = subparsers.add_parser(
        "plan-manipulation",
        help="Plan a MuJoCo one-shot push command for a target articulated joint position",
    )
    manipulation_parser.add_argument(
        "mjcf",
        type=Path,
        help=(
            "Path to an MJCF model. A dynamics_identification.json artifact is also accepted and "
            "will use its mjcf_optimized_path."
        ),
    )
    manipulation_parser.add_argument("--target-q", type=float, required=True, help="Target hinge angle or slide position")
    manipulation_parser.add_argument("--joint-name", type=str, default=None, help="Joint name to plan for")
    manipulation_parser.add_argument("--joint-id", type=int, default=None, help="Joint id to plan for")
    manipulation_parser.add_argument("--initial-q", type=float, default=None, help="Initial joint position")
    manipulation_parser.add_argument(
        "--mode",
        choices=["initial_velocity", "pulse_force"],
        default="initial_velocity",
        help="Search either an initial joint velocity or a short generalized-force pulse",
    )
    manipulation_parser.add_argument(
        "--dynamics-identification",
        type=Path,
        default=None,
        help="Optional dynamics_identification.json to copy fitted parameter diagnostics into the plan",
    )
    manipulation_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for manipulation_plan.json (defaults to <mjcf_dir>/downstream_manipulation)",
    )
    manipulation_parser.add_argument("--duration-s", type=float, default=1.5, help="Rollout duration")
    manipulation_parser.add_argument("--sim-dt", type=float, default=None, help="Override MuJoCo timestep")
    manipulation_parser.add_argument("--num-candidates", type=int, default=81, help="Grid search candidate count")
    manipulation_parser.add_argument("--max-initial-qvel", type=float, default=8.0, help="Max absolute initial qvel candidate")
    manipulation_parser.add_argument("--max-pulse-force", type=float, default=5.0, help="Max absolute pulse force/torque candidate")
    manipulation_parser.add_argument("--pulse-duration-s", type=float, default=0.15, help="Pulse duration for --mode pulse_force")
    manipulation_parser.add_argument("--tolerance", type=float, default=0.03, help="Success tolerance in joint units")
    manipulation_parser.add_argument(
        "--enable-contact",
        action="store_true",
        help="Enable MJCF geom contacts during manipulation rollout",
    )
    manipulation_parser.add_argument(
        "--gravity-mode",
        choices=["model", "zero"],
        default="model",
        help="Gravity used during manipulation rollout",
    )

    demo_parser = subparsers.add_parser(
        "generate-il-demos",
        help="Generate ballistic dynamics-conditioned imitation demos from a MuJoCo joint",
    )
    demo_parser.add_argument("mjcf", type=Path, help="MJCF model or dynamics_identification.json")
    demo_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for il_dataset.npz")
    demo_parser.add_argument(
        "--task",
        choices=["impulse-to-target", "timing-gate", "energy-budget"],
        default="impulse-to-target",
        help="Ballistic showcase task. v1 uses the same one-kick teacher for all tasks.",
    )
    demo_parser.add_argument("--num-episodes", type=int, default=256)
    demo_parser.add_argument("--release-duration-s", type=float, default=1.0)
    demo_parser.add_argument("--condition-source", choices=["oracle", "estimated"], default="oracle")
    demo_parser.add_argument("--joint-name", type=str, default=None)
    demo_parser.add_argument("--joint-id", type=int, default=None)
    demo_parser.add_argument("--sim-dt", type=float, default=None)
    demo_parser.add_argument("--max-initial-qvel", type=float, default=8.0)
    demo_parser.add_argument("--num-teacher-candidates", type=int, default=121)
    demo_parser.add_argument("--response-samples", type=int, default=64)
    demo_parser.add_argument("--tolerance", type=float, default=0.05)
    demo_parser.add_argument("--qdot-tolerance", type=float, default=0.25)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument("--enable-contact", action="store_true")
    demo_parser.add_argument("--gravity-mode", choices=["model", "zero"], default="zero")

    train_parser = subparsers.add_parser(
        "train-il-policy",
        help="Train a BC-MLP policy for ballistic one-kick imitation",
    )
    train_parser.add_argument("dataset", type=Path, help="Path to il_dataset.npz")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--condition-mode", choices=["none", "oracle", "estimated", "noisy-estimated"], default="oracle")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--hidden-layers", type=int, default=2)
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--noisy-condition-std", type=float, default=0.05)

    eval_parser = subparsers.add_parser(
        "eval-il-policy",
        help="Evaluate a ballistic IL policy with kick-then-release MuJoCo rollouts",
    )
    eval_parser.add_argument("policy", type=Path, help="Path to policy.pt")
    eval_parser.add_argument("mjcf", type=Path, help="MJCF model or dynamics_identification.json")
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--num-episodes", type=int, default=64)
    eval_parser.add_argument("--release-duration-s", type=float, default=1.0)
    eval_parser.add_argument("--joint-name", type=str, default=None)
    eval_parser.add_argument("--joint-id", type=int, default=None)
    eval_parser.add_argument("--sim-dt", type=float, default=None)
    eval_parser.add_argument("--response-samples", type=int, default=64)
    eval_parser.add_argument("--tolerance", type=float, default=0.05)
    eval_parser.add_argument("--qdot-tolerance", type=float, default=0.25)
    eval_parser.add_argument("--seed", type=int, default=1)
    eval_parser.add_argument("--max-abs-initial-qvel", type=float, default=8.0)
    eval_parser.add_argument("--enable-contact", action="store_true")
    eval_parser.add_argument("--gravity-mode", choices=["model", "zero"], default="zero")

    x5_parser = subparsers.add_parser(
        "prepare-arx-x5-model",
        help="Copy and patch the ARX X5A URDF package so MuJoCo can load its meshes",
    )
    x5_parser.add_argument("--output-dir", type=Path, required=True)
    x5_parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing ARX_Model checkout. If omitted, the command clones the public repository.",
    )
    x5_parser.add_argument("--repo-url", default="https://github.com/ARXroboticsX/ARX_Model.git")
    x5_parser.add_argument("--no-validate-mujoco", action="store_true")

    contact_parser = subparsers.add_parser(
        "generate-contact-il-demos",
        help="Generate V1 part-level contact impulse demos with an idealized X5 end-effector push",
    )
    contact_parser.add_argument("mjcf", type=Path, help="Articulated object MJCF or dynamics_identification.json")
    contact_parser.add_argument("--output-dir", type=Path, required=True)
    contact_parser.add_argument("--robot-model", type=Path, default=None, help="Prepared ARX X5A URDF/MJCF path stored in the manifest")
    contact_parser.add_argument("--robot-end-effector-body", default="link6")
    contact_parser.add_argument("--task", choices=["impulse-to-target", "timing-gate", "energy-budget"], default="impulse-to-target")
    contact_parser.add_argument("--num-episodes", type=int, default=256)
    contact_parser.add_argument("--release-duration-s", type=float, default=1.0)
    contact_parser.add_argument("--contact-duration-s", type=float, default=0.12)
    contact_parser.add_argument("--condition-source", choices=["oracle", "estimated"], default="estimated")
    contact_parser.add_argument("--joint-name", type=str, default=None)
    contact_parser.add_argument("--joint-id", type=int, default=None)
    contact_parser.add_argument("--sim-dt", type=float, default=None)
    contact_parser.add_argument("--max-force", type=float, default=20.0)
    contact_parser.add_argument("--num-teacher-candidates", type=int, default=101)
    contact_parser.add_argument("--response-samples", type=int, default=64)
    contact_parser.add_argument("--tolerance", type=float, default=0.05)
    contact_parser.add_argument("--qdot-tolerance", type=float, default=0.25)
    contact_parser.add_argument("--seed", type=int, default=0)
    contact_parser.add_argument("--enable-contact", action="store_true")
    contact_parser.add_argument("--gravity-mode", choices=["model", "zero"], default="zero")

    contact_eval_parser = subparsers.add_parser(
        "eval-contact-il-policy",
        help="Evaluate a BC policy that outputs V1 contact impulse actions",
    )
    contact_eval_parser.add_argument("policy", type=Path, help="Path to policy.pt")
    contact_eval_parser.add_argument("mjcf", type=Path, help="Articulated object MJCF or dynamics_identification.json")
    contact_eval_parser.add_argument("--output-dir", type=Path, required=True)
    contact_eval_parser.add_argument("--num-episodes", type=int, default=64)
    contact_eval_parser.add_argument("--release-duration-s", type=float, default=1.0)
    contact_eval_parser.add_argument("--joint-name", type=str, default=None)
    contact_eval_parser.add_argument("--joint-id", type=int, default=None)
    contact_eval_parser.add_argument("--sim-dt", type=float, default=None)
    contact_eval_parser.add_argument("--response-samples", type=int, default=64)
    contact_eval_parser.add_argument("--tolerance", type=float, default=0.05)
    contact_eval_parser.add_argument("--qdot-tolerance", type=float, default=0.25)
    contact_eval_parser.add_argument("--seed", type=int, default=1)
    contact_eval_parser.add_argument("--max-force", type=float, default=40.0)
    contact_eval_parser.add_argument("--max-contact-duration-s", type=float, default=0.25)
    contact_eval_parser.add_argument("--enable-contact", action="store_true")
    contact_eval_parser.add_argument("--gravity-mode", choices=["model", "zero"], default="zero")

    viz_parser = subparsers.add_parser(
        "visualize-il-dataset",
        help="Write an SVG summary of ballistic/contact IL free-response trajectories",
    )
    viz_parser.add_argument("dataset", type=Path, help="Path to il_dataset.npz or contact_il_dataset.npz")
    viz_parser.add_argument("--output-svg", type=Path, default=None)
    viz_parser.add_argument("--max-trajectories", type=int, default=32)
    viz_parser.add_argument("--title", default=None)

    render_parser = subparsers.add_parser(
        "render-il-rollout",
        help="Render one ballistic/contact IL rollout as a MuJoCo animation",
    )
    render_parser.add_argument("dataset", type=Path, help="Path to il_dataset.npz or contact_il_dataset.npz")
    render_parser.add_argument("mjcf", type=Path, help="Articulated object MJCF or dynamics_identification.json")
    render_parser.add_argument("--output-path", type=Path, default=None, help="Output .mp4/.gif path or frame directory")
    render_parser.add_argument("--robot-model", type=Path, default=None, help="Optional prepared ARX X5 URDF/MJCF to add to the rendered scene")
    render_parser.add_argument("--dynamics-artifact", type=Path, default=None, help="Optional dynamics_identification.json used to overwrite object dynamics")
    render_parser.add_argument("--episode-index", type=int, default=0)
    render_parser.add_argument(
        "--trajectory-source",
        choices=["dataset", "resimulate"],
        default="dataset",
        help="Render stored free_response q(t), or resimulate the stored action in the current MJCF",
    )
    render_parser.add_argument("--joint-name", type=str, default=None)
    render_parser.add_argument("--joint-id", type=int, default=None)
    render_parser.add_argument("--sim-dt", type=float, default=None)
    render_parser.add_argument("--width", type=int, default=960)
    render_parser.add_argument("--height", type=int, default=720)
    render_parser.add_argument("--fps", type=int, default=30)
    render_parser.add_argument("--camera-name", type=str, default=None)
    render_parser.add_argument("--camera-azimuth", type=float, default=135.0)
    render_parser.add_argument("--camera-elevation", type=float, default=-25.0)
    render_parser.add_argument("--camera-distance", type=float, default=None)
    render_parser.add_argument("--object-visual-scale", type=float, default=1.0)
    render_parser.add_argument("--playback-slowdown", type=float, default=1.0)
    render_parser.add_argument("--robot-base-pos", type=float, nargs=3, default=(-0.35, -0.65, 0.0))
    render_parser.add_argument("--robot-base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.8))
    render_parser.add_argument("--robot-motion", choices=["static", "ik-pulse"], default="ik-pulse")
    render_parser.add_argument("--robot-ee-body", default="x5_link8")
    render_parser.add_argument("--robot-approach-distance", type=float, default=0.08)
    render_parser.add_argument("--robot-pulse-distance", type=float, default=0.06)
    render_parser.add_argument("--dynamics-body-name", type=str, default=None)
    render_parser.add_argument("--dynamics-joint-name", type=str, default=None)
    render_parser.add_argument("--enable-contact", action="store_true")
    render_parser.add_argument(
        "--allow-robot-self-collision",
        action="store_true",
        help="Disable robot self-collision rejection in the IK pulse visualizer",
    )
    render_parser.add_argument("--self-collision-penetration-tolerance-m", type=float, default=0.0)
    render_parser.add_argument(
        "--allow-robot-object-collision",
        action="store_true",
        help="Disable robot-object collision rejection in the IK pulse visualizer",
    )
    render_parser.add_argument("--object-collision-penetration-tolerance-m", type=float, default=0.0)
    render_parser.add_argument("--gravity-mode", choices=["model", "zero"], default="zero")
    render_parser.add_argument("--write-frames", action="store_true", help="Write PNG frames instead of a video container")
