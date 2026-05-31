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
