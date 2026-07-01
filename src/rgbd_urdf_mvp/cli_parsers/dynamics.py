from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
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
