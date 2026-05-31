from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
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
