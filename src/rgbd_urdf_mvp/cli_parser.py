from __future__ import annotations

import argparse

from .cli_parsers import batch, core, dynamics, generation, manipulation, perception, recording


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI by registering workflow-specific subcommands."""
    parser = argparse.ArgumentParser(description="RGB-D to URDF MVP scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    core.register(subparsers)
    batch.register(subparsers)
    perception.register(subparsers)
    dynamics.register(subparsers)
    manipulation.register(subparsers)
    generation.register(subparsers)
    recording.register(subparsers)

    return parser
