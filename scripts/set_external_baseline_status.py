#!/usr/bin/env python3
"""Set an explicit external-baseline status without fabricating metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("status")
    parser.add_argument("--stage")
    parser.add_argument("--reason")
    parser.add_argument("--protocol")
    parser.add_argument("--oracle-input", action="append", default=[])
    args = parser.parse_args()

    payload = (
        json.loads(args.metrics_path.read_text(encoding="utf-8"))
        if args.metrics_path.exists()
        else {"schema": "external-baseline-result-v1"}
    )
    payload["status"] = args.status
    if args.stage:
        payload["stage"] = args.stage
    if args.reason:
        payload["reason"] = args.reason
    if args.protocol:
        payload["protocol"] = args.protocol
    if args.oracle_input:
        payload["oracle"] = {
            "used_for_inference": True,
            "inputs": args.oracle_input,
        }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
