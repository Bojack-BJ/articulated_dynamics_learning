#!/usr/bin/env python3
"""Promote a reviewed analytic pseudo joint artifact to relation supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    joints = source.get("joints", [])
    if not joints:
        raise ValueError(f"No pseudo joints in {args.source}")
    payload = {
        "schema_version": 1,
        "annotation_source": "pseudo-reviewed",
        "source_artifact": str(args.source.resolve()),
        "joints": [{**joint, "annotation_source": "pseudo-reviewed"} for joint in joints],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "joint_count": len(joints)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
