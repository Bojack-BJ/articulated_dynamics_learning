#!/usr/bin/env python3
"""Read official Arti4D interaction CSVs directly from scene archives."""

import argparse
import csv
import io
import json
from pathlib import PurePosixPath

from remotezip import RemoteZip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()
    rows = []
    for url in args.urls:
        split = PurePosixPath(url).stem.removeprefix("arti4d-")
        with RemoteZip(url, headers={"User-Agent": "Mozilla/5.0"}) as archive:
            names = archive.namelist()
            cue_paths = sorted(name for name in names if name.endswith("/matched_cues.csv"))
            for cue_path in cue_paths:
                sequence = PurePosixPath(cue_path).parent.name
                cues = csv.DictReader(io.StringIO(archive.read(cue_path).decode("utf-8")))
                for cue in cues:
                    start = int(cue["CUE_START"])
                    end = int(cue["CUE_END"])
                    rows.append({
                        "split": split,
                        "sequence": sequence,
                        "axis_name": cue["AXIS_NAME"],
                        "cue_start": start,
                        "cue_end": end,
                        "interaction_frames": end - start + 1,
                        "verification": cue.get("VERIFICATION", ""),
                        "source_url": url,
                    })
    print(json.dumps({"count": len(rows), "interactions": rows}, indent=2))


if __name__ == "__main__":
    main()
