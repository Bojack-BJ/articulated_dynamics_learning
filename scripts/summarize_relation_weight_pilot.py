#!/usr/bin/env python3
import json
import sys
from pathlib import Path


KEYS = (
    "edge_f1",
    "joint_type_accuracy",
    "axis_error_deg",
    "axis_error_median_deg",
    "axis_error_p90_deg",
    "axis_line_error_normalized",
    "illegal_graph_rate",
)


for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    payload = json.loads(path.read_text())
    if "rows" in payload:
        metrics = next(row for row in payload["rows"] if row.get("category") == "all")
    else:
        metrics = payload.get("overall", payload.get("metrics", payload))
    print(path)
    for key in KEYS:
        candidates = (key, f"test_{key}", f"val_{key}")
        value = next((metrics[name] for name in candidates if name in metrics), None)
        print(f"  {key}: {value}")
