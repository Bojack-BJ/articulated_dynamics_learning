#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/Users/lxt/Documents/New project}"
manifest="$repo_root/configs/hoi4d_raw_pilot_v1.csv"
download_root="$repo_root/data/hoi4d_articulation_v1/_download/pilot_raw_v1"
log_root="$download_root/logs"
proxy="${HTTPS_PROXY:-http://127.0.0.1:7890}"
base_url="https://huggingface.co/datasets/Livioni/hoi4d/resolve/main"

mkdir -p "$download_root" "$log_root"

download_one() {
  local archive="$1"
  local expected_size="$2"
  local output="$download_root/$archive"
  local partial="$output.partial"
  local log="$log_root/${archive%.tar.gz}.log"

  if [[ -f "$output" ]] && [[ "$(stat -f %z "$output")" == "$expected_size" ]]; then
    echo "already verified: $archive" | tee -a "$log"
    return
  fi

  curl -x "$proxy" -L --fail --retry 50 --retry-all-errors --retry-delay 10 \
    --continue-at - -o "$partial" "$base_url/$archive?download=true" >>"$log" 2>&1
  local actual_size
  actual_size="$(stat -f %z "$partial")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "size mismatch for $archive: got $actual_size expected $expected_size" | tee -a "$log" >&2
    return 1
  fi
  tar -tzf "$partial" >/dev/null
  mv "$partial" "$output"
  echo "verified: $archive ($actual_size bytes)" | tee -a "$log"
}
export -f download_one
export download_root log_root proxy base_url

awk -F, 'NR > 1 {print $2, $3}' "$manifest" | \
  xargs -P 3 -n 2 bash -c 'download_one "$0" "$1"'

python3 - "$manifest" "$download_root/download_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest, output = map(Path, sys.argv[1:])
rows = list(csv.DictReader(manifest.open()))
root = output.parent
summary = {
    "status": "complete",
    "sequence_count": len(rows),
    "total_bytes": sum(int(row["size_bytes"]) for row in rows),
    "archives": [],
}
for row in rows:
    path = root / row["archive"]
    size = path.stat().st_size
    summary["archives"].append({
        "sequence": row["sequence"],
        "archive": str(path),
        "size_bytes": size,
        "verified_size": size == int(row["size_bytes"]),
    })
output.write_text(json.dumps(summary, indent=2) + "\n")
PY

echo "HOI4D raw pilot download complete: $download_root"
