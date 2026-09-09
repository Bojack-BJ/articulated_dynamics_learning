#!/usr/bin/env bash
set -euo pipefail

root="/Users/lxt/Documents/New project/data/hoi4d_articulation_v1/_download"
partial="$root/HOI4D_annotations.zip.partial"
archive="$root/HOI4D_annotations.zip"
expected=22252507455

if screen -ls 2>&1 | grep -q '[.]hoi4d_annotations'; then
  session="screen-running"
else
  session="screen-not-found"
fi

file="$partial"
[[ -f "$archive" ]] && file="$archive"
if [[ ! -f "$file" ]]; then
  echo "session=$session file=missing"
  exit 1
fi

before=$(stat -f %z "$file")
sleep 5
after=$(stat -f %z "$file")
rate=$(( (after - before) / 5 ))
percent=$(awk -v n="$after" -v total="$expected" 'BEGIN { printf "%.2f", 100*n/total }')

if (( rate > 0 )); then
  status="downloading"
  eta=$(awk -v n="$after" -v total="$expected" -v r="$rate" 'BEGIN { printf "%.2f", (total-n)/r/3600 }')
elif [[ -f "$archive" ]]; then
  status="download-complete"
  eta="0"
else
  status="stalled-or-retrying"
  eta="unknown"
fi

echo "status=$status"
echo "session=$session"
echo "file=$file"
echo "bytes=$after/$expected percent=${percent}%"
echo "rate_bytes_s=$rate eta_hours=$eta"
ls -lh "$file"
echo "recent_errors:"
grep -aE 'curl:|size mismatch|sha256 mismatch' "$root/annotations_pipeline.log" 2>/dev/null | tail -n 5 || true
