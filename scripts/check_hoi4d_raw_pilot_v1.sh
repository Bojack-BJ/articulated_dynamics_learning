#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/Users/lxt/Documents/New project}"
manifest="$repo_root/configs/hoi4d_raw_pilot_v1.csv"
root="$repo_root/data/hoi4d_articulation_v1/_download/pilot_raw_v1"

expected=$(awk -F, 'NR > 1 {sum += $3} END {printf "%.0f", sum}' "$manifest")
downloaded=0
complete=0
partial=0
while IFS=, read -r _ archive size _; do
  [[ "$archive" == "archive" ]] && continue
  if [[ -f "$root/$archive" ]]; then
    downloaded=$((downloaded + $(stat -f %z "$root/$archive")))
    complete=$((complete + 1))
  elif [[ -f "$root/$archive.partial" ]]; then
    downloaded=$((downloaded + $(stat -f %z "$root/$archive.partial")))
    partial=$((partial + 1))
  fi
done < "$manifest"

percent=$(awk -v n="$downloaded" -v total="$expected" 'BEGIN {printf "%.2f", 100*n/total}')
# macOS screen returns 1 even when it lists one detached session.
if [[ "$(screen -ls 2>/dev/null || true)" == *".hoi4d_raw_pilot_v1"* ]]; then
  status="downloading"
elif [[ "$complete" == 8 ]]; then
  status="complete"
else
  status="stopped"
fi

echo "status=$status"
echo "archives_complete=$complete/8 active_partial=$partial"
echo "bytes=$downloaded/$expected percent=${percent}%"
du -sh "$root" 2>/dev/null || true
find "$root" -maxdepth 1 -type f \( -name '*.partial' -o -name '*.tar.gz' \) -exec ls -lh {} \;
