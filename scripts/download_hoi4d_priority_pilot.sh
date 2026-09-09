#!/usr/bin/env bash
set -euo pipefail

manifest="${1:?usage: download_hoi4d_priority_pilot.sh MANIFEST OUTPUT_DIR [PROXY]}"
output_dir="${2:?usage: download_hoi4d_priority_pilot.sh MANIFEST OUTPUT_DIR [PROXY]}"
proxy="${3:-http://127.0.0.1:18888}"
base_url="https://huggingface.co/datasets/Livioni/hoi4d/resolve/main"
mkdir -p "$output_dir/logs"

download_one() {
  local archive="${1//$'\r'/}"
  [[ -n "$archive" ]] || return 0
  local output="$output_dir/$archive"
  local partial="$output.partial"
  local url="$base_url/$archive?download=true"
  local log="$output_dir/logs/${archive%.tar.gz}.log"
  local expected actual

  expected="$(HTTPS_PROXY="$proxy" curl -sSIL "$url" | tr -d '\r' | awk '
    tolower($1) == "x-linked-size:" {value=$2}
    END {print value}
  ')"
  if [[ -z "$expected" ]]; then
    echo "could not resolve size: $archive" | tee -a "$log" >&2
    return 1
  fi
  if [[ -f "$output" ]] && [[ "$(stat -c %s "$output")" == "$expected" ]]; then
    tar -tzf "$output" >/dev/null
    echo "already verified: $archive" | tee -a "$log"
    return
  fi
  HTTPS_PROXY="$proxy" curl -L --fail --retry 50 --retry-all-errors \
    --retry-delay 10 --continue-at - -o "$partial" "$url" >>"$log" 2>&1
  actual="$(stat -c %s "$partial")"
  if [[ "$actual" != "$expected" ]]; then
    echo "size mismatch: $archive got=$actual expected=$expected" | tee -a "$log" >&2
    return 1
  fi
  tar -tzf "$partial" >/dev/null
  mv "$partial" "$output"
  echo "verified: $archive bytes=$actual" | tee -a "$log"
}
export output_dir proxy base_url
export -f download_one

tail -n +2 "$manifest" | cut -d, -f3 | tr -d '\r' | xargs -P 4 -n 1 bash -c 'download_one "$0"'
echo "priority HOI4D pilot download complete"
