#!/usr/bin/env bash
set -euo pipefail

download_pid="${1:?download PID required}"
root="/Users/lxt/Documents/New project/data/hoi4d_articulation_v1/_download"
partial="$root/HOI4D_annotations.zip.partial"
archive="$root/HOI4D_annotations.zip"
log="$root/annotations_finalize.log"
expected_size=22252507455
expected_sha="41eeddb9cd52d0db6bf1df18a743cf5623f4214438efc13ef5a9986db6022984"
remote_dir="/lumos-vePFS/suzhou/Users/lixiaotong/datasets/HOI4D_articulation_v1/_download/raw"

exec >>"$log" 2>&1
echo "$(date -Iseconds) waiting for curl pid $download_pid"
while kill -0 "$download_pid" 2>/dev/null; do
  sleep 60
done

actual_size=$(stat -f %z "$partial")
if [[ "$actual_size" != "$expected_size" ]]; then
  echo "$(date -Iseconds) size mismatch: got $actual_size, expected $expected_size"
  exit 1
fi

actual_sha=$(shasum -a 256 "$partial" | awk '{print $1}')
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "$(date -Iseconds) sha256 mismatch: got $actual_sha"
  exit 1
fi

mv "$partial" "$archive"
ssh dev-c "mkdir -p '$remote_dir'"
scp "$archive" "dev-c:$remote_dir/HOI4D_annotations.zip.partial"
ssh dev-c "mv '$remote_dir/HOI4D_annotations.zip.partial' '$remote_dir/HOI4D_annotations.zip'"
echo "$(date -Iseconds) verified and uploaded to dev-c"
