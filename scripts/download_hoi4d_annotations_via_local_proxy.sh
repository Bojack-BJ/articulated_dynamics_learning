#!/usr/bin/env bash
set -euo pipefail

root="/Users/lxt/Documents/New project/data/hoi4d_articulation_v1/_download"
partial="$root/HOI4D_annotations.zip.partial"
archive="$root/HOI4D_annotations.zip"
url="https://huggingface.co/datasets/yinloonga/HOI4D/resolve/main/HOI4D_annotations.zip?download=true"
expected_size=22252507455
expected_sha="41eeddb9cd52d0db6bf1df18a743cf5623f4214438efc13ef5a9986db6022984"
remote_dir="/lumos-vePFS/suzhou/Users/lixiaotong/datasets/HOI4D_articulation_v1/_download/raw"

mkdir -p "$root"
curl -x http://127.0.0.1:7890 -L --fail \
  --retry 50 --retry-all-errors --retry-delay 10 --continue-at - \
  -o "$partial" "$url"

actual_size=$(stat -f %z "$partial")
[[ "$actual_size" == "$expected_size" ]] || {
  echo "size mismatch: got $actual_size, expected $expected_size" >&2
  exit 1
}

actual_sha=$(shasum -a 256 "$partial" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "sha256 mismatch: got $actual_sha" >&2
  exit 1
}

mv "$partial" "$archive"
ssh dev-c "mkdir -p '$remote_dir'"
scp "$archive" "dev-c:$remote_dir/HOI4D_annotations.zip.partial"
ssh dev-c "mv '$remote_dir/HOI4D_annotations.zip.partial' '$remote_dir/HOI4D_annotations.zip'"
echo "verified and uploaded to dev-c at $(date -Iseconds)"
