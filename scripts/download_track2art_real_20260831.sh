#!/usr/bin/env bash
set -euo pipefail

root="/Users/lxt/Documents/New project/data/track2art_real_new_20260831"
raw="$root/raw"
mkdir -p "$raw"

download() {
  local id="$1"
  local name="$2"
  local output="$raw/$name"
  curl -x http://127.0.0.1:7890 --http1.1 -L --fail \
    --retry 50 --retry-all-errors --retry-delay 5 --continue-at - \
    -o "$output.partial" \
    "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
  mv "$output.partial" "$output"
  unzip -tq "$output"
}

download 1ix0ffwcutzPr3iwDP4l0--ZxLFzH61OM 0831_microwave_hand_close_2.zip &
download 1qx4XPAofnF5pEOLjPMQCguceR6rgbXby 0831_microwave_hand_open_2.zip &
download 1rlslJ5BVHpT7cl4IgarA-AQUGSiO0_SP 0831_ofen_freefall_2.zip &
wait

download 1akpH-gChIFN358vprAu5OA2uqPT3S7iv 0831_ofen_freefall.zip &
download 1sFwmNC_OfCH7NXb2GZD7skZud6x8tZQZ 0831_ofen_hand_close.zip &
download 1NwY1vMJ4-7iCgZnsaj1Zgv97a1AizkS- 0831_ofen_hand_open.zip &
wait

echo "all six archives downloaded and verified at $(date -Iseconds)"
