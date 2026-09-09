#!/usr/bin/env bash
set -euo pipefail
scene=$1
gpu=$2
repo=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning
data_root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/track2art_real_20260831/extracted
out=$repo/outputs/real_0831_${scene/ofen/oven}_track2art_v1
py=/root/Users/miniconda3/envs/particulate/bin/python
while tmux has-session -t "mask0831_$scene" 2>/dev/null; do sleep 10; done
if [[ "$scene" == ofen_hand_open ]]; then
  while tmux has-session -t doorfix0831_open 2>/dev/null; do sleep 10; done
elif [[ "$scene" == ofen_freefall_2 ]]; then
  while tmux has-session -t doorfix0831_freefall 2>/dev/null; do sleep 10; done
fi
if [[ "$scene" == microwave* ]]; then
  CUDA_VISIBLE_DEVICES=$gpu /root/Users/miniconda3/envs/sam2/bin/python /tmp/segment_real_0831_objects.py \
    "$data_root/$scene" "$out" --sam-root /lumos-vePFS/suzhou/Users/lixiaotong/sam2 \
    --prepare /tmp/prepare_phystwin_tracking_episode.py > "$out/mask_corrected.log" 2>&1
fi
test -f "$out/episode.json"
reference=$($py -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata"]["recommended_tracking"]["reference_source_frame"])' "$out/episode.json")
TRACK2ART_PYTHON=$py TRACK_FRAME_STRIDE=1 bash "$repo/scripts/run_real_scene_track2art_remote.sh" \
  "$scene" "$gpu" "$reference" "$(basename "$out")" > "$out/pipeline.log" 2>&1
