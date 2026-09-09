#!/usr/bin/env bash
set -euo pipefail
data_root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/track2art_real_20260831/extracted
repo=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning
sam_root=/lumos-vePFS/suzhou/Users/lixiaotong/sam2
py=/root/Users/miniconda3/envs/sam2/bin/python
scenes=(microwave_hand_close_2 microwave_hand_open_2 ofen_freefall ofen_freefall_2 ofen_hand_close ofen_hand_open)
for gpu in 0 1 2 3 4 5; do
  scene=${scenes[$gpu]}
  out=$repo/outputs/real_0831_${scene/ofen/oven}_track2art_v1
  mkdir -p "$out"
  tmux new-session -d -s "mask0831_$scene" "CUDA_VISIBLE_DEVICES=$gpu $py /tmp/segment_real_0831_objects.py $data_root/$scene $out --sam-root $sam_root --prepare /tmp/prepare_phystwin_tracking_episode.py > $out/mask_pipeline.log 2>&1"
done
