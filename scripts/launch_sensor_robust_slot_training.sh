#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-3}
seed=${2:-20260831}
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
python=/root/Users/miniconda3/envs/particulate/bin/python
epochs=${EPOCHS:-100}
out="$workspace/outputs/neural_head_pivot_b_pilot_v1/sensor_robust_slots_main_${epochs}ep_seed_${seed}"

mkdir -p "$out"
started=$(date +%s)
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
  -m rgbd_urdf_mvp train-motion-part-slots \
  "$data/cotracker_learning_manifest.tsv" --output-dir "$out" \
  --max-slots 16 --epochs "$epochs" --object-batch-size 16 --data-loader-workers 16 \
  --learning-rate 3e-4 --rigid-loss-weight 0.2 --pairwise-loss-weight 0.35 \
  --track-dropout-ratio 0.1 \
  --geometry-noise-std 0.006 --depth-bias-probability 0.5 \
  --depth-bias-std 0.015 --sampled-depth-spike-probability 0.05 \
  --device cuda --seed "$seed" >"$out/train.log" 2>&1
finished=$(date +%s)
printf '{"seed":%s,"started_unix":%s,"finished_unix":%s,"wall_seconds":%s}\n' \
  "$seed" "$started" "$finished" "$((finished-started))" >"$out/runtime.json"
