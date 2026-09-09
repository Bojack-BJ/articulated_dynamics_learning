#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
code=/tmp/track2art_arti4d_relation_v3
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
python=/root/Users/miniconda3/envs/particulate/bin/python
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial="$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt"
manifest="$workspace/outputs/arti4d_relation_finetune_v1/slot_decoder_5scene_manifest.tsv"
out="$workspace/outputs/arti4d_relation_finetune_v1/relation_5scene_low_lr_5ep"
gpu=${1:-6}

mkdir -p "$out"
started=$(date +%s)
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$code/src" "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head \
  "$manifest" "$slot" --output-dir "$out" --epochs 5 --object-batch-size 32 \
  --learning-rate 2e-5 --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --initial-relation-model "$initial" --device cuda --seed 20260831 \
  >"$out/train.log" 2>&1
finished=$(date +%s)
printf '{"started_unix":%s,"finished_unix":%s,"wall_seconds":%s}\n' \
  "$started" "$finished" "$((finished-started))" >"$out/runtime.json"
