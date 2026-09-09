#!/usr/bin/env bash
set -euo pipefail

code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial="$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt"
manifest=/tmp/real_only_train_manifest.tsv
out="$workspace/outputs/real_only_arti4d_hoi4d_full_finetune_v1"

mkdir -p "$out"
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$code/src" "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head \
  "$manifest" "$slot" --output-dir "$out" --epochs 5 --object-batch-size 16 \
  --learning-rate 5e-6 --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --initial-relation-model "$initial" --unfreeze-slot-backbone \
  --slot-unfreeze-scope all --slot-learning-rate-scale 0.2 \
  --slot-assignment-loss-weight 1.0 --slot-existence-loss-weight 0.25 \
  --device cuda --seed 20260831 >"$out/train.log" 2>&1
