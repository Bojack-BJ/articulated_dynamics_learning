#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots_view_dropout_v1/motion_part_slots.pt
initial=$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
base=$workspace/outputs/balanced_real_generalization_v1
manifest=$base/fold1_mixed/manifest.tsv
out=$base/fold1_mixed_view_dropout_ckptfix

mkdir -p "$out"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$out" --epochs 30 --object-batch-size 16 \
  --learning-rate 3e-5 --weight-decay 1e-4 \
  --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --axis-loss-weight 2.0 --axis-line-loss-weight 1.0 \
  --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --initial-relation-model "$initial" --ignore-initial-slot-state \
  --unfreeze-slot-backbone --slot-unfreeze-scope all --slot-learning-rate-scale 0.25 \
  --slot-assignment-loss-weight 1.0 --slot-dice-loss-weight 0.5 \
  --slot-pairwise-loss-weight 0.25 --slot-rigid-loss-weight 0.1 \
  --slot-existence-loss-weight 0.25 --device cuda --seed 20260908 \
  >"$out/train.log" 2>&1
