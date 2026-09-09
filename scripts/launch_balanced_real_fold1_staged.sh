#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-7}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
base=$workspace/outputs/balanced_real_generalization_v1
manifest=$base/fold1_mixed/manifest.tsv
slot=$partnet/../partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial=$base/fold1_mixed/slot_relation_head.pt
slot_stage=$base/fold1_mixed_stage_slot10
relation_stage=$base/fold1_mixed_stage_relation10

common=(
  --object-batch-size 16 --weight-decay 1e-4
  --axis-geometry-branch --axis-head-type vector_neuron
  --vector-pivot-parameterization analytic_plane_residual_v1
  --joint-type-loss-weight 2.0 --axis-loss-weight 2.0 --axis-line-loss-weight 1.0
  --slot-geometry-representation invariant_v1
  --quality-weighted-trajectories --robust-segment-weights
  --slot-assignment-loss-weight 1.0 --slot-dice-loss-weight 0.5
  --slot-pairwise-loss-weight 0.25 --slot-rigid-loss-weight 0.1
  --slot-existence-loss-weight 0.25 --device cuda --seed 20260908
)

mkdir -p "$slot_stage" "$relation_stage"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$slot_stage" --epochs 10 --learning-rate 2e-5 \
  --initial-relation-model "$initial" --relation-train-scope slot_only \
  --unfreeze-slot-backbone --slot-unfreeze-scope all --slot-learning-rate-scale 0.5 \
  "${common[@]}" >"$slot_stage/train.log" 2>&1

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$relation_stage" --epochs 10 --learning-rate 1e-5 \
  --initial-relation-model "$slot_stage/slot_relation_head.pt" \
  --relation-train-scope all "${common[@]}" >"$relation_stage/train.log" 2>&1
