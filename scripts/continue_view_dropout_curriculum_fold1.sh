#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
slot=$partnet/../partnet_core_v1_training/cotracker_slots_view_dropout_v1/motion_part_slots.pt
root=$workspace/outputs/balanced_real_generalization_v1
manifest=$root/fold1_mixed/manifest.tsv
initial_dir=$root/fold1_mixed_view_dropout_ckptfix
extended=$root/fold1_mixed_view_dropout_extended20
slot_stage=$root/fold1_mixed_view_dropout_slot10
type_stage=$root/fold1_mixed_view_dropout_type10

while [[ ! -s "$initial_dir/training_summary.json" || ! -s "$initial_dir/slot_relation_head_final.pt" ]]; do
  sleep 30
done

common=(
  --object-batch-size 16 --weight-decay 1e-4
  --axis-geometry-branch --axis-head-type vector_neuron
  --vector-pivot-parameterization analytic_plane_residual_v1
  --slot-geometry-representation invariant_v1
  --quality-weighted-trajectories --robust-segment-weights
  --slot-assignment-loss-weight 1.0 --slot-dice-loss-weight 0.5
  --slot-pairwise-loss-weight 0.25 --slot-rigid-loss-weight 0.1
  --slot-existence-loss-weight 0.25 --device cuda --seed 20260908
)

mkdir -p "$extended" "$slot_stage" "$type_stage"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$extended" --epochs 20 --learning-rate 1e-5 \
  --joint-type-loss-weight 2.0 --axis-loss-weight 2.0 --axis-line-loss-weight 1.0 \
  --initial-relation-model "$initial_dir/slot_relation_head_final.pt" \
  --unfreeze-slot-backbone --slot-unfreeze-scope all --slot-learning-rate-scale 0.25 \
  "${common[@]}" >"$extended/train.log" 2>&1

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$slot_stage" --epochs 10 --learning-rate 2e-5 \
  --joint-type-loss-weight 2.0 --axis-loss-weight 2.0 --axis-line-loss-weight 1.0 \
  --initial-relation-model "$extended/slot_relation_head_best_slot.pt" \
  --relation-train-scope slot_only --unfreeze-slot-backbone \
  --slot-unfreeze-scope all --slot-learning-rate-scale 0.5 \
  "${common[@]}" >"$slot_stage/train.log" 2>&1

# Freeze the selected slot representation and prioritize discrete type recovery.
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$type_stage" --epochs 10 --learning-rate 5e-6 \
  --joint-type-loss-weight 6.0 --axis-loss-weight 0.5 --axis-line-loss-weight 0.25 \
  --initial-relation-model "$slot_stage/slot_relation_head_best_slot.pt" \
  --relation-train-scope all "${common[@]}" >"$type_stage/train.log" 2>&1
