#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: launch_balanced_real_fold1.sh real_only|mixed [gpu]}
gpu=${2:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
arti=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
slot=$partnet/../partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial=$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
if [[ "$mode" == real_only ]]; then
  run_name=fold1_real_only_scratch
else
  run_name=fold1_mixed
fi
out=$workspace/outputs/balanced_real_generalization_v1/$run_name
manifest=$out/manifest.tsv

case "$mode" in
  real_only) fraction=0.0; initial_args=() ;;
  mixed) fraction=0.5; initial_args=(--initial-relation-model "$initial") ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$out"
if [[ "$mode" == real_only ]]; then
  random_slot=$out/random_slot.pt
  PYTHONPATH=$code/src "$python" "$code/scripts/make_random_slot_checkpoint.py" \
    "$slot" "$random_slot" --seed 20260908
  slot=$random_slot
fi
PYTHONPATH=$code/src "$python" "$code/scripts/build_balanced_real_partnet_manifest.py" \
  "$partnet/cotracker_learning_manifest.tsv" "$arti" "$manifest" \
  --real-repeats 20 --partnet-fraction "$fraction" --seed 20260908 \
  --train rh201_stove_oven --train rh201_cabinet --train rh201_top_drawer \
  --train rh078_blue_drawer --train rh078_right_drawer_1 \
  --val rh201_fridge --val rh078_cabinet_right

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head "$manifest" "$slot" \
  --output-dir "$out" --epochs 30 --object-batch-size 16 \
  --learning-rate 3e-5 --weight-decay 1e-4 \
  --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --axis-loss-weight 2.0 --axis-line-loss-weight 1.0 \
  --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --unfreeze-slot-backbone --slot-unfreeze-scope all --slot-learning-rate-scale 0.25 \
  --slot-assignment-loss-weight 1.0 --slot-dice-loss-weight 0.5 \
  --slot-pairwise-loss-weight 0.25 --slot-rigid-loss-weight 0.1 \
  --slot-existence-loss-weight 0.25 "${initial_args[@]}" \
  --device cuda --seed 20260908 >"$out/train.log" 2>&1
