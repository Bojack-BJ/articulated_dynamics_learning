#!/usr/bin/env bash
set -euo pipefail

fold=${1:?usage: launch_balanced_real_arti4d_cv.sh FOLD GPU}
gpu=${2:?usage: launch_balanced_real_arti4d_cv.sh FOLD GPU}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
arti=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
slot=$partnet/../partnet_core_v1_training/cotracker_slots_view_dropout_v1/motion_part_slots.pt
initial=$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
root=$workspace/outputs/balanced_real_generalization_v1/arti4d_stratified_cv
out=$root/fold${fold}
manifest=$out/manifest.tsv

case "$fold" in
  1)
    train=(rh201_stove_oven rh201_cabinet rh201_top_drawer rh078_blue_drawer rh078_cabinet_right)
    val=(rh201_fridge rh078_right_drawer_1)
    ;;
  2)
    train=(rh201_stove_oven rh201_fridge rh201_top_drawer rh078_right_drawer_1 rh078_cabinet_right)
    val=(rh201_cabinet rh078_blue_drawer)
    ;;
  3)
    train=(rh201_fridge rh201_cabinet rh078_blue_drawer rh078_right_drawer_1)
    val=(rh201_stove_oven rh201_top_drawer rh078_cabinet_right)
    ;;
  *) echo "fold must be 1, 2, or 3" >&2; exit 2 ;;
esac

args=()
for name in "${train[@]}"; do args+=(--train "$name"); done
for name in "${val[@]}"; do args+=(--val "$name"); done

mkdir -p "$out"
PYTHONPATH=$code/src "$python" "$code/scripts/build_balanced_real_partnet_manifest.py" \
  "$partnet/cotracker_learning_manifest.tsv" "$arti" "$manifest" \
  --real-repeats 20 --partnet-fraction 0.5 --seed "$((20260908 + fold))" \
  "${args[@]}"

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
  --slot-existence-loss-weight 0.25 --device cuda --seed "$((20260908 + fold))" \
  >"$out/train.log" 2>&1
