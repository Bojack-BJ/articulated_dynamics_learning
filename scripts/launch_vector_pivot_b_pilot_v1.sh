#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
data_root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
python=/root/Users/miniconda3/envs/particulate/bin/python
slot_model=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial="$workspace/outputs/neural_head_final_matrix_v1/no_rotation_all/seed_20260831/slot_relation_head.pt"
output_root="$workspace/outputs/neural_head_pivot_b_pilot_v1"

run_train() {
  local gpu=${1:-7}
  local out="$output_root/frozen_10ep_seed_20260831"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
    -m rgbd_urdf_mvp train-slot-relation-head \
    "$data_root/cotracker_learning_manifest.tsv" "$slot_model" \
    --output-dir "$out" --epochs 10 --object-batch-size 32 \
    --learning-rate 3e-4 --axis-geometry-branch --axis-head-type vector_neuron \
    --vector-pivot-parameterization analytic_plane_residual_v1 \
    --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
    --temporal-occlusion-augmentation --temporal-occlusion-probability 0.7 \
    --temporal-occlusion-min-fraction 0.15 --temporal-occlusion-max-fraction 0.30 \
    --temporal-occlusion-track-fraction 0.5 --initial-relation-model "$initial" \
    --device cuda --seed 20260831 >"$out/train.log" 2>&1
}

run_pivot_only() {
  local gpu=${1:-7}
  local out="$output_root/pivot_only_10ep_seed_20260831"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
    -m rgbd_urdf_mvp train-slot-relation-head \
    "$data_root/cotracker_learning_manifest.tsv" "$slot_model" \
    --output-dir "$out" --epochs 10 --object-batch-size 32 \
    --learning-rate 3e-4 --axis-geometry-branch --axis-head-type vector_neuron \
    --vector-pivot-parameterization analytic_plane_residual_v1 \
    --relation-train-scope vector_pivot_only \
    --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
    --temporal-occlusion-augmentation --temporal-occlusion-probability 0.7 \
    --temporal-occlusion-min-fraction 0.15 --temporal-occlusion-max-fraction 0.30 \
    --temporal-occlusion-track-fraction 0.5 --initial-relation-model "$initial" \
    --device cuda --seed 20260831 >"$out/train.log" 2>&1
}

run_pure_pivot() {
  local gpu=${1:-7}
  local out="$output_root/pure_vn_c_15ep_seed_20260831"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
    -m rgbd_urdf_mvp train-slot-relation-head \
    "$data_root/cotracker_learning_manifest.tsv" "$slot_model" \
    --output-dir "$out" --epochs 15 --object-batch-size 32 \
    --learning-rate 3e-4 --axis-geometry-branch --axis-head-type vector_neuron \
    --vector-pivot-parameterization pure_plane_residual_v1 \
    --relation-train-scope vector_pivot_only \
    --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
    --temporal-occlusion-augmentation --temporal-occlusion-probability 0.7 \
    --temporal-occlusion-min-fraction 0.15 --temporal-occlusion-max-fraction 0.30 \
    --temporal-occlusion-track-fraction 0.5 --initial-relation-model "$initial" \
    --device cuda --seed 20260831 >"$out/train.log" 2>&1
}

run_eval() {
  local checkpoint=$1 name=$2 gpu=${3:-6}
  local out="$output_root/$name"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
    "$workspace/scripts/evaluate_relation_head_by_category.py" \
    "$data_root/cotracker_learning_manifest.tsv" "$slot_model" "$checkpoint" \
    "$data_root/catalog.json" --output-json "$out/full_test_metrics.json" \
    --output-csv "$out/full_test_per_object.csv" --device cuda \
    >"$out/eval.log" 2>&1
}

case "${1:-}" in
  train) run_train "${2:-7}" ;;
  pivot_only) run_pivot_only "${2:-7}" ;;
  pure_pivot) run_pure_pivot "${2:-7}" ;;
  eval) run_eval "$2" "$3" "${4:-6}" ;;
  *) echo "usage: $0 train [GPU] | pivot_only [GPU] | pure_pivot [GPU] | eval CHECKPOINT NAME [GPU]" >&2; exit 2 ;;
esac
