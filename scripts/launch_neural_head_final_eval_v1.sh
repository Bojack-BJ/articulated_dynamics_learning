#!/usr/bin/env bash
set -euo pipefail

variant=$1
seed=$2
gpu=$3

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
data_root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
matrix_root="$workspace/outputs/neural_head_final_matrix_v1"
checkpoint="$matrix_root/$variant/seed_$seed/slot_relation_head.pt"
output="$matrix_root/$variant/seed_$seed"
python=/root/Users/miniconda3/envs/particulate/bin/python
evaluator=/tmp/evaluate_relation_head_by_category.py
slot_model=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt

run_eval() {
  local manifest=$1
  local name=$2
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" "$evaluator" \
    "$manifest" "$slot_model" "$checkpoint" "$data_root/catalog.json" \
    --output-json "$output/${name}_metrics.json" --device cuda \
    >"$output/${name}_eval.log" 2>&1
}

run_eval "$data_root/cotracker_learning_manifest.tsv" full_test
run_eval "$matrix_root/aligned20_relation_manifest.tsv" aligned20
