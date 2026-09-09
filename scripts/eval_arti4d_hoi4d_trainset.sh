#!/usr/bin/env bash
set -euo pipefail

python=/root/Users/miniconda3/envs/particulate/bin/python
code=/tmp/track2art_arti4d_relation_v3
manifest=/tmp/real_train_unique8_eval.tsv
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
catalog=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1/catalog.json
baseline=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
finetuned=${1:-/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/arti4d_hoi4d_review3_pilot_v1/relation_all_5ep/slot_relation_head.pt}
out=${2:-/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/arti4d_hoi4d_review3_pilot_v1/trainset_eval}

mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$code/src"
"$python" /tmp/evaluate_relation_head_by_category.py \
  "$manifest" "$slot" "$baseline" "$catalog" \
  --output-json "$out/baseline.json" --device cuda
"$python" /tmp/evaluate_relation_head_by_category.py \
  "$manifest" "$slot" "$finetuned" "$catalog" \
  --output-json "$out/finetuned.json" --device cuda
