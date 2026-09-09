#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
arti=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
root=$workspace/outputs/balanced_real_generalization_v1
manifest=$root/fold1_train_unique.tsv
catalog=$root/fold1_catalog.json

PYTHONPATH=$code/src "$python" "$code/scripts/build_balanced_real_partnet_manifest.py" \
  "$partnet/cotracker_learning_manifest.tsv" "$arti" "$manifest.tmp" \
  --real-repeats 1 --partnet-fraction 0 --seed 20260908 \
  --train rh201_stove_oven --train rh201_cabinet --train rh201_top_drawer \
  --train rh078_blue_drawer --train rh078_right_drawer_1 \
  --val rh201_fridge --val rh078_cabinet_right
awk -F '\t' 'BEGIN{OFS="\t"} NR==1{print;next} $4=="train"{$4="test"; print}' \
  "$manifest.tmp" >"$manifest"
"$python" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); json.dump({"objects":[{"object_id":r["object_id"],"category":r["object_id"].split("_r")[0]} for r in rows]},open(sys.argv[2],"w"))' "$manifest" "$catalog"

for variant in real_only_scratch mixed; do
  model=$root/fold1_$variant/slot_relation_head.pt
  out=$root/fold1_$variant/trainset_eval
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
    "$code/scripts/evaluate_relation_head_by_category.py" \
    "$manifest" "$partnet/../partnet_core_v1_training/cotracker_slots/motion_part_slots.pt" \
    "$model" "$catalog" --output-json "$out/per_object.json" --device cuda
done
