#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
python=/root/Users/miniconda3/envs/particulate/bin/python
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial="$workspace/outputs/neural_head_pivot_b_pilot_v1/pivot_only_10ep_seed_20260831/slot_relation_head.pt"
out="$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831"
gpu=${1:-7}

mkdir -p "$out"
started=$(date +%s)
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head \
  "$data/cotracker_learning_manifest.tsv" "$slot" \
  --output-dir "$out" --epochs 10 --object-batch-size 32 \
  --learning-rate 1e-4 --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories \
  --temporal-occlusion-augmentation --temporal-occlusion-probability 0.7 \
  --temporal-occlusion-min-fraction 0.15 --temporal-occlusion-max-fraction 0.30 \
  --temporal-occlusion-track-fraction 0.5 --initial-relation-model "$initial" \
  --device cuda --seed 20260831 >"$out/train.log" 2>&1
finished=$(date +%s)
printf '{"started_unix":%s,"finished_unix":%s,"wall_seconds":%s}\n' \
  "$started" "$finished" "$((finished-started))" >"$out/runtime.json"

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
  "$workspace/scripts/evaluate_relation_head_by_category.py" \
  "$data/cotracker_learning_manifest.tsv" "$slot" "$out/slot_relation_head.pt" \
  "$data/catalog.json" --output-json "$out/full_test_metrics.json" \
  --output-csv "$out/full_test_per_object.csv" --device cuda \
  >"$out/full_test_eval.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
  "$workspace/scripts/evaluate_relation_head_by_category.py" \
  "$workspace/outputs/neural_head_final_matrix_v1/aligned20_relation_manifest.tsv" \
  "$slot" "$out/slot_relation_head.pt" "$data/catalog.json" \
  --output-json "$out/aligned20_metrics.json" \
  --output-csv "$out/aligned20_per_object.csv" --device cuda \
  >"$out/aligned20_eval.log" 2>&1
