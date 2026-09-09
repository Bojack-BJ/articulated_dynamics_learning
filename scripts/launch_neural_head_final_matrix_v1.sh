#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
python=/root/Users/miniconda3/envs/particulate/bin/python
manifest=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1/cotracker_learning_manifest.tsv
slot_model=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
output_root="$workspace/outputs/neural_head_final_matrix_v1"

common=(
  -m rgbd_urdf_mvp train-slot-relation-head
  "$manifest" "$slot_model"
  --epochs 50
  --object-batch-size 32
  --learning-rate 3e-4
  --axis-geometry-branch
  --axis-head-type vector_neuron
  --joint-type-loss-weight 2.0
  --slot-geometry-representation invariant_v1
  --temporal-occlusion-augmentation
  --temporal-occlusion-probability 0.7
  --temporal-occlusion-min-fraction 0.15
  --temporal-occlusion-max-fraction 0.30
  --temporal-occlusion-track-fraction 0.5
  --device cuda
)

run_all_unfreeze() {
  local variant=$1 seed=$2 gpu=$3
  local out="$output_root/$variant/seed_$seed"
  mkdir -p "$out"
  local rotation=()
  if [[ "$variant" == haar_all ]]; then
    rotation=(
      --rotation-augmentation
      --rotation-augmentation-probability 1.0
      --rotation-augmentation-mode uniform_quaternion
      --rotation-augmentation-scope slot_and_relation_geometry
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" \
    "$python" "${common[@]}" --output-dir "$out" --seed "$seed" \
    --unfreeze-slot-backbone --slot-unfreeze-scope all \
    --slot-learning-rate-scale 0.1 "${rotation[@]}" \
    >"$out/train.log" 2>&1
}

run_frozen_then_decoder() {
  local seed=$1 gpu=$2
  local frozen="$output_root/no_rotation_frozen/seed_$seed"
  local decoder="$output_root/no_rotation_decoder/seed_$seed"
  mkdir -p "$frozen" "$decoder"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" \
    "$python" "${common[@]}" --output-dir "$frozen" --seed "$seed" \
    >"$frozen/train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" \
    "$python" "${common[@]}" --output-dir "$decoder" --seed "$seed" \
    --epochs 10 --initial-relation-model "$frozen/slot_relation_head.pt" \
    --unfreeze-slot-backbone --slot-unfreeze-scope decoder \
    --slot-learning-rate-scale 0.1 \
    >"$decoder/train.log" 2>&1
}

case "${1:-}" in
  all)
    run_all_unfreeze "$2" "$3" "$4"
    ;;
  frozen_decoder)
    run_frozen_then_decoder "$2" "$3"
    ;;
  *)
    echo "usage: $0 all VARIANT SEED GPU | frozen_decoder SEED GPU" >&2
    exit 2
    ;;
esac
