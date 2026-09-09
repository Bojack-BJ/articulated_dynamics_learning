#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
base=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1/rh201_cabinet
out=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/arti4d_single_scene_scratch_overfit_v1
random_slot=$out/random_slot.pt
manifest=$out/manifest.tsv

mkdir -p "$out"
cd "$code"
PYTHONPATH=src "$python" scripts/make_random_slot_checkpoint.py "$base" "$random_slot"
printf 'object_id\ttracks_path\tfeatures_npz\tsplit\trelation_gt_path\n' > "$manifest"
printf 'cabinet_scratch_train\t%s\t%s\ttrain\t%s\n' \
  "$data/track_quality/motion_part_tracks_with_quality.json" \
  "$data/tracking/cotracker_features.npz" "$data/relation_gt.json" >> "$manifest"
printf 'cabinet_scratch_val\t%s\t%s\tval\t%s\n' \
  "$data/track_quality/motion_part_tracks_with_quality.json" \
  "$data/tracking/cotracker_features.npz" "$data/relation_gt.json" >> "$manifest"

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src "$python" -m rgbd_urdf_mvp train-slot-relation-head \
  "$manifest" "$random_slot" --output-dir "$out" --epochs 100 --object-batch-size 1 \
  --learning-rate 3e-4 --weight-decay 0 --axis-geometry-branch \
  --axis-head-type vector_neuron --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --slot-assignment-loss-weight 2.0 \
  --slot-existence-loss-weight 1.0 --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --relation-train-scope all --unfreeze-slot-backbone --slot-unfreeze-scope all \
  --slot-learning-rate-scale 1.0 --device cuda --seed 20260908 > "$out/train.log" 2>&1
