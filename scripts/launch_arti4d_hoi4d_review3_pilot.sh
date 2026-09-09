#!/usr/bin/env bash
set -euo pipefail

root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/HOI4D_articulation_v1
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
partnet=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
arti=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
initial="$workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt"
out="$workspace/outputs/arti4d_hoi4d_review3_pilot_v1"
manifest="$out/train_manifest.tsv"

scenes=(
  ZY20210800001_H1_C14_N28_S155_s01_T1
  ZY20210800001_H1_C3_N46_S114_s03_T2
  ZY20210800001_H1_C4_N35_S21_s04_T2
)

prepare_scene() {
  local scene=$1 gpu=$2
  local episode="$root/priority15_track2art_v4/$scene/episode.json"
  local scene_out="$out/hoi4d/$scene"
  local tracks="$scene_out/tracking/part_tracks.json"
  local features="$scene_out/tracking/cotracker_features.npz"
  local quality="$scene_out/track_quality/motion_part_tracks_with_quality.json"
  mkdir -p "$scene_out/tracking" "$scene_out/track_quality"
  if [[ -s "$tracks" && -s "$features" && -s "$quality" ]]; then
    echo "Reusing prepared HOI4D scene: $scene"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$code/src" "$python" \
    -m rgbd_urdf_mvp track-part-pixels "$episode" \
    --output-json "$tracks" --device cuda \
    --cotracker-repo /lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/co-tracker \
    --cotracker-checkpoint /lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/co-tracker/ckpt/scaled_offline.pth \
    --reference-frame -1 --frame-stride 3 --seed-stride-px 10 \
    --max-tracks-per-part-view 500 --max-queries-per-forward 512 \
    --visibility-threshold 0.5 --dynamic-reseeding --dynamic-reseed-bidirectional \
    --reseed-interval-frames 8 --reseed-coverage-radius-px 12 \
    --reseed-max-tracks-per-frame-view 32 --reseed-max-tracks-per-view 300 \
    --repair-temporal-depth-spikes --export-cotracker-features \
    --cotracker-features-output "$features" >"$scene_out/track.log" 2>&1
  PYTHONPATH="$code/src" "$python" -m rgbd_urdf_mvp compute-track-quality \
    "$tracks" --output-dir "$scene_out/track_quality" --mask-bad-timesteps \
    --max-step-m 0.12 --keep-query-connected-segment \
    --min-query-connected-frames 4 --query-connected-max-gap-frames 2 \
    >"$scene_out/quality.log" 2>&1
}

mkdir -p "$out"
prepare_scene "${scenes[0]}" 6 &
p0=$!
prepare_scene "${scenes[1]}" 7 &
p1=$!
wait "$p0" "$p1"
prepare_scene "${scenes[2]}" 7

PYTHONPATH="$code/src" "$python" "$code/scripts/build_arti4d_relation_manifest.py" \
  "$partnet/cotracker_learning_manifest.tsv" "$arti" "$manifest" --repeat 10 \
  --train rh201_stove_oven --train rh201_cabinet --train rh201_top_drawer \
  --train rh078_blue_drawer --train rh078_right_drawer_1 \
  --val rh201_fridge --val rh078_cabinet_right

for scene in "${scenes[@]}"; do
  tracks="$out/hoi4d/$scene/track_quality/motion_part_tracks_with_quality.json"
  features="$out/hoi4d/$scene/tracking/cotracker_features.npz"
  gt="$root/priority15_manual_gt_v1/$scene/relation_gt_manual.json"
  for replica in $(seq 0 5); do
    printf 'hoi4d_%s_r%02d\t%s\t%s\ttrain\t%s\n' \
      "$scene" "$replica" "$tracks" "$features" "$gt" >>"$manifest"
  done
done

train_out="$out/relation_all_5ep"
mkdir -p "$train_out"
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$code/src" "$python" \
  -m rgbd_urdf_mvp train-slot-relation-head \
  "$manifest" "$slot" --output-dir "$train_out" --epochs 5 --object-batch-size 32 \
  --learning-rate 2e-5 --axis-geometry-branch --axis-head-type vector_neuron \
  --vector-pivot-parameterization analytic_plane_residual_v1 \
  --joint-type-loss-weight 2.0 --slot-geometry-representation invariant_v1 \
  --quality-weighted-trajectories --robust-segment-weights \
  --initial-relation-model "$initial" --unfreeze-slot-backbone \
  --slot-unfreeze-scope decoder --slot-learning-rate-scale 0.1 \
  --slot-assignment-loss-weight 1.0 \
  --slot-existence-loss-weight 0.25 \
  --slot-assignment-consistency-loss-weight 0.25 \
  --device cuda --seed 20260831 \
  >"$train_out/train.log" 2>&1
