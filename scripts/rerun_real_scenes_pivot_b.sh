#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
data_workspace=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning
python=/root/Users/miniconda3/envs/particulate/bin/python
slot_model=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
relation_model=${3:-"$workspace/outputs/neural_head_pivot_b_pilot_v1/pivot_only_10ep_seed_20260831/slot_relation_head.pt"}
variant=${4:-pivot_b}
output_root=${5:-"$workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes"}
quality_mode=${6:-binary}
min_trajectory_quality=${7:-0}
extra_infer_args=("${@:8}")
quality_args=()
if [[ "$quality_mode" == quality ]]; then
  quality_args+=(--quality-weighted-trajectories --min-trajectory-quality "$min_trajectory_quality")
fi

scene=$1
gpu=${2:-6}
case "$scene" in
  scene*) source_name="real_${scene}_track2art_dense_filtered_v1" ;;
  drawer_hand*) source_name="real_${scene}_track2art_dense_filtered_v1" ;;
  *) echo "unknown scene: $scene" >&2; exit 2 ;;
esac
source_dir="$data_workspace/outputs/$source_name"
out="$output_root/$scene"
mkdir -p "$out"

if [[ ! -s "$out/joint_inference_${variant}.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$workspace/src" "$python" \
    -m rgbd_urdf_mvp infer-slot-relation-head \
    "$source_dir/track_quality/motion_part_tracks_with_quality.json" \
    "$source_dir/tracking/cotracker_features.npz" "$slot_model" "$relation_model" \
    --output-json "$out/joint_inference_${variant}.json" --device cuda \
    "${quality_args[@]}" "${extra_infer_args[@]}"
fi

PYTHONPATH="$data_workspace/src" "$python" \
  -m rgbd_urdf_mvp visualize-object-mask-flow-html \
  "$source_dir/slots/motion_part_tracks_slots.json" \
  --joint-inference "$out/joint_inference_${variant}.json" \
  --background-episode "$source_dir/episode.json" --background-exclude-object-mask \
  --background-max-points 10000 --max-tracks 1200 --axis-remap x,y,z \
  --output-html "$out/viewer_${variant}.html"
