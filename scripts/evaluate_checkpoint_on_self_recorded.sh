#!/usr/bin/env bash
set -euo pipefail

relation=${1:?usage: evaluate_checkpoint_on_self_recorded.sh RELATION LABEL GPU}
label=${2:?usage: evaluate_checkpoint_on_self_recorded.sh RELATION LABEL GPU}
gpu=${3:-5}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
base_slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots_view_dropout_v1/motion_part_slots.pt
out_root=$workspace/outputs/balanced_real_generalization_v1/self_recorded_eval/$label

scenes=(scene29 scene30 scene31 scene32 scene34 drawer_hand drawer_hand2)
mkdir -p "$out_root"
slot=$out_root/slot_from_relation.pt
PYTHONPATH=$code/src "$python" "$code/scripts/extract_slot_checkpoint_from_relation.py" \
  "$base_slot" "$relation" "$slot"

for scene in "${scenes[@]}"; do
  source_dir=$data/outputs/real_${scene}_track2art_dense_filtered_v1
  out=$out_root/$scene
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" -m rgbd_urdf_mvp \
    infer-motion-part-slots \
    "$source_dir/track_quality/motion_part_tracks_with_quality.json" \
    "$source_dir/tracking/cotracker_features.npz" "$slot" \
    --output-json "$out/slots.json" --device cuda \
    --min-visible-frames 4 --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" -m rgbd_urdf_mvp \
    infer-slot-relation-head \
    "$source_dir/track_quality/motion_part_tracks_with_quality.json" \
    "$source_dir/tracking/cotracker_features.npz" "$slot" "$relation" \
    --output-json "$out/joints.json" --device cuda \
    --quality-weighted-trajectories
  PYTHONPATH=$code/src "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$out/slots.json" --joint-inference "$out/joints.json" \
    --background-episode "$source_dir/episode.json" --background-exclude-object-mask \
    --background-persistent --background-voxel-size-m 0.02 \
    --background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z \
    --output-html "$out/viewer.html"
done
