#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
arti=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
base=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
models=$workspace/outputs/balanced_real_generalization_v1
viewers=$models/fold1_viewers

export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src
for scene in rh201_fridge rh078_cabinet_right; do
  scene_root=$arti/$scene
  tracks=$scene_root/track_quality/motion_part_tracks_with_quality.json
  features=$scene_root/tracking/cotracker_features.npz
  episode=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["input_episode_path"])' "$tracks")
  mkdir -p "$viewers/$scene"
  for variant in real_only_scratch mixed; do
    relation=$models/fold1_$variant/slot_relation_head.pt
    slot=$viewers/$scene/slot_$variant.pt
    "$python" "$code/scripts/extract_slot_checkpoint_from_relation.py" "$base" "$relation" "$slot"
    "$python" -m rgbd_urdf_mvp infer-motion-part-slots "$tracks" "$features" "$slot" \
      --output-json "$viewers/$scene/slots_$variant.json" --device cuda \
      --min-visible-frames 4 --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15
    "$python" -m rgbd_urdf_mvp infer-slot-relation-head "$tracks" "$features" \
      "$slot" "$relation" --output-json "$viewers/$scene/joints_$variant.json" --device cuda
    "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
      "$viewers/$scene/slots_$variant.json" \
      --joint-inference "$viewers/$scene/joints_$variant.json" \
      --gt-joint-annotation "$scene_root/relation_gt.json" \
      --background-episode "$episode" --background-exclude-object-mask \
      --background-persistent --background-voxel-size-m 0.02 \
      --background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z \
      --output-html "$viewers/$scene/viewer_$variant.html"
  done
done
