#!/usr/bin/env bash
set -euo pipefail

scene=$1
gpu=${2:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
models=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs
base_slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
baseline=$models/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
slot_only=$models/arti4d_relation_finetune_v1/slot_decoder_5scene_5ep/slot_relation_head.pt
relation_only=$models/arti4d_relation_finetune_v1/relation_5scene_low_lr_5ep/slot_relation_head.pt
slot_only_slot=$models/arti4d_relation_finetune_v1/slot_decoder_5scene_5ep/motion_part_slots.pt
out=$root/$scene
tracks=$out/track_quality/motion_part_tracks_with_quality.json
features=$out/tracking/cotracker_features.npz
episode=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["input_episode_path"])' "$tracks")
viewers=$out/relation_viewers_final
mkdir -p "$viewers"

if [[ ! -f "$slot_only_slot" ]]; then
  "$python" "$code/scripts/extract_slot_checkpoint_from_relation.py" \
    "$base_slot" "$slot_only" "$slot_only_slot"
fi

export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src
for variant in baseline slot_only relation_only; do
  relation=${!variant}
  slot=$base_slot
  [[ "$variant" == slot_only ]] && slot=$slot_only_slot
  slot_json=$viewers/slots_${variant}.json
  prediction=$viewers/joint_inference_${variant}.json
  "$python" -m rgbd_urdf_mvp infer-motion-part-slots "$tracks" "$features" "$slot" \
    --output-json "$slot_json" --device cuda --min-visible-frames 4 \
    --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15
  "$python" -m rgbd_urdf_mvp infer-slot-relation-head "$tracks" "$features" \
    "$slot" "$relation" --output-json "$prediction" --device cuda
  "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html "$slot_json" \
    --joint-inference "$prediction" --gt-joint-annotation "$out/relation_gt.json" \
    --background-episode "$episode" \
    --background-exclude-object-mask --background-persistent \
    --background-voxel-size-m 0.02 --background-max-points 12000 \
    --max-tracks 1200 --axis-remap x,y,z --output-html "$viewers/viewer_${variant}.html"
done

override=$viewers/oracle_assignment_override.json
"$python" "$code/scripts/build_oracle_slot_override.py" "$tracks" \
  "$viewers/joint_inference_relation_only.json" "$override"
"$python" -m rgbd_urdf_mvp infer-slot-relation-head "$tracks" "$features" \
  "$base_slot" "$relation_only" --output-json "$viewers/joint_inference_oracle.json" \
  --trajectory-assignment-override "$override" --device cuda
"$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
  "$viewers/slots_relation_only.json" --joint-inference "$viewers/joint_inference_oracle.json" \
  --gt-joint-annotation "$out/relation_gt.json" \
  --background-episode "$episode" --background-exclude-object-mask \
  --background-persistent --background-voxel-size-m 0.02 --background-max-points 12000 \
  --max-tracks 1200 --axis-remap x,y,z --output-html "$viewers/viewer_oracle.html"
