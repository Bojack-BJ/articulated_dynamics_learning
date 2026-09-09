#!/usr/bin/env bash
set -euo pipefail

scene=$1
gpu=${2:-0}
src=${TRACK2ART_SRC:-/tmp/track2art_arti4d_relation_v3/src}
python=${TRACK2ART_PYTHON:-/root/Users/miniconda3/envs/particulate/bin/python}
root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
model_root=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
relation=$model_root/arti4d_relation_finetune_v1/holdout_cabinet_right_5ep/slot_relation_head.pt
out=$root/$scene
tracks=$out/track_quality/motion_part_tracks_with_quality.json
features=$out/tracking/cotracker_features.npz
prediction=$out/relation_viewers/joint_inference_finetuned.json
override=$out/relation_viewers/oracle_assignment_override.json

case "$scene" in
  rh201_stove_oven) episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_stove_oven/annotations/track2art_manual_v1/episode.live_sam2_video_f0026_v0_1788697095.json ;;
  rh201_fridge) episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_fridge/annotations/track2art_manual_v1/episode.live_sam2_video_f0005_v0_1788699102.json ;;
  rh201_cabinet) episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_cabinet/annotations/track2art_manual_v1/episode.live_sam2_video_f0041_v0_1788699497.json ;;
  rh078_cabinet_right) episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_subsets/arti4d_cabinet_right/annotations/track2art_manual_v1/episode.live_sam2_video_f0038_v0_1788692207.json ;;
  *) echo "Unknown scene: $scene" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$src
"$python" /tmp/track2art_arti4d_relation_v3/scripts/build_oracle_slot_override.py \
  "$tracks" "$prediction" "$override"

for variant in gated oracle; do
  output=$out/relation_viewers/joint_inference_${variant}.json
  extra=(--slot-existence-threshold 0.8 --gate-selected-edges \
    --min-edge-observability 0.1 --min-axis-observability 0.1 --min-axis-confidence 0.2)
  if [[ "$variant" == oracle ]]; then
    extra+=(--trajectory-assignment-override "$override")
  fi
  "$python" -m rgbd_urdf_mvp infer-slot-relation-head \
    "$tracks" "$features" "$slot" "$relation" --output-json "$output" \
    --device cuda "${extra[@]}"
  "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$out/slots/motion_part_tracks_slots.json" --joint-inference "$output" \
    --background-episode "$episode" --background-exclude-object-mask \
    --background-persistent --background-voxel-size-m 0.02 \
    --background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z \
    --output-html "$out/relation_viewers/viewer_${variant}.html"
done
