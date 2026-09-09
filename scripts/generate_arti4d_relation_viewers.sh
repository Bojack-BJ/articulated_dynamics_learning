#!/usr/bin/env bash
set -euo pipefail

scene=$1
gpu=${2:-0}
src=${TRACK2ART_SRC:-/tmp/track2art_arti4d_relation_v3/src}
python=${TRACK2ART_PYTHON:-/root/Users/miniconda3/envs/particulate/bin/python}
root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
model_root=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
baseline=$model_root/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
finetuned=$model_root/arti4d_relation_finetune_v1/holdout_cabinet_right_5ep/slot_relation_head.pt
out=$root/$scene

case "$scene" in
  rh201_stove_oven)
    episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_stove_oven/annotations/track2art_manual_v1/episode.live_sam2_video_f0026_v0_1788697095.json ;;
  rh201_fridge)
    episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_fridge/annotations/track2art_manual_v1/episode.live_sam2_video_f0005_v0_1788699102.json ;;
  rh201_cabinet)
    episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_manual_pilot_raw/rh201_cabinet/annotations/track2art_manual_v1/episode.live_sam2_video_f0041_v0_1788699497.json ;;
  rh078_cabinet_right)
    episode=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/arti4d_subsets/arti4d_cabinet_right/annotations/track2art_manual_v1/episode.live_sam2_video_f0038_v0_1788692207.json ;;
  *) echo "Unknown scene: $scene" >&2; exit 2 ;;
esac

tracks=$out/track_quality/motion_part_tracks_with_quality.json
features=$out/tracking/cotracker_features.npz
mkdir -p "$out/slots" "$out/relation_viewers"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$src

"$python" -m rgbd_urdf_mvp infer-motion-part-slots \
  "$tracks" "$features" "$slot" \
  --output-json "$out/slots/motion_part_tracks_slots.json" --device cuda \
  --min-visible-frames 4 --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15

for variant in baseline finetuned; do
  checkpoint=${!variant}
  prediction=$out/relation_viewers/joint_inference_${variant}.json
  "$python" -m rgbd_urdf_mvp infer-slot-relation-head \
    "$tracks" "$features" "$slot" "$checkpoint" \
    --output-json "$prediction" --device cuda
  "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$out/slots/motion_part_tracks_slots.json" --joint-inference "$prediction" \
    --background-episode "$episode" --background-exclude-object-mask \
    --background-persistent --background-voxel-size-m 0.02 \
    --background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z \
    --output-html "$out/relation_viewers/viewer_${variant}.html"
done
