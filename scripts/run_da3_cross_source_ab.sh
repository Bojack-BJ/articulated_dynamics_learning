#!/usr/bin/env bash
set -euo pipefail

scene=$1
gpu=$2
root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/da3_cross_source_pilot_v1/$scene
repo=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
slot=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt
relation=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
cotracker=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/co-tracker

cd "$repo"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu
for variant in raw temporal_fused; do
  if [[ $variant == raw ]]; then
    episode=$root/episode_view0_raw.json
  else
    episode=$root/temporal_fused/episode_sensor_da3_fused.json
  fi
  out=$root/ab/$variant
  mkdir -p "$out/tracking" "$out/track_quality" "$out/slots" "$out/viewer"
  "$python" -m rgbd_urdf_mvp track-part-pixels "$episode" \
    --output-json "$out/tracking/part_tracks.json" --device cuda \
    --cotracker-repo "$cotracker" --cotracker-checkpoint "$cotracker/ckpt/scaled_offline.pth" \
    --reference-frame -1 --frame-stride 2 --seed-stride-px 8 \
    --max-tracks-per-part-view 600 --max-queries-per-forward 512 \
    --visibility-threshold 0.5 --dynamic-reseeding --dynamic-reseed-bidirectional \
    --export-cotracker-features --cotracker-features-output "$out/tracking/cotracker_features.npz" \
    --reseed-interval-frames 8 --reseed-coverage-radius-px 10 \
    --reseed-max-tracks-per-frame-view 48 --reseed-max-tracks-per-view 384 \
    --repair-temporal-depth-spikes --depth-consistency-window-radius-px 2 \
    --depth-consistency-max-delta-m 0.08 > "$out/tracking/track.log" 2>&1
  "$python" -m rgbd_urdf_mvp compute-track-quality "$out/tracking/part_tracks.json" \
    --output-dir "$out/track_quality" --mask-bad-timesteps --max-step-m 0.05 >/dev/null
  "$python" -m rgbd_urdf_mvp infer-motion-part-slots \
    "$out/track_quality/motion_part_tracks_with_quality.json" "$out/tracking/cotracker_features.npz" "$slot" \
    --output-json "$out/slots/motion_part_tracks_slots.json" --device cuda \
    --min-visible-frames 4 --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15 >/dev/null
  "$python" -m rgbd_urdf_mvp infer-slot-relation-head \
    "$out/track_quality/motion_part_tracks_with_quality.json" "$out/tracking/cotracker_features.npz" \
    "$slot" "$relation" --output-json "$out/viewer/joint_inference_partnet.json" --device cuda >/dev/null
  "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$out/slots/motion_part_tracks_slots.json" --joint-inference "$out/viewer/joint_inference_partnet.json" \
    --background-episode "$episode" --background-exclude-object-mask --background-persistent \
    --background-voxel-size-m 0.02 --background-max-points 12000 --max-tracks 1200 \
    --axis-remap x,y,z --output-html "$out/viewer/viewer.html" >/dev/null
done
