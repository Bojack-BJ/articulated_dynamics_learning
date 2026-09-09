#!/usr/bin/env bash
set -euo pipefail

worker_index=${1:?worker index required}
worker_count=${2:?worker count required}
gpu=${3:?GPU required}

root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/HOI4D_articulation_v1
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
manifest=/tmp/hoi4d_priority30_sequences.csv
episode_root=$root/priority15_track2art_v4
gt_root=$root/priority15_review_v5
output_root=$root/priority15_flow_viewers_v3

mapfile -t names < <(
  tail -n +2 "$manifest" | awk -F, '$4 ~ /^(C3|C4|C6|C14)$/ {gsub("/", "_", $2); print $2}'
)

for index in "${!names[@]}"; do
  (( index % worker_count == worker_index )) || continue
  name=${names[$index]}
  episode=$episode_root/$name/episode.json
  gt=$gt_root/$name/relation_gt.json
  out=$output_root/$name
  tracks=$out/part_tracks.json
  viewer=$out/viewer_gt_axis.html
  mkdir -p "$out"
  if [[ ! -s "$tracks" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
      -m rgbd_urdf_mvp track-part-pixels "$episode" \
      --output-json "$tracks" --device cuda \
      --cotracker-repo /lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/co-tracker \
      --cotracker-checkpoint /lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/co-tracker/ckpt/scaled_offline.pth \
      --reference-frame -1 --frame-stride 3 --seed-stride-px 10 \
      --max-tracks-per-part-view 500 --max-queries-per-forward 512 \
      --visibility-threshold 0.5 --dynamic-reseeding --dynamic-reseed-bidirectional \
      --reseed-interval-frames 8 --reseed-coverage-radius-px 12 \
      --reseed-max-tracks-per-frame-view 32 --reseed-max-tracks-per-view 300 \
      --repair-temporal-depth-spikes >"$out/track.log" 2>&1
  fi
  PYTHONPATH=$code/src "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$tracks" --gt-joint-annotation "$gt" --background-episode "$episode" \
    --background-exclude-object-mask --background-persistent \
    --background-voxel-size-m 0.01 --background-max-points 12000 \
    --max-tracks 1200 --axis-remap x,y,z --output-html "$viewer" \
    >"$out/viewer.log" 2>&1
done
