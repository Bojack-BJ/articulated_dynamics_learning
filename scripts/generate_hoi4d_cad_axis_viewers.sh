#!/usr/bin/env bash
set -euo pipefail

worker_index=${1:?worker index required}
worker_count=${2:?worker count required}
root=/lumos-vePFS/suzhou/Users/lixiaotong/datasets/HOI4D_articulation_v1
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
viewer_root=$root/priority15_flow_viewers_v3

mapfile -t names < <(find "$viewer_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
for index in "${!names[@]}"; do
  (( index % worker_count == worker_index )) || continue
  name=${names[$index]}
  tracks=$viewer_root/$name/part_tracks.json
  episode=$root/priority15_track2art_v4/$name/episode.json
  cad_gt=$root/priority15_cad_axis_v1/$name/relation_gt_cad.json
  output=$viewer_root/$name/viewer_cad_axis.html
  [[ -s "$tracks" && -s "$episode" && -s "$cad_gt" ]] || continue
  PYTHONPATH=$code/src "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$tracks" --gt-joint-annotation "$cad_gt" --background-episode "$episode" \
    --background-exclude-object-mask --background-persistent \
    --background-voxel-size-m 0.01 --background-max-points 12000 \
    --max-tracks 1200 --axis-remap x,y,z --output-html "$output" \
    >"$viewer_root/$name/viewer_cad.log" 2>&1
done
