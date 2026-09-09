#!/usr/bin/env bash
set -euo pipefail

scene=$1
requested_variant=${2:-all}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
root=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs/arti4d_relation_features_v1
out=$root/$scene
tracks=$out/track_quality/motion_part_tracks_with_quality.json
viewers=$out/relation_viewers_final
episode=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["input_episode_path"])' "$tracks")
export PYTHONPATH=$code/src

"$python" "$code/scripts/apply_oracle_slot_override.py" \
  "$viewers/slots_relation_only.json" "$viewers/oracle_assignment_override.json" \
  "$viewers/slots_oracle.json"

variants=(baseline slot_only relation_only oracle)
[[ "$requested_variant" != all ]] && variants=("$requested_variant")
for variant in "${variants[@]}"; do
  slot_variant=$variant
  "$python" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$viewers/slots_${slot_variant}.json" \
    --joint-inference "$viewers/joint_inference_${variant}.json" \
    --gt-joint-annotation "$out/relation_gt.json" \
    --background-episode "$episode" --background-exclude-object-mask \
    --background-persistent --background-voxel-size-m 0.02 \
    --background-max-points 12000 --max-tracks 1200 --axis-remap x,y,z \
    --output-html "$viewers/viewer_${variant}.html"
done
