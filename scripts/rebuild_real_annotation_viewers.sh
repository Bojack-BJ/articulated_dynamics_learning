#!/usr/bin/env bash
set -uo pipefail

checkpoint=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
output=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes_quality_trained

for scene in scene30 scene31 scene32 scene34 drawer_hand drawer_hand2; do
  if ! /tmp/rerun_real_scenes_pivot_b.sh \
    "$scene" 0 "$checkpoint" pivot_b_quality_trained "$output" quality 0; then
    printf 'Failed to rebuild %s\n' "$scene" >&2
  fi
done
