#!/usr/bin/env bash
set -euo pipefail

checkpoint=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/quality_weighted_relation_10ep_seed_20260831/slot_relation_head.pt
output=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes_quality_trained

tmux new-session -d -s pb_quality_trained_a \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene29 6 $checkpoint pivot_b_quality_trained $output quality && /tmp/rerun_real_scenes_pivot_b.sh scene31 6 $checkpoint pivot_b_quality_trained $output quality && /tmp/rerun_real_scenes_pivot_b.sh scene34 6 $checkpoint pivot_b_quality_trained $output quality && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand 6 $checkpoint pivot_b_quality_trained $output quality' > /tmp/pb_quality_trained_a.log 2>&1"

tmux new-session -d -s pb_quality_trained_b \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene30 7 $checkpoint pivot_b_quality_trained $output quality && /tmp/rerun_real_scenes_pivot_b.sh scene32 7 $checkpoint pivot_b_quality_trained $output quality && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand2 7 $checkpoint pivot_b_quality_trained $output quality' > /tmp/pb_quality_trained_b.log 2>&1"

tmux ls | grep pb_quality_trained
