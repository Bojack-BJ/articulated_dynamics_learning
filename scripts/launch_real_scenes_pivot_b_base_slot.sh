#!/usr/bin/env bash
set -euo pipefail

checkpoint=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/pivot_only_10ep_seed_20260831/slot_relation_head_relation_only.pt
output=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes_base_slot

tmux new-session -d -s pb_base_a \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene29 6 $checkpoint pivot_b_base_slot $output && /tmp/rerun_real_scenes_pivot_b.sh scene31 6 $checkpoint pivot_b_base_slot $output && /tmp/rerun_real_scenes_pivot_b.sh scene34 6 $checkpoint pivot_b_base_slot $output && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand 6 $checkpoint pivot_b_base_slot $output' > /tmp/pb_base_a.log 2>&1"

tmux new-session -d -s pb_base_b \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene30 7 $checkpoint pivot_b_base_slot $output && /tmp/rerun_real_scenes_pivot_b.sh scene32 7 $checkpoint pivot_b_base_slot $output && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand2 7 $checkpoint pivot_b_base_slot $output' > /tmp/pb_base_b.log 2>&1"
