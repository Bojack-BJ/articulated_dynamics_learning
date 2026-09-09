#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
checkpoint="$workspace/outputs/neural_head_pivot_b_pilot_v1/relation_corruption_sensor_joint_20260831_15ep_seed_20260831/slot_relation_head.pt"
output="$workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes_sensor_joint_20260831"
runner=/tmp/rerun_real_scenes_pivot_b.sh
variant=sensor_joint_20260831

tmux kill-session -t sensor_joint_real_a 2>/dev/null || true
tmux kill-session -t sensor_joint_real_b 2>/dev/null || true

tmux new-session -d -s sensor_joint_real_a \
  "bash -lc '$runner scene29 6 $checkpoint $variant $output quality && $runner scene31 6 $checkpoint $variant $output quality && $runner scene34 6 $checkpoint $variant $output quality && $runner drawer_hand 6 $checkpoint $variant $output quality' > /tmp/sensor_joint_real_a.log 2>&1"

tmux new-session -d -s sensor_joint_real_b \
  "bash -lc '$runner scene30 7 $checkpoint $variant $output quality && $runner scene32 7 $checkpoint $variant $output quality && $runner drawer_hand2 7 $checkpoint $variant $output quality' > /tmp/sensor_joint_real_b.log 2>&1"

tmux ls | grep sensor_joint_real
