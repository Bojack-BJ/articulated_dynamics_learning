#!/usr/bin/env bash
set -euo pipefail

tmux new-session -d -s pb_real_a \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene29 6 && /tmp/rerun_real_scenes_pivot_b.sh scene31 6 && /tmp/rerun_real_scenes_pivot_b.sh scene34 6 && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand 6' > /tmp/pb_real_a.log 2>&1"

tmux new-session -d -s pb_real_b \
  "bash -lc '/tmp/rerun_real_scenes_pivot_b.sh scene30 7 && /tmp/rerun_real_scenes_pivot_b.sh scene32 7 && /tmp/rerun_real_scenes_pivot_b.sh drawer_hand2 7' > /tmp/pb_real_b.log 2>&1"

tmux ls | grep -E 'pb_(haar|real)'
