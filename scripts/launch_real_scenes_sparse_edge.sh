#!/usr/bin/env bash
set -euo pipefail

workspace=/lumos-vePFS/suzhou/Users/lixiaotong/neural_head_so3_pilot_workspace
checkpoint="$workspace/outputs/neural_head_pivot_b_pilot_v1/relation_corruption_sparse_edge_20260831_10ep_seed_20260831/slot_relation_head.pt"
output="$workspace/outputs/neural_head_pivot_b_pilot_v1/real_scenes_sparse_edge_20260831"
runner=/tmp/rerun_real_scenes_pivot_b.sh
variant=sparse_edge_20260831
infer_args="--gate-selected-edges --edge-threshold 0.5"

tmux kill-session -t sparse_edge_real_a 2>/dev/null || true
tmux kill-session -t sparse_edge_real_b 2>/dev/null || true

tmux new-session -d -s sparse_edge_real_a \
  "bash -c '$runner scene29 4 $checkpoint $variant $output quality 0 $infer_args && $runner scene31 4 $checkpoint $variant $output quality 0 $infer_args && $runner scene34 4 $checkpoint $variant $output quality 0 $infer_args && $runner drawer_hand 4 $checkpoint $variant $output quality 0 $infer_args' > /tmp/sparse_edge_real_a.log 2>&1"

tmux new-session -d -s sparse_edge_real_b \
  "bash -c '$runner scene30 5 $checkpoint $variant $output quality 0 $infer_args && $runner scene32 5 $checkpoint $variant $output quality 0 $infer_args && $runner drawer_hand2 5 $checkpoint $variant $output quality 0 $infer_args' > /tmp/sparse_edge_real_b.log 2>&1"

tmux ls | grep sparse_edge_real
