#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
code=/tmp/track2art_arti4d_relation_v3
python=/root/Users/miniconda3/envs/particulate/bin/python
data=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1
out=/lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots_view_dropout_v1

mkdir -p "$out"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$code/src "$python" \
  -m rgbd_urdf_mvp train-motion-part-slots "$data/cotracker_learning_manifest.tsv" \
  --output-dir "$out" --max-slots 16 --hidden-dim 128 \
  --encoder-layers 2 --decoder-layers 2 --attention-heads 4 \
  --epochs 100 --learning-rate 3e-4 --weight-decay 1e-4 \
  --rigid-loss-weight 0.2 --dice-loss-weight 0.5 --pairwise-loss-weight 0.35 \
  --existence-loss-weight 0.25 --track-dropout-ratio 0.1 \
  --view-dropout-probability 0.7 --max-dropped-views 2 \
  --pair-samples-per-object 4096 --object-batch-size 16 \
  --data-loader-workers 16 --device cuda --seed 0 >"$out/train.log" 2>&1
