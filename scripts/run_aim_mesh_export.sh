#!/usr/bin/env bash
# Export AiM's official TSDF component meshes for paper-style voxel IoU.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 AIM_ROOT DATASET_DIR RUN_DIR LOG_PATH" >&2
  exit 2
fi

aim_root=$1
dataset_dir=$2
run_dir=${3%/}/
log_path=$4

export PYTHONPATH="/tmp:${aim_root}:${aim_root}/submodules/diff-gaussian-rasterization1/build/lib.linux-x86_64-cpython-310:${aim_root}/submodules/simple-knn/build/lib.linux-x86_64-cpython-310:${aim_root}/lib/pointops/build/lib.linux-x86_64-cpython-310${PYTHONPATH:+:${PYTHONPATH}}"

exec /root/Users/miniconda3/envs/aim/bin/python /tmp/render_main_compat.py \
  --source_path "$dataset_dir" \
  --model_path "$run_dir" \
  --is_blender \
  --eval \
  --random_bg_color \
  --iterations 13000 >"$log_path" 2>&1
