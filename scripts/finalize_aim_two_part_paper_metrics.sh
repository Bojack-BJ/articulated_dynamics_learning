#!/usr/bin/env bash
# Finish paper-style AiM metrics after asynchronous TSDF mesh exports complete.
set -euo pipefail

root=${1:?usage: $0 PROJECT_ROOT}
run_root="$root/outputs/aim_two_part_front_oscillate_v1/runs_5k8k"
record_root="$root/outputs/aim_two_part_front_oscillate_v1/recordings"
eval_root="$root/outputs/aim_two_part_front_oscillate_v1/evaluation"
metrics_root="$eval_root/paper_metrics"

# Renderer PIDs are intentionally passed as positional arguments so this
# process never mistakes itself for an export process while polling.
shift
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
done

cd "$root"
mkdir -p "$metrics_root"
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_aim_mesh_voxel_iou.py \
  "$record_root/partnet_46556_front_oscillate_v1/episode.json" \
  "$run_root/partnet_46556_front_oscillate_v1_retry" --state end \
  --output-json "$metrics_root/partnet_46556/mesh_voxel_iou.json"
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_aim_mesh_voxel_iou.py \
  "$record_root/partnet_12540_front_oscillate_v1/episode.json" \
  "$run_root/partnet_12540_front_oscillate_v1" --state end \
  --output-json "$metrics_root/partnet_12540/mesh_voxel_iou.json"
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_aim_mesh_voxel_iou.py \
  "$record_root/partnet_10849_front_oscillate_v1/episode.json" \
  "$run_root/partnet_10849_front_oscillate_v1_retry" --state end \
  --output-json "$metrics_root/partnet_10849/mesh_voxel_iou.json"

PYTHONPATH=src ./.venv/bin/python scripts/evaluate_aim_motion_axis.py \
  --output-dir "$metrics_root" \
  --object partnet_9388 "$run_root/partnet_9388_front_oscillate_v1" "$eval_root/partnet_9388_front_oscillate_v1.json" "$record_root/partnet_9388_front_oscillate_v1/episode.json" \
  --object partnet_46556 "$run_root/partnet_46556_front_oscillate_v1_retry" "$eval_root/partnet_46556_front_oscillate_v1.json" "$record_root/partnet_46556_front_oscillate_v1/episode.json" \
  --object partnet_12540 "$run_root/partnet_12540_front_oscillate_v1" "$eval_root/partnet_12540_front_oscillate_v1.json" "$record_root/partnet_12540_front_oscillate_v1/episode.json" \
  --object partnet_10849 "$run_root/partnet_10849_front_oscillate_v1_retry" "$eval_root/partnet_10849_front_oscillate_v1.json" "$record_root/partnet_10849_front_oscillate_v1/episode.json"
PYTHONPATH=src ./.venv/bin/python scripts/summarize_aim_paper_metrics.py \
  "$metrics_root" --output-dir "$metrics_root"
