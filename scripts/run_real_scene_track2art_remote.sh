#!/usr/bin/env bash
set -euo pipefail

SCENE="$1"
GPU="$2"
REFERENCE_FRAME="$3"
ROOT="${TRACK2ART_ROOT:-/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning}"
PYTHON_BIN="${TRACK2ART_PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_NAME="${4:-real_scene${SCENE}_track2art_v1}"
OUT="$ROOT/outputs/$OUTPUT_NAME"
# Dense temporal sampling is the validated real-data default. Revolute arcs
# become poorly observable at stride 2 on the recorded oven/microwave scenes.
FRAME_STRIDE="${TRACK_FRAME_STRIDE:-1}"
VISIBILITY_THRESHOLD="${TRACK_VISIBILITY_THRESHOLD:-0.5}"
SEED_STRIDE_PX="${TRACK_SEED_STRIDE_PX:-8}"
SPATIAL_DBSCAN_EPS_M="${TRACK_SPATIAL_DBSCAN_EPS_M:-0.03}"
SPATIAL_DBSCAN_MIN_SAMPLES="${TRACK_SPATIAL_DBSCAN_MIN_SAMPLES:-5}"
MAX_STEP_M="${TRACK_MAX_STEP_M:-0.05}"

cd "$ROOT"
mkdir -p "$OUT/tracking" "$OUT/track_quality" "$OUT/slots" "$OUT/kinematics"
export CUDA_VISIBLE_DEVICES="$GPU"

PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp track-part-pixels "$OUT/episode.json" \
  --output-json "$OUT/tracking/object_tracks_cotracker.json" --device cuda \
  --cotracker-repo co-tracker --cotracker-checkpoint co-tracker/ckpt/scaled_offline.pth \
  --reference-frame "$REFERENCE_FRAME" --frame-stride "$FRAME_STRIDE" --seed-stride-px "$SEED_STRIDE_PX" \
  --max-tracks-per-part-view 512 --max-queries-per-forward 512 --visibility-threshold "$VISIBILITY_THRESHOLD" \
  --strict-object-mask-consistency --export-cotracker-features \
  --cotracker-features-output "$OUT/tracking/cotracker_features.npz" \
  --dynamic-reseeding --reseed-interval-frames 20 --reseed-coverage-radius-px 10 \
  --reseed-bbox-scale 1.15 --reseed-max-tracks-per-frame-view 48 --reseed-max-tracks-per-view 192

PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp compute-track-quality \
  "$OUT/tracking/object_tracks_cotracker.json" --output-dir "$OUT/track_quality" \
  --mask-bad-timesteps --max-step-m "$MAX_STEP_M" \
  --spatial-dbscan-eps-m "$SPATIAL_DBSCAN_EPS_M" \
  --spatial-dbscan-min-samples "$SPATIAL_DBSCAN_MIN_SAMPLES"

PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp infer-motion-part-slots \
  "$OUT/track_quality/motion_part_tracks_with_quality.json" \
  "$OUT/tracking/cotracker_features.npz" \
  /lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt \
  --output-json "$OUT/slots/motion_part_tracks_slots.json" --device cuda \
  --min-visible-frames 4 --min-visible-ratio 0.15 --max-trajectory-jump-m 0.15

PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp infer-slot-relation-head \
  "$OUT/track_quality/motion_part_tracks_with_quality.json" \
  "$OUT/tracking/cotracker_features.npz" \
  /lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt \
  /lumos-vePFS/suzhou/Users/lixiaotong/articulated_relation_training/outputs/partnet_core_v1_training/relation_heads_batched_final/cotracker_only/phase1_frozen/slot_relation_head.pt \
  --output-json "$OUT/kinematics/joint_inference_neural.json" --device cuda

# Independent geometric fit for real-data diagnosis. It consumes the same
# cleaned slot assignments and never changes the neural prediction.
PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp estimate-part-poses \
  "$OUT/slots/motion_part_tracks_slots.json" --method tracks --quality-weighted \
  --anchor-selection lowest-motion \
  --output-json "$OUT/kinematics/part_poses_analytic.json"

PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp infer-joints \
  "$OUT/kinematics/part_poses_analytic.json" \
  --output-json "$OUT/kinematics/joint_inference_analytic.json" \
  --mujoco-prior off --quality-weighted-replay --robust-track-model-trim-ratio 0.15 \
  --orient-parent-by-motion

for MODE in neural analytic; do
  PYTHONPATH=src "$PYTHON_BIN" -m rgbd_urdf_mvp visualize-object-mask-flow-html \
    "$OUT/slots/motion_part_tracks_slots.json" \
    --joint-inference "$OUT/kinematics/joint_inference_${MODE}.json" \
    --background-episode "$OUT/episode.json" --background-exclude-object-mask \
    --background-max-points 10000 --max-tracks 1200 --axis-remap x,y,z \
    --output-html "$OUT/viewer_${MODE}_annotation.html"
done
