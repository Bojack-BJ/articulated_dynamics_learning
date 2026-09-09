#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/outputs"
LOCAL_ROOT="outputs"

for scene in 30 32 34; do
  name="real_scene${scene}_track2art_v1"
  mkdir -p "${LOCAL_ROOT}/${name}/tracking" \
    "${LOCAL_ROOT}/${name}/track_quality" \
    "${LOCAL_ROOT}/${name}/slots" \
    "${LOCAL_ROOT}/${name}/kinematics"

  scp "dev-h:${REMOTE_ROOT}/${name}/tracking/object_tracks_cotracker.json" \
    "${LOCAL_ROOT}/${name}/tracking/"
  scp "dev-h:${REMOTE_ROOT}/${name}/tracking/object_tracks_cotracker.runtime.json" \
    "${LOCAL_ROOT}/${name}/tracking/"
  scp "dev-h:${REMOTE_ROOT}/${name}/track_quality/track_quality_summary.json" \
    "${LOCAL_ROOT}/${name}/track_quality/"
  scp "dev-h:${REMOTE_ROOT}/${name}/track_quality/timestep_quality_summary.json" \
    "${LOCAL_ROOT}/${name}/track_quality/"
  scp "dev-h:${REMOTE_ROOT}/${name}/track_quality/motion_part_tracks_with_quality.json" \
    "${LOCAL_ROOT}/${name}/track_quality/"
  scp "dev-h:${REMOTE_ROOT}/${name}/slots/motion_part_tracks_slots.json" \
    "${LOCAL_ROOT}/${name}/slots/"
  scp "dev-h:${REMOTE_ROOT}/${name}/kinematics/joint_inference_neural.json" \
    "${LOCAL_ROOT}/${name}/kinematics/"
done
