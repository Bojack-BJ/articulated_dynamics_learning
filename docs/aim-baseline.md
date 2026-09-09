# AiM Baseline Integration

This integration runs the official [AiM](https://github.com/haoai-1997/AiM) implementation as an
independent baseline. AiM is a dynamic 3D Gaussian reconstruction and segmentation method; it does
not consume this project's lifted 3D tracks or predicted part masks.

## Deployment

The pinned upstream revision is `1303e0d89524cff01c49985048c9344b8d3ea096`. The isolated deployment
on `dev-h` uses:

```text
/root/Users/lixiaotong/third_party/AiM
/root/Users/miniconda3/envs/aim
```

Apply [`patches/aim-1303e0d-dataset-reader.patch`](../patches/aim-1303e0d-dataset-reader.patch)
to the upstream checkout. It fixes two input-reader defects without changing the AiM algorithm:

- RGBA arrays must use unsigned `uint8`, not signed `byte`.
- Blender `FovX` and `FovY` were assigned in reverse order for non-square images.

The upstream environment also depends on `pytorch3d.ops.ball_query`. The available PyTorch3D binary
would replace AiM's pinned Torch 1.13/CUDA 11.7 stack with Torch 2.0/CUDA 11.8. Apply
[`patches/aim-1303e0d-pointops-ball-query.patch`](../patches/aim-1303e0d-pointops-ball-query.patch)
instead. It preserves the original PyTorch3D path when available and otherwise performs the same
radius-filtered K-nearest-neighbor query through AiM's bundled `pointops` CUDA extension.

For short integration runs, also apply
[`patches/aim-1303e0d-short-run-psnr.patch`](../patches/aim-1303e0d-short-run-psnr.patch).
The upstream code assumes that the final static-stage step always runs evaluation and therefore calls
`.item()` on a Python float otherwise. The patch makes the scalar conversion type-safe and does not
change PSNR computation or optimization.

Finally, apply [`patches/aim-1303e0d-seg-import.patch`](../patches/aim-1303e0d-seg-import.patch).
The upstream segmentation entry point imports `training_report` from a missing `train.py`; the
function is shipped in `train_report.py`.

## Episode Conversion

AiM expects separate `start`, `motion`, and `end` Blender datasets. Convert a recorded episode with:

```bash
PYTHONPATH=src python scripts/export_episode_to_aim.py \
  outputs/path/to/episode.json \
  --output-dir outputs/aim_inputs/object_id \
  --boundary-frame-count 3 \
  --motion-frame-stride 1 \
  --test-stride 8
```

The exporter uses only RGB, object masks, depth, camera poses, and timestamps. It does not read part
masks, simulator part IDs, joint types, or joint parameters. The export manifest records this contract
and the camera-convention conversion.

### AiM-Inspired Acquisition Approximation

`record-mujoco` supports a two-stage protocol that is closer to AiM's input
structure, but its numeric camera settings are project-defined:

```bash
PYTHONPATH=src python -m rgbd_urdf_mvp record-mujoco path/to/model.xml \
  --category refrigerator \
  --object-id example \
  --output-dir outputs/aim_style_protocol_recordings \
  --duration-s 8 \
  --fps 15 \
  --control-mode staggered \
  --all-joints \
  --segmentation-masks \
  --recording-protocol aim_style \
  --aim-static-scan-views 24 \
  --aim-static-scan-elevation-deg 15 \
  --aim-interaction-camera-orbits 1.0 \
  --aim-interaction-elevation-amplitude-deg 10
```

The output keeps the normal `episode.json` contract for downstream RGB-D tools and adds:

```text
<episode>/
  aim_protocol/
    static_scan/
      view_000/{rgb,depth,mask}.*
      ...
      cameras.json
    interaction/
      cameras.json
      joint_states.json
```

The static scan is rendered before any simulation step. Every view therefore has exactly the same
joint positions. The interaction stage contains one camera per timestep. Its orbit has an independent
phase modulation and elevation trajectory, so camera azimuth is not a fixed linear function of joint
position. `export_episode_to_aim.py` automatically uses the dense static scan as AiM's `start` stage;
the interaction frames remain the `motion` stage. AiM's official optimizer samples ten `end`
training cameras at once, so sufficiently long `aim_style` episodes automatically reserve enough
final interaction observations for that operation. This does not add a third acquisition stage.

[`configs/record_partnet_aim_style.yaml`](../configs/record_partnet_aim_style.yaml) is the reproducible
PartNet-Mobility approximation. Change the model, category, and object ID per
asset. Do not describe its 24 static views, 120 interaction frames, 15 Hz, or
camera orbit as the official AiM acquisition protocol.

The paper specifies a start-state scan plus a monocular interaction video and
reports 200 video frames for two/three-part objects and 500 for more complex
objects. It says the interaction camera follows a trajectory around the object.
The 100 random upper-hemisphere views described in the paper apply to
**two-state baselines**, not to AiM's own start-state scan. The released AiM
loader consumes every frame listed in the supplied Blender
`transforms_*.json`; it does not hard-code a native static-view count, orbit,
or FPS.

Therefore an exact `aim_native` reproduction must preserve the official
released dataset's images, transforms, ordering, and timestamps. If those
camera-generation parameters are unavailable for our PartNet object, report
this configuration as `aim_style_approx`, not as native. Native-dataset results
and same-object equal-input adapter results belong in separate tables.

## Official Execution

The deployment keeps AiM's pinned Torch 1.13.1/CUDA 11.7 environment. The three upstream CUDA
extensions are loaded from their CPython 3.10 build directories:

```bash
export AIM_ROOT=/root/Users/lixiaotong/third_party/AiM
export AIM_PYTHON=/root/Users/miniconda3/envs/aim/bin/python
export PYTHONPATH="$AIM_ROOT:$AIM_ROOT/submodules/diff-gaussian-rasterization1/build/lib.linux-x86_64-cpython-310:$AIM_ROOT/submodules/simple-knn/build/lib.linux-x86_64-cpython-310:$AIM_ROOT/lib/pointops/build/lib.linux-x86_64-cpython-310"
```

Run official reconstruction and motion training:

```bash
CUDA_VISIBLE_DEVICES=6 "$AIM_PYTHON" "$AIM_ROOT/train_main.py" \
  --source_path /root/Users/lixiaotong/aim_datasets/object_id \
  --model_path /root/Users/lixiaotong/aim_runs/object_id \
  --is_blender --eval --random_bg_color
```

Run official motion segmentation from the saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES=6 "$AIM_PYTHON" "$AIM_ROOT/seg_main.py" \
  --source_path /root/Users/lixiaotong/aim_datasets/object_id \
  --model_path /root/Users/lixiaotong/aim_runs/object_id \
  --is_blender --eval --random_bg_color
```

The integration smoke test used `partnet_single_revolute_12531` with five static and five motion
optimization steps. It completed both entry points and produced the saved Gaussian/deformation
checkpoint, `motion.json`, `motion_seg_final/segmented_point.ply`, and trajectory point clouds. These
short-run artifacts validate execution only and must not be used as benchmark results.

## Comparability Constraint

The current standard recordings contain only three unique camera viewpoints. That is sufficient for
an integration smoke test but is not a fair reconstruction benchmark for AiM. A final comparison
should record an orbit sequence with static start/end hold periods and articulated motion between
them. AiM's published mesh-part IoU is also not identical to track-level RI/ARI or visible-point IoU;
the metrics must be aligned before reporting a ranking.

## Point-Cloud IoU

AiM's published mesh/voxel IoU and this project's observed-point IoU measure different domains and
must be reported separately. The former remains the paper-reproduction metric. The latter transfers
AiM's predicted Gaussian labels to a labeled fused RGB-D point cloud and applies one-to-one Hungarian
matching on point labels, making it comparable to the current point-domain segmentation metrics.

```bash
PYTHONPATH=src python scripts/evaluate_aim_pointcloud_iou.py \
  /path/to/aim_run/motion_seg_final/segmented_point.ply \
  /path/to/pointcloud_4d_partseg/frames/frame_0179.ply \
  --reference-part-ids-from-tracks /path/to/motion_part_tracks_slots.json \
  --output-json /path/to/aim_run/pointcloud_iou.json
```

The evaluator reports two forms at multiple distance thresholds normalized by the reference bounding
box diagonal:

- `covered_only_metrics` measures segmentation only where AiM reconstructed nearby geometry. This is
  useful for diagnosing label quality but can hide reconstruction failures.
- `coverage_aware_metrics` keeps every observed reference point in the IoU denominator. Missing AiM
  geometry therefore lowers the score instead of being silently excluded.

Always report `geometry_coverage` next to point IoU. A result with fewer than two GT part labels is an
integration or geometry-coverage diagnostic, not a valid part-segmentation benchmark case.
For direct comparison with the slot benchmark, use `--reference-part-ids-from-tracks` for both AiM
and track predictions. This removes fixed or untracked simulator links from the fused PLY reference
domain and keeps exactly the GT parts represented by the benchmark tracks.

## Camera-Protocol Pilot

A three-object robustness pilot compares the standard fixed three-view protocol with a continuous
two-orbit camera trajectory. Both use 120 frames at 15 FPS and identical object motion, rendering,
and optimization budgets. Evaluation uses the standard three-view `frame_116` point cloud and GT
part-ID domain for both protocols, so the metric domain does not change with single-view visibility.

The dense-orbit AiM run increases mean 2%-bbox geometry coverage from 0.614 to 0.645, while mean
point IoU decreases from 0.286 to 0.214 and ARI decreases from 0.197 to 0.126. This pilot does not
support limited camera coverage as the sole cause of AiM's segmentation gap. The full report is
[`outputs/aim_dense_protocol_v1/summary.md`](../outputs/aim_dense_protocol_v1/summary.md).

This remains a robustness diagnostic rather than an AiM upper bound. The orbit recording contains
one camera stream per timestep, whereas the standard protocol contains three simultaneous streams.
The `aim_style` approximation above addresses missing dense start-state
coverage without changing AiM's official reconstruction, segmentation, or
RANSAC parameters. It does not reproduce an exact official camera path.

The three-object result is reported in
[`outputs/aim_style_protocol_v1/summary.md`](../outputs/aim_style_protocol_v1/summary.md).

## Native Motion-Axis Evaluation

AiM writes one screw-motion estimate per segmented Gaussian component to
`motion.json`. Component zero is the static identity motion and must not be
treated as a joint proposal. The remaining entries provide `motion_type`,
`axis`, `center`, `R`, and `t`.

Evaluate these native outputs against simulation GT with:

```bash
PYTHONPATH=src python scripts/evaluate_aim_motion_axis.py \
  --object partnet_103069 /path/to/aim_run /path/to/aim_iou.json /path/to/episode.json \
  --object partnet_10638 /path/to/aim_run /path/to/aim_iou.json /path/to/episode.json \
  --output-dir outputs/aim_style_protocol_v1/axis
```

The evaluator recovers the AiM component ID from the segmented PLY palette,
uses the existing point-IoU Hungarian assignment to map that component to a GT
part, and then compares the native screw type/axis with the corresponding
simulation joint. Axis error is sign invariant and conditional on a correct
joint type. Axis-line distance is reported only for type-correct revolute
joints. GT is used only by this evaluation adapter.

## Extended Baseline Evaluation

The paper-facing extended evaluation keeps the acquisition protocols and official AiM algorithm
fixed. Shared 3-view is the primary equal-input comparison, continuous orbit is a moving-camera
robustness ablation, and `aim_style_approx` is an AiM-favorable acquisition
ablation. Only runs on official released cameras/transforms should be labeled
`aim_native`.

The stratified 20-object held-out subset is listed in
[`configs/aim_baseline_extended_v1.tsv`](../configs/aim_baseline_extended_v1.tsv). The remote runner
prepares the common frame-116 point domain, runs official AiM with the same 5k static plus 8k motion
budget, and evaluates AiM and Hybrid with identical GT part IDs:

```bash
PYTHONPATH=src python scripts/run_aim_baseline_extended.py \
  configs/aim_baseline_extended_v1.tsv \
  --stage all \
  --recordings-root /path/to/partnet_core_v1_recordings \
  --hybrid-root /path/to/hybrid/test \
  --dataset-root /path/to/aim_datasets/shared_extended_v1 \
  --run-root /path/to/aim_runs/shared_extended_v1 \
  --report-root /path/to/aim_reports/aim_baseline_extended_v1 \
  --aim-root /path/to/AiM \
  --aim-python /path/to/aim/python \
  --gpus 0,1,2,3
```

Generate the unified paper report after adding any completed protocol-ablation manifest:

```bash
PYTHONPATH=src python scripts/summarize_aim_baseline_extended.py \
  outputs/aim_baseline_extended_v1/run_manifest.csv \
  --additional-manifest outputs/aim_baseline_extended_v1/protocol_pilot_manifest.csv \
  --failure-details outputs/aim_baseline_extended_v1/failure_details.json \
  --output-root outputs/aim_baseline_extended_v1
```

The report includes per-object segmentation and part-count diagnostics, complexity buckets,
protocol-paired deltas, Pearson/Spearman coverage correlations, and unmodified official failure
records. See
[`outputs/aim_baseline_extended_v1/summary.md`](../outputs/aim_baseline_extended_v1/summary.md).
