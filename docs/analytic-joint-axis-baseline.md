# Analytic Joint-Axis Baseline

The analytic baseline evaluates whether explicit relative-motion fitting can
replace or diagnose the neural Relation Head axis prediction. It is an
evaluation-only tool and does not modify training or inference checkpoints.

## Estimators

- Revolute joints: robust per-frame parent/child SE(3), relative transforms,
  SO(3) logarithms, sign-aligned robust axis averaging, and a least-squares
  hinge-line estimate.
- Prismatic joints: relative translations followed by PCA/SVD direction
  fitting.
- Low-motion and insufficient-visibility cases are rejected rather than being
  assigned an arbitrary axis. Every estimate includes motion, residual,
  valid-frame, and confidence diagnostics.

Track poses are fitted from consecutive-frame correspondences. This supports
dynamically seeded tracks whose query/reference frames differ.

## Oracle Decomposition

The evaluator reports six settings:

1. GT parts + GT edges + GT type + analytic axis.
2. GT parts + GT edges + GT type + neural axis.
3. Predicted parts + GT edges + GT type + analytic axis.
4. Predicted parts + predicted edges/type + analytic axis.
5. Full neural pipeline.
6. Predicted parts + predicted neural edges + analytic joint type/axis.

The sixth setting fits both revolute and prismatic hypotheses and compares
them using the same normalized child-point replay RMSE. This avoids comparing
the revolute estimator's angular residual directly with the prismatic
estimator's translation residual. The selected model supplies both joint type
and axis; the neural relation head still supplies topology proposals.

GT labels are used only for this simulation diagnostic. They are never used by
the deployable inference path.

```bash
PYTHONPATH=src python scripts/evaluate_analytic_joint_axis_oracles.py \
  outputs/partnet_core_v1/cotracker_learning_manifest.tsv \
  outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt \
  outputs/partnet_core_v1_training/relation_heads_full_slot_final/cotracker_only/phase3_full_slot_finetune/slot_relation_head.pt \
  --catalog outputs/partnet_core_v1/catalog.json \
  --output-dir outputs/partnet_core_v1_training/analytic_axis_oracle_cotracker \
  --device cuda
```

Outputs:

- `analytic_axis_oracle_report.json`: aggregate, category, joint-type, motion,
  track-count, confidence-bin, coordinate-frame, and slot track-selection
  diagnostics.
- `analytic_axis_per_joint.json/csv`: all joints under all six settings.
- `catastrophic_axis_audit.json/csv`: neural errors above the configured
  threshold and the analytic comparison.

## Controlled Ours Ablation

The paper ablation does not report the full feature/backend Cartesian product.
It first selects a feature path and a kinematic backend on validation data:

- features: CoTracker-only, TAPIP-only, and Hybrid;
- backends: full neural, neural type plus analytic axis, and fully analytic
  type plus axis.

The feature ablation fixes the selected backend. The backend ablation fixes
the selected feature. Their shared best configuration appears once, producing
five unique test configurations. `Ours Main` is this validation-selected
combination. Test metrics are never used for selection.

The summarizer retains `ours_five_config_per_object.csv` and
`ours_five_config_per_joint.csv` as the raw analysis tables. It also emits
macro-averaged category and difficulty tables, where difficulty is defined by
GT part count: 2, 3-4, and at least 5 parts. Aggregate tables must remain
derivable from these raw rows.

```bash
PYTHONPATH=src python scripts/summarize_ours_controlled_ablation.py \
  outputs/partnet_core_v1_training/ours_controlled_ablation_v1 \
  --slot-training-root outputs/partnet_core_v1_training \
  --segmentation-report \
    outputs/partnet_core_v1_training/slot_benchmark_three_methods_test/benchmark_metrics.json
```

Axis angle uses `abs(dot)`, so the axis is undirected. Axis-line distance is
reported only for type-correct revolute joints and is normalized by the object
bbox diagonal. Prismatic joints have no axis-position error.

The report also includes strict paired comparisons on identical joints,
confidence/residual subsets, neural-analytic disagreement cases, and
end-to-end metrics that assign a 90-degree penalty to missing edges, wrong
types, or unavailable axes. Conditional axis means must not be compared when
their evaluated joint sets differ.

## Relation to a Relative-SE(3) Encoder

The fully analytic baseline already implements the proposed explicit
parent-child relative-SE(3) computation. It robustly fits per-frame parent and
child poses, forms `inv(T_parent) @ T_child`, and estimates a revolute hinge or
prismatic direction from the relative transforms. A separate
`relative_se3` neural encoder would therefore reuse the same intermediate
transforms and expose their SE(3) log/twist sequence to a learned head; it
would not be a new analytic baseline.

The evaluator also traces the track support used by the learned geometry
branch:

- visible tracks: visible in at least one sampled timestep;
- hard-assigned tracks: the slot is the track's maximum-probability slot;
- selected tracks: deterministic top-k by soft slot probability;
- effective tracks: `1 / sum(normalized_weight**2)`.

There is no assignment-probability threshold before the top-k operation.
Visibility masks invalid tracks, then every slot ranks all remaining tracks by
its soft probability. Consequently, selected count is normally equal to the
configured cap; hard-assigned and effective counts are the informative support
statistics.

It reports spatial RMS, weighted spatial RMS, maximum pairwise extent,
visibility duration, displacement, and correlations with neural axis error.
These diagnostics test whether low effective track count means that soft
pooling has collapsed onto a small spatial patch.

## Temporal Resolution Pilot

The original PartNet CoTracker dataset contains 120-frame, 8-second episodes
recorded at 15 FPS. CoTracker used `frame_stride=4`, producing only 30 tracking
timesteps at 3.75 Hz. TAPIP used all 120 episode frames at 15 Hz; older TAPIP
artifacts incorrectly inherited the CoTracker 3.75 Hz metadata.

Generate an isolated high-resolution pilot without overwriting the baseline:

```bash
PYTHONPATH=src python scripts/run_partnet_temporal_resolution_pilot.py \
  outputs/partnet_core_v1/catalog.json \
  --output-root outputs/partnet_temporal_pilot_30fps_15hz \
  --recording-fps 30 \
  --tracking-frame-stride 2 \
  --gpus 0,1,2 \
  --jobs 3 \
  --resume
```

This produces 240 source frames and 120 tracking timesteps per eight-second
episode. Use `--reuse-recording-root` with `--tracking-frame-stride 1` to build
a 15-FPS/15-Hz control group from existing episodes. This separates the effect
of denser tracking from the effect of smoother source video.
