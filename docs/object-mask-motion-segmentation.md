# Object-Mask Motion Segmentation V1

## Goal

Validate whether object-mask RGB-D lifted CoTracker tracks contain enough motion
signal to recover articulated refrigerator parts without ground-truth part masks.

V1 focuses on diagnostics and controlled clustering experiments. It does not
perform automatic `K` selection, split/merge refinement, or joint-aware EM.

## Stages

1. **Baseline reproduction**: run the legacy connected-components baseline on
   `refrigerator038-045`, confirm the giant-component failure, and save cluster
   count, largest-cluster ratio, GT composition, and downstream metrics.
2. **Diagnostics**: report same-GT/cross-GT edge ratios, cross-edge types,
   giant-cluster GT composition, cluster SE(3) residuals, per-frame residuals,
   and per-GT-part residual breakdown when `original_part_id` is available.
3. **Track cleanup**: optionally filter tracks with too few visible frames,
   excessive `depth_m` jumps, or excessive 3D trajectory discontinuities. Always
   report dropped-track statistics.
4. **Base bridge check**: split tracks into static-confident, moving-confident,
   and ambiguous groups, then run lowest-motion `40%` / `60%` removal sanity
   checks to see whether base tracks bridge moving parts.
5. **Clustering MVP**: build a weighted spatial kNN graph and run fixed-`K`
   spectral clustering for `K=2..8`. Sweep edge ablations instead of tuning one
   opaque score.
6. **Cluster validation**: fit per-frame rigid SE(3) transforms for each cluster
   and report rigid RMSE, per-frame RMSE, inlier ratio, dominant GT part, and GT
   purity. V1 marks high-residual clusters only; it does not auto split/merge.
7. **Downstream evaluation**: export each fixed-`K` result as `part_tracks.json`
   and run the existing pose/joint pipeline. Note that current MuJoCo GT
   evaluation is ID-based and does not yet solve object-mask cluster to GT-part
   matching.

## CLI

Legacy connected baseline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/recordings_refrigerators_staged_dense_mps/refrigerator039/pointcloud_4d_partseg/part_tracks_objectified.json \
  --mode connected \
  --output-json outputs/flow_tracking_eval/refrigerator039/connected_tracks.json \
  --diagnostics-json outputs/flow_tracking_eval/refrigerator039/connected_diagnostics.json
```

kNN spectral fixed-`K` sweep:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/flow_tracking_eval/refrigerator039/object_tracks.json \
  --mode knn-spectral \
  --output-json outputs/flow_tracking_eval/refrigerator039/motion_part_tracks_knn.json \
  --diagnostics-json outputs/flow_tracking_eval/refrigerator039/diagnostics_knn.json \
  --sweep-output-dir outputs/flow_tracking_eval/refrigerator039/sweep_knn \
  --quality-filter \
  --min-visible-frames 8 \
  --max-trajectory-jump-m 0.25 \
  --static-motion-threshold-m 0.01 \
  --moving-motion-threshold-m 0.03 \
  --knn-k 12 \
  --k-min 2 \
  --k-max 8 \
  --spectral-k 4 \
  --edge-ablation all
```

The selected `--spectral-k` result is written to `--output-json`. Every swept
ablation/`K` assignment is written to `--sweep-output-dir` when provided.

## Edge Ablations

- `A`: rigidity plus spatial kNN.
- `B`: rigidity, endpoint displacement similarity, and visibility overlap.
- `C`: rigidity, displacement magnitude curve similarity, velocity cosine,
  endpoint displacement similarity, visibility overlap, and spatial distance
  penalty.

## Current Evaluation Limitation

The existing `evaluate-kinematic-model` compares inferred joints to MuJoCo GT by
`child_part_id`. Object-mask clusters do not naturally preserve GT part IDs, so
that report may show no matched GT joints even when motion clusters improve. Use
the simulation-only evaluator below for cluster-to-GT diagnostics.

## Simulation-Only Cluster-to-GT Evaluation

Use `evaluate-object-mask-kinematics` when the track artifact contains
`original_part_id` labels. This is a diagnostic evaluator only; it never feeds
GT matching back into inference.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp evaluate-object-mask-kinematics \
  outputs/flow_tracking_eval/refrigerator039/joint_inference_knn.json \
  --part-poses outputs/flow_tracking_eval/refrigerator039/part_poses_knn.json \
  --output-json outputs/flow_tracking_eval/refrigerator039/object_mask_kinematic_evaluation.json \
  --output-csv outputs/flow_tracking_eval/refrigerator039/object_mask_kinematic_evaluation.csv \
  --matching-metric iou
```

The evaluator reports:

- predicted-cluster to GT-part overlap and IoU matrices
- per-cluster purity, per-GT coverage, mean purity, mean GT coverage, and
  largest-cluster ratio
- Hungarian-style one-to-one cluster/GT matching using IoU or raw overlap
- directed joint coverage after remapping parent and child cluster IDs
- undirected joint coverage when either direction matches a GT joint pair
- reverse-directed match count to separate true orientation errors from
  missing/incorrect GT parent metadata
- per-joint remap details, including mapped parent/child GT ids and parent/child
  cluster motion scores, cluster GT composition, purity, and rigid SE(3) RMSE
- joint type and axis errors for matched GT joints
- revolute `pivot_error_m`, which is the shortest distance between the
  predicted and GT rotation-axis lines, not point-to-point pivot distance
- no prismatic pivot/position error; pure prismatic joints are evaluated by
  axis direction because the origin point is not physically identifiable

For fixed-`K` sweeps, run this command on each exported
`motion_part_tracks_<ablation>_K<k>.json -> part_poses -> joint_inference`
chain and aggregate the optional one-row CSV summaries.

Use the per-joint `parent_cluster_debug` and `child_cluster_debug` fields to
diagnose pivot errors. Large pivot errors usually indicate that the matched
child cluster is mixed or non-rigid, not that the axis estimator failed. For
example, a matched joint with high child-cluster rigid RMSE and low GT purity is
a split/cleanup target before trying more complex joint fitting.

The evaluator also writes a lightweight `failure_reason_guess` and
`cleanup_recommendation` per predicted joint. Typical values are:

- `child_cluster_mixed_or_nonrigid`: the child cluster has low GT purity and
  high rigid RMSE or low inlier ratio. Split or clean this cluster before
  trusting pivot estimates.
- `pivot_sensitive_to_child_cluster_outliers`: the axis direction is reasonable
  but the pivot is far from GT, usually because a few child tracks pull the
  fitted pivot away from the true hinge.
- `pivot_offset_with_good_axis`: the joint direction is plausible but the joint
  line is translated to the wrong location.

## V2-Lite Cleanup Before Joint Inference

Do not start with full joint-aware EM. The first cleanup pass should be local
and diagnostic:

1. Run the cluster-to-GT evaluator.
2. Find moving child clusters with `gt_purity < 0.7`, `rigid_rmse_m > 0.03`, or
   `inlier_ratio < 0.5`.
3. Locally split only those clusters, or remove high residual outlier tracks.
4. Re-run `estimate-part-poses -> infer-joints -> evaluate-object-mask-kinematics`.

Generate local split candidates from an evaluation artifact:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp local-split-motion-cluster \
  outputs/flow_tracking_eval/refrigerator039/motion_part_tracks_knn.json \
  --evaluation-json outputs/flow_tracking_eval/refrigerator039/object_mask_kinematic_evaluation.json \
  --output-dir outputs/flow_tracking_eval/refrigerator039/local_split \
  --local-k-min 2 \
  --local-k-max 3 \
  --knn-k 8 \
  --edge-ablation B
```

The command reads `failure_reason_guess` and `cleanup_recommendation`, then
splits only recommended child clusters. It writes:

- `local_split_summary.json`
- one candidate `motion_part_tracks_split_cluster_<id>_<ablation>_K<k>.json`
  per local split setting

`local_split_summary.json` also reports soft candidate-selection metrics for
each subcluster:

- `track_count`
- `bbox_diag_m`
- `mean_motion_m` / `median_motion_m`
- `visible_frame_ratio`
- `rigid_rmse_m` / `inlier_ratio`
- `candidate_selection.selection_score_no_gt`
- `candidate_selection.diagnostic_score_with_gt`, only when simulation
  `original_part_id` labels are available

These metrics are deliberately not hard filters. Small real parts can have small
bbox and motion magnitude, so V2.1 uses them as ranking/debug signals rather
than reject rules. A useful child proposal should usually have enough rigid
coverage, motion, and visibility to constrain a joint, but the final decision
should come from downstream pose/joint replay and stability checks.

You can also force a cluster manually:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp local-split-motion-cluster \
  outputs/flow_tracking_eval/refrigerator039/motion_part_tracks_knn.json \
  --split-cluster-id 4 \
  --output-dir outputs/flow_tracking_eval/refrigerator039/local_split_manual \
  --local-k-min 2 \
  --local-k-max 3
```

Evaluate a candidate by feeding it through the normal downstream path:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  outputs/flow_tracking_eval/refrigerator039/local_split/motion_part_tracks_split_cluster_4_B_K2.json \
  --method tracks \
  --output-json outputs/flow_tracking_eval/refrigerator039/local_split/part_poses_split_cluster_4_B_K2.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/flow_tracking_eval/refrigerator039/local_split/part_poses_split_cluster_4_B_K2.json \
  --output-json outputs/flow_tracking_eval/refrigerator039/local_split/joint_inference_split_cluster_4_B_K2.json \
  --mujoco-prior off

PYTHONPATH=src python3 -m rgbd_urdf_mvp evaluate-object-mask-kinematics \
  outputs/flow_tracking_eval/refrigerator039/local_split/joint_inference_split_cluster_4_B_K2.json \
  --part-poses outputs/flow_tracking_eval/refrigerator039/local_split/part_poses_split_cluster_4_B_K2.json \
  --output-json outputs/flow_tracking_eval/refrigerator039/local_split/object_mask_eval_split_cluster_4_B_K2.json
```

Or build a candidate table for all local split outputs:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp evaluate-local-split-candidates \
  outputs/flow_tracking_eval/refrigerator039/local_split/local_split_summary.json \
  --output-dir outputs/flow_tracking_eval/refrigerator039/local_split/candidate_eval \
  --output-json outputs/flow_tracking_eval/refrigerator039/local_split/candidate_eval.json \
  --output-csv outputs/flow_tracking_eval/refrigerator039/local_split/candidate_eval.csv \
  --mujoco-prior off
```

This command runs `estimate-part-poses -> infer-joints ->
evaluate-object-mask-kinematics` for every split candidate and writes one row per
subcluster. Important columns:

- `selection_score_no_gt`: soft split proposal score, no GT labels
- `joint_selection_score_no_gt`: joint-level no-GT ranking score combining
  candidate quality, saturated replay score, and anti-degeneracy terms
- `joint_replay_error_m`: 3D track replay RMSE of the selected joint model
- `joint_replay_score`: `exp(-joint_replay_error_m / 0.05)`, so very small
  replay errors do not dominate all other terms
- `motion_normalized_replay_error`: replay error divided by child mean motion,
  diagnostic only
- `degeneracy_penalty`: soft penalty for small track count, small bbox, low
  motion, and `base_like_low_motion`
- `joint_type`: inferred candidate joint type
- `diagnostic_axis_error_gt` and `diagnostic_pivot_error_gt`: simulation-only
  GT metrics when `original_part_id` and MuJoCo metadata are available
- `axis_stability_bootstrap` and `pivot_stability_bootstrap`: reserved for a
  later bootstrap stability pass

The goal is to check whether high no-GT proposal scores align with lower joint
replay error and stable downstream joint estimates. If they do not align, the
proposal score is missing a joint-level term.

Do not rank candidates by raw `joint_replay_error_m` alone. Near-static
base-like fragments can achieve artificially low replay error because almost any
degenerate joint model explains little motion. Prefer `joint_selection_score_no_gt`
for no-GT ranking, then compare these three orderings during diagnostics:

- highest `selection_score_no_gt`
- lowest `joint_replay_error_m`
- highest `joint_selection_score_no_gt`

## V3 EM-Lite: Joint-Aware Reassignment

The fixed-`K` spectral stage should be treated as a part-proposal generator, not
as the final segmentation. It can recover useful motion signal, but two failure
modes remain common:

- Extra motion-consistent clusters can become fake parts or fake joints. For
  example, a mixed lower-door/freezer/base cluster may be explainable by a
  prismatic model even though it is not a valid semantic joint.
- True moving parts can be under-covered. A right-door cluster may have high
  purity but low GT coverage, so the estimated axis direction becomes unstable
  even if the joint pair is correctly matched.

These are not evaluator problems. They are model-selection and track-assignment
problems caused by the current pipeline shape:

```text
tracks -> fixed-K clustering -> part poses -> joint inference
```

The next stage should add one joint-aware cleanup pass:

```text
initial clusters
  -> fit SE(3) part poses and joints
  -> suppress duplicate/unsupported extra parts
  -> recover missing tracks for clean matched parts
  -> refit poses and joints once
```

This is an EM-like loop, but V3 should start with one iteration only. Full EM
with repeated split/merge/reassignment is deferred until the one-step behavior is
understood.

### Extra Part Suppression

For every predicted child joint, mark it as an extra/duplicate candidate when it
has several of these no-GT symptoms:

- low child rigid consistency
- low parent-child motion contrast
- small or fragmented track support
- high overlap in motion mode with an existing moving part
- joint replay does not improve over merging into a neighboring part or base
- duplicate joint type/axis/q trajectory relative to an already supported child

In simulation diagnostics, unmatched GT status and low GT purity can confirm the
failure mode, but they must not be used for inference.

Candidate actions:

- merge into the nearest compatible moving cluster
- merge back into base/ambiguous
- delete the joint proposal from the usable kinematic graph

Merge decisions should compare rigid replay and joint replay before and after
the merge. A fake part should be removed only if the merge does not significantly
worsen the no-GT reconstruction/replay objective.

### Coverage Recovery

For a clean matched moving part with high purity but low coverage, recover tracks
from nearby mixed or ambiguous clusters:

1. Keep the current parent/child joint model fixed.
2. For each candidate track outside the child cluster, compute residuals under:
   base/static model, current assigned cluster model, and the child joint model.
3. Reassign the track to the child only if child replay residual is lower by a
   margin and the reassignment does not break local spatial/visibility
   consistency.
4. Refit the child SE(3) trajectory and joint axis/pivot once.

This targets cases where a real door cluster is pure but incomplete. Higher
coverage should improve axis stability and reduce angle error, especially when
the original child cluster only covers a small or short-motion region.

### One-Step Objective

Use a soft track-to-part cost rather than a hard nearest-cluster rule:

```text
track_to_part_cost =
  rigid_replay_residual
+ joint_replay_residual
+ spatial_smoothness_cost
+ visibility_or_quality_cost
+ duplicate_or_too_small_part_penalty
```

The first implementation should report a table before changing the main result:

- candidate track id / current cluster / proposed cluster
- rigid residual before and after reassignment
- joint replay residual before and after reassignment
- spatial distance to the proposed part
- visibility overlap with the proposed part
- accepted or rejected

Then produce a refit artifact in a separate directory, for example:

```text
em_lite/
  reassignment_summary.json
  motion_part_tracks_em_lite.json
  part_poses_em_lite.json
  joint_inference_em_lite.json
  object_mask_eval_em_lite.json
```

### Acceptance Criteria

V3 EM-lite is useful only if it improves downstream kinematic metrics without
creating more fake joints:

- unmatched/fake predicted joint count decreases or stays constant
- directed/undirected joint coverage does not decrease
- high-purity but low-coverage parts recover more tracks
- axis angle error decreases for recovered real joints
- revolute axis-line position error does not increase significantly
- part-mask path and earlier fixed-`K` diagnostics remain unchanged

For refrigerator debugging, the target behavior is:

- suppress or merge the extra mixed prismatic proposal
- recover missing right-door tracks from mixed clusters
- refit the right-door axis with better coverage

## Diagnostic Visualization

Use PLY diagnostics when numeric scores are inconclusive:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-object-mask-diagnostics \
  outputs/flow_tracking_eval/refrigerator039/motion_part_tracks_knn.json \
  --joint-inference outputs/flow_tracking_eval/refrigerator039/joint_inference_knn.json \
  --evaluation-json outputs/flow_tracking_eval/refrigerator039/object_mask_kinematic_evaluation.json \
  --local-split-summary outputs/flow_tracking_eval/refrigerator039/local_split/local_split_summary.json \
  --candidate-eval outputs/flow_tracking_eval/refrigerator039/local_split/candidate_eval.json \
  --output-dir outputs/flow_tracking_eval/refrigerator039/diagnostic_viz \
  --frame first \
  --top-n-candidates 8 \
  --make-matplotlib \
  --make-animation \
  --plot-projections xz,xy \
  --animation-fps 12 \
  --animation-max-tracks 1000 \
  --animation-color-by pred_cluster
```

The command writes:

- `clusters_pred_<frame>.ply`: predicted cluster colors
- `clusters_gt_<frame>.ply`: GT-part colors when `original_part_id` exists
- `clusters_motion_<frame>.ply`: motion magnitude heat color
- `largest_cluster_gt_<frame>.ply`: largest predicted cluster colored by GT
- `purity_error_<frame>.ply`: non-dominant GT tracks inside each predicted
  cluster in red
- `joints_pred_vs_gt.ply`: parent/child points plus predicted red axes and GT
  green axes
- `local_split_candidates_topN_<frame>.ply`: top local split candidates colored
  by rank
- `overview_<projection>.png`: static 2D matplotlib panels for cluster, GT,
  motion, and purity-error inspection
- `track_replay_<projection>_<color>.mp4` or `.gif`: projected temporal replay
  of 3D tracks, with GIF fallback when ffmpeg is unavailable

These files are diagnostic-only. Open them in MeshLab, CloudCompare, or Open3D
to inspect whether the freezer failure is caused by mixed clusters, weak motion,
insufficient spatial coverage, or a GT/visual pivot mismatch.

The animation is intentionally a 2D projection replay rather than a matplotlib
3D interactive widget. It is faster, easier to export, and better suited for
checking whether a cluster moves as one part over time. Use `--plot-projections
all` to generate `xy`, `xz`, and `yz` replays.

For interactive 3D point-flow debugging, generate the Plotly HTML viewer:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-object-mask-flow-html \
  outputs/flow_tracking_eval/refrigerator039/motion_part_tracks_knn.json \
  --joint-inference outputs/flow_tracking_eval/refrigerator039/joint_inference_knn.json \
  --evaluation-json outputs/flow_tracking_eval/refrigerator039/object_mask_kinematic_evaluation.json \
  --output-html outputs/flow_tracking_eval/refrigerator039/flow_viewer.html \
  --frame-stride 3 \
  --trail-length 8 \
  --max-tracks 800 \
  --color-by pred_cluster
```

The HTML viewer embeds the selected tracks and exposes browser-side controls:

- `Frame`: replay lifted 3D tracks over time.
- `Visible track count`: choose how many embedded tracks to draw without
  regenerating the file. Tracks are sorted by endpoint motion, so lower counts
  prioritize the most informative moving tracks.
- `Trail length`: control the temporal point-flow tail length.
- `Color by`: switch between predicted cluster, GT part, and motion magnitude.
- `Layers`: toggle trajectory trails, joint axes, and coordinate axes.

This viewer is the preferred tool for checking whether a freezer/largest cluster
contains mixed motion modes. PLY files remain useful as raw 3D debug exports,
but the HTML viewer is better for point-flow inspection.

`infer-joints` has a small robust replay option that trims the largest 3D track
model residuals before comparing revolute and prismatic models:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/flow_tracking_eval/refrigerator039/part_poses_knn.json \
  --output-json outputs/flow_tracking_eval/refrigerator039/joint_inference_knn_trimmed.json \
  --mujoco-prior off \
  --robust-track-model-trim-ratio 0.2
```

This option does not fix a bad `part_poses.json` by itself, because the SE(3)
poses may already be contaminated by a mixed cluster. Treat it as a robustness
diagnostic and use local cluster split/refit as the main fix for large pivot
errors.

## Optional Parent Orientation Heuristic

`infer-joints` can optionally flip a predicted joint so the lower-motion cluster
is the parent:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/flow_tracking_eval/refrigerator039/part_poses_knn.json \
  --output-json outputs/flow_tracking_eval/refrigerator039/joint_inference_knn_oriented.json \
  --mujoco-prior off \
  --orient-parent-by-motion \
  --parent-orientation-motion-margin-m 0.005
```

This is off by default. Use it for object-mask diagnostics where the selected
anchor cluster may be more dynamic than the child cluster. The output joint
metrics include `parent_orientation` with the parent/child motion scores and
whether a flip was applied.
