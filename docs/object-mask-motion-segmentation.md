# Object-Mask Motion Segmentation V1

## Goal

Validate whether object-mask RGB-D lifted CoTracker tracks contain enough motion
signal to recover articulated refrigerator parts without ground-truth part masks.

V1 focuses on diagnostics and controlled clustering experiments. It does not
perform automatic `K` selection, split/merge refinement, or joint-aware EM.

## Dynamic reseeding for disocclusion

Object-mask tracking can optionally add tracks for surfaces that first become
visible after the reference frame, such as an interior drawer revealed by an
opening refrigerator door:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp track-part-pixels episode.json \
  --output-json object_tracks.json \
  --dynamic-reseeding \
  --reseed-interval-frames 5 \
  --reseed-coverage-radius-px 12 \
  --reseed-bbox-scale 1.2 \
  --reseed-max-tracks-per-frame-view 64 \
  --reseed-max-tracks-per-view 256
```

At each checkpoint, the tracker samples foreground pixels with valid depth
that are not covered by an existing visible track. Candidates must lie inside
a robust 3D object bounding box expanded by `--reseed-bbox-scale`. This
proximity gate rejects most unrelated foreground objects while retaining newly
exposed internal surfaces.

Dynamic tracks record `seed_source: dynamic_reseed` and their own
`query_frame_index`. Samples before this birth frame are always marked
invisible. Existing tracks are not invalidated solely because they leave a
propagated object mask: an opening articulated part may legitimately expand
beyond the original silhouette. `mask_consistent` remains available as a soft
quality diagnostic.

The mechanism also works with simulation part masks. It then reseeds each known
part separately, providing a controlled diagnostic for parts fully occluded in
the first frame. GT part IDs are not required in object-mask inference mode.

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

### Sequential RANSAC Rigid Proposals

`sequential-ransac` is an optional proposal generator. Unlike kNN spectral
clustering, each proposal is supported by one shared per-frame rigid SE(3)
model. It does not choose the final kinematic graph: a real door can be split
into several locally consistent RANSAC patches when RGB-D lifting or tracking
noise makes a global rigid fit exceed the inlier threshold. Feed its output into
the post-segmentation group-level merge diagnostic below.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/flow_tracking_eval/refrigerator045/em_lite/baseline/motion_part_tracks_knn.json \
  --mode sequential-ransac \
  --output-json outputs/flow_tracking_eval/refrigerator045/ransac/motion_part_tracks_ransac.json \
  --diagnostics-json outputs/flow_tracking_eval/refrigerator045/ransac/diagnostics.json \
  --ransac-iterations 256 \
  --ransac-inlier-threshold-m 0.025 \
  --ransac-min-inliers 16 \
  --ransac-spatial-link-m 0.45 \
  --ransac-assignment-threshold-m 0.05
```

Then run the normal pose and joint stack, followed by group-level merge:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  outputs/flow_tracking_eval/refrigerator045/ransac/motion_part_tracks_ransac.json \
  --method tracks \
  --output-json outputs/flow_tracking_eval/refrigerator045/ransac/part_poses_ransac.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/flow_tracking_eval/refrigerator045/ransac/part_poses_ransac.json \
  --output-json outputs/flow_tracking_eval/refrigerator045/ransac/joint_inference_ransac.json \
  --mujoco-prior off

PYTHONPATH=src python3 scripts/run_post_spectral_merge_diagnostic.py \
  outputs/flow_tracking_eval/refrigerator045/ransac/motion_part_tracks_ransac.json \
  --joint-inference outputs/flow_tracking_eval/refrigerator045/ransac/joint_inference_ransac.json \
  --output-dir outputs/flow_tracking_eval/refrigerator045/ransac/postmerge \
  --segmentation-source sequential-ransac \
  --enable-post-spectral-merge \
  --post-merge-mode greedy \
  --generate-viewer
```

The script name is retained for compatibility, but it is now source-agnostic:
it operates on any predicted motion-part artifact. It tests nearby proposals
against shared revolute/prismatic group models and emits a separate
`motion_part_tracks_postmerge.json`; it never overwrites the RANSAC proposal
artifact.

### Threshold Guide

The following parameters are intentionally exposed because their correct scale
depends on depth noise, object size, motion excitation, and track density. They
are not simulation GT priors.

| Parameter | Default | Effect | Tuning guidance |
| --- | ---: | --- | --- |
| `--ransac-inlier-threshold-m` | `0.025` m | Maximum trimmed shared-SE(3) replay RMSE for a track to join a rigid proposal. | Start at `0.025`. Raise to `0.030-0.040` if one physical door is split by depth/track noise. Lower it if door and drawer merge. This alone cannot resolve over-segmentation; follow with group merge. |
| `--ransac-assignment-threshold-m` | `0.050` m | Threshold for attaching ambiguous/leftover tracks to an extracted proposal. | Keep above the inlier threshold. Raising it improves coverage but can contaminate a clean proposal. |
| `--ransac-min-inliers` | `12` tracks | Minimum consensus needed to instantiate another rigid proposal. | Raise to suppress tiny noise patches; lower only for genuinely small parts or sparse observations. |
| `--ransac-max-models` | `8` | Maximum sequential proposals. | Set near the expected upper bound for visible moving parts plus a small margin. It is a safety cap, not a semantic part count. |
| `--ransac-sample-size` | `4` tracks | Tracks used for each SE(3) hypothesis. | Keep `3-5`. Three is the geometric minimum; larger values reduce accidental hypotheses but are more sensitive to mixed samples. |
| `--ransac-iterations` | `128` | Hypotheses evaluated for each extracted model. | Use `256` for dense refrigerator tracks; increase when consensus extraction is unstable, not as a substitute for better motion models. |
| `--ransac-spatial-link-m` | `0.35` m | Reference-space locality for an inlier consensus component. | Raise for a large door whose valid tracks span a wide area. Lower if spatially distant, motion-similar regions are attached. |
| `--static-motion-threshold-m` | `0.010` m | Tracks treated as confident base/static. | Keep conservative so hinge-adjacent moving tracks stay ambiguous rather than being permanently assigned to base. |
| `--moving-motion-threshold-m` | `0.030` m | Tracks eligible for moving-proposal extraction. | Lower only when excitation is weak; otherwise low-motion base/hinge tracks become moving candidates. |
| `--no-ransac-base-stabilize` | off | Disables base-frame stabilization. | Use only as a diagnostic when static base tracks are visibly contaminated; stabilization is usually preferred when camera/object global motion exists. |

For `refrigerator045`, `0.025 m` is the safer current operating point: it
preserves good joint geometry but produces fragmented proposals. `0.035 m`
improves some coverage but can admit locally wrong joint explanations. The
recommended approach is therefore `0.025 m` proposals followed by shared-joint
merge, rather than using a large inlier threshold as the only cleanup mechanism.

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

`xyz_world` from the MuJoCo/SAPIEN recorder is already z-up, so the flow and
PLY viewers preserve `x,y,z` by default. For an adapter that writes image-style
y-down coordinates, explicitly add `--axis-remap x,z,-y`.

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

## Frozen CoTracker Representation Probe

The learning-based exploration path first tests whether CoTracker's frozen
update-transformer tokens already separate physical parts. This is a
simulation-only diagnostic and does not use GT labels during tracking.

Export one L2-normalized, temporally pooled 384-D feature per emitted 3D track:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp track-part-pixels \
  outputs/recordings/refrigerator039/episode.json \
  --output-json outputs/recordings/refrigerator039/part_tracks_features.json \
  --device mps \
  --cotracker-repo co-tracker \
  --cotracker-checkpoint co-tracker/ckpt/scaled_offline.pth \
  --export-cotracker-features \
  --cotracker-features-output outputs/recordings/refrigerator039/cotracker_features.npz
```

Probe same-part versus cross-part similarity and hidden-only clustering. The
label field must come from simulator diagnostics and is never consumed by
inference:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp probe-cotracker-features \
  outputs/recordings/refrigerator039/part_tracks_features.json \
  outputs/recordings/refrigerator039/cotracker_features.npz \
  --label-field original_part_id \
  --output-json outputs/recordings/refrigerator039/cotracker_feature_probe.json \
  --output-embedding-csv outputs/recordings/refrigerator039/cotracker_feature_pca.csv
```

The probe reports same/cross-part cosine distributions, pairwise AUC, hidden-only
cosine-k-means purity, mean GT coverage, ARI, and NMI. The PCA CSV is intended
for plotting or interactive inspection.

If the frozen representation is informative, it can be added as a bounded soft
prior to the existing geometric kNN affinity:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/recordings/refrigerator039/part_tracks_features.json \
  --mode knn-spectral \
  --spectral-k 5 \
  --cotracker-features-npz outputs/recordings/refrigerator039/cotracker_features.npz \
  --learned-affinity-floor 0.7
```

This learned affinity is disabled by default. The multiplier is bounded to
`[floor, 1]`, so the frozen representation can downweight a geometric edge but
cannot create nonlocal edges or remove an edge outright. Compare it against the
unchanged geometric baseline before considering a trainable affinity head.

## Pairwise Affinity Head

The trainable pairwise head predicts whether two local CoTracker tracks belong
to the same rigid part. It combines frozen CoTracker descriptors with symmetric
pair geometry and motion features. Training uses simulator-only
`original_part_id` labels; inference does not consume GT part labels.

Create a tab-separated manifest with one object per row:

```text
object_id	tracks_path	features_npz	split
refrigerator038	/path/to/object_tracks.json	/path/to/cotracker_features.npz	train
refrigerator039	/path/to/object_tracks.json	/path/to/cotracker_features.npz	val
refrigerator040	/path/to/object_tracks.json	/path/to/cotracker_features.npz	test
```

The split must be object-level. Putting tracks from the same object in both
training and evaluation measures memorization rather than generalization.

Train and evaluate the local affinity model:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-pairwise-affinity \
  configs/pairwise_affinity_manifest.tsv \
  --output-dir outputs/pairwise_affinity \
  --knn-k 12 \
  --hard-mining-k 8 \
  --max-positive-negative-ratio 2.0 \
  --hard-negative-weight 2.0 \
  --device mps

PYTHONPATH=src python3 -m rgbd_urdf_mvp evaluate-pairwise-affinity \
  configs/pairwise_affinity_manifest.tsv \
  outputs/pairwise_affinity/pairwise_affinity.pt \
  --split test \
  --output-json outputs/pairwise_affinity/test_metrics.json \
  --device mps
```

Training combines spatial neighbors with feature-similar and motion-similar
cross-part hard negatives, plus spatially distant same-part hard positives.
Positive pairs are downsampled when needed to enforce the configured class
ratio. Simulator labels are used only to mine training pairs. The head scores
only existing spatial kNN edges in the default inference path.

Use the checkpoint as a bounded soft prior during spectral clustering:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/recordings/refrigerator040/object_tracks.json \
  --mode knn-spectral \
  --spectral-k 5 \
  --cotracker-features-npz outputs/recordings/refrigerator040/cotracker_features.npz \
  --pairwise-affinity-model outputs/pairwise_affinity/pairwise_affinity.pt \
  --pairwise-affinity-floor 0.5 \
  --diagnostics-json outputs/recordings/refrigerator040/pairwise_diagnostics.json
```

This path is experimental and disabled by default. A fixed spectral K still
forces K clusters, so the affinity head can improve boundaries but cannot by
itself solve model selection or remove over-segmented parts. Use a held-out
object evaluation before enabling it in a production pipeline.

To diagnose whether fixed-K spectral clustering is the bottleneck, the
`learned-connected` mode scores every pair of moving-confident tracks and uses
a probability threshold plus connected components to infer the part count:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/recordings/refrigerator040/object_tracks.json \
  --mode learned-connected \
  --cotracker-features-npz outputs/recordings/refrigerator040/cotracker_features.npz \
  --pairwise-affinity-model outputs/pairwise_affinity/pairwise_affinity.pt \
  --pairwise-connect-threshold 0.9 \
  --output-json outputs/recordings/refrigerator040/motion_parts_learned_connected.json
```

This is a diagnostic rather than a recommended inference path. A model trained
only on local kNN pairs may be poorly calibrated on dense nonlocal pairs, and a
single false-positive bridge can merge two complete parts under connected
components. Train with nonlocal hard negatives and evaluate on held-out objects
before interpreting this mode as a learned part-count estimator.

The safer hybrid experiment uses learned affinity only to propose coherent
minimal samples for sequential RANSAC. Shared-SE(3) replay remains the final
inlier criterion and no fixed K is required:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-motion-parts \
  outputs/recordings/refrigerator040/object_tracks.json \
  --mode sequential-ransac \
  --cotracker-features-npz outputs/recordings/refrigerator040/cotracker_features.npz \
  --pairwise-affinity-model outputs/pairwise_affinity/pairwise_affinity.pt \
  --ransac-learned-seed \
  --output-json outputs/recordings/refrigerator040/motion_parts_learned_ransac.json
```

Always compare this against both unmodified sequential RANSAC and the
learned-affinity kNN-spectral path. The flag is disabled by default.

## DETR-Style Motion-Part Slots

The slot path predicts an episode-local rigid object ID for every track. Slot
IDs are permutation invariant: Hungarian matching aligns predicted slots with
simulator `original_part_id` labels during training. Each slot is additionally
regularized by differentiable weighted Kabsch replay, so tracks assigned to one
slot are encouraged to share a per-frame SE(3). New training defaults address
the common giant-slot failure with six complementary mechanisms:

- inverse-part-frequency assignment loss;
- matched-slot Dice loss and spatial hard-negative pairwise loss;
- object-scale-normalized rigid replay;
- canonical object geometry plus shared rigid/noise augmentation;
- part-stratified track dropout;
- topology-balanced episode sampling for rare multi-part objects.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-motion-part-slots \
  configs/pairwise_affinity_manifest.tsv \
  --output-dir outputs/motion_part_slots \
  --max-slots 8 \
  --epochs 100 \
  --rigid-loss-weight 0.2 \
  --dice-loss-weight 0.5 \
  --pairwise-loss-weight 0.35 \
  --device mps

PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-motion-part-slots \
  outputs/recordings/refrigerator040/object_tracks.json \
  outputs/recordings/refrigerator040/cotracker_features.npz \
  outputs/motion_part_slots/motion_part_slots.pt \
  --output-json outputs/recordings/refrigerator040/motion_part_tracks_slots.json \
  --slot-existence-threshold 0.5 \
  --device mps
```

Inference uses the learned slot-existence logits to suppress inactive slots.
The output includes `slot_segmentation_evaluation` when simulator
`original_part_id` is available. Prefer its one-to-one Hungarian mIoU/F1,
pairwise same-part F1, GT collision count, and over/under-segmentation counts.
Independent per-GT best-cluster coverage can reuse one giant predicted cluster
for several GT parts and must not be used as the primary slot metric.

Post-RANSAC refinement is conservative: it keeps a model assignment unless a
different slot's rigid replay residual is substantially lower. The initial slot
ID and confidence remain in every output track for before/after diagnostics.
Use object-held-out splits; same-object training only validates implementation.

### Diagnostic Pairwise Joint Proposal Head

The first feedforward articulation experiment learns an ordered relation for
every `(parent_slot, child_slot)` pair. Each slot combines its learned token
with a soft aggregation of the complete canonical 3D trajectory descriptor.
The pair feature concatenates the parent and child representations, their
difference, and their elementwise product. Independent heads predict edge
existence, joint type, axis direction, and a point on the revolute axis line.

This is intentionally an unconstrained diagnostic. It reports cycles, multiple
parents, root count, and legal-tree rate, but does not repair the graph or alter
the existing optimization pipeline. MuJoCo part/joint metadata is used only as
simulation training/evaluation supervision.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-slot-relation-head \
  configs/motion_part_slots_refrigerator_microwave.tsv \
  outputs/flow_tracking_eval/slot_benchmark_refrigerator_microwave/model/motion_part_slots.pt \
  --output-dir outputs/flow_tracking_eval/slot_relation_head \
  --epochs 50 \
  --device mps

PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-slot-relation-head \
  outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/refrigerator044/object_tracks.json \
  outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/learning_features_v1/refrigerator044/cotracker_features.npz \
  outputs/flow_tracking_eval/slot_benchmark_refrigerator_microwave/model/motion_part_slots.pt \
  outputs/flow_tracking_eval/slot_relation_head/slot_relation_head.pt \
  --output-json outputs/flow_tracking_eval/slot_relation_head/refrigerator044.json \
  --device mps
```

Axis angular loss is sign invariant. For revolute joints, the position loss is
the perpendicular distance to the GT axis line, not Euclidean distance between
two arbitrary pivot points. Prismatic joints do not receive a pivot loss.

The primary `axis_error_deg` metric is conditional on correct joint type. A
revolute prediction classified as prismatic contributes to joint-type error,
not axis error. `axis_error_deg_all_gt_pairs` preserves the unconditional value
as a diagnostic for detecting type-confusion outliers.

The relation head uses two distinct physical losses:

- weighted Kabsch SE(3) replay belongs to the upstream slot model and teaches
  track-to-part assignments to form rigid groups;
- joint-model replay belongs to the relation head and replays child tracks with
  the predicted revolute axis line or prismatic direction after fitting only
  the per-frame joint coordinate.

An experimental dual-branch axis estimator avoids compressing the 3D motion
too early. Enable it with `--axis-geometry-branch`. Edge and joint-type
prediction continue to use the fused slot-relation representation. Axis and
pivot prediction instead use:

1. ordered per-track tokens containing canonical reference position,
   displacement, velocity, visibility, and normalized time;
2. a shared temporal GRU over uniformly sampled trajectory frames;
3. differentiable soft pooling with the slot assignment probabilities; and
4. a separate parent-child geometry head.

This keeps appearance-conditioned tracker features useful for part identity
while forcing axis regression to consume explicit 3D motion. The default
remains the original fused head for checkpoint compatibility.

`--geometry-encoder-type track_gru_transformer` enables the cross-track
ablation. It preserves the track dimension after temporal encoding, selects at
most `--geometry-max-tracks` tracks per slot by soft assignment probability,
applies a permutation-equivariant Transformer without track positional
encoding, and pools only after cross-track interaction. The baseline
`track_gru_average` mode remains the default. Training summaries report model
parameter count, effective tracks per slot, and normalized pooling entropy.

The first PartNet CoTracker ablation used 32 trajectory frames, at most 32
tracks per slot, one cross-track Transformer layer, full SO(3) augmentation,
and 50 epochs. The cross-track decoder-finetune model improved test axis mean
from 54.52 to 47.67 degrees and median from 55.40 to 48.52 degrees relative to
`track_gru_average`, but P90 worsened from 66.26 to 81.18 degrees. The frozen
variant did not improve test median. Cross-track interaction is therefore
still experimental: it helps central cases but does not control catastrophic
axis failures and is not the default.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-slot-relation-head \
  configs/relation_training_partnet_three_methods.tsv \
  outputs/partnet_core_v1_training/cotracker_slots/motion_part_slots.pt \
  --output-dir outputs/partnet_core_v1_training/cotracker_axis_trajectory \
  --axis-geometry-branch \
  --geometry-encoder-type track_gru_transformer \
  --trajectory-samples 32 \
  --trajectory-hidden-dim 128 \
  --geometry-max-tracks 64 \
  --rotation-augmentation \
  --rotation-augmentation-probability 1.0 \
  --rotation-augmentation-scope slot_and_relation_geometry \
  --unfreeze-slot-backbone \
  --slot-unfreeze-scope decoder \
  --slot-learning-rate-scale 0.1 \
  --device cuda
```

Direct angular axis loss is weighted more strongly than joint replay. This is
important because replay can partially compensate an inaccurate axis through
its fitted joint coordinate. Joint replay is therefore a physical consistency
regularizer, not a replacement for axis supervision. Joint-type CE is balanced
by the observed training-joint frequencies.

The slot backbone remains frozen by default. `--unfreeze-slot-backbone` enables
low-rate joint training with slot assignment and existence losses; use
`--slot-learning-rate-scale` to control its optimizer group. Shared SO(3)
augmentation is also experimental and disabled by default. Enable it with
`--rotation-augmentation --rotation-augmentation-probability 0.25`. It rotates
relation-level 3D evidence, axis lines, and replay points consistently while
leaving the non-equivariant slot encoder input unchanged.

On the current held-out refrigerator044/045 and microwave044/045 split, the
recommended frozen/no-augmentation configuration reduced correctly typed axis
error from approximately 4.08 degrees to 2.12 degrees. Fine-tuning slots gave
2.02 degrees and better edge F1, but slightly worse axis-line error. With only
eight training objects, 25% SO(3) augmentation degraded axis error to 6.63
degrees, so more rotation-diverse data or an equivariant relation architecture
is required before enabling it by default. The only held-out prismatic joint
remains misclassified; this is reported as type error rather than contaminating
the primary axis metric.

# TAPIP3D Remote Tracking Experiment

TAPIP3D is CUDA-only in its reference implementation (`xformers`, `torch-scatter`, and custom point operators). The local project therefore prepares the recorded RGB-D sequence and imports remote results; it does not attempt to run TAPIP3D on macOS/MPS.

Prepare one camera view with the exact existing object-mask seed tracks, then copy the NPZ to a CUDA TAPIP3D checkout:

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp prepare-tapip3d-input \
  outputs/recordings_refrigerators_staged_dense_mps/refrigerator038/episode.json \
  --view-index 0 --frame-stride 4 \
  --seed-tracks outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/refrigerator038/object_tracks.json \
  --output-npz /tmp/refrigerator038_view0_tapip3d.npz
```

On the CUDA host, run the official TAPIP3D inference with that NPZ. It preserves the supplied `query_point` values and ignores the additional project-only `track_ids` key:

```bash
python inference.py --input_path refrigerator038_view0_tapip3d.npz \
  --checkpoint checkpoints/tapip3d_final.pth --device cuda --resolution_factor 2
```

Export the frozen TAPIP3D UpdateFormer tokens immediately before the flow head,
then copy the output NPZ back. The exporter reruns inference with a forward hook
because the reference result NPZ stores trajectories but not internal tokens. It
stores per-timestep tokens plus an L2-normalized concatenation of temporal mean,
standard deviation, and endpoint delta. The output keeps the same `track_ids` /
`embeddings` contract as the CoTracker probe input:

```bash
python /path/to/articulated_dynamics_learning/scripts/export_tapip3d_encoder_features.py \
  --tapip-root . --input refrigerator038_view0_tapip3d.npz \
  --result outputs/inference/.../refrigerator038_view0_tapip3d.result.npz \
  --checkpoint checkpoints/tapip3d_final.pth \
  --output tapip3d_updateformer_features.npz
```

Import TAPIP3D's `coords` / `visibs` world trajectories into the existing pose, segmentation, joint, and feature-probe paths:

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp import-tapip3d-tracks \
  /tmp/refrigerator038_view0_tapip3d.npz remote/refrigerator038_view0_tapip3d.result.npz \
  outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/refrigerator038/object_tracks.json \
  --output-json outputs/tapip3d/refrigerator038_view0_tracks.json

PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp probe-track-features \
  outputs/tapip3d/refrigerator038_view0_tracks.json remote/tapip3d_encoder_features.npz \
  --output-json outputs/tapip3d/refrigerator038_view0_feature_probe.json
```

The single-view commands above are adapter smoke tests. The production
CoTracker path runs each camera independently, lifts tracks into the shared
world frame, and then concatenates all views. TAPIP3D comparisons must use the
same protocol. Prepare, infer, export, and import each view separately, then
merge them before segmentation or representation learning:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp merge-tapip3d-views \
  outputs/tapip3d/view0_tracks.json \
  outputs/tapip3d/view1_tracks.json \
  outputs/tapip3d/view2_tracks.json \
  --output-tracks outputs/tapip3d/multiview_tracks.json \
  --features outputs/tapip3d/view0_tokens.npz \
             outputs/tapip3d/view1_tokens.npz \
             outputs/tapip3d/view2_tokens.npz \
  --output-features outputs/tapip3d/multiview_updateformer_features.npz
```

This merge does not deduplicate cross-view observations. Each camera supplies
independent tracks with globally unique IDs; their trajectories are comparable
because TAPIP3D predicts them in the same world frame.

## Per-Track Slot Head

Pairwise affinity supervises graph edges but does not force all observations of
one physical part to share one identity. The experimental track-slot head
instead predicts a part slot for every track. Hungarian matching aligns slots
with simulator part labels independently for each training object, so raw GT
part IDs do not need to be consistent across objects.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-track-slot-head \
  configs/track_slot_cotracker_refrigerators_038_040.tsv \
  --output-dir outputs/track_slot_head \
  --max-slots 8 --device mps

PYTHONPATH=src python3 -m rgbd_urdf_mvp predict-track-slots \
  path/to/object_tracks.json path/to/features.npz \
  outputs/track_slot_head/track_slot_head.pt \
  --output-json outputs/track_slot_head/predicted_slots.json \
  --output-viewer outputs/track_slot_head/viewer_slots.html \
  --device mps
```

Prediction generates the interactive flow viewer by default. Use `--no-viewer`
only for non-debug batch runs. The current head uses inverse-frequency slot CE
to prevent the static base from collapsing smaller moving parts, plus a
balanced pairwise partition loss as an auxiliary objective.
