# ICRA revision draft: equivariant relation geometry and real-world robustness

This document is a LaTeX-ready revision guide for
`ICRA_template__Xiaotong_-2.pdf`. The editable TeX source is not present in the
workspace, so the PDF itself has not been modified. Numbers marked
**diagnostic** must not be placed in the main comparison table.

## 1. Required correction to the current manuscript

The current Method section describes a generic relation MLP that directly
regresses a free 3D axis, while the Training section states that consistent
SO(3) augmentation is used. Subsequent controlled experiments show that this
combination is not structurally equivariant: Haar augmentation collapses the
old head toward broadly inaccurate 60--70 degree predictions. The revised
paper must not present augmentation alone as rotation invariance.

The current main numbers (116 objects, old full-neural head) remain the frozen
baseline until the VN candidate passes the same end-to-end evaluation contract.
In particular, the PDF's 2.40 degree median axis error cannot yet be replaced by
the VN full-test value of 12.39 degrees.

## 2. Replacement Method text

### SO(3)-equivariant joint geometry

For each ordered parent--child slot pair, we separate invariant relation
classification from equivariant geometric prediction. Edge existence and joint
type are predicted from scalar features formed from norms, dot products,
visibility, motion magnitude, and learned appearance conditioning. Joint axes
are predicted by a lightweight vector-neuron branch whose vector channels are
constructed only from explicit relative 3D trajectories, center differences,
and relative rigid-motion summaries. Vector layers use scalar gating, vector
linear combinations, dot products, norms, and cross products; the axis output
contains no unconstrained world-coordinate bias. Consequently, rotating all
explicit geometric inputs by `Q in SO(3)` rotates the predicted undirected axis
by the same `Q`, while edge and type probabilities remain invariant.

### Pivot-B revolute axis-line parameterization

Directly regressing an arbitrary 3D pivot is over-parameterized because points
along the joint axis represent the same line. For revolute joints, Pivot-B uses
an analytic hinge-line estimate obtained from relative motion as an equivariant
base point, then predicts a learned residual restricted to the plane normal to
the predicted axis. The analytic estimate is computed solely from observations
and does not use ground-truth joint parameters. This hybrid parameterization
retains feed-forward learned type and axis prediction while enforcing the gauge
freedom of an axis line.

### Track-quality-aware temporal aggregation

The relation geometry branch consumes complete trajectories rather than only
endpoint displacement. Per-track and per-timestep quality scores down-weight
depth discontinuities, short visible fragments, and inconsistent rigid motion.
For diagnostic analytic fitting, contiguous valid segments are fitted
independently and combined through robust consensus. Low-observability joints
are reported separately rather than silently mapped to a canonical world axis.

## 3. Replacement Training text

We train part discovery and relation prediction in stages. The slot network is
first trained on persistent 3D trajectories. The relation classifier and
equivariant geometry branch are then trained using frozen predicted slots, with
consistent rotations applied to all explicit 3D quantities. Opaque CoTracker
appearance descriptors are used only as scalar conditioning and cannot directly
produce world-coordinate vectors. Paired original/rotated samples supervise
axis equivariance and edge/type consistency. Temporal occlusion, depth spikes,
track fragmentation, and recovery after occlusion are introduced as structured
trajectory corruptions. A final low-learning-rate joint fine-tuning stage is
selected only on validation metrics and must not degrade slot ARI/RI.

## 4. New controlled ablation table

The following is the completed 121-object / 275-joint VN diagnostic. It uses a
different evaluation population from the PDF's frozen 116-object table and
therefore belongs in an ablation until a common-intersection rerun is complete.

| Relation geometry | Edge F1 | Type acc. | Axis mean | Axis median | Axis P90 | Axis-line / bbox | Slot ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy VN pivot | 0.9544 | 0.9055 | 26.34 | 12.39 | 73.94 | 0.1499 | 0.9598 |
| Pivot-B direct switch | 0.9544 | 0.9055 | 26.34 | 12.39 | 73.94 | 0.1296 | 0.9598 |
| Pivot-B, pivot-only 10 epochs | 0.9544 | 0.9055 | 26.34 | 12.39 | 73.94 | **0.0833** | 0.9598 |
| Pure VN-C, pivot-only 15 epochs | 0.9544 | 0.9055 | 26.34 | 12.39 | 73.94 | 0.1338 | 0.9598 |

Pivot-B reduces normalized axis-line error by 44.4% relative to the legacy VN
pivot without changing slots, topology, type, or axis direction. Pure VN-C is
the strictly neural ablation; its smaller gain shows that the restricted pivot
representation was defective, but that the observation-derived hinge proposal
remains useful for line localization.

## 5. SO(3) audit paragraph

We evaluate each joint under five fixed Haar rotations of the entire explicit
3D object representation. Across 2,750 paired predictions, the VN/Pivot-B
relation geometry branch obtains 0.00026 degree median and 0.345 degree P90
axis-equivariance error; revolute P90 is 0.532 degree. Four pairs exceed 10
degrees and none exceed 30 degrees. This audit validates the relation geometry
branch, not rotation invariance of upstream slot assignments. The paper should
report both the isolated relation audit and a full-pipeline rotated-object audit.

## 6. Real-data evaluation paragraph

On seven manually annotated RGB-D scenes (nine GT joints), the current robust
pipeline obtains scene-macro part IoU 0.549 and ARI 0.373. Directed edge
precision/recall/F1 are 0.500/0.667/0.571. Six matched joints are type-correct,
but their axis error remains 51.51 degrees mean, 54.61 degrees median, and 80.81
degrees P90. These results expose a substantial simulation-to-real gap caused by
slot under/over-segmentation, fragmented visibility, correlated depth errors,
and insufficiently diverse real trajectory noise. They should be reported as a
limitation or robustness study, not as evidence that Pivot-B alone solves real
joint recovery.

## 7. Evaluation protocol clarification

Primary axis statistics should be computed on recovered, type-correct joints.
Zero-motion and pre-declared low-observability joints should be excluded from
the primary axis metric using a threshold fixed from GT/input excitation before
evaluating predictions, and retained in a separate coverage/abstention table.
End-to-end performance must additionally report edge recall/F1, type-correct
joint recall, and failure-penalized axis error so that exclusion does not hide
missed joints. Predicted confidence must never determine which GT examples enter
the primary benchmark.

## 8. HOI4D extension status

We preselect a 381-sequence articulation-focused HOI4D core subset containing
laptops, storage furniture, safes, and trash cans. HOI4D annotations and camera
parameters are being acquired first; RGB/depth clips will be selected only after
the annotation inventory identifies usable articulation segments. This dataset
is intended for real-noise adaptation and robustness evaluation. No HOI4D result
should enter the manuscript until the subset has been validated for joint type,
axis supervision, object masks, and train/test leakage.

## 9. Remaining gates before replacing the main method

1. Rerun old head, VN, and VN+Pivot-B on the identical 116-object/270-joint test intersection.
2. Report canonical and five-Haar full-pipeline results, not relation-only equivariance.
3. Complete TAPIP3D-only and Hybrid feature-path reruns with the selected relation head.
4. Report primary observable-joint metrics and all-joint coverage/failure metrics together.
5. Evaluate the fixed checkpoint on the annotated real scenes and the curated HOI4D subset.
6. Update the abstract only after one validation-selected configuration passes these gates.

## 10. Revised experiment narrative and table allocation

The current Table III and Table IV should not remain as two equal-weight main
ablation tables. They answer development questions rather than demonstrating
the revised method's central contributions, and Table IV uses Hybrid slots even
though the selected system uses CoTracker-only slots.

### Main paper

1. Keep the object-aligned external comparison table, with protocol and oracle
   columns. This establishes the paper-level comparison.
2. Keep one compact end-to-end table for the selected CoTracker-only model on
   the fixed 116-object intersection. Report segmentation, topology, type,
   axis mean/median/P90, axis-line, and Joint@20 together.
3. Replace the current controlled-ablation tables with one contribution-aligned
   table containing three blocks:
   - free 3D axis regression versus vector-neuron axis prediction, evaluated on
     canonical inputs and five-Haar rotated pairs;
   - legacy pivot versus Pure VN-C versus Pivot-B, isolating axis-line error;
   - no quality handling versus quality weighting and structured occlusion/noise
     training, reporting both PartNet and real-scene robustness.
4. Keep the complexity figure because it supports a distinct and important
   limitation claim about weakly excited and high-part-count objects.

### Supplementary material

Move the CoTracker/TAPIP3D/Hybrid table to the supplement and describe it as
trajectory-source selection. The main text needs only one sentence: CoTracker
was selected by validation slot-assignment loss and used for all main results.
The small numerical differences do not justify a full main-paper analysis.

Move full-neural/neural-type-plus-analytic/full-analytic to the supplement as an
estimator decomposition. Analytic fitting is valuable because it diagnoses
observability and supplies the Pivot-B hinge proposal, but it is not a competing
version of the final method. The main text should report only the conclusion:
learned classification is stronger, while observation-derived geometry improves
line localization when incorporated through an equivariant parameterization.

### Proposed main ablation layout

| Design | Canonical axis median/P90 | Rotated-pair median/P90 | Axis-line | Type acc. | Real axis median | Purpose |
|---|---:|---:|---:|---:|---:|---|
| Old free-axis head | TBD | existing failure audit | legacy | TBD | TBD | exposes canonical-axis shortcut |
| VN axis | 12.39 / 73.94 | 0.00026 / 0.345 | 0.1499 | 0.9055 | TBD | validates structural SO(3) equivariance |
| VN + Pivot-B | 12.39 / 73.94 | 0.00026 / 0.345 | **0.0833** | 0.9055 | 54.61 | validates line parameterization |
| VN + Pivot-B + robust training | TBD | TBD | TBD | TBD | TBD | validates real-noise robustness |

All rows must be regenerated on one common object/joint intersection before
publication. The table deliberately leaves cells as TBD instead of combining
incompatible pilots. Rotation augmentation is meaningful here only when paired
with the old non-equivariant head and the structurally equivariant VN head; a
standalone “with/without augmentation” row would conflate architecture and data.

## 11. Frozen canonical results and immediate training protocol

The external canonical predictions are frozen in
`outputs/paper_experiments/external_canonical_v2/`. Canonical segmentation is available for
all successful adapters, but axis prediction is not a common output of every method. The paper
must distinguish unsupported axis metrics from failed runs and from results obtained under a
different native protocol. AiM and ReArt axis numbers are diagnostic native/pilot results;
GaussianArt uses oracle parts; PARIS is restricted to a two-part subset. These results must not
be presented as a single paired axis leaderboard.

On the 121-object/275-joint PartNet test, three-seed averages are:

| VN setting | Edge F1 | Type acc. | Axis mean | Axis median | Axis P90 | Axis-line | Slot ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| no rotation, frozen slot | 0.9540 | 0.9079 | 27.58 | 13.78 | 74.28 | 0.1522 | 0.9583 |
| no rotation, all slot components | 0.9650 | 0.9152 | 26.46 | 13.27 | 73.71 | 0.1505 | 0.9584 |
| Haar, all slot components | 0.9650 | 0.9079 | 26.40 | 12.77 | 76.06 | 0.1513 | 0.9592 |

Haar augmentation therefore does not materially improve canonical accuracy. Its paper claim must
come from a paired rotated-object robustness audit. The proposed same-day adaptation experiment
uses PartNet initialization, Arti4D supervision where joint labels are valid, and a new real-data
adaptation split grouped by physical object. The previously annotated seven real scenes remain a
held-out test set. Open/close recordings of the same appliance must not be split independently
across train and test.

The paper-facing axis table layout and its populated rotation-robustness block are defined in
`docs/paper-axis-comparison-table.md`. Axis prediction remains in the main paper, with a paired
PartNet block and a visibly separated native-protocol diagnostic block.
