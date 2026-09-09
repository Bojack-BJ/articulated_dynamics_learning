# Current Experimental Results and Recommended Paper Organization

## 1. Purpose and status

This document consolidates the experimental evidence currently available in the repository. It is intended as an input to writing the paper's Experiments section, not as a replacement for the raw result files.

The results fall into three different evaluation contracts and must not be averaged or compared without qualification:

1. **Ours controlled held-out evaluation:** 116 common PartNet-Mobility test objects and 270 GT joints. This is the authoritative source for the main method, feature ablations, kinematic-backend ablations, category analysis, and complexity analysis.
2. **Object-aligned external baseline suite:** 20 PartNet objects evaluated with common segmentation metrics, while each external method retains its native or favorable acquisition protocol and declared oracle assumptions. This is a protocol-specific external comparison, not an identical-input comparison.
3. **Reproduction and diagnostic pilots:** small subsets used to study temporal resolution, analytic observability, SO(3) shortcuts, quality weighting, AiM protocol sensitivity, and baseline failure modes. These results explain behavior but should not be mixed into the main averages.

The validation-selected main configuration under the fixed-connected-body collapsed kinematic ontology is:

```text
CoTracker-only features + neural slot decomposition + full neural relation head
```

This supersedes the earlier Hybrid selection, which used validation assignment losses computed under the pre-collapse ontology. The analytic estimators remain evaluation backends and diagnostics; they are not the selected main inference path.

## 2. Recommended Experiments section

A clean paper organization is:

1. **Experimental setup**
   - datasets and split;
   - recording and input protocols;
   - metrics;
   - implementation and model-selection details;
   - external baseline assumptions.
2. **Main held-out results for Ours**
   - segmentation, topology, type, axis, and axis-line metrics on 116 objects / 270 joints.
3. **Controlled method ablations**
   - CoTracker vs TAPIP3D vs Hybrid;
   - full neural vs neural type plus analytic axis vs full analytic.
4. **Generalization with articulation complexity**
   - GT part-count buckets 2, 3--4, and >=5;
   - category-level failure analysis.
5. **External baseline comparison**
   - common segmentation table on the aligned 20-object subset;
   - explicit protocol and oracle labels;
   - common-success/failure reporting.
6. **Analysis and diagnostics**
   - neural vs analytic axis decomposition;
   - observability and temporal resolution;
   - SO(3) shortcut audit;
   - runtime scope.
7. **Baseline reproduction study**
   - AiM native/paper-like sanity checks and failure localization;
   - preferably placed in the supplement unless it is a central paper contribution.

The object-mask optimization-path experiments, EM-lite routing, and dynamics identification are useful secondary studies. They should be presented as supplementary development analyses or a downstream extension rather than mixed into the feedforward main table.

## 3. Evaluation contracts

### 3.1 Main PartNet-Mobility setup

The default simulated interaction contains 120 source frames recorded over eight seconds at 15 FPS. Each timestep contains three calibrated RGB-D views. The deployable inputs are RGB-D observations, camera intrinsics/extrinsics, and an object-level mask. Semantic part labels and joint metadata are used only for training supervision and evaluation.

The controlled comparison uses the common intersection across all feature/backend configurations:

- 116 held-out test objects;
- 270 GT joints;
- categories: coffee machine, dishwasher, door, drawer, microwave, oven, and refrigerator;
- complexity buckets: 2 parts, 3--4 parts, and >=5 parts.

Feature and backend selection are validation-only:

- feature path: minimum validation slot-assignment loss;
- backend: maximum validation Joint@20, followed by failure-penalized axis error and joint-type accuracy;
- selected feature: CoTracker only, validation assignment loss 0.108469;
- comparison losses under the same ontology: Hybrid 0.131409 and TAPIP3D only 0.290504;
- selected backend: full neural.

### 3.2 Metrics

Segmentation metrics:

- Hungarian one-to-one Point IoU;
- Adjusted Rand Index (ARI);
- ordinary Rand Index (RI);
- exact predicted part-count rate.

Kinematic metrics:

- GT-pair edge recall;
- joint-type accuracy;
- undirected axis angular error using `acos(abs(dot(pred, gt)))`;
- revolute axis-line distance normalized by object bounding-box diagonal;
- failure-penalized axis error, which assigns a penalty when the edge or type is incorrect;
- Joint@10 and Joint@20, requiring an end-to-end recovered joint with angular error below the threshold.

`edge_recall` is used instead of Edge F1 in the controlled oracle-decomposition evaluation because this evaluator measures recovery of GT pairs and does not enumerate every unmatched predicted edge.

### 3.3 External baseline contract

The external suite aligns object identity and segmentation metrics but intentionally preserves method-specific input protocols. It is therefore:

```text
object-aligned, metric-aligned, protocol-specific
```

It is not a strict identical-input benchmark. Oracle information is reported explicitly and unsupported outputs remain blank.

## 4. Main result: Ours controlled held-out evaluation

> **Selection update (2026-08-22):** CoTracker-only is the validation-selected feature path after retraining and evaluating all paths with the fixed-connected-body collapsed ontology. Tables below that label Hybrid as Ours Main are retained as historical results and must not be copied into the final manuscript without regeneration. The authoritative replacement artifacts are under `outputs/paper_experiments/cotracker_selection_v2`.

### 4.1 Five controlled configurations

| Configuration | Point IoU | ARI | RI | Edge recall | Type acc. | Axis mean | Axis median | Axis-line | Penalized axis | Joint@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoTracker + full neural | **0.678** | **0.750** | **0.894** | 0.878 | **0.885** | 12.27 deg | **1.40 deg** | 0.084 | **26.96 deg** | **0.685** |
| TAPIP3D + full neural | 0.625 | 0.657 | 0.845 | 0.852 | 0.863 | 19.56 deg | 1.58 deg | 0.099 | 36.25 deg | 0.570 |
| Hybrid + full neural (legacy selection) | 0.669 | 0.716 | 0.874 | 0.856 | **0.885** | 14.09 deg | 1.67 deg | 0.119 | 28.71 deg | 0.659 |
| Hybrid + neural type + analytic axis | 0.669 | 0.716 | 0.874 | 0.856 | 0.885 | 17.45 deg | 5.34 deg | **0.016** | 35.18 deg | 0.548 |
| Hybrid + full analytic | 0.669 | 0.716 | 0.874 | 0.856 | 0.596 | 12.56 deg | 3.14 deg | 0.017 | 44.40 deg | 0.493 |

### 4.2 Main interpretation

These segmentation metrics are recomputed on one method-independent source-frame GT point cloud, after collapsing fixed-body chains into the same kinematic-part ontology used by the external suite. The earlier 0.894 Hybrid IoU was measured on Hybrid's own track population and is retained only as a legacy track-domain diagnostic.

On the newly collapsed common domain, CoTracker-only has the highest validation ranking and is Ours Main. Its 116-object segmentation metrics are Point IoU 0.683, ARI 0.765, RI 0.899, exact part count 0.595, under-segmentation 0.336, and over-segmentation 0.069. Hybrid obtains Point IoU 0.653, ARI 0.708, RI 0.872, exact part count 0.612, under-segmentation 0.319, and over-segmentation 0.069.

The full neural relation head is the validation-selected end-to-end backend. It preserves high type accuracy and gives the best Hybrid Joint@20. Analytic fitting estimates axis-line position much more accurately, but loses end-to-end success because joint-type selection and low-observability cases are less reliable.

CoTracker plus the full neural backend obtains a slightly higher test Joint@20 than Hybrid (0.685 vs 0.659), despite weaker segmentation. This is not used to reselect the method after testing: Hybrid was selected by validation assignment loss. The distinction should be stated to avoid test-set model selection.

The large gap between mean and median neural axis error is important. Typical type-correct predictions are highly accurate, with medians around 1.4--1.7 degrees, but a small catastrophic tail raises the mean to 12--20 degrees. The paper should therefore report mean, median, and thresholded joint success rather than mean axis error alone.

## 5. Feature-path ablation

The unified feature ablation uses the same 116 objects, kinematic-part ontology, and source-frame GT reference points for all three methods.

| Feature path | Point IoU | ARI | RI | Exact part count |
|---|---:|---:|---:|---:|
| **CoTracker only** | **0.683** | **0.765** | **0.899** | 0.595 |
| TAPIP3D only | 0.625 | 0.663 | 0.850 | 0.586 |
| Hybrid | 0.653 | 0.708 | 0.872 | **0.612** |

CoTracker gives the strongest common-domain overlap and is selected by validation loss. Hybrid has a slightly higher exact part-count rate, but that test statistic is not used to reselect the model. TAPIP3D-only remains weakest overall. The previous tracker-domain ablation can still diagnose assignment quality on each tracker's own samples, but it is not used as the paper's Point IoU table.

## 6. Kinematic-backend ablation

The three Hybrid backend variants isolate the role of learned and analytic inference:

- **Full neural:** learned topology, type, axis, and line parameters.
- **Neural type + analytic axis:** learned topology/type, then relative-SE(3) analytic axis fitting.
- **Full analytic:** neural topology proposals followed by analytic type selection and axis fitting through point replay.

Key conclusions:

1. Neural type prediction is substantially stronger than analytic model selection: type accuracy 0.885 vs 0.596.
2. Analytic geometry is stronger for revolute line localization: normalized axis-line error about 0.016--0.017 vs 0.119.
3. The full neural backend has the best Hybrid end-to-end Joint@20: 0.659 vs 0.548 and 0.493.
4. The analytic backend is best treated as an interpretable baseline, confidence-aware fallback, and oracle diagnostic rather than the main method.

## 7. Complexity scaling

### 7.1 Ours Main by GT part count

| GT parts | Objects | Point IoU | ARI | RI | Edge recall | Type acc. | Penalized axis | Joint@20 | Exact part count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 47 | 0.991 | 0.984 | 0.992 | 1.000 | 1.000 | 7.07 deg | 0.915 | 0.936 |
| 3--4 | 46 | 0.930 | 0.980 | 0.992 | 0.884 | 0.920 | 23.34 deg | 0.743 | 0.826 |
| >=5 | 23 | 0.626 | 0.839 | 0.935 | 0.808 | 0.836 | 36.01 deg | 0.551 | 0.391 |

The method is nearly saturated on two-part objects. Performance degrades monotonically with articulation complexity, with the dominant deterioration occurring in exact part-count recovery, edge recall, and failure-penalized kinematics. This is the most important limitation analysis in the current results.

The complexity table also supports a stronger claim than a global average: the method's advantage comes from retaining useful decomposition on multi-part objects, but >=5-part articulation remains far from solved.

### 7.2 Category analysis

For Ours Main:

- refrigerator: IoU 0.995, ARI 0.998, Joint@20 1.000 over five objects;
- door: IoU 0.929, ARI 0.973, Joint@20 0.793 over 75 objects;
- drawer: IoU 0.866, ARI 0.943, Joint@20 0.828 over 24 objects;
- dishwasher: IoU 0.850, ARI approximately 1.000, Joint@20 0.800 over four objects;
- oven: IoU 0.747, Joint@20 0.311 over three objects;
- coffee machine: IoU 0.426, Joint@20 0.215 over four objects;
- microwave: one object, so its perfect segmentation and 0.5 Joint@20 should not be overinterpreted.

Coffee machines and ovens are the clearest category failures. They combine complex topology, small parts, and less separable motion. Refrigerator performance is high in this held-out set, but the sample count is small and should be reported alongside N.

## 8. Analytic axis oracle and error decomposition

The Hybrid analytic oracle uses GT parts, GT edges, and GT type, then estimates the axis from relative motion. It isolates the information contained in the observed tracks from segmentation, topology, and type errors.

### 8.1 Overall oracle comparison

| Axis backend on GT pairs/type | Evaluated joints | Mean | Median | P90 | Errors >80 deg | Revolute line error |
|---|---:|---:|---:|---:|---:|---:|
| Analytic relative motion | 229 | 17.10 deg | 5.63 deg | 56.82 deg | 3 | 0.0146 bbox |
| Neural axis | 270 | 18.75 deg | **1.88 deg** | 82.87 deg | 30 | 0.133 bbox |

This comparison reveals complementary behavior:

- the neural head has a much sharper central distribution;
- the neural head has a larger catastrophic near-orthogonal tail;
- the analytic estimator has poorer median accuracy but much better line localization and far fewer >80-degree failures.

### 8.2 By joint type

Under the GT-parts/GT-edges/GT-type analytic oracle:

- revolute: 114 evaluated joints, mean 5.31 degrees, median 2.09 degrees, P90 11.15 degrees;
- prismatic: 115 evaluated joints, mean 28.79 degrees, median 20.31 degrees, P90 64.80 degrees.

The analytic estimator is therefore a strong revolute baseline but is not robust for prismatic motion in the current observations. Short translation, depth noise, and near-static tracks make the principal translation direction poorly observable.

### 8.3 Confidence and excitation

For high-confidence analytic fits, the mean/median errors fall to approximately 7.48/2.47 degrees with no >80-degree failures. High-motion joints similarly reach about 6.45-degree mean error, while low-motion joints exceed 50 degrees on average.

This is strong evidence that kinematic accuracy is governed by motion observability, not only estimator capacity. A confidence-aware hybrid between learned and analytic predictions is a plausible future direction, but it is not part of the current main method.

## 9. Temporal-resolution experiment

This is a five-object training-set stress test, not a held-out result.

| Temporal profile | Backend | Axis mean | Axis median | >80 deg | Joint@10 | Joint@20 |
|---|---|---:|---:|---:|---:|---:|
| 15 FPS recording / 3.75 Hz tracks | Full neural | 11.78 deg | 1.24 deg | 1 | 0.875 | 0.875 |
| 30 FPS recording / 15 Hz tracks | Full neural | 11.86 deg | 1.24 deg | 1 | 0.875 | 0.875 |
| 15 FPS recording / 3.75 Hz tracks | Predicted parts + analytic | 5.56 deg | 2.31 deg | 0 | 0.750 | 1.000 |
| 30 FPS recording / 15 Hz tracks | Predicted parts + analytic | **3.01 deg** | **2.05 deg** | 0 | **1.000** | 1.000 |

Denser temporal sampling improves explicit relative-motion fitting, reducing analytic mean error by 2.55 degrees. It does not improve the old neural checkpoint because the model was not retrained for the new temporal distribution. The experiment supports higher temporal resolution for geometric fitting, but does not yet establish a held-out gain for the learned model.

## 10. SO(3) augmentation and orientation-shortcut audit

The 10-epoch diagnostic compares orientation augmentation on the same oracle joint set.

| Augmentation | Edge F1 | Type acc. | Axis mean | Axis median | P90 | >80 deg |
|---|---:|---:|---:|---:|---:|---:|
| None | 0.859 | 0.920 | 24.75 deg | 3.58 deg | 88.08 deg | 47 |
| Yaw only | 0.856 | 0.913 | 46.57 deg | 46.34 deg | 85.28 deg | 40 |
| Random axis <=15 deg | **0.869** | 0.905 | **23.28 deg** | **3.05 deg** | 86.77 deg | 49 |
| Random axis <=30 deg | **0.869** | 0.909 | 26.52 deg | 7.74 deg | 86.86 deg | 45 |
| Haar SO(3) | 0.850 | 0.902 | 58.80 deg | 64.83 deg | 68.34 deg | 0 |

The experiment exposes a canonical-orientation shortcut, but aggressive Haar augmentation does not solve it. The absence of >80-degree errors under Haar is caused by collapse toward broad 60--70-degree errors, not true equivariance. Mild random-axis augmentation is the least harmful variant but does not remove the catastrophic tail. This belongs in an analysis or supplementary section, not the main ablation table.

## 11. External baseline suite

### 11.1 Protocol and oracle inventory

| Method | Protocol | Success | Oracle / scope |
|---|---|---:|---|
| Ours Hybrid | Continuous RGB-D interaction | 20/20 | None declared |
| AiM | AiM-style cross-protocol interaction | 20/20 | None declared |
| ReArt | Native 4D point cloud | 20/20 | None declared |
| DTA | Two-state multiview RGB-D | 17/20 | GT part count |
| ArtGS | Two-state multiview RGB-D | 20/20 | GT part count |
| VideoArtGS | Continuous monocular interaction | 14/20 | Joint count, types, and parent topology |
| GaussianArt | Two-state multiview | 20/20 | Part count, semantic initialization, and GT motion metadata |
| PARIS | Two-state RGB, two-part scope | 4/5 | Native two-part scope |
| Ditto | Two fused point clouds, one-joint scope | 0/5 | Native one-joint scope; checkpoint unavailable |

### 11.2 Common segmentation metrics

| Method | Metric N | Point IoU | ARI | RI |
|---|---:|---:|---:|---:|
| **Ours Hybrid** | 20 | **0.731** | **0.846** | **0.940** |
| AiM | 17 | 0.228 | 0.062 | 0.549 |
| ReArt | 16 | 0.221 | 0.046 | 0.506 |
| DTA | 17 | 0.481 | 0.511 | 0.814 |
| ArtGS | 20 | 0.409 | 0.592 | 0.836 |
| VideoArtGS | 14 | 0.238 | 0.392 | 0.688 |

GaussianArt and PARIS do not expose predicted semantic part labels or mappable part meshes in the current official-output adapters, so common IoU/ARI/RI are unsupported rather than zero. Ditto was not run because the official released checkpoint links return HTTP 404.

The main conclusion is that Ours has the strongest common segmentation result on the aligned subset. However, the methods are not information-equivalent: DTA and ArtGS know the GT part count, while VideoArtGS receives an even stronger kinematic inventory. These oracle labels must remain visible in the table or caption.

Under the unified observable kinematic domain, external Ours has IoU 0.625 on its 20-object subset and controlled Ours Main has IoU 0.669 on 116 objects. The ontology and reference-domain definitions are now aligned; the remaining difference reflects the object subset and acquisition/export path rather than two definitions of Point IoU.

### 11.3 External complexity analysis

| GT parts | Method | Point IoU | ARI | RI | Underseg. |
|---|---|---:|---:|---:|---:|
| 2 | Ours | **1.000** | **1.000** | **1.000** | 0.000 |
| 2 | AiM | 0.390 | 0.067 | 0.562 | 0.000 |
| 2 | ReArt | 0.377 | 0.077 | 0.537 | 0.000 |
| 2 | DTA | 0.677 | 0.472 | 0.794 | 0.000 |
| 2 | ArtGS | 0.694 | 0.781 | 0.917 | 0.000 |
| 2 | VideoArtGS | 0.386 | 0.523 | 0.778 | 0.000 |
| 3--4 | Ours | **0.873** | **0.973** | **0.996** | 0.286 |
| 3--4 | AiM | 0.252 | 0.088 | 0.590 | 1.000 |
| 3--4 | ReArt | 0.252 | 0.038 | 0.706 | 0.250 |
| 3--4 | DTA | 0.424 | 0.476 | 0.843 | 0.286 |
| 3--4 | ArtGS | 0.408 | 0.590 | 0.903 | 0.000 |
| 3--4 | VideoArtGS | 0.156 | 0.135 | 0.532 | 0.200 |
| >=5 | Ours | **0.440** | **0.639** | **0.854** | 0.750 |
| >=5 | AiM | 0.095 | 0.040 | 0.509 | 1.000 |
| >=5 | ReArt | 0.092 | 0.030 | 0.369 | 1.000 |
| >=5 | DTA | 0.365 | 0.598 | 0.794 | 0.600 |
| >=5 | ArtGS | 0.231 | 0.477 | 0.727 | 0.375 |
| >=5 | VideoArtGS | 0.201 | 0.546 | 0.773 | 0.400 |

Every method degrades with complexity. AiM and ReArt strongly under-segment the >=5-part objects in this cross-protocol setting. Ours retains the best common IoU/ARI/RI but also under-segments 75% of these difficult objects. This is evidence of a relative advantage, not evidence that complex articulation is solved.

### 11.4 Native kinematic outputs

The currently available method-native numbers are:

- VideoArtGS: axis error 11.02 degrees and axis-position/line error 0.123;
- GaussianArt: axis error 47.11 degrees, axis-position error 0.163, and motion error 17.05 in its native reported units;
- PARIS: axis error 44.49 degrees, axis-position error 0.395, and novel-view PSNR 26.94.

These values are not yet a unified kinematic benchmark. Their oracle assumptions, units, matching domains, and supported outputs differ. They should be reported in a method-specific appendix table unless adapters establish a common metric contract.

## 12. External runtime

| Method | Measured N | Mean | Median | Scope |
|---|---:|---:|---:|---|
| Ours Hybrid | 117 | 4.30 s | 4.25 s | Slot segmentation only; excludes tracking and relation head |
| ReArt | 19 | 67.52 s | 68.34 s | Official native inference |
| ArtGS | 20 | 863.05 s | 882.38 s | Official end-to-end optimization |
| DTA | 17 | 2280.03 s | 2105.89 s | Official end-to-end optimization |

No end-to-end Ours speedup should be claimed from this table because the scopes differ. The result does show that the feedforward slot stage is lightweight compared with per-object optimization, but tracker extraction and relation inference still need a complete timing measurement.

## 13. AiM protocol and reproduction studies

These experiments should be separated into a reproduction/failure-analysis subsection. They do not replace the 20-object aligned baseline.

### 13.1 Shared-view and camera-protocol pilot

On a three-object pilot:

| Method / protocol | Point IoU | ARI | Geometry coverage |
|---|---:|---:|---:|
| Hybrid / shared 3-view | 0.635 | 0.706 | 0.358 |
| Hybrid / continuous orbit | 0.274 | 0.112 | 0.125 |
| AiM / shared 3-view | 0.286 | 0.197 | 0.614 |
| AiM / continuous orbit | 0.214 | 0.126 | 0.645 |

The orbit raises AiM geometry coverage slightly but reduces segmentation quality. The drawer is the clearest counterexample: coverage rises from 0.390 to 0.687 while IoU falls from 0.284 to 0.213. Camera coverage is therefore not the sole explanation for AiM under-segmentation.

Across the extended shared-view evaluation, correlation between AiM geometry coverage and Point IoU is approximately zero (Pearson 0.051 over 18 successful objects). High-coverage failures include coffee machine 103069, with coverage 0.910 but only 2 predicted parts for 6 GT parts.

### 13.2 Paper-like object and temporal-budget sanity

Four PartNet IDs listed by AiM were evaluated with 200 frames for simpler objects and 500 frames for complex objects, but with the project's automatic staggered schedule and incomplete 5k/8k optimization budget:

| Object | Pred./GT | Point IoU | ARI | RI | Coverage |
|---|---:|---:|---:|---:|---:|
| Storage-47024 | 4/2 | 0.275 | 0.003 | 0.495 | 0.657 |
| Fridge-11304 | 2/3 | 0.206 | 0.016 | 0.506 | 0.655 |
| Storage-47648 | 2/7 | 0.099 | 0.027 | 0.512 | 0.676 |
| Table-31249 | 3/5 | 0.165 | 0.045 | 0.556 | 0.757 |

The larger frame budget alone does not recover paper-level decomposition. This result is explicitly paper-like rather than an exact reproduction because motion schedules, optimization completion, and the published volumetric metric are not all aligned.

### 13.3 Sequential-RANSAC sensitivity on Storage-47648

Using frozen upstream Gaussian trajectories, the official 10% minimum-inlier rule recovers one post-merge component. Relaxing the ratio to 2.5% yields three post-merge components for the current orbit, still below seven GT parts.

The audit establishes three failure sources:

1. the global minimum-inlier threshold suppresses small moving components;
2. largest-connected-component filtering is secondary but removes additional support;
3. upstream Gaussian trajectories remain mixed or weakly separable, so threshold relaxation alone cannot recover all parts.

This is a diagnostic-only sensitivity analysis. Relaxed-threshold results must not replace the official baseline.

### 13.4 Favorable observation sanity on Storage-47024

Moving closer increased object occupancy from 0.477 to 0.546 and geometry coverage from 0.657 to 0.752, but segmentation remained 4/2 and Point IoU decreased from 0.275 to 0.243. A synchronized four-view upper bound exposed an official implementation incompatibility and failed before RANSAC.

The supported conclusion is that camera distance alone is not the primary bottleneck. The evidence instead points to deformation optimization and motion-representation quality before sequential model fitting.

### 13.5 Paper-style TSDF/voxel metrics

Four simple two-part front-oscillation runs were evaluated using AiM TSDF component meshes voxelized at 4 mm:

| Object | Full-object voxel IoU | Part voxel IoU | Pred./GT meshes | Type acc. | Axis angle |
|---|---:|---:|---:|---:|---:|
| partnet_10849 | 0.024 | 0.015 | 2/2 | n/a | n/a |
| partnet_12540 | 0.033 | 0.015 | 2/2 | 1.0 | 82.67 deg |
| partnet_46556 | 0.032 | 0.017 | 2/2 | 1.0 | 12.19 deg |
| partnet_9388 | 0.052 | 0.024 | 2/2 | 0.0 | n/a |

Aggregate part voxel IoU is 0.0176 and full-object voxel IoU is 0.0349. Joint-type accuracy is 0.667 over matched joints, with a type-correct axis mean of 47.43 degrees. Diagnostic ICP does not recover the mesh overlap, suggesting that the discrepancy is not only a global frame offset.

These values show that changing from observed-point metrics to paper-style volumetric metrics does not explain the reproduction gap in the current generated data chain. They are still not a reproduction of the exact paper assets.

### 13.6 Exact-paper A2 status

The simultaneous-motion A2 run for Storage-47024 reaches three predicted components including static, and the recovered moving components include revolute and prismatic hypotheses. The current single-frame evaluation domain is incomplete: it contains only two GT parts and cannot map the two predicted joints reliably. At a 2% bbox coverage threshold, covered-only Point IoU is 0.601 and ARI is 0.337, but the kinematic evaluator reports zero matched joints because of this GT-domain mismatch.

This result is promising reproduction evidence but remains provisional. It must be reevaluated on a multi-frame-union GT domain before being quoted as the exact-paper kinematic result.

## 14. Optimization-path development diagnostics

These experiments predate the feedforward main model and are useful for explaining design choices.

### 14.1 Quality weighting

On refrigerator038 and refrigerator045:

| Variant | Axis mean | Pivot mean | Stability | GT coverage | Runtime |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.71 deg | 0.0302 m | 0.514 | 0.688 | 175 s |
| Quality-weighted part poses | **5.37 deg** | **0.0274 m** | 0.517 | 0.688 | 207 s |
| Quality-weighted replay | 5.71 deg | 0.0302 m | 0.514 | 0.688 | 204 s |
| Quality-weighted affinity | 4.02 deg | 0.0310 m | **0.522** | 0.636 | 234 s |

Part-pose weighting is the safest improvement. Affinity weighting can reduce axis error but changes graph connectivity and loses coverage. Combining part-pose and affinity weighting becomes unstable in this two-object pilot, with axis error around 24.3 degrees and pivot error around 0.251 m.

### 14.2 EM-lite and stability-aware routing

The routing experiments establish that unmatched clusters cannot be blindly merged into the static parent. Some unmatched clusters contain useful moving-part support, and parent merging can reduce fake-joint count while worsening pivot accuracy. A leave-one-spatial-bin-out stability diagnostic improved route agreement with GT-aware diagnostic selection from 3/6 to 4/6 on refrigerator038/045.

The conclusion is methodological rather than a final benchmark number: local split proposals require multi-term selection using motion, rigid consistency, coverage, replay, and stability. Neither purity nor replay error alone is sufficient.

These results motivated moving from hand-designed clustering/refinement toward the learned slot feedforward path.

## 15. Dynamics-identification extension

The repository also includes downstream physical parameter identification after kinematic recovery. This is a separate experiment from the articulation reconstruction benchmark.

The current microwave forced-response validation uses a distinct torque-pulse recording, known generalized-force replay, contact-disabled rollout, and zero-gravity hinge debugging. Reported trajectory-level performance is:

- joint-position RMSE approximately 0.0089 rad;
- joint-velocity RMSE approximately 0.033 rad/s;
- identified damping/friction matches simulator GT;
- effective joint-inertia error approximately 5.2%.

Earlier free-decay experiments exposed a mass/damping identifiability problem: passive trajectories mainly constrain damping-to-inertia ratios. The forced-response episode resolves this by injecting a known excitation. This should appear as a downstream extension or supplementary experiment, not in the main segmentation/kinematics table.

## 16. Failure inventory and unsupported comparisons

The external suite currently contains several method-specific limitations:

- DTA fails on some high-part-count objects during adapter/common evaluation and uses GT part count.
- VideoArtGS succeeds on 14/20 and assumes the joint inventory/topology.
- GaussianArt's released outputs contain oracle-conditioned kinematic aggregates but no mappable predicted semantic labels or part meshes for common segmentation metrics.
- PARIS is restricted to two-part objects and currently succeeds on 4/5 applicable objects.
- Ditto cannot currently be evaluated because official checkpoint links are unavailable.
- AiM aligned metrics exclude incomplete GT domains, reducing metric N below success N.
- ReArt native-format inputs preserve the official 4-frame structure, but the aligned PartNet objects are a cross-dataset generalization test rather than the anonymous official Sapiens test sequences.

Unsupported metrics must remain blank. They should never be imputed from GT labels, another method's output, or native paper numbers.

## 17. Recommended paper tables and figures

### Main paper

**Table 1: Ours held-out main and controlled ablations**

- five configurations from the 116-object common test;
- Point IoU, ARI, RI, edge recall, type accuracy, axis mean/median, axis-line, and Joint@20.

**Table 2: Complexity scaling**

- Ours Main over 2, 3--4, and >=5 GT parts;
- segmentation, exact K, edge/type, penalized axis, and Joint@20.

**Table 3: External common segmentation**

- aligned 20-object subset;
- requested N, success N, metric N, IoU, ARI, RI;
- protocol and oracle columns shown explicitly.

**Figure 1: Per-object performance sorted by complexity/error**

- use the retained per-object CSV;
- order by GT part count and then error;
- show segmentation and kinematic error separately rather than combining incompatible scales.

**Figure 2: Qualitative comparison**

- simple, medium, and complex objects;
- GT mesh/parts, Ours, and available external predictions;
- include predicted axes only when the method supports them.

### Supplementary material

- category-level table;
- neural vs analytic oracle decomposition;
- SO(3) augmentation audit;
- temporal-resolution stress test;
- runtime with scope labels;
- external method failure table;
- AiM protocol/reproduction study;
- optimization-path quality weighting and EM-lite diagnostics;
- dynamics-identification extension.

## 18. Raw result index

Authoritative result files:

```text
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/summary.md
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/ours_five_config_test.csv
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/ours_five_config_per_object.csv
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/ours_five_config_per_joint.csv
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/ours_five_config_by_category.csv
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/ours_five_config_by_complexity.csv
outputs/partnet_core_v1_training/slot_benchmark_three_methods_test/benchmark_summary.csv
outputs/partnet_core_v1_training/ours_controlled_ablation_v1/hybrid/test/analytic_axis_oracle_report.json
outputs/external_baseline_suite_v1/summary.md
outputs/external_baseline_suite_v1/segmentation_summary.csv
outputs/external_baseline_suite_v1/complexity_summary.csv
outputs/external_baseline_suite_v1/failures.json
outputs/external_baseline_suite_v1/runtime_comparison_v2/all_methods/summary.md
outputs/benchmarks/temporal_profile_comparison/summary.json
outputs/partnet_core_v1_training/so3_difficulty_10ep/ablation_summary.json
outputs/aim_baseline_extended_v1/summary.md
outputs/external_baselines_v1/aim_protocol_audit_v1/summary.md
outputs/external_baselines_v1/aim_protocol_audit_v1/storage47648_ransac_sensitivity/summary/summary.md
outputs/external_baselines_v1/aim_simple_object_observation_sanity_v1/summary.md
outputs/aim_two_part_front_oscillate_v1/evaluation/paper_metrics/summary.md
```

All per-object traces are retained for the controlled Ours comparison and the external baseline suite. This supports difficulty-gradient plots, category analysis, and failure-case selection without reconstructing values from aggregate tables.

## 19. Concise experimental story

The current evidence supports the following paper narrative:

1. A feedforward slot model operating on object-mask-derived 3D trajectories can recover articulated parts without part masks at inference time.
2. On a common GT reference-point domain, CoTracker gives the highest overlap (0.678 IoU), while Hybrid gives the highest exact part-count rate (0.681) and is selected as Ours Main by validation loss.
3. A neural relation head gives the strongest end-to-end kinematic recovery, while analytic relative-motion fitting provides accurate revolute geometry and exposes observability failures.
4. Performance remains strong on two-part and 3--4-part objects but declines substantially for >=5-part structures, identifying complex motion decomposition as the principal unresolved challenge.
5. On an object-aligned external subset, Ours has the strongest common segmentation metrics even against baselines with known-part-count or stronger kinematic oracles.
6. AiM and ReArt results should be interpreted as protocol/cross-dataset generalization. Detailed AiM audits show that camera coverage alone does not explain failure; upstream deformation quality, small-component thresholds, and motion-component merging all contribute.
7. The results do not justify claiming that all external methods were reproduced under their original benchmark distributions. They do justify a controlled comparison on a broader common PartNet domain, provided protocol and oracle differences remain explicit.

## 20. Real-scene diagnostic: PhysTwin scene27

The first scene27 microwave run exposed two independent failure sources.

First, the PhysTwin calibration files store OpenCV camera-to-world poses with camera `+Y` pointing down, while the RGB-D back-projection path expects camera `+Y` pointing up. The episode adapter now applies the corresponding camera-frame basis conversion and records both the source and target conventions in `episode.json`. Direct comparison against the provided scene point clouds confirms the corrected projection to numerical precision; the previous convention produced 4--20 cm errors.

Second, some lifted tracks still contain depth-surface jumps. With the current 5 cm timestep mask, all retained adjacent steps are below 5 cm, but many 2--5 cm discontinuities remain. Tightening the threshold to 2 cm changes the inferred axis without materially reducing joint replay error. The current mask also removes only the timestep receiving a large jump; it does not reject a track segment that jumps onto an incorrect surface and then remains there. A future filter should mask both sides of a discontinuity and detect persistent depth-surface changes rather than relying only on a single-step threshold.

The corrected scene27 result also shows estimator sensitivity:

- geometric relative-SE(3) fitting and the neural relation head predict similar revolute directions (8.24 degrees apart) and axis lines (2.7 cm apart);
- the robust analytic model has lower internal replay residual but differs by about 47--54 degrees from those two estimates;
- therefore low replay residual alone is not sufficient to select a real-scene axis when tracks contain correlated depth errors and incomplete spatial support.

The next real-scene tests should determine each scene's object mask from its metadata and RGB views, run the same camera-convention validation, and report track discontinuity statistics before comparing joint estimators.
