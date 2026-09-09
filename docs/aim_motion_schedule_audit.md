# AiM Motion-Schedule Audit

The released AiM repository exposes the dataset loader, optimization schedule, and
segmentation implementation, but it does not include object-specific PartNet motion
scripts for Storage-47024, Fridge-11304, Storage-47648, or Table-31249. Exact official
joint ordering and static intervals therefore cannot be reconstructed without guessing.

| Item | AiM official / inferred | Current paper-like recording | Match? |
|---|---|---|---|
| Motion image count | Paper: 200 for two/three-part objects and 500 for complex objects | 200 or 500 by object | Yes |
| Start-state image count | Paper appendix: 100 multi-view start-state images | 24-view orbit | No |
| Frame rate | The paper specifies image counts, not a physical FPS in the cited dataset description | 15 FPS | Project choice, not paper-aligned |
| Joint range | Paper reports large object-specific ranges | Full available MJCF limit | Approximate |
| Joint ordering | Not available in released code/assets | Deterministic staggered MJCF order | Unknown |
| Static intervals | Not available | Per-joint inactive portions of staggered slots | Unknown |
| Simultaneous motion | Not recoverable from released scripts | Primarily sequential, with controller settling | Unknown |
| Camera path | Monocular moving interaction camera | One smooth orbit with elevation perturbation | Compatible, not exact |
| Start scan semantics | Fixed start state, multiple cameras | Fixed start state, 24-view orbit | Semantic match only |
| End scan | Separate end dataset; all cameras are assigned time 1.0, implying one fixed end state | Current `aim_style` uses the final 12 interaction frames | No |
| Static iterations | Official default 20,000 | Existing sanity run 5,000 | No |
| Motion iterations | Official default 30,000 | Existing sanity run 8,000 | No |

The controlled `aim_style_fixed_end` variant records an additional 12-view orbit after
the interaction while holding every articulated joint at the final state. It does not
replace or modify the existing `aim_style` protocol.

The current configuration is therefore **paper-like rather than a strict
reproduction**. It matches the start/motion/end semantics and the 200/500 interaction
image counts, but not the 100-image start scan or the official optimization budget. The
exporter also uses an evaluation split: for the 500-image Storage-47648 interaction,
437 cameras are used for optimization and the remaining cameras are held out, while all
500 remain available for evaluation.

## Official 3DGS Render Audit

The learned representation can be checked before sequential RANSAC by rerendering every
interaction camera with AiM's official `render_mix` path:

```bash
python scripts/render_aim_3dgs_reconstruction.py \
  /path/to/AiM \
  /path/to/aim/run \
  --source-path /path/to/aim/dataset \
  --output aim_3dgs_reconstruction.mp4 \
  --include-test-cameras
```

For the completed Storage-47648 checkpoint at iteration 13,000 (5,000 static plus
8,000 motion iterations), the 500-frame render has:

| Metric | Value |
|---|---:|
| Full-frame PSNR mean | 28.83 dB |
| Object-mask PSNR mean / median | 20.31 / 20.25 dB |
| Object-mask MAE mean | 0.0953 |
| Mean object pixel ratio | 9.46% |

The full-frame metric is inflated by the black background. More importantly, the error
is strongly time-dependent: frame 0 has object PSNR 10.23 dB and object MAE 0.231,
whereas frame 400 reaches 29.42 dB and 0.0157. Early renders contain visible Gaussian
trails and displaced geometry, while late renders are substantially cleaner.

This localizes part of the failure upstream of sequential RANSAC: the dynamic Gaussian
representation does not fit all interaction times uniformly. It does not prove that
photometric error is the only cause of under-segmentation. Even a visually accurate
render can be produced by non-rigid per-Gaussian trajectories that do not admit a
single Kabsch transform per physical part. Render quality and trajectory rigidity must
therefore be audited separately.

## Storage-47648 Effective-Motion Audit

The object-specific audit is generated with:

```bash
PYTHONPATH=src python scripts/audit_aim_storage_effective_motion.py \
  path/to/partnet_47648/episode.json \
  path/to/partnet_47648/aim_run \
  path/to/partnet_47648/aim.json \
  --output-dir outputs/aim_protocol_audit_storage47648_v1
```

The implemented schedule contains six non-overlapping motion intervals. Four revolute
joints sweep about 120 degrees and two prismatic joints move about 0.16 m. The camera
follows an independent continuous orbit. This is favorable for motion separability, but
it is not claimed as an exact reproduction because the released AiM assets do not
contain the object-specific official temporal schedule.

The effective-motion audit shows that the failure is not explained only by weak visible
motion:

| GT part | Visible frames | Max 2D centroid motion | Max 3D centroid motion | Share of initial dynamic Gaussians | Below global 10% |
|---|---:|---:|---:|---:|---|
| 2 | 100.0% | 45.2 px | 0.977 m | 36.8% | No |
| 3 | 59.2% | 155.1 px | 0.497 m | 21.7% | No |
| 4 | 55.2% | 97.9 px | 0.466 m | 18.6% | No |
| 5 | 46.4% | 40.1 px | 0.492 m | 11.8% | No |
| 6 | 94.2% | 79.6 px | 0.329 m | 5.8% | Yes |
| 7 | 42.8% | 62.7 px | 0.136 m | 4.4% | Yes |
| 8 | 55.8% | 99.5 px | 0.186 m | 0.8% | Yes |

The Gaussian shares above use nearest-GT-surface assignment for support-size
diagnostics. Only 6.4% of dynamic Gaussians fall within the strict common evaluator's
2%-bbox correspondence radius, so nearest-assignment distances must be inspected before
using these assignments as semantic ground truth.

The official sequential-RANSAC threshold is 10% of the complete initial dynamic set.
Parts 6, 7, and 8 therefore cannot independently pass that threshold under the
diagnostic assignment. This agrees with the logged behavior: after the first accepted
component, later raw proposals contain enough candidates, but connected-component
filtering leaves only about 2.2k-4.0k points against a roughly 5.4k minimum. The result
supports a small-part/model-selection bottleneck in addition to any optimization
quality limitation.

The strict 20k/30k runs fail near motion iteration 9k with a CUDA invalid-configuration
error and save no intermediate motion checkpoints. Consequently, the requested 2k-to-9k
dynamic-ratio convergence trend cannot be reconstructed from current artifacts. A future
rerun must save deformation and motion point-cloud checkpoints explicitly before this
question can be answered.

## Storage-47648 Camera-Trajectory Ablation

A controlled camera-timing ablation compares the existing phase-modulated orbit with a
`front_loaded_orbit`. Both recordings use the same model, 500 frames at 15 Hz, identical
joint trajectories, the same elevation variation, and one nominal camera orbit. The
front-loaded trajectory completes approximately 180 degrees by the end of articulated
motion and completes the remaining hemisphere afterward using smoothstep easing.

Both predictions are evaluated against the same control frame-499 GT reference:

| Variant | Moving components | Predicted/GT parts | Dynamic Gaussians | Point IoU | ARI | RI | Coverage at 2% bbox |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current orbit | 3 | 4/7 | 55,326 | 0.223 | 0.157 | 0.613 | 0.635 |
| Front-loaded orbit | 1 | 2/7 | 36,843 | 0.110 | -0.008 | 0.481 | 0.486 |

The front-loaded trajectory does not recover the small parts. Nearest-GT-surface support
for parts 6/7/8 decreases from approximately 7.2%/5.4%/0.7% of initial dynamic
Gaussians to 3.2%/2.9%/0.1%. This controlled result does not support the hypothesis that
insufficient front-hemisphere coverage during articulation is the primary cause of the
under-segmentation. Concentrating the interaction views on the front hemisphere reduces
multi-view dynamic reconstruction coverage and makes the decomposition worse.

This conclusion is specific to the tested trajectory. The control preserves the
existing sinusoidally phase-modulated implementation, whose realized sweep is about 342
degrees rather than exactly 360 degrees. The front-loaded trajectory realizes about 360
degrees. No AiM optimization, segmentation, connected-component, or RANSAC parameter was
changed.

## Storage-47648 Sequential-RANSAC Sensitivity

The sensitivity audit freezes the completed current-orbit and front-loaded-v3 Gaussian
trajectories and reruns only sequential RANSAC. It scans global minimum-inlier ratios of
10%, 7.5%, 5%, and 2.5%, with largest-connected-component filtering enabled and disabled.
These runs are diagnostic only; the official 10% baseline output remains unchanged.

Lowering the threshold materially increases the number of pre-merge proposals. The
current orbit changes from one proposal at 10% to four at 2.5%; front-loaded-v3 changes
from one to two proposals with connected-component filtering, or three without it.
Largest-connected-component filtering is therefore a secondary bottleneck: disabling
it retains more Gaussians, but changes component count only for front-loaded-v3 at 2.5%.

The official merge/model-selection stage also collapses proposals. For the current
orbit, two proposals at 5-7.5% become one and four proposals at 2.5% become three. For
front-loaded-v3, two proposals at 5% become one and three proposals at 2.5% become two.
Even the most permissive setting remains far below seven GT parts.

Simulation-only GT-oracle rigid fits further show that thresholding is not the only
failure. Several GT parts have noisy within-part Gaussian trajectories or weak residual
separation from another part's rigid model. Components recovered at permissive
thresholds also remain semantically mixed. The diagnostic therefore supports a combined
failure: the global 10% threshold suppresses small-part proposals, but upstream Gaussian
motion ambiguity and final merging prevent threshold relaxation from recovering a clean
seven-part decomposition.

Detailed outputs are stored under:

```text
outputs/external_baselines_v1/aim_protocol_audit_v1/
  storage47648_ransac_sensitivity/
```
# 100-View Factorial Reproduction Audit

The clean reproduction audit uses the same 100-view fixed start scan for all
optimization conditions. Part masks are exported only as diagnostic sidecars;
they are never passed to AiM reconstruction, deformation training, or
segmentation.

| Object | Static / motion iterations | Static PSNR | Predicted motion components | Result |
|---|---:|---:|---:|---|
| Storage-47024 | 5k / 8k | 22.91 dB | 3 | Success |
| Storage-47024 | 20k / 8k | 23.59 dB | 2 | Success |
| Storage-47648 | 5k / 8k | 27.88 dB | 2 | Success |
| Storage-47648 | 20k / 8k | 28.51 dB | 1 | Success |
| Storage-47648 | 20k / 30k | 28.53 dB at the end of the static stage | N/A | Official CUDA extension fails near motion iteration 9k |

Increasing static optimization from 5k to 20k improves canonical reconstruction
PSNR by only about 0.6--0.7 dB. It does not improve motion decomposition:
Storage-47648 becomes more under-segmented, and all accepted motion hypotheses
in the completed runs are reported as `theta=0` prismatic models.

## Moving-Part Reconstruction

The following PSNR is evaluated only on GT articulated-part pixels. This is a
simulation-only diagnostic and is not used by AiM.

| Object | Static / motion iterations | Moving-part mean PSNR | Temporal quartile moving-part PSNR |
|---|---:|---:|---|
| Storage-47024 | 5k / 8k | 19.14 dB | 10.25 / 10.26 / 29.25 / 25.75 dB |
| Storage-47024 | 20k / 8k | 18.62 dB | 10.65 / 10.05 / 27.77 / 25.01 dB |
| Storage-47648 | 5k / 8k | 16.76 dB | 6.68 / 5.84 / 23.51 / 31.11 dB |
| Storage-47648 | 20k / 8k | 16.33 dB | 6.65 / 5.67 / 21.78 / 31.31 dB |

The first half of the interaction remains severely underfit even after the
100-view start scan and 20k static optimization. More static optimization does
not improve moving-part PSNR, so the primary remaining reproduction gap is the
dynamic deformation representation rather than the canonical start model.

The strict 20k/30k run is also blocked by a reproducible implementation failure.
With `CUDA_LAUNCH_BLOCKING=1`, Storage-47648 still exits with
`invalid configuration argument` immediately after the periodic reassignment
around motion iteration 9000. This is not treated as a successful 30k result and
the official algorithm is not patched to hide the failure.
