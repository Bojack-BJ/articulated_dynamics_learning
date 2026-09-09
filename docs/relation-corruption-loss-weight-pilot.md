# Relation corruption and loss-weight pilot

Date: 2026-09-04

## Setup

- Warm start: `robust_segment_relation_10ep_seed_20260831`
- Training: 5 epochs, seed 20260831, learning rate 5e-5
- Existing augmentation: temporal occlusion
- Added relation-level corruption: segmentwise drift plus isolated trajectory spikes
- Default: axis / axis-line weights = 2 / 1
- Hard-loss: axis / axis-line weights = 4 / 2
- Type weight remains 2 because its training loss is already near saturation

## Results

| Split | Variant | Edge F1 | Type acc. | Axis mean | Axis median | Axis P90 | Axis-line |
|---|---|---:|---:|---:|---:|---:|---:|
| Full test (121 objects) | Robust baseline | 0.9577 | 0.9055 | 26.01 deg | 12.47 deg | 73.10 deg | 0.08097 |
| Full test (121 objects) | Corruption, 2/1 | 0.9594 | 0.9091 | 26.04 deg | 12.52 deg | 73.04 deg | 0.07977 |
| Full test (121 objects) | Corruption, 4/2 | 0.9594 | 0.9055 | 25.78 deg | 12.28 deg | 71.66 deg | 0.07915 |
| Aligned-20 | Robust baseline | 0.9032 | 0.8356 | 42.68 deg | 43.90 deg | 84.57 deg | 0.08822 |
| Aligned-20 | Corruption, 2/1 | 0.9091 | 0.8493 | 42.50 deg | 43.40 deg | 84.31 deg | 0.08626 |
| Aligned-20 | Corruption, 4/2 | 0.9091 | 0.8493 | 42.39 deg | 43.07 deg | 83.83 deg | 0.08626 |
| Full test (121 objects) | Excitation gate, 4/2 | 0.9594 | 0.9091 | 26.02 deg | 12.53 deg | 73.00 deg | 0.07990 |
| Full test (121 objects) | Gate + hard focal, 4/2 | 0.9594 | 0.9055 | 25.78 deg | 12.59 deg | 71.67 deg | 0.07926 |
| Aligned-20 | Excitation gate, 4/2 | 0.9091 | 0.8493 | 42.38 deg | 43.18 deg | 83.82 deg | 0.08650 |
| Aligned-20 | Gate + hard focal, 4/2 | 0.9091 | 0.8493 | 42.21 deg | 42.62 deg | 83.69 deg | 0.08647 |

## Decision

The corruption curriculum gives a small, consistent gain in edge/type robustness and
axis-line error, but does not materially fix the axis long tail. Doubling axis and
axis-line weights yields another small axis gain, especially full-test P90, while
slightly reducing full-test type accuracy relative to the 2/1 corruption run.

Keep `4/2` as the next candidate, but do not replace the main checkpoint yet. The
next experiment should weight difficult observable joints at the sample level and
separate low-observability abstention from supervised geometry learning. Global loss
scaling cannot make unobservable trajectories informative.

The follow-up confirms that excitation gating alone does not improve the long tail.
Adding capped hard-example focal weighting recovers roughly the same result as global
4/2 weighting, rather than exceeding it. This rejects the hypothesis that low-motion
joints are the primary source of harmful axis gradients in the current training set.
The remaining domain gap is more likely caused by missing joint corruption patterns:
spatially coherent depth failures, track identity switches, partial-part contamination,
and errors propagated through predicted slot assignments.

This pilot only corrupts relation-level trajectories. Sensor/depth corruption and
predicted-slot interface corruption require a separate joint-slot training pilot.
