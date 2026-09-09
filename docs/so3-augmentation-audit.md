# SO(3) Augmentation and Equivariance Audit

The Relation Head now uses Haar-uniform unit-quaternion rotations by default.
The legacy independent-Euler sampler remains available only as an explicit
ablation:

```bash
--rotation-augmentation --rotation-augmentation-mode euler
```

Use `--rotation-augmentation-mode uniform_quaternion` for new training runs.
The augmentation probability is unchanged; set it to `1.0` when testing the
effect of removing all canonical-orientation examples.

An optional undirected consistency objective is available and remains disabled
by default:

```bash
--rotation-augmentation \
--rotation-augmentation-probability 1.0 \
--axis-equivariance-loss-weight 0.1
```

It compares the original and rotated predictions with
`1 - abs(dot(Q * axis_original, axis_rotated))`.

## Audit

```bash
PYTHONPATH=src python scripts/audit_relation_so3_equivariance.py \
  path/to/learning_manifest.tsv \
  path/to/motion_part_slots.pt \
  path/to/slot_relation_head.pt \
  --catalog path/to/catalog.json \
  --output-dir outputs/so3_audit \
  --device cuda
```

The audit performs:

- a 100,000-sample Haar-uniform distribution check;
- raw/augmented GT and prediction canonical-axis histograms;
- category, joint-type, and motion-bin breakdowns;
- model-level undirected axis equivariance evaluation;
- geometry-field registry validation;
- spherical direction plots and per-joint CSV export.

## Coordinate-Frame Interpretation

The current augmentation is a post-lifting change of 3D coordinate basis:

```text
geometry' = Q geometry
axis' = Q axis
pivot' = Q pivot
```

RGB and CoTracker image descriptors remain unchanged. This is internally
consistent when those descriptors are treated as frame-invariant appearance
or correspondence features. It is not equivalent to physically rotating the
object relative to the camera, which would require rerendering/retracking or
recomputing visual features after changing the object-camera transform.

With the default `relation_geometry` scope, the Slot encoder still consumes
the original explicit 3D geometry. The optional
`slot_and_relation_geometry` scope rotates the explicit slot geometry before
recomputing slot tokens, while preserving the opaque CoTracker descriptor.
The descriptor is not declared SO(3)-equivariant.

Camera rays are not currently consumed by the Relation Head. Their frame is
recorded as unresolved so future use cannot silently introduce a mismatch.
Camera extrinsics are used to lift RGB-D into world-space tracks and are not
inputs after lifting; they should therefore not be transformed for the current
post-lifting augmentation. If camera rays/extrinsics become model inputs, their
transformation rule must be made explicit.

## Rotation-Difficulty Diagnostic

A 10-epoch cross-track Transformer diagnostic uses identical data, seeds, and
decoder fine-tuning while varying only augmentation difficulty:

| Augmentation | Test edge F1 | Type accuracy | Axis mean | Axis median | Axis P90 | >80 deg |
|---|---:|---:|---:|---:|---:|---:|
| None | 0.859 | 0.920 | 24.75 | 3.58 | 88.08 | 47 |
| Yaw `[-180, 180]` | 0.856 | 0.913 | 46.57 | 46.34 | 85.28 | 40 |
| Random-axis up to 15 deg | 0.869 | 0.905 | 23.28 | 3.05 | 86.77 | 49 |
| Random-axis up to 30 deg | 0.869 | 0.909 | 26.52 | 7.74 | 86.86 | 45 |
| Haar SO(3) | 0.850 | 0.902 | 58.80 | 64.83 | 68.34 | 0 |

Axis statistics in this table use the GT-pair/type-correct oracle view so all
neural checkpoints are compared on the same 275 joints. The low medians but
very high P90 values without broad rotations expose a canonical-orientation
shortcut. Haar augmentation removes those isolated greater-than-80-degree
failures by collapsing toward broadly inaccurate 60--70-degree predictions;
it does not solve equivariant axis estimation.

On the same tracks, the explicit analytic relative-SE(3) estimator obtains a
5.14-degree median. This isolates the main failure to the learned axis
representation/regression rather than proving that the lifted tracks lack
joint-axis information.
