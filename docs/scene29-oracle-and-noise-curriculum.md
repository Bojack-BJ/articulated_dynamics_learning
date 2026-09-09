# Scene29 oracle diagnostics and corruption curriculum

## Diagnostic result

The scene contains two annotated prismatic drawers, but only one is sufficiently
excited. With GT part tracks, the moving drawer travels 12.7 cm and full-path PCA
recovers its direction with 6.1 degrees axis error. The other drawer travels only
4.2 mm and must be treated as low-observability rather than a normal axis target.

For the moving `slot7 -> slot1` relation:

| Trajectory assignment | Revolute probability | Prismatic probability | Prediction |
|---|---:|---:|---|
| Predicted slots | 0.821 | 0.176 | revolute |
| GT three-part trajectory aggregation | 0.514 | 0.483 | revolute |
| Base + inactive drawer versus moving drawer | 0.710 | 0.287 | revolute |

The GT-assignment experiment overrides only trajectory aggregation and retains the
predicted slot latents. It is therefore a trajectory-oracle diagnostic, not a full
oracle-slot result. The probability shift demonstrates substantial assignment
contamination. Failure to cross the decision boundary demonstrates an additional
slot-latent/type-prior or relation-geometry domain gap.

## Actual module order

The deployed pipeline is not a strict `slot -> type -> axis geometry` cascade.
It is:

1. RGB-D and CoTracker produce 3D trajectories and opaque appearance features.
2. The slot model jointly produces slot latents, existence, and soft track assignments.
3. The relation model consumes slot latents and assignment-weighted trajectory tokens.
4. Edge, type, axis, confidence, observability, and pivot are parallel outputs from a
   shared pair representation.

Consequently, type and axis must see compatible corruptions. Training them on separate,
independently corrupted representations would introduce another train-test mismatch.

## Corruption layers

### Sensor and reconstruction corruption

Apply before slot inference: depth quantization, range-dependent depth noise, missing
depth, edge flying pixels, small intrinsics/extrinsics perturbations, and view dropout.
This layer trains the slot model against realistic 3D track and fusion errors.

### Tracker corruption

Apply to trajectories: contiguous occlusion, delayed onset, early termination, drift,
isolated depth spikes, ID switches, and recovery as a new segment. Corruption parameters
must be fitted to measured CoTracker statistics from the real scenes rather than chosen
only for visual plausibility.

### Slot-interface corruption

Apply between slot and relation models: soft assignment leakage, split one rigid part
across two slots, merge base and child tracks, omit a low-support part, and add a false
small slot. Preserve the clean GT relation target and pass both corrupted assignments
and correspondingly corrupted/recomputed slot latents to the relation model.

### Relation observability corruption

Use the same corrupted pair input for edge, type, and geometry outputs. Mask direct axis
and pivot losses when the input excitation is below a fixed GT/input-derived threshold;
supervise low-observability confidence and abstention instead. Add clean/corrupted paired
consistency only when the corruption preserves the underlying relation.

## Recommended staged experiment

1. Measure real noise distributions: gap length, track lifetime, step-size tail, depth
   invalid rate, assignment leakage, split/merge rate, and per-view support.
2. Train the slot model with sensor/tracker corruption and retain clean/noisy paired
   assignment consistency.
3. Freeze the slot model and cache clean, naturally predicted, and synthetically
   corrupted slot interfaces. Train the relation head on a balanced mixture.
4. Fine-tune slot decoder and relation head jointly at low slot learning rate using mild
   corruption; do not unfreeze CoTracker in this stage.
5. Evaluate clean PartNet, fixed Haar rotation audit, synthetic corruption severity
   curves, and all seven real scenes. Report predicted-slot and trajectory-oracle results
   separately.

The first pilot should target drawer and door samples and use three seeds. Promotion
requires no meaningful clean-set regression, improved corrupted-set type/axis metrics,
and correction of scene29 without using GT assignments at inference.
