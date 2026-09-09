# PartNet-Mobility Recording and Split Strategy

## Dataset source

Use the full SAPIEN PartNet-Mobility v0 archive as the primary asset source.
GAPartNet is a 1,045-object annotated subset of the 2,347 SAPIEN objects. The
current RGB-D motion pipeline requires only visual meshes, the URDF link tree,
joint type/axis/limits, and simulator link IDs. All are available in the
original PartNet-Mobility assets.

Keep GAPartNet annotations as optional simulation diagnostics:

- `semantics_gapartnet.txt`: normalized GAPart link semantics;
- `link_annotation_gapartnet.json`: GAPart category and link OBB;
- `mobility_v2.json`: convenient JSON kinematic hierarchy;
- remeshed geometry and precomputed point samples.

Do not consume these enhanced fields during inference. Link masks, kinematic
GT, and object bounds can be generated from the original URDF and meshes.

## Dataset splits

Generate deterministic category- and articulation-stratified splits:

```bash
python3 scripts/build_partnet_mobility_splits.py \
  /root/Users/lixiaotong/datasets/PartNet-Mobility/partnet-mobility-v0.zip \
  --output-dir outputs/partnet_mobility_splits/core_v1 \
  --preset core
```

The default seed is `20260720`, with 70/15/15 train/validation/test fractions.
Stratification uses both object category and articulation profile:

- `single_revolute`
- `single_prismatic`
- `multi_revolute`
- `multi_prismatic`
- `mixed`

Core v1 contains 940 objects from 14 categories, split into 673 train, 133
validation, and 134 test objects. Object IDs are globally unique across splits.

Prepare every split asset and convert its URDF to a recording MJCF. Bad source
assets are logged and skipped so a single missing mesh cannot abort the batch:

```bash
PYTHONPATH=src python3 scripts/prepare_partnet_profile_samples.py \
  /path/to/partnet-mobility-v0.zip \
  --split-catalog outputs/partnet_mobility_splits/core_v1/catalog.tsv \
  --output-dir outputs/partnet_core_v1 \
  --continue-on-error
```

Relation-head experiments use two phases: train the relation head with a frozen
slot backbone, then continue from that checkpoint while fine-tuning only slot
queries and the Transformer decoder at `0.1x` learning rate. CoTracker-only,
TAPIP-only, and hybrid runs can use separate GPUs:

```bash
PYTHONPATH=src python3 scripts/run_three_relation_training.py \
  configs/relation_training_three_methods_example.tsv \
  --output-root outputs/relation_partnet_3way \
  --frozen-epochs 50 \
  --finetune-epochs 25 \
  --resume
```

Each phase reports epoch time and peak CUDA memory. Keep both checkpoints:
decoder fine-tuning is selected only when validation metrics improve, because
it can improve one tracker while degrading another.
Because GAPartNet IDs are a strict subset of SAPIEN IDs, GAP assets must inherit
the split assigned to the same SAPIEN object ID.

## Episode suite

Each object should have separate episode purposes rather than one universal
motion:

1. `geometry`: static multiview coverage.
2. `isolated_joint`: move one joint while all others remain fixed.
3. `staggered_multi_joint`: excite multiple joints with distinct timing.
4. `dynamic_excitation`: velocity and pulse variation for system ID.
5. `staged_visibility`: expose initially occluded nested parts before seeding.

## Multi-joint identifiability

Synchronous equal-speed motion is not a valid default. It makes distinct links
motion-equivalent and can collapse them into one inferred rigid part. Assign
each movable joint a deterministic but different excitation signature:

- different onset time;
- different movement duration;
- different peak velocity;
- alternating direction where limits permit;
- stationary gaps between major transitions;
- at least one isolated interval per joint.

For a four-joint object, an example schedule is:

| Joint | Onset | Duration | Peak speed | Direction |
|---|---:|---:|---:|---|
| 0 | 0.5 s | 1.5 s | 0.45 normalized/s | positive |
| 1 | 2.5 s | 1.0 s | 0.70 normalized/s | positive |
| 2 | 4.0 s | 1.8 s | 0.35 normalized/s | negative |
| 3 | 6.2 s | 1.2 s | 0.60 normalized/s | positive |

An additional partially overlapping episode can test robustness, but it must
not replace isolated/staggered excitation used to establish identifiability.

## Dispatch policy

Recording configuration should be selected from both category and URDF joint
profile:

- category controls camera distance, look-at, field of view, and likely visible
  faces;
- profile controls episode length, excitation schedule, reference frames, and
  whether staged visibility is required;
- object bounding box controls camera scale automatically;
- joint limits control safe position/velocity amplitudes.

Start with single-DOF assets for pipeline regression. Add multi-DOF assets only
after staggered scheduling and multi-reference track seeding are enabled.
