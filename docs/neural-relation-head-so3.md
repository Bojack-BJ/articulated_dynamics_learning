# Neural Relation Head SO(3) Pilot

This pilot fixes coordinate-frame leakage in joint-axis prediction without
changing the paper checkpoint or the default `direct` head.

## Architectures

`direct` preserves the existing unconstrained MLP/GRU head. It remains the
no-augmentation baseline and the Haar-collapse reproduction.

`equivariant_proposal` builds no-GT candidates from the complete lifted
trajectory: full-path translation PCA, weighted per-frame Kabsch transforms,
relative-SE(3) rotation logs, and a least-squares hinge line. The network only
predicts scalar candidate weights, edge logits, and type logits. Unoriented
axes are aggregated through the principal eigenvector of
`sum_k w_k a_k a_k^T`; no valid candidate produces an explicit
low-observability result rather than a fallback world axis.

`vector_neuron` uses the same equivariant geometric basis and invariant scalar
gates. Because PCA and rotation-log axes are unoriented, the gated axis uses a
second-moment eigenspace rather than a sign-sensitive vector sum. Its pivot is
a pair center plus an oriented equivariant center-delta offset projected
perpendicular to the predicted axis. Opaque
tracker/slot features condition scalar weights only. Neither equivariant head
contains a biased `Linear(..., 3)` axis/pivot output.

## Paired training

The original and a shared-Haar-rotated sample are forwarded as a pair. All
explicit 3D geometry, GT axes, pivots, and replay points rotate together.
Losses cover undirected axis equivariance, revolute line equivariance, edge
probability consistency, and joint-type probability consistency. The slot
model is frozen in Stage 1. Stage 2 updates only slot queries and decoder at
`0.1 x` relation learning rate.

## Pilot

Prepare the deterministic `24 / 8 / 12` object split without training:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_neural_head_so3_pilot.py \
  <learning_manifest.tsv> <cotracker_slot_checkpoint.pt> \
  --phase prepare
```

Run four variants, three seeds, and paired audits:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_neural_head_so3_pilot.py \
  <learning_manifest.tsv> <cotracker_slot_checkpoint.pt> \
  --phase stage1 --devices cuda:0,cuda:1,cuda:2 --max-parallel 3
```

Aggregate metrics and apply Stage-1 gates:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_neural_head_so3_pilot.py \
  <learning_manifest.tsv> <cotracker_slot_checkpoint.pt> --phase summarize
```

Stage 2 refuses to launch unless a new architecture passes the axis, type,
edge, graph-legality, candidate-coverage, and paired-equivariance gates:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_neural_head_so3_pilot.py \
  <learning_manifest.tsv> <cotracker_slot_checkpoint.pt> --phase stage2
```

Artifacts are isolated under `outputs/neural_head_so3_pilot_v1/`. Existing
paper results and default checkpoints are not overwritten. Scene 31 remains a
slot-count/segmentation failure and is not claimed as an axis-head fix.
