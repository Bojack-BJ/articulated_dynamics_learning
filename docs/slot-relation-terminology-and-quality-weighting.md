# Slot / Relation Terminology and Quality Weighting

## Minimal method taxonomy

The pipeline has two learned modules, not four independent methods:

1. **Slot model**: assigns tracks to parts and decides which slots exist.
   - **Base slot** is the original frozen CoTracker-only slot checkpoint.
   - **Joint-tuned slot** is that checkpoint after its slot queries/decoder were fine-tuned together with the relation objective.
   - **Slot regression** means the tuned model changes part count or track assignments for the worse.
2. **Relation head**: consumes the selected parent/child slot trajectories and predicts graph edges, joint type, and axis.
   - **Old head** is the unconstrained coordinate-regression baseline.
   - **VN head** is the SO(3)-equivariant axis/type head.
   - **Pivot-B** is not another segmentation or type model. It is the revolute pivot representation used with VN: analytic hinge point plus a learned equivariant planar residual.
   - **Relation-head regression** means type or axis becomes worse while holding slot assignments fixed.

Quality weighting is an input aggregation option for the VN relation head, not a new head. It consumes existing per-timestep and per-track quality scores.

## Quality weighting

For visible observation `(track i, timestep t)`, the relation head uses

```text
w_it = sqrt(track_quality_score_i) * timestep_quality_score_it
```

The weights are used consistently in temporal feature pooling, slot pooling, weighted Kabsch, translation PCA, rotation-log candidates, and observability. Legacy checkpoints keep binary visibility unless quality weighting is explicitly enabled or saved in the checkpoint.

## Current result

On the 121-object CoTracker-only full test, quality-trained VN + Pivot-B changes:

| Metric | Pivot-B | + quality-trained | Delta |
|---|---:|---:|---:|
| Edge F1 | 0.9544 | 0.9577 | +0.0034 |
| Type accuracy | 0.9055 | 0.9055 | 0 |
| Axis mean | 26.34 deg | 26.08 deg | -0.27 deg |
| Axis median | 12.39 deg | 11.62 deg | -0.77 deg |
| Axis P90 | 73.94 deg | 73.90 deg | -0.04 deg |
| Axis-line | 0.08331 | 0.08150 | -2.17% |

The improvement is real but small. Soft quality weighting helps median and axis-line behavior, but does not fix the long tail. The next robust change should fit continuous valid segments separately and combine them by consensus, with IRLS/Huber residual weighting and hard rejection of depth spikes or ID switches.
