# Main-paper axis comparison design

Axis prediction should remain in the main paper. The table uses two explicitly separated
blocks because not every released baseline exposes joints under the same observation and part
discovery protocol. Empty entries mean that the method does not expose a compatible quantity,
not that the experiment is pending.

## Table X: Articulation and axis estimation

### A. Object-aligned PartNet evaluation

| Method | Input | Parts | N | Edge F1 ↑ | Type Acc. ↑ | Axis Err. ↓ | Axis-Line ↓ | Joint@20 ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Track2Art, old head | continuous RGB-D tracks | predicted | common intersection | TBD | TBD | TBD | TBD | TBD |
| Track2Art, VN | continuous RGB-D tracks | predicted | common intersection | TBD | TBD | TBD | TBD | TBD |
| Track2Art, VN + Pivot-B | continuous RGB-D tracks | predicted | common intersection | TBD | TBD | TBD | TBD | TBD |
| VideoArtGS | monocular interaction | predicted | 14 successful | -- | 1.000 | 11.02 | 0.123 | -- |
| GaussianArt | two-state multi-view | oracle | 20 | -- | -- | 47.11 | 0.163 | -- |
| PARIS | two-state multi-view RGB | predicted | 4/5, two-part | -- | -- | 44.49 | 0.395 | -- |

The three Track2Art rows must be regenerated on one identical object/joint intersection. The
currently available VN full-test averages use 121 objects and 275 joints and therefore should not
be copied into this block until the common-intersection export is complete.

### B. Native-protocol diagnostic results

| Method | Dataset/protocol | N | Type Acc. ↑ | Axis Err. ↓ | Qualification |
|---|---|---:|---:|---:|---|
| ReArt | official Sapiens subset | native subset | 0.216 | 16.46 | converted relative-SE(3) diagnostic; not official ReArt metric |
| AiM | AiM-style PartNet pilot | 3 objects | 0.667 | 40.45 | conditional on matched components |

This block is still in the main table but visually separated and rendered in a smaller font. The
caption must state that values in A and B are not a single paired leaderboard.

## Rotation robustness companion

| Relation design | Canonical median / P90 ↓ | Rotated median / P90 ↓ | ΔP90 ↓ | Type Δ ↓ |
|---|---:|---:|---:|---:|
| Old free-axis head | 1.82 / 85.55 | 58.10 / 83.96 | -1.59 | not exported |
| VN, no Haar training | 16.03 / 80.44 | 15.98 / 80.98 | +0.54 | not exported |
| VN, Haar training | 15.72 / 80.60 | 15.72 / 80.67 | +0.07 | not exported |

This companion ablation uses whole-object Haar rotations and identical objects, tracks, slot
checkpoint, and evaluation code. It is the evidence for rotation robustness; relation-only
equivariance numbers are reported separately as an architecture sanity check.
These rotation rows use all 275 GT joint pairs for both canonical and rotated samples, rather than
conditioning the canonical number on predicted type. This prevents the comparison population from
changing between the two columns. The existing audit did not export rotated edge/type predictions;
that missing export should be added before the final camera-ready table.

## Caption draft

**Articulation and axis estimation.** Block A reports object-aligned evaluation wherever released
outputs permit a compatible joint estimate. Block B retains native-protocol diagnostics that use a
different dataset or conversion and must not be compared as paired samples. Axis error is the
undirected angular error over recovered type-correct joints; Axis-Line is normalized by object
scale. We additionally report edge/type recovery so that conditional axis accuracy cannot hide
missed joints. Oracle part input and restricted success subsets are marked explicitly.
