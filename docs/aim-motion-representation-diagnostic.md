# AiM Motion-Representation Diagnostic

## Purpose

This diagnostic separates three possible causes of poor AiM segmentation:

1. insufficient image occupancy or temporal visibility;
2. unsupported acquisition structure;
3. an upstream deformation representation that does not form clean rigid-part trajectories.

The official AiM reconstruction, dynamic/static split, sequential RANSAC, merge logic,
and thresholds remain unchanged.

## Storage-47024

The paper-like and close-monocular recordings use identical 200-frame, 15 Hz joint
trajectories. Moving-part image occupancy and geometry coverage improve in the close
recording, but both runs predict four parts for a two-part object.

Per-GT-part trajectory fitting maps the observed GT reference points to AiM Gaussian
trajectories at `t=1`, then fits a separate SE(3) transform at `t={0,0.5,1}`:

| Protocol | GT part | Mean motion | Mean rigid RMSE | Cross/own replay ratio |
|---|---:|---:|---:|---:|
| Paper-like | 2 | 0.077 m | 0.082 m | 1.17 |
| Paper-like | 4 | 0.046 m | 0.046 m | 1.45 |
| Close monocular | 2 | 0.069 m | 0.075 m | 1.34 |
| Close monocular | 4 | 0.037 m | 0.059 m | 1.29 |

The fitted rigid residual is comparable to the estimated motion itself, while applying
the other part's motion model increases replay error by only about `1.17-1.45x`.
Therefore the learned trajectories are neither cleanly rigid within each GT part nor
strongly separable across parts.

## Synchronized Four-View Compatibility

The exporter assigns all four cameras at a simulation timestep the same normalized
deformation time. The released AiM loader reads these observations, but the trained
deformation evaluates to exactly zero displacement at `t={0,0.5,1}`:

```text
mean_deformation = 0
```

Official `seg_main.py` passes this value as the strict RANSAC inlier threshold. Because
no residual satisfies `error < 0`, `ransac_propose_two_stage()` never updates
`best_errors`, and the caller fails at `torch.isfinite(None)`.

This is an implementation/protocol incompatibility. It is not a valid four-view
upper-bound score and must not be used as direct evidence about AiM's supported
monocular protocol.

## Minimal Hinge-Door Sanity

The toy model contains one fixed base and one large, unoccluded hinge door. The door
rotates through 85 degrees over 200 frames at 15 Hz. The input uses a supported
monocular AiM-style interaction sequence and unchanged official optimization and
segmentation.

AiM completes and returns one moving component plus static (`2/2`), proving the
renderer/exporter/optimizer chain can execute. However, common-domain evaluation gives:

| Pred./GT | Point IoU | ARI | RI | Coverage @2% bbox |
|---:|---:|---:|---:|---:|
| 2/2 | 0.447 | 0.094 | 0.648 | 0.989 |

The door's mapped Gaussian trajectories have median displacement `0`, and their rigid
RMSE remains comparable to mean motion. Thus correct part count does not imply correct
part decomposition.

## Conclusion

Camera distance alone is not the dominant bottleneck. The grouped multi-camera
experiment is unsupported by the current implementation and should remain a
compatibility diagnostic. Most importantly, even the favorable monocular toy case
produces mixed labels and weak deformation trajectories despite full geometry coverage
and an 85-degree hinge motion.

The next AiM audit should compare the released official dataset renderer/export format
against this project's RGB-D-to-Blender adapter at the level of:

- image masks and alpha/background treatment;
- camera normalization and scene radius;
- deformation-time sampling;
- start/motion/end coordinate normalization;
- per-Gaussian motion magnitude and static leakage.
