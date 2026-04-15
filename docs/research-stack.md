# Research Stack Replacement Points

## Current Scaffold

The repository is organized so the current heuristic backends can be replaced without changing the top-level pipeline contracts.

## Reconstruction

Replace `GenerationFirstReconstructor` with:

- multi-view RGB-D object extraction
- canonical mesh generation
- depth reprojection verification
- mesh cleanup and support/confidence estimation

## Articulation Initialization

Replace `CategoryPriorParticulateAdapter` with:

- PARTICULATE execution on the canonical mesh
- post-processing into a legal kinematic tree
- part segmentation and joint candidate cleanup

## Temporal Refit

Replace `SlidingWindowRefitter` with:

- dense RGB-D / pointcloud alignment
- joint axis, origin, and limit optimization
- temporal optimization over part poses and generalized coordinates

## Dynamics Stage

The current repository stops at the `URDF + q + qdot` contract. Stage 3 should consume:

- exported URDF / MJCF
- refit `q`
- refit `qdot`
- recorded action logs

and then add:

- dynamic parameter identification
- contact modeling
- forward rollout evaluation
- manipulation-planning-facing prediction interfaces
