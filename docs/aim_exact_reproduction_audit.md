# AiM Exact-Paper Reproduction Audit

This document separates values that are explicitly available from the AiM paper or
released repository from project-side choices. It is not a claim that every row is a
bitwise reproduction of the authors' private synthetic generator.

| Object | Paper frames | GT parts | Joint type | Start | End range | Camera | Notes |
|---|---:|---:|---|---|---|---|---|
| Storage-45135 | 200 | Pending paper asset audit | 3 generated slides | Pending | Pending | Moving monocular interaction; 100 start views | Paper-listed simple object. The public model maps to three generated slide joints (`joint_0..2`), but the paper-to-asset range mapping is not public. |
| Storage-47024 | 200 | 3 | Revolute + prismatic | 0, 0 | +90 deg, +0.70 m | Moving monocular interaction; 100 start views | Generated MJCF: `joint_0` hinge [0, 1.7090], `joint_1` slide [0, 0.816]. Both published targets are valid. |
| Storage-47648 | 500 | 7 | 4 revolute + 2 prismatic | 0 | +120, -120, -60, +60 deg; +0.10, +0.16 m | Moving monocular interaction; 100 start views | Generated `joint_0..5` maps to four hinges then two slides. Axis signs encode part of the published direction; q target magnitudes are valid. |
| Table-31249 | 500 | 5 | 2 revolute + 2 prismatic | Pending appendix audit | Pending appendix audit | Moving monocular interaction; 100 start views | Generated MJCF: two slides and two hinges (`joint_0`, `joint_1`, `joint_3`, `joint_4`). Exact end ranges still need a public appendix transcription. |

## Verified released-code facts

- The paper describes 100 multi-view observations of the start state, followed by 200
  interaction images for two/three-part scenes and 500 images for complex scenes.
- The public `seg_main.py` uses a sequential-RANSAC minimum inlier count of
  `0.1 * num_dynamic`.
- The public repository exposes the optimization and segmentation implementation but
  does not include object-specific PartNet motion scripts, camera trajectories, or
  per-joint time intervals for the four objects above.
- The paper describes motion windows at normalized times 0, 0.5, and 1.0. It does
  not provide enough released information to reconstruct each object's full temporal
  schedule. We therefore do not infer one from the final ranges.

## Project-side controlled replacement

`paper_simultaneous` is an independently named recorder mode. For each explicit
joint range it uses:

```text
s(t) = t^2 (3 - 2t)
q_k(t) = q_start_k + s(t) (q_end_k - q_start_k)
```

All configured joints share the same normalized time. It guarantees target values of
the start state, midpoint, and end state, while the episode records the actual MuJoCo
state and target trajectory so controller tracking error remains observable.

This mode does not alter legacy `staggered`, `track`, `free`, or front-camera
protocols. The main causal comparison is Storage-47024 with identical camera,
frame count, static scan, optimizer, and seed. Its comparison control is
`paper_sequential`, which uses the same explicit start/end values in isolated time
slots. This avoids conflating timing with the legacy `staggered` mode's use of full
MJCF joint ranges and alternating directions.

## Required preflight before an exact run

1. Resolve the generated MJCF joint names for every listed PartNet object.
2. Confirm every configured start/end value lies within its generated joint range.
3. Render frames at normalized time 0, 0.5, and 1.0 and verify all target joints
   overlap for the entire interaction.
4. Record target-versus-actual max deviation from `episode.json`/dynamics logs.
5. Mark unavailable fields as unavailable rather than substituting MJCF full limits.

## Interpretation boundaries

The resulting experiment is an exact-object, paper-budget, paper-range reproduction
attempt with an explicit simultaneous-motion schedule. It is not an assertion that
the unpublished original camera path, random seed, or temporal schedule are exactly
known. Any remaining mismatch must be investigated through the requested static
geometry, representation, and adapter audits before modifying AiM core algorithms.
