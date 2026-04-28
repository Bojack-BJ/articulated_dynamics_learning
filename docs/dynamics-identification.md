# Dynamics Identification

This stage assumes the kinematic structure is already fixed:

```text
episode.json
  + articulation_artifact.json
  + inferred MJCF
  -> identify-dynamics
  -> identify-dynamics-mjx
  -> optimized mass / damping / frictionloss
  -> optimized MJCF + dynamics_identification.json
```

The current implementation is an optimization-first MVP:

- it loads the inferred MJCF into MuJoCo
- it reads observed `q(t)` tracks from `articulation_artifact.json`
- it re-simulates the free-motion or forced-excitation episode
- it replays known generalized forces from `dynamics_log.jsonl` when the
  recording provides one
- it optimizes:
  - moving-part mass
  - joint damping
  - joint frictionloss
- it minimizes rollout mismatch in `q(t)` and `qdot(t)`

It currently uses finite-difference gradient descent around MuJoCo rollouts.
It is not exact end-to-end autodiff through MuJoCo.

The parallel `identify-dynamics-mjx` path uses JAX autodiff through MuJoCo MJX
rollouts. It keeps the same input/output contract so you can compare the two
optimizers on the same object.

## Run It

If you already exported inferred articulation:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --render-gl-backend cgl
```

Or use the YAML example:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave.yaml
```

For a torque-pulse episode, use the forced-response YAML:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave_torque_pulse.yaml
```

MJX autodiff path:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics-mjx \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --jax-platform metal \
  --enable-pjrt-compatibility
```

Or use the MJX YAML example:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave_mjx.yaml
```

Outputs:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/dynamics_identification/
  dynamics_identification.json
  <object-id>.mjcf.identified.xml
  simulated_view_<view-index>.mp4
  comparison_view_<view-index>.mp4
```

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/dynamics_identification_mjx/
  dynamics_identification.json
  <object-id>.mjcf.identified.xml
```

The JSON artifact includes:

- optimizer history
- best rollout loss
- force replay diagnostics under `simulation_options.force_replay`
- per-part mass estimates
- per-joint damping / frictionloss estimates
- observed joint trajectories
- simulated trajectories from the best-fit model
- trajectory error metrics for `q` and `qdot`
- simulator-GT parameter diagnostics when the original MJCF is available,
  including joint effective inertia error
- optional rollout rendering artifacts

Plot optimizer history:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp plot-dynamics-identification \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json
```

This writes `optimization_history.svg` next to the input artifact. Each loss
metric gets its own panel, so `total_loss`, `q_mse`, `qdot_mse`, and optional
`prior_penalty` are not overlaid. Each optimized mass/damping/friction
parameter also gets its own panel. When simulator GT is available from the
recording MJCF, the parameter panel includes a dashed horizontal ground-truth
line. Use `--linear-loss` if you do not want the default log-scale loss plots.

The inferred MJCF is initialized from the exported articulation proxies. When `export-inferred-articulation` can resolve the sibling `fusion_manifest.json`,
those proxies are sized from the reference-frame fused pointcloud for each part. The artifact marks this with `metadata.source =
"pointcloud-reference-bounds"`. If the pointcloud cannot be resolved, the
exporter falls back to the bounds already stored in `part_poses.json`.

By default, dynamics identification disables MuJoCo contacts between the
inferred proxy geoms. These proxy boxes are coarse geometry/inertial estimates
and can overlap at the joint, so enabling contacts too early can cause the
rollout to jam before the door/drawer reaches the observed state. The optimized
MJCF written by this stage records that policy with `contype="0"` and
`conaffinity="0"` on all geoms. Use `--enable-contact` only when the proxy
geometry has been validated for contact-rich identification.

If the inferred proxy geometry or joint axis produces spurious gravity torque,
use `--gravity-mode zero` for the identification rollout. This is useful for
door-like vertical hinge tests where the intended dynamics are dominated by the
known input torque, damping, and joint-axis inertia. The optimized MJCF records
this choice with `<option gravity="0 0 0">`.

## Debugging History And Current Status

This section records the important failure modes found during the microwave
MVP. It is meant to prevent repeating the same conclusions when the pipeline is
scaled to more objects.

### 1. Collision Proxies Were Not Accurate Enough For Contact

The exported URDF/MJCF uses simple box proxies derived from part pointcloud
bounds. These proxies are useful for a kinematic/collision placeholder, but
they are not yet reliable contact geometry. In early dynamics rollouts the
boxes overlapped near the hinge and MuJoCo contacts prevented the door from
following the observed motion. The symptom was a comparison video where the
door appeared stuck or jammed.

Current behavior:

- `identify-dynamics` disables geom contacts by default.
- The optimized MJCF writes `contype="0"` and `conaffinity="0"` unless
  `--enable-contact` is passed.
- Contact-rich system ID is intentionally deferred until the collision proxies
  are validated or replaced with better geometry.

### 2. Inertial Initialization Was Too Crude

The first MJCF export used a fixed placeholder-like heuristic for inertial
values. That made the starting effective joint inertia too far from the source
model, and it also made mass comparisons misleading. The exporter now sizes
the proxy boxes from the reference-frame fused pointcloud when
`fusion_manifest.json` is available.

Current behavior:

- `articulation.json` records proxy metadata such as
  `metadata.source = "pointcloud-reference-bounds"`.
- The inferred MJCF initializes mass and diagonal inertia from those proxy
  boxes.
- The dynamics report compares both raw body mass/inertia and joint effective
  inertia. For single-DOF behavior, joint effective inertia `M[dof,dof]` is the
  more meaningful metric than body mass alone.

### 3. Passive Free Motion Could Not Identify Absolute Mass

A passive free-decay episode without known external input mostly constrains
ratios such as damping/effective-inertia. Many different mass and damping
combinations can produce similar trajectories. This is not an optimizer bug;
it is an identifiability issue in the experiment design.

Current behavior:

- Free-motion recordings are still useful for checking `q(t)` extraction and
  qualitative rollout behavior.
- For parameter identification, use a forced-response episode with a known
  generalized force.
- The recorder now supports `--excitation-mode pulse|prbs|sine` and writes a
  sim-rate `dynamics_log.jsonl`.

### 4. Forced-Response Recording Needed Clean Initial Conditions

The original free-mode recorder had a fallback initial velocity when
`initial-qvel` was zero. That was helpful for passive motion, but it polluted
forced-response recordings with an unknown impulse. The result was a mismatch
between the intended known input and the actual initial state.

Current behavior:

- When `--excitation-mode` is not `none`, the recorder respects
  `--initial-qvel 0.0`.
- It does not auto-generate a fallback free initial velocity unless explicitly
  requested through the free-motion settings.
- The forced-response config writes a separate object id, for example
  `microwave011_torque_pulse`, so it does not overwrite the free-motion
  episode.

### 5. The Optimizer Initially Ignored Known Forces

The first dynamics identifier only replayed autonomous/free dynamics. Running
it on a torque-pulse recording treated the observed forced response as if it
were free motion, so the fitted parameters were not meaningful.

Current behavior:

- `identify-dynamics` reads `dynamics_log.jsonl` when available.
- It maps recorded source joint names, such as `microjoint`, to inferred joint
  names, such as `Microwave011_door_joint`, using the MuJoCo prior stored in
  `articulation_artifact.json`.
- The output records replay diagnostics under
  `simulation_options.force_replay`.

### 6. Local Finite-Difference Optimization Had Bad Basins

The MuJoCo finite-difference optimizer can be misled by local discontinuities
from joint limits, dry friction, or contact. On the microwave pulse episode,
starting near damping `0.1` gave a local finite-difference direction that moved
away from the correct damping near `1.0`.

Current behavior:

- `identify-dynamics` runs a small coordinate-search warm start before local
  finite-difference gradient descent.
- This is still a pragmatic optimizer, not a globally reliable solver.
- The SVG plot from `plot-dynamics-identification` should be checked to verify
  that loss actually falls and parameters do not just hit arbitrary bounds.

### 7. Autodiff/MJX Is Useful But Environment-Sensitive

MJX gives a cleaner autodiff path than finite differences, but local setup on
macOS Metal was brittle. We saw cases where sandboxed execution reported
backend errors that were not reproducible in the user's real local terminal.
JAX Metal also produced compatibility errors such as
`UNIMPLEMENTED: default_memory_space is not supported` in some environments.

Current behavior:

- `identify-dynamics-mjx` exists as a parallel backend with the same artifact
  contract.
- The default JAX platform is CPU for predictable behavior.
- Use `probe-mjx --jax-platform metal --enable-pjrt-compatibility` before
  requesting Metal.
- Treat sandbox-only MuJoCo rendering, PyTorch MPS, or JAX Metal failures as
  possible false negatives and retry in the real local environment when needed.

### 8. Geometry Errors Can Create Fake Gravity Torque

The inferred door axis and proxy center of mass can be slightly different from
the source MJCF. With normal gravity, this creates gravity torque that does not
exist in the intended hinge model, and the optimizer may compensate by changing
friction or mass. This was visible as a good-ish trajectory fit but poor raw
mass/friction interpretation.

Current behavior:

- `--gravity-mode zero` is available for door-like hinge identification when
  the goal is to estimate input-driven effective inertia and damping.
- Normal-gravity rollout remains available for cases where gravity is part of
  the target behavior.
- The current microwave torque-pulse validation is considered healthy under
  zero gravity:
  - `q_rmse` is about `0.009 rad`
  - `qdot_rmse` is about `0.033 rad/s`
  - damping and friction match simulator GT
  - joint effective inertia error is about `5%`

### 9. Current Recommended Dynamics Validation Path

For a clean single-DOF door/drawer dynamics check:

1. Record a free-motion episode for qualitative behavior and the original RGB-D
   artifact.
2. Record a separate forced-response episode with known generalized force.
3. Run pointcloud fusion, part pose, joint inference, and inferred articulation
   export on the forced-response episode.
4. Run `identify-dynamics` with force replay enabled by the episode log.
5. Use `--gravity-mode zero` first for door-like hinge debugging.
6. Inspect `q_rmse`, `qdot_rmse`, damping/friction error, and effective joint
   inertia error.
7. Use the comparison video and optimizer SVG plot before trusting the
   optimized MJCF.

## Rollout Video Rendering

`identify-dynamics` and `identify-dynamics-mjx` can render the optimized rollout
and save a side-by-side video against the original recorded RGB frames:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --render-gl-backend cgl
```

Backend choices:

- `cgl`: recommended on macOS when running from a normal local GUI terminal.
- `glfw`: fallback if CGL fails and a display session is available.
- `none`: disable video rendering, useful for batch/headless runs.
- `auto`: let MuJoCo choose; this can fail in sandboxed or SSH/headless contexts.

The videos are written next to `dynamics_identification.json` as
`simulated_view_<view-index>.mp4` and `comparison_view_<view-index>.mp4`. The
comparison video uses original RGB on the left and the simulated rollout on the
right. If rendering is unavailable, the JSON artifact stores
`render_artifacts.available = false` and a diagnostic `reason` instead of
failing the optimization.

## Current Limitations

- The current MVP supports `control_mode=free` recordings.
- It does not yet replay closed-loop `track` control episodes.
- Absolute mass can be weakly identifiable from passive single-DOF motion.
  In practice, damping and friction are often better constrained than mass.
- Contact-rich identification is not implemented yet.
- The current MJX path assumes all observed joints share the same timestamp schedule.
- The repository defaults `identify-dynamics-mjx` to `--jax-platform cpu`.
  Use `--jax-platform metal` explicitly after `probe-mjx` succeeds on your
  local machine.

## How This Connects To A Lagrangian Network

Later, if you switch from direct MuJoCo system identification to a learned
dynamics model, keep the same contract:

```text
input:  q, qdot, u
target: qdd  or  next-state
```

Recommended progression:

1. Use the current optimization artifact as a structured baseline.
   It gives you:
   - joint type
   - limits
   - initial mass / damping / friction guesses
   - clean `q(t), qdot(t)` trajectories

2. Build a Lagrangian model in generalized coordinates.
   Predict:
   - mass matrix `M(q)`
   - potential energy `V(q)`
   - optional dissipation `D(q, qdot)`

3. Train against acceleration or one-step rollout targets.
   For this project, the practical target is:
   - differentiate smoothed `q(t)` to get `qdot(t), qdd(t)`
   - train `f(q, qdot, u) -> qdd`

4. Keep friction/contact outside the pure conservative core.
   A robust decomposition is:
   - Lagrangian core for rigid-body dynamics
   - separate residual head for friction/contact impulses

5. Use the system-ID MJCF as a teacher or prior.
   Useful strategies:
   - behavior cloning on MuJoCo rollouts from the identified model
   - initialize a learned damping head near the fitted damping/friction values
   - compare learned rollout error against the identified-physics baseline

In short: the current optimizer gives you a physically grounded stage-3
baseline; the Lagrangian-network path should be layered on top of the same
`URDF/MJCF + q,qdot` artifact contract, not replace the earlier perception and
kinematic stages.
