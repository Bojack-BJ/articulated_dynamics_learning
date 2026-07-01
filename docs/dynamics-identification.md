# Dynamics Identification

This document summarizes the current Stage 3 dynamics-identification path. Stage 3 starts after the kinematic structure has already been inferred and exported:

```text
episode.json
  + articulation_artifact.json
  + inferred MJCF
  -> identify-dynamics or identify-dynamics-mjx
  -> optimized mass / damping / frictionloss
  -> optimized MJCF + dynamics_identification.json
```

The implementation is still an MVP, but the current baseline is usable for single-DOF door/drawer-style experiments. The most reliable path today is the MuJoCo finite-difference optimizer with known-force replay from a forced-response recording. The MJX path is available as a parallel autodiff backend, but CPU is the default because JAX Metal setup is environment-sensitive.

## Current Capabilities

- Load inferred MJCF from the Stage 2 articulation export.
- Read observed `q(t)` and `qdot(t)` tracks from `articulation_artifact.json`.
- Replay known generalized forces from `dynamics_log.jsonl` when the episode was recorded with `--excitation-mode`.
- Optimize moving-part mass, joint damping, and joint frictionloss against rollout mismatch.
- Disable contacts by default to avoid invalid contact forces from coarse proxy geometry.
- Initialize inertial parameters from pointcloud-sized collision proxies instead of a fixed placeholder box.
- Save an optimized MJCF with the fitted dynamics parameters.
- Report `q/qdot` trajectory errors and simulator-GT parameter diagnostics when the source MJCF is available.
- Render optimized rollout videos and side-by-side comparison videos when MuJoCo rendering is available.
- Plot optimization history with one panel per loss metric and one panel per optimized parameter, including simulator-GT reference lines when available.

## Run It

Finite-difference MuJoCo path:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --render-gl-backend cgl
```

YAML entry point:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave.yaml
```

Forced-response microwave example:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave_torque_pulse.yaml
```

MJX autodiff path:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics-mjx \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --jax-platform cpu
```

Probe Metal before using it:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp probe-mjx --jax-platform metal --enable-pjrt-compatibility
```

## Outputs

MuJoCo finite-difference output:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/dynamics_identification/
  dynamics_identification.json
  <object-id>.mjcf.identified.xml
  simulated_view_<view-index>.mp4
  comparison_view_<view-index>.mp4
  optimization_history.svg
```

MJX autodiff output:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/dynamics_identification_mjx/
  dynamics_identification.json
  <object-id>.mjcf.identified.xml
```

The JSON artifact includes optimizer history, fitted parameters, observed/simulated trajectories, force-replay diagnostics, trajectory error metrics, simulator-GT parameter diagnostics, and optional render diagnostics.

Plot optimizer history:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp plot-dynamics-identification \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json
```

Use `--linear-loss` if you do not want log-scale loss panels.

## Current Validation Snapshot

The current microwave torque-pulse validation uses a separate forced-response recording, known generalized force replay, contact-disabled rollout, and `--gravity-mode zero` for hinge debugging.

Representative result for `microwave011_torque_pulse`:

- `q_rmse`: about `0.0089 rad`
- `qdot_rmse`: about `0.033 rad/s`
- damping relative error against simulator GT: `0.0`
- frictionloss relative error against simulator GT: `0.0`
- effective joint inertia relative error: about `5.2%`
- raw body mass relative error remains large, about `87.9%`
- raw body inertia relative error remains large, about `141%`

The large raw mass/inertia error is expected for now. The inferred MJCF uses pointcloud-derived proxy geometry, while the simulator GT uses the original asset body/mesh decomposition. For single-DOF dynamics, the more meaningful diagnostic is the effective joint inertia `M[dof,dof]`, because the observed motion constrains the inertia seen by the joint rather than the full body mass distribution.

## Resolved Issues And Implemented Fixes

| Issue | Why it mattered | Current solution |
| --- | --- | --- |
| Coarse proxy contacts jammed the rollout | Box proxies can overlap near hinges, so MuJoCo contact forces can stop the door before it follows the observed video. | Contacts are disabled by default in `identify-dynamics`; optimized MJCF geoms get `contype="0"` and `conaffinity="0"` unless `--enable-contact` is passed. |
| Inertial initialization used fixed placeholders | A fixed box heuristic produced poor starting inertia and made mass diagnostics misleading. | Exported MJCF now initializes mass/inertia from pointcloud-sized part proxies when `fusion_manifest.json` is available. |
| Passive free motion could not identify absolute mass | Free decay mostly constrains damping/effective-inertia ratios; many mass/damping pairs can fit the same trajectory. | Recorder supports forced-response episodes through `--excitation-mode pulse|prbs|sine` and writes sim-rate `dynamics_log.jsonl`. |
| Forced-response recordings had polluted initial conditions | The old free-motion fallback could inject an unintended initial velocity even when testing known external force. | Excitation episodes respect `--initial-qvel 0.0`; forced-response configs use separate object ids, such as `microwave011_torque_pulse`. |
| The optimizer ignored known applied forces | A torque-pulse episode was initially treated as autonomous motion, making fitted parameters meaningless. | `identify-dynamics` reads `dynamics_log.jsonl`, maps source joint names to inferred joint names, and replays forces into `qfrc_applied`. |
| Finite-difference descent could start in a bad local basin | Joint limits, dry friction, and rollout discontinuities can make a local finite-difference step point the wrong way. | The optimizer now runs a coordinate-search warm start before local finite-difference gradient descent; optimization history should still be inspected. |
| Geometry errors produced fake gravity torque | Small errors in inferred axis or center of mass can create gravity torques that the original source model did not effectively have. | `--gravity-mode zero` is available for door-like hinge debugging when the intended experiment is dominated by known input torque and damping. |
| Rendering failed in sandbox/headless contexts | MuJoCo rendering may fail under sandboxed execution even when it works in the local GUI terminal. | Rendering is optional; the artifact stores `render_artifacts.available=false` and a reason instead of failing optimization. Use `--render-gl-backend cgl` locally on macOS. |
| Loss/parameter plots were hard to read | Overlaying many metrics on one plot made optimizer behavior difficult to diagnose. | `plot-dynamics-identification` now writes one panel per loss metric and one panel per parameter, with GT reference lines when available. |

## Recommended Validation Procedure

For a clean single-DOF dynamics experiment:

1. Record a normal free-motion episode for qualitative inspection.
2. Record a separate forced-response episode with a known generalized force.
3. Run pointcloud fusion, part pose estimation, joint inference, and inferred articulation export on the forced-response episode.
4. Run `identify-dynamics` with force replay from `dynamics_log.jsonl`.
5. Start with `--gravity-mode zero` for door-like hinges to isolate input-driven dynamics from geometry-induced gravity torque.
6. Inspect `q_rmse`, `qdot_rmse`, damping/friction error, effective joint inertia error, comparison video, and `optimization_history.svg`.
7. Only enable contacts after the collision proxies have been validated.

## Rollout Video Rendering

`identify-dynamics` and `identify-dynamics-mjx` can render the optimized rollout and save a side-by-side video against the original recorded RGB frames:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --render-gl-backend cgl
```

Backend choices:

- `cgl`: recommended on macOS from a normal local GUI terminal.
- `glfw`: fallback if CGL fails and a display session is available.
- `none`: disable video rendering for batch/headless runs.
- `auto`: let MuJoCo choose; this can fail in sandboxed or SSH/headless contexts.

If rendering is unavailable, the JSON artifact stores `render_artifacts.available = false` and a diagnostic `reason`.

## Remaining Limitations

- The current pipeline is most reliable for single-DOF objects.
- Contact-rich dynamics identification is deferred until collision proxies are accurate enough.
- Raw body mass and body inertia are still hard to compare when inferred proxy geometry differs from simulator GT geometry.
- Passive free-motion episodes remain weakly identifiable for absolute mass.
- The finite-difference optimizer is a pragmatic baseline, not a globally reliable solver.
- MJX autodiff exists, but CPU is the stable default; Metal should be used only after `probe-mjx` succeeds in the real local environment.
- Closed-loop `track` control replay is not yet the main supported dynamics-ID path.

## Meeting Log Summary

- We extended the RGB-D-to-articulation pipeline with a Stage 3 dynamics-identification module that consumes the inferred articulation artifact and MJCF, then fits moving-part mass, joint damping, and joint frictionloss.
- The first major issue was identifiability: passive free motion cannot uniquely determine absolute mass and damping, because it mainly constrains a damping/effective-inertia ratio. The fix is to record a separate forced-response episode with a known generalized force and replay that force during optimization.
- The recorder now supports forced excitation (`pulse`, `prbs`, `sine`) and writes `dynamics_log.jsonl` at simulation rate. Excitation recordings also preserve clean initial conditions, so `initial-qvel=0` is respected.
- The dynamics optimizer now reads the force log, maps the simulator joint name to the inferred joint name, and applies the recorded force schedule during rollout.
- Coarse pointcloud proxy geometry caused contact jamming in rollout videos. The current default disables contacts during dynamics ID and writes the same contact policy into the optimized MJCF.
- The inferred MJCF no longer uses a fixed placeholder inertial box. It initializes proxy size, mass, and diagonal inertia from the fused pointcloud bounds for each inferred part.
- Normal gravity can amplify small geometry or axis errors into fake hinge torques. For the current microwave hinge test, zero-gravity identification is used to focus on input torque, damping, friction, and effective joint inertia.
- Current microwave forced-response validation is healthy at the trajectory level: `q_rmse` is about `0.0089 rad`, `qdot_rmse` is about `0.033 rad/s`, damping/friction match simulator GT, and effective joint inertia error is about `5.2%`.
- Raw mass and raw inertia errors are still large because inferred pointcloud proxy geometry is not the same representation as the source MJCF mesh/body decomposition. For single-DOF behavior, effective joint inertia and rollout error are currently more meaningful than raw body-mass error.
- We added optimizer diagnostics: `dynamics_identification.json`, optimized MJCF, optional rollout comparison video, and `optimization_history.svg` with separate panels for loss terms and parameters.
- MJX autodiff has been added as a parallel backend with the same artifact contract. CPU is the stable default; JAX Metal remains environment-sensitive and should be probed separately.
- Next steps are to improve geometry/proxy quality, scale the forced-response protocol to more objects, add richer excitation for better mass observability, and later use the identified MuJoCo model as a teacher/baseline for a Lagrangian-network dynamics model.

## Connection To A Lagrangian Network

The learned-dynamics path should reuse the same artifact contract rather than replace the perception or kinematic stages:

```text
input:  q, qdot, u
target: qdd or next-state
```

Recommended use:

1. Use the current system-ID artifact as a structured physics baseline.
2. Train a Lagrangian core to predict mass matrix `M(q)` and potential energy `V(q)`.
3. Keep dissipation, friction, and contact as separate residual terms instead of forcing them into the conservative Lagrangian core.
4. Use identified MuJoCo rollouts as teacher data or as a comparison baseline.
5. Report learned rollout error against the same `q/qdot/u` trajectories used by `identify-dynamics`.
