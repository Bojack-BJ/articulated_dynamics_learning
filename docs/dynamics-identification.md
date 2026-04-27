# Dynamics Identification

This stage assumes the kinematic structure is already fixed:

```text
episode.json
  + articulation_artifact.json
  + inferred MJCF
  -> identify-dynamics
  -> optimized mass / damping / frictionloss
  -> optimized MJCF + dynamics_identification.json
```

The current implementation is an optimization-first MVP:

- it loads the inferred MJCF into MuJoCo
- it reads observed `q(t)` tracks from `articulation_artifact.json`
- it re-simulates the free-motion episode
- it optimizes:
  - moving-part mass
  - joint damping
  - joint frictionloss
- it minimizes rollout mismatch in `q(t)` and `qdot(t)`

It currently uses finite-difference gradient descent around MuJoCo rollouts.
It is not exact end-to-end autodiff through MuJoCo.

## Run It

If you already exported inferred articulation:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml
```

Or use the YAML example:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/identify_dynamics_microwave.yaml
```

Outputs:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/dynamics_identification/
  dynamics_identification.json
  <object-id>.mjcf.identified.xml
```

The JSON artifact includes:

- optimizer history
- best rollout loss
- per-part mass estimates
- per-joint damping / frictionloss estimates
- observed joint trajectories
- simulated trajectories from the best-fit model

## Current Limitations

- The current MVP supports `control_mode=free` recordings.
- It does not yet replay closed-loop `track` control episodes.
- Absolute mass can be weakly identifiable from passive single-DOF motion.
  In practice, damping and friction are often better constrained than mass.
- Contact-rich identification is not implemented yet.

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
