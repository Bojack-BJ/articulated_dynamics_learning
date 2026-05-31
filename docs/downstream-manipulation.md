# Downstream Manipulation

This module turns an inferred or identified MuJoCo articulation into a small downstream manipulation demo.

```text
inferred/identified MJCF
  + optional dynamics_identification.json
  + target joint position
  -> plan-manipulation
  -> manipulation_plan.json
```

The first supported task is a single-joint reach primitive: find an initial joint velocity or a short generalized-force pulse that moves a hinge or slide joint toward a desired position. This is intentionally joint-space first. It gives a clean physics-backed target for policy learning before adding a full robot arm, contacts, grasp selection, or end-effector planning.

## Why Condition A Visual Policy On Dynamics

If mass, effective joint inertia, damping, and friction are known or predicted, they are useful context for a visual policy. Two objects can look similar but need different push speeds or impulses. A policy can consume these dynamics values as extra conditioning:

```text
observation = image/features + robot state + articulated state
condition   = mass, effective_joint_inertia, damping, frictionloss, limits
action      = end-effector velocity / push primitive / target impulse
```

The planner exports this as `policy_conditioning.features`, `policy_conditioning.values`, and `policy_conditioning.by_name` in `manipulation_plan.json`.

## Run It

Plan an initial-velocity push from an optimized dynamics artifact:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp plan-manipulation \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json \
  --joint-name door_joint \
  --target-q 0.8 \
  --mode initial_velocity \
  --gravity-mode zero
```

Plan a short generalized-force pulse:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp plan-manipulation \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --dynamics-identification outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json \
  --joint-name door_joint \
  --target-q 0.8 \
  --mode pulse_force \
  --max-pulse-force 3.0 \
  --pulse-duration-s 0.12
```

## Output

```text
downstream_manipulation/
  manipulation_plan.json
```

The artifact includes:

- task metadata: joint, initial position, target position, rollout duration
- command: selected initial `qvel` or force pulse
- policy conditioning vector: mass, effective inertia, damping, friction, armature, limits
- rollout samples for the best command
- candidate summary from the grid search
- success and final error metrics

## Current Limitations

- This is not yet a full robot-arm planner. It plans the articulated joint command that a robot push should induce.
- Contacts are disabled by default, matching the dynamics-ID default for coarse proxy geometry.
- The planner uses a deterministic grid search, not MPC or trajectory optimization.
- For hardware use, the selected joint-space velocity or impulse still needs to be mapped to an end-effector velocity controller and contact point on the moving part.

