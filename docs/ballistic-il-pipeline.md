# Ballistic IL Pipeline

This document describes the downstream pipeline for proving that identified
joint dynamics are useful for manipulation. The key rule is:

```text
kick once -> release -> evaluate free joint response
```

The task is deliberately not a sustained velocity controller. The policy predicts
one initial joint velocity, then MuJoCo applies no further control. Success
depends on effective joint inertia, damping, and friction.

## 0. Setup

Install MuJoCo, visualization utilities, tracking if needed, and the IL baseline:

```bash
source .venv/bin/activate
python -m pip install -e ".[simulation,viz,tracking,imitation]"
```

Use these shell variables in the examples below:

```bash
export OBJECT_ID=microwave011_torque_pulse
export JOINT_NAME=door_joint
export MODEL_XML=examples/mujoco_models/microwave011.xml
export RUN_ROOT=outputs/recordings/$OBJECT_ID
```

## 1. Record Data

For dynamics identification, prefer a forced-response recording with a known
generalized force. This gives the optimizer enough information to fit damping
and effective inertia more meaningfully than passive motion alone.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco "$MODEL_XML" \
  --object-id "$OBJECT_ID" \
  --category microwave \
  --output-dir outputs/recordings \
  --control-mode free \
  --joint-name "$JOINT_NAME" \
  --initial-qvel 0.0 \
  --excitation-mode pulse \
  --excitation-force 1.2 \
  --excitation-start-s 0.05 \
  --excitation-duration-s 0.20 \
  --frames 48 \
  --frame-dt 0.04 \
  --sim-dt 0.01 \
  --camera-mode triview \
  --rgb-format png \
  --depth-format png \
  --segmentation-masks \
  --part-segmentation-masks
```

Expected key outputs:

```text
$RUN_ROOT/
  episode.json
  dynamics_log.jsonl
  assets/
```

`dynamics_log.jsonl` is important. It stores the known force pulse used later by
`identify-dynamics`.

## 2. Build 4D Point Cloud And Articulation

Run the usual perception and kinematics stages. For a full batch, use the
existing end-to-end scripts/configs. The important final artifacts are:

```text
$RUN_ROOT/pointcloud_4d_partseg/
  fusion_manifest.json
  part_tracks.json
  part_poses.json
  joint_inference.json
  inferred_articulation/
    articulation_artifact.json
    urdf/$OBJECT_ID.mjcf.xml
```

Single-object command sketch:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  "$RUN_ROOT/episode.json" \
  --output-dir "$RUN_ROOT/pointcloud_4d" \
  --pixel-stride 8 \
  --voxel-size-m 0.02

PYTHONPATH=src python3 -m rgbd_urdf_mvp track-part-pixels \
  "$RUN_ROOT/pointcloud_4d/fusion_manifest.json" \
  --output-json "$RUN_ROOT/pointcloud_4d_partseg/part_tracks.json"

PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  "$RUN_ROOT/pointcloud_4d_partseg/part_tracks.json" \
  --output-json "$RUN_ROOT/pointcloud_4d_partseg/part_poses.json" \
  --method tracks

PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  "$RUN_ROOT/pointcloud_4d_partseg/part_poses.json" \
  --output-json "$RUN_ROOT/pointcloud_4d_partseg/joint_inference.json" \
  --mujoco-prior auto

PYTHONPATH=src python3 -m rgbd_urdf_mvp export-inferred-articulation \
  "$RUN_ROOT/episode.json" \
  "$RUN_ROOT/pointcloud_4d_partseg/part_poses.json" \
  "$RUN_ROOT/pointcloud_4d_partseg/joint_inference.json" \
  --output-dir "$RUN_ROOT/pointcloud_4d_partseg/inferred_articulation"
```

If the object already came from Particulate/remote articulation, the downstream
pipeline can start from the exported MJCF/articulation package instead.

## 3. Identify Dynamics

Fit the inferred MJCF to the timed joint trajectory and known force replay:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp identify-dynamics \
  "$RUN_ROOT/episode.json" \
  "$RUN_ROOT/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json" \
  "$RUN_ROOT/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml" \
  --output-dir "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification" \
  --gravity-mode zero \
  --render-gl-backend none
```

Expected key outputs:

```text
$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/
  dynamics_identification.json
  <object>.identified.xml
```

Use `dynamics_identification.json` as the input to the downstream IL pipeline.
It resolves to the optimized MJCF and carries the fitted parameter diagnostics.

## 4. Generate Ballistic IL Demos

Generate teacher demonstrations by randomized MuJoCo rollouts. Each episode
samples object dynamics, initial joint state, and target joint state. The teacher
searches for one `initial_qvel`, then releases the joint.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp generate-il-demos \
  "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --output-dir "$RUN_ROOT/downstream_il/dataset" \
  --joint-name "$JOINT_NAME" \
  --task impulse-to-target \
  --num-episodes 1024 \
  --release-duration-s 1.0 \
  --condition-source estimated \
  --max-initial-qvel 8.0 \
  --num-teacher-candidates 121 \
  --response-samples 64 \
  --gravity-mode zero
```

Outputs:

```text
$RUN_ROOT/downstream_il/dataset/
  il_dataset.npz
  il_dataset_manifest.json
```

Dataset arrays:

```text
obs            [N, 4]      q0, qdot0, target_q, release_duration_s
condition      [N, 7]      log inertia, log damping, log friction, ratios, limits
action         [N, 1]      teacher initial_qvel
free_response  [N, T, 3]   timestamp, q, qdot after release
```

## 5. Train Policies

Train the no-condition baseline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-il-policy \
  "$RUN_ROOT/downstream_il/dataset/il_dataset.npz" \
  --output-dir "$RUN_ROOT/downstream_il/bc_no_cond" \
  --condition-mode none \
  --epochs 100 \
  --batch-size 64
```

Train the dynamics-conditioned policy:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-il-policy \
  "$RUN_ROOT/downstream_il/dataset/il_dataset.npz" \
  --output-dir "$RUN_ROOT/downstream_il/bc_estimated" \
  --condition-mode estimated \
  --epochs 100 \
  --batch-size 64
```

Optional robustness baseline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-il-policy \
  "$RUN_ROOT/downstream_il/dataset/il_dataset.npz" \
  --output-dir "$RUN_ROOT/downstream_il/bc_noisy_estimated" \
  --condition-mode noisy-estimated \
  --noisy-condition-std 0.05
```

Each policy directory contains:

```text
policy.pt
training_metrics.json
```

## 6. Rollout And Evaluate

Evaluate no-condition:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp eval-il-policy \
  "$RUN_ROOT/downstream_il/bc_no_cond/policy.pt" \
  "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --output-dir "$RUN_ROOT/downstream_il/eval_no_cond" \
  --joint-name "$JOINT_NAME" \
  --num-episodes 256 \
  --release-duration-s 1.0 \
  --gravity-mode zero
```

Evaluate dynamics-conditioned:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp eval-il-policy \
  "$RUN_ROOT/downstream_il/bc_estimated/policy.pt" \
  "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --output-dir "$RUN_ROOT/downstream_il/eval_estimated" \
  --joint-name "$JOINT_NAME" \
  --num-episodes 256 \
  --release-duration-s 1.0 \
  --gravity-mode zero
```

Evaluation output:

```text
il_eval.json
```

Report these metrics:

- `success_rate`
- `final_error_mean`
- `settling_qdot_mean`
- `overshoot_mean`

The policy is only allowed to choose the kick. The rollout evaluator applies
that kick as an initial joint velocity and then sets no further control or force.

## 7. Recommended Paper Ablations

Run at least:

```text
BC-no-cond
BC-oracle-cond
BC-estimated-cond
BC-noisy-estimated-cond
```

Recommended table columns:

```text
condition_mode | success_rate | final_error_mean | settling_qdot_mean | overshoot_mean
```

The main claim should be that dynamics-conditioned policies perform better on
free-response ballistic tasks, especially under held-out dynamics. Avoid
claiming raw mass is fully identifiable; emphasize effective joint inertia,
damping, friction, and their ratios.

## 8. Roadmap Beyond V0

The current implementation is intentionally the smallest validation of the
dynamics signal. It should not be the final manipulation showcase.

### V0: Joint-Space Ballistic Sanity Check

Current implementation:

```text
state:  q0, qdot0, target_q, release_duration
cond:   effective inertia, damping, friction, ratios, limits
action: initial joint qvel
eval:   set qvel once, release, measure free response
```

This proves the basic dependency: the same `q0 -> target_q` request needs
different kick strengths under different joint dynamics. It is useful for quick
ablation, but it is also close to a model-based solve when `q`, target, and
physical parameters are known.

Add this baseline before using V0 as a paper result:

```text
model-based / analytic single-DOF solver
grid-search teacher
BC-no-cond
BC-dynamics-cond
```

The learned policy should be interpreted as learning a compact mapping to the
teacher, not as solving the full manipulation problem.

### V1: Contact Impulse / Part-Level Push

Current upgrade:

```text
state:  q0, qdot0, target_q, part geometry / contact candidates
cond:   identified joint dynamics
action: contact point, push direction, push speed or impulse, contact duration
eval:   apply a short force/contact to the moving part, release, measure response
```

This is more defensible because the action no longer directly sets joint
velocity. The policy must account for geometry and lever arm:

```text
tau = r x F
```

A known dynamics model helps choose how large the impulse should be, but the
task is no longer reducible to a one-dimensional qvel lookup.

Prepare the ARX X5A model package:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp prepare-arx-x5-model \
  --source-dir /path/to/ARX_Model \
  --output-dir outputs/arx_x5_model
```

If `--source-dir` is omitted, the command clones
`https://github.com/ARXroboticsX/ARX_Model.git`. The command copies `X5/X5A`,
patches `package://X5A/meshes/...` URDF mesh paths to relative paths, and writes:

```text
outputs/arx_x5_model/
  arx_x5_model_manifest.json
  X5A/urdf/X5A.mujoco.urdf
  X5A/meshes/*.STL
```

Generate V1 contact-impulse demos:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp generate-contact-il-demos \
  "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --output-dir "$RUN_ROOT/downstream_contact_il/dataset" \
  --robot-model outputs/arx_x5_model/X5A/urdf/X5A.mujoco.urdf \
  --robot-end-effector-body link6 \
  --joint-name "$JOINT_NAME" \
  --num-episodes 512 \
  --release-duration-s 1.0 \
  --contact-duration-s 0.12 \
  --max-force 40.0 \
  --num-teacher-candidates 121 \
  --gravity-mode zero
```

Train a BC policy on the 8D contact action:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp train-il-policy \
  "$RUN_ROOT/downstream_contact_il/dataset/contact_il_dataset.npz" \
  --output-dir "$RUN_ROOT/downstream_contact_il/bc_estimated" \
  --condition-mode estimated
```

Evaluate with contact-force rollout:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp eval-contact-il-policy \
  "$RUN_ROOT/downstream_contact_il/bc_estimated/policy.pt" \
  "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --output-dir "$RUN_ROOT/downstream_contact_il/eval_estimated" \
  --joint-name "$JOINT_NAME" \
  --max-force 40.0 \
  --gravity-mode zero
```

Visualize a generated dataset:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-il-dataset \
  "$RUN_ROOT/downstream_contact_il/dataset/contact_il_dataset.npz" \
  --output-svg "$RUN_ROOT/downstream_contact_il/dataset/contact_il_dataset.svg" \
  --title "V1 contact impulse teacher"
```

Render one rollout as a MuJoCo animation:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp render-il-rollout \
  "$RUN_ROOT/downstream_contact_il/dataset/contact_il_dataset.npz" \
  examples/mujoco_models/Microwave011.xml \
  --output-path "$RUN_ROOT/downstream_contact_il/dataset/lightwheel_x5_scaled_ep045_collisionchecked.mp4" \
  --episode-index 45 \
  --joint-name microjoint \
  --robot-model outputs/arx_x5_model/X5A/urdf/X5A.mujoco.urdf \
  --dynamics-artifact "$RUN_ROOT/pointcloud_4d_partseg/dynamics_identification/dynamics_identification.json" \
  --dynamics-body-name Microwave011_door \
  --object-visual-scale 0.6 \
  --playback-slowdown 2.5 \
  --robot-motion ik-pulse \
  --robot-ee-body x5_link6 \
  --robot-base-pos -0.05 -0.25 -0.10 \
  --robot-base-euler 0 0 -1.0 \
  --robot-pulse-distance 0.04 \
  --robot-approach-distance 0.10 \
  --enable-contact \
  --fps 20 \
  --width 800 \
  --height 608
```

Episode 45 in the current smoke dataset starts from an open microwave door
(`q0=-0.98 rad`) and moves to a near-closed target (`target_q=-0.12 rad`), a
`0.86 rad` door motion. That makes the ballistic response much easier to see
than the first dataset episode.

`render-il-rollout` defaults to `--trajectory-source dataset`, which renders
the stored teacher free-response `q(t)`. This is the safest default because the
dataset stores per-episode physical conditions but not the exact randomized
MuJoCo mass/damping scale factors needed to reproduce every episode from the
base MJCF. For paper/demo visualization, prefer passing the original Lightwheel
object MJCF and `--dynamics-artifact`; this keeps the original mesh/texture
assets while overwriting the compiled MuJoCo model's identified body mass,
inertia, joint damping, and joint friction for rendering or resimulation. Use
`--trajectory-source resimulate` when you explicitly want to apply the saved
action in the current MJCF. Add `--write-frames` to write a PNG frame directory
instead of an MP4/GIF container.

When `--robot-model` is provided, the renderer writes a temporary combined MJCF
next to the video, with the original object scene plus a prefixed X5 visual
model. `--robot-base-pos x y z` and `--robot-base-euler r p y` control the
robot placement. The default `--robot-motion ik-pulse` uses a kinematic planner:
for each rendered frame it computes a target point from the saved teacher
contact point and push direction, then solves a damped-least-squares IK problem
with MuJoCo's body Jacobian and writes the resulting X5 revolute joint qpos into
the scene. The end effector starts at the contact point, moves along the push
direction for the short pulse, then retracts while the door continues its free
response. A sidecar `<video>.ik_debug.json` is written with per-frame target,
end-effector position, joint qpos, and IK error. This is a visualization of the
pulse execution; it does not yet make the X5 arm dynamics cause the door motion
through MuJoCo contact constraints. Pass `--enable-contact` when rendering so
the IK line search rejects candidate joint poses with robot-object or robot-self
penetration. The sidecar also reports robot-object and robot-self contacts and
penetration depths. Use `--allow-robot-object-collision` or
`--allow-robot-self-collision` only for debugging raw IK behavior.

The raw Microwave011 Lightwheel scene has `model.stat.extent` around `1.10m`,
which makes the microwave look too large relative to the X5. For visualization,
`--object-visual-scale 0.6` scales the object mesh, body offsets, joint anchors,
inertial offsets, and saved contact point together. This is a visual unit
calibration for the demo; the underlying dataset q trajectory is unchanged.
For quantitative full-contact experiments, replace this heuristic with an
explicit Lightwheel unit calibration step.

On macOS, MuJoCo rendering needs a CoreGraphics/OpenGL context. If the command
runs inside a sandbox or headless shell and fails while creating
`mujoco.Renderer`, run the same command from a normal local terminal.

Current V1 implementation notes:

- The X5A model is prepared and stored as the robot resource used in the
  manifest.
- The renderer can show a kinematic X5 IK pulse that follows the teacher contact
  point and push direction.
- The teacher/evaluator rollout is still an idealized end-effector impulse, not
  full X5 arm contact dynamics.
- The teacher uses MuJoCo `mj_applyFT` to apply a short force at a point on the
  moving part, then releases the object for free response.
- This is the intended bridge between V0 joint-space qvel and V2 full robot-arm
  velocity pushing.
- Teacher contact-point selection should use a high-leverage point on the moving
  part, such as the farthest box corner from the joint axis. Using the inertial
  center can make the lever arm too short and force the teacher to saturate.
- For the first learned V1 policy, prefer a constrained action head:
  deterministic geometry/contact projection plus learned force and duration.
  Unconstrained 8D regression of contact point, direction, force, and duration
  can have low MSE but poor rollout because small point/direction errors create
  wrong torque.

### V2: Robot-Arm Velocity Push

Final showcase:

```text
state:  RGB-D / 4D point cloud, robot state, q0, target_q
cond:   identified effective inertia / damping / friction
action: short end-effector velocity command or motion primitive
eval:   robot contacts the part for 100-200 ms, releases, joint free-runs
```

This is the strongest downstream story. The robot must convert the desired
joint impulse into an executable push through contact. Dynamics-conditioned
policies should choose stronger/faster pushes for heavy, damped, or high-friction
objects and gentler pushes for light objects.

Recommended progression:

```text
V0 for sanity and debugging
V1 for the first serious paper ablation
V2 for final robot/manipulation showcase
```

## Troubleshooting

- If teacher success is low, increase `--max-initial-qvel`, increase
  `--num-teacher-candidates`, or shorten `--release-duration-s`.
- If all methods perform similarly, make the dynamics randomization wider or use
  more OOD eval episodes.
- If free response is dominated by gravity artifacts, use `--gravity-mode zero`
  until geometry and center of mass are reliable.
- If contact artifacts jam the motion, keep contacts disabled for this benchmark.
