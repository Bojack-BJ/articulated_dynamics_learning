# MuJoCo Recording

## Install

Simulation extras:

```bash
pip install -e '.[simulation]'
```

Visualization extras for PNG output and MP4 preview:

```bash
pip install -e '.[viz]'
```

## Basic Recording

Record a `sim RGB-D + camera poses + action_log` episode from a MuJoCo model:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco examples/mujoco_models/door_hinge.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --frames 60 \
  --perturbation-scale 0.4
```

The recorder writes an episode under `outputs/recordings/<object-id>/episode.json`.

## USD To MJCF

If you already have a USD asset, use the all-in-one converter:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp.usd_to_mjcf Lightwheel/Refrigerator031/Refrigerator031.usd --category refrigerator
```

This:

1. Flattens the USD and converts meshes to OBJ
2. Extracts joints from USD Physics
3. Generates a MuJoCo MJCF XML

Output:

- `examples/mujoco_models/Refrigerator031.xml`
- `examples/mujoco_models/Refrigerator031_obj/`

Then record directly:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco examples/mujoco_models/Refrigerator031.xml \
  --category refrigerator \
  --object-id Refrigerator031 \
  --output-dir outputs/recordings \
  --duration-s 8 \
  --fps 20 \
  --all-joints
```

If you want the lower-level inspection path:

```bash
usdcat --flatten Lightwheel/Refrigerator031/Refrigerator031.usd | \
  PYTHONPATH=src python3 -m rgbd_urdf_mvp.usda_to_obj - \
  --out-dir examples/mujoco_models/Refrigerator031_obj

PYTHONPATH=src python3 -m rgbd_urdf_mvp.usd_joint_parser \
  Lightwheel/Refrigerator031/Refrigerator031.usd \
  --output /tmp/refrigerator031_joints.xml
```

## Common Options

Duration and FPS:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20
```

PNG RGB and depth:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --frames 60 \
  --rgb-format png \
  --depth-format png
```

MP4 preview:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20 \
  --video
```

Tri-view capture:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20 \
  --camera-mode triview \
  --camera-triview-spacing-deg 35 \
  --video
```

Disk-light tri-view capture:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20 \
  --camera-mode triview \
  --rgb-format png \
  --depth-format png \
  --mask-format png
```

Override MP4 playback FPS:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20 \
  --video \
  --video-fps 30
```

## Recorder Behavior

- `track` mode: finds the target hinge/slide joint and applies tracking control with optional perturbation
- `free` mode: sets initial state and runs without external control
- `--random-initial-qpos`: randomizes controlled joint positions within limits
- `--auto-initial-qvel-from-limits`: infers an initial velocity from joint limits and reset pose
- `--auto-initial-qvel-min-abs`: lower bound for inferred initial speed
- `--auto-initial-qvel-max-abs`: upper bound for inferred initial speed
- `--auto-initial-qvel-direction-mode`: `away-from-qpos0`, `toward-lower`, or `toward-upper`
- `--camera-mode orbit`: single sweeping camera
- `--camera-mode triview`: three fixed cameras
- `--write-concat-assets`: opt back into duplicated stitched raw RGB/depth/mask assets under `assets/concat`

Tri-view writes by default:

- `assets/view_0/`
- `assets/view_1/`
- `assets/view_2/`
- `episode_concat.mp4`
- `episode_view_0.mp4`
- `episode_view_1.mp4`
- `episode_view_2.mp4`

If you pass `--write-concat-assets`, tri-view additionally writes:

- `assets/concat/`

To compact an older triview episode that still has duplicated `assets/concat/` raw
frames, run:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp compact-mujoco-recording \
  outputs/recordings/<object-id>/episode.json
```

To transcode an older episode from `ppm/pgm` assets to `png` in place and
update `episode.json`, run:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp repack-mujoco-recording \
  outputs/recordings/<object-id>/episode.json
```

## Useful Flags For Real Assets

- `--segmentation-masks`: render binary object masks
- `--part-segmentation-masks`: render indexed MuJoCo-prior part masks
- `--disable-target-mesh-collision`: disable target mesh collisions during rollout
- `--hide-clear-meshes`: hide helper meshes such as `*_Clear` from RGB/depth/mask rendering
