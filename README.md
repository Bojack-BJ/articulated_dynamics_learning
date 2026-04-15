# Articulated Dynamics Learning From RGB-D


This repository implements a runnable research scaffold for the MVP plan:

- `sim RGB-D + camera poses -> canonical mesh`
- `canonical mesh -> category-prior articulation init`
- `temporal refit -> q, qdot`
- `articulation -> kinematic + collision URDF`

The current implementation is intentionally scaffold-first:

- It defines the artifact contracts you asked for.
- It keeps a clean replacement boundary for a real `3D generation` backend and a real `PARTICULATE` backend.
- It provides a default heuristic backend for `door` and `drawer` so the pipeline, tests, and URDF export are executable now.

## What is implemented

- Episode schema for RGB-D sequences with camera poses and action logs
- Generation-first reconstruction stage with cleanup of low-support hallucinated components
- Single-DOF articulation initialization with `door -> revolute` and `drawer -> prismatic`
- Sliding-window temporal refit with smoothed `q` and derived `qdot`
- URDF export with visual meshes, collision box proxies, nominal inertial placeholders, and a MuJoCo MJCF stub
- Example episodes for `door` and `drawer`
- Unit tests for schema loading, temporal smoothing, and end-to-end door/drawer output

## What is still a placeholder

- Real RGB-D fusion / generative 3D model inference
- Real PARTICULATE invocation
- Dense RGB-D alignment for joint refit
- Real SAPIEN / MuJoCo runtime integration

The code is structured so those backends can replace the current heuristics without changing the pipeline contracts.

## Layout

- `src/rgbd_urdf_mvp/models.py`: public data contracts
- `src/rgbd_urdf_mvp/reconstruction.py`: generation-first reconstruction scaffold
- `src/rgbd_urdf_mvp/articulation.py`: articulation init adapter
- `src/rgbd_urdf_mvp/refit.py`: temporal smoothing and `qdot` extraction
- `src/rgbd_urdf_mvp/urdf.py`: URDF export
- `src/rgbd_urdf_mvp/pipeline.py`: end-to-end orchestration
- `examples/episodes/*`: sample door and drawer episodes

## Run

Validate an episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode examples/episodes/door/episode.json
```

Run the pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/drawer/episode.json --output-dir outputs/drawer_demo
```

Fuse a recorded MuJoCo episode into a time-indexed 4D point cloud:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud outputs/recordings/refrigerator031_door_only/episode.json \
  --output-dir outputs/recordings/refrigerator031_door_only/pointcloud_4d \
  --pixel-stride 8 \
  --voxel-size-m 0.02
```

Build a self-contained HTML viewer for inspection:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/refrigerator031_door_only/pointcloud_4d/fusion_manifest.json
```

Render target-object masks for an existing MuJoCo episode, then re-fuse:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp render-mujoco-masks \
  outputs/recordings/refrigerator031_door_only/episode.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/refrigerator031_door_only/episode.json \
  --output-dir outputs/recordings/refrigerator031_door_only/pointcloud_4d_masked \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback
```

What these commands do:

- `validate-episode` only checks the episode JSON structure and category support. It does not run reconstruction or export anything.
- `run` takes a valid episode and executes the full scaffold pipeline: reconstruction, articulation initialization, refit, and URDF/MJCF export.

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## MuJoCo recording for articulated objects

Install simulation extras:

```bash
pip install -e '.[simulation]'
```

If you want PNG outputs and/or an MP4 preview video, also install visualization extras:

```bash
pip install -e '.[viz]'
```

Record a `sim RGB-D + camera poses + action_log` episode from a MuJoCo model:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco examples/mujoco_models/door_hinge.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --frames 60 \
  --perturbation-scale 0.4
```

If you already have a USD asset and want to turn it into a MuJoCo-usable model, you can use the all-in-one `usd-to-mjcf` command:

By default, `--output-prefix` is `examples/mujoco_models/<usd_stem>` and the script only writes the final MJCF plus the OBJ directory.
`--category` is case-insensitive and supports common Lightwheel families, including `microwave`, `coffeemachine`, `dishwasher`, `electrickettle`, `oven`, `rangehood`, `sink`, `standmixer`, `stove`, `stovetop`, `toaster`, and `toasteroven`.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp.usd_to_mjcf Lightwheel/Refrigerator031/Refrigerator031.usd --category refrigerator
```

This automatically:
1. Flattens the USD and converts meshes to OBJ files (with UVs)
2. Extracts joint definitions from the USD Physics schema
3. Generates a complete MJCF XML with all meshes and joints

Output files:
- `examples/mujoco_models/Refrigerator031.xml` - ready to use MJCF
- `examples/mujoco_models/Refrigerator031_obj/` - directory with OBJ files

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

If you want to inspect the extracted joints separately, you can still run the parser directly:

```bash
# Step 1: Flatten USD and convert meshes to OBJ (no intermediate USDA file saved)
usdcat --flatten Lightwheel/Refrigerator031/Refrigerator031.usd | \
  PYTHONPATH=src python3 -m rgbd_urdf_mvp.usda_to_obj - \
  --out-dir examples/mujoco_models/Refrigerator031_obj

# Optional: inspect joints
PYTHONPATH=src python3 -m rgbd_urdf_mvp.usd_joint_parser \
  Lightwheel/Refrigerator031/Refrigerator031.usd \
  --output /tmp/refrigerator031_joints.xml

# Then record with the final MJCF written by usd_to_mjcf
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco examples/mujoco_models/Refrigerator031.xml \
  --category refrigerator \
  --object-id Refrigerator031 \
  --output-dir outputs/recordings \
  --all-joints
```

## Recording options

Control recording duration / frame rate:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20
```

To record PNGs instead of PPM/PGM:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --frames 60 \
  --rgb-format png \
  --depth-format png
```

To also write an MP4 preview video during recording:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 3.0 \
  --fps 20 \
  --video
```

To switch camera mode:

- `--camera-mode orbit` (default): single camera with azimuth sweep.
- `--camera-mode triview`: three fixed cameras at the same height/lookat, with azimuths centered on `--camera-azimuth-start-deg` and spaced by `--camera-triview-spacing-deg`.
  - RGB/Depth frames are saved in four folders:
    - stitched: `assets/concat/`
    - per-view: `assets/view_0/`, `assets/view_1/`, `assets/view_2/`
  - with `--video`, four MP4 files are written:
    - stitched: `episode_concat.mp4`
    - per-view: `episode_view_0.mp4`, `episode_view_1.mp4`, `episode_view_2.mp4`

Example (fixed tri-view):

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

By default the MP4 playback frame rate is `1/frame_dt` (so it matches the recorded episode timeline). You can override playback FPS with:

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

The recorder:

- loads a MuJoCo articulated model (MJCF XML)
- in `track` mode: finds the first hinge/slide joint and applies tracking control with random perturbation
- in `free` mode: only sets initial state (e.g. `--initial-qvel`) and then applies no external force during rollout
  - optional: `--random-initial-qpos` randomizes each controlled joint position within its limits at reset (seeded by `--seed`)
  - optional: `--auto-initial-qvel-from-limits` infers each controlled joint's initial velocity direction and a conservative impulse magnitude from its limits and initialized qpos
  - optional: `--auto-initial-qvel-min-abs` sets the minimum inferred speed floor in auto mode
  - optional: `--auto-initial-qvel-max-abs` sets the maximum inferred speed to avoid violent limit overshoot in auto mode
  - optional: `--auto-initial-qvel-direction-mode` can force inferred direction: `away-from-qpos0` (default), `toward-lower`, or `toward-upper`
- renders RGB (`.ppm` or `.png`) and depth (`.pgm` 16-bit millimeters, or 16-bit `.png`)
- writes an episode at `outputs/recordings/<object-id>/episode.json`

### From Recording To Fused Point Cloud

Recommended flow: record with per-view masks enabled, then fuse, then inspect in the HTML viewer.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 4 \
  --fps 20 \
  --camera-mode triview \
  --camera-triview-spacing-deg 90 \
  --segmentation-masks \
  --part-segmentation-masks

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/door-mj-001/episode.json \
  --output-dir outputs/recordings/door-mj-001/pointcloud_4d \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/door-mj-001/pointcloud_4d/fusion_manifest.json
```

This writes a self-contained viewer to:

- `outputs/recordings/door-mj-001/pointcloud_4d/viewer.html`

Open that HTML file in a browser to inspect the fused 4D point cloud. Useful checks:

- `Current`: inspect a single time slice without temporal accumulation
- `Current + Static`: compare the current frame against the persistent/static structure
- `Static Only`: check whether the fused static geometry is sensible before debugging motion
- `Front / Side / Top`: switch to orthographic presets when perspective view is hard to interpret
- keep `ghost history` low at first; a large value can make misalignment look worse than it is

### Part Segmentation + Visualize

If you want the fused point cloud to carry MuJoCo-prior part labels and have the viewer color by part, use this path:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-partseg-001 \
  --output-dir outputs/recordings \
  --duration-s 4 \
  --fps 20 \
  --camera-mode triview \
  --camera-triview-spacing-deg 90 \
  --segmentation-masks \
  --part-segmentation-masks

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/door-mj-partseg-001/episode.json \
  --output-dir outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/fusion_manifest.json
```

This produces:

- indexed per-part masks in the episode manifest: `part_mask_path` / `part_mask_paths_by_view`
- a fused `pointcloud_4d.ply` with an extra `part_id` vertex property
- a `fusion_manifest.json` with `part_segmentation`, `part_ids_present`, and `part_point_counts`
- an HTML viewer that defaults to `Color: Part` when part labels are present

If the episode was already recorded, you can add part masks afterward and then rebuild the viewer:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp render-mujoco-masks \
  outputs/recordings/door-mj-001/episode.json \
  --part-segmentation-masks \
  --output-episode outputs/recordings/door-mj-001/episode_partseg.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/door-mj-001/episode_partseg.json \
  --output-dir outputs/recordings/door-mj-001/pointcloud_4d_partseg \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/door-mj-001/pointcloud_4d_partseg/fusion_manifest.json
```

Viewer checks for part segmentation:

- `Color: Part`: color points by `part_id`
- `Color: Height`: fall back to geometric coloring when you want to ignore labels
- left-side legend: confirm which part ids are present and how many fused points each part owns
- `Static Only` plus `Color: Part`: useful for checking whether the static base and moving links were separated cleanly

Notes:

- If the asset has problematic self-collision from visual meshes, add `--disable-target-mesh-collision` to `record-mujoco`.
- If the asset contains visual helper meshes such as `*_Clear` door/glass panels, add `--hide-clear-meshes` to keep those surfaces out of RGB/depth/mask rendering.
- If you add `--part-segmentation-masks`, the recorder also writes indexed per-part masks using MuJoCo body/geom priors.
- The fused `pointcloud_4d.ply` will then contain an extra `part_id` vertex property, and `fusion_manifest.json` will include the `part_segmentation` spec plus per-part point counts.
- If you already recorded an episode without masks, render them afterward and then re-fuse:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp render-mujoco-masks \
  outputs/recordings/door-mj-001/episode.json \
  --part-segmentation-masks

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/door-mj-001/episode.json \
  --output-dir outputs/recordings/door-mj-001/pointcloud_4d_masked \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/door-mj-001/pointcloud_4d_masked/fusion_manifest.json
```

Then run the normal Stage-2 scaffold pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run outputs/recordings/door-mj-001/episode.json --output-dir outputs/door_mj_run
```

## Replacement points for the real research stack

- Replace `GenerationFirstReconstructor` with:
  - multi-view RGB-D object extraction
  - canonical mesh generation
  - depth reprojection verification and cleanup
- Replace `CategoryPriorParticulateAdapter` with:
  - PARTICULATE execution on canonical mesh
  - post-processing to legal kinematic tree
- Replace `SlidingWindowRefitter` with:
  - dense RGB-D / point cloud alignment
  - joint axis/origin/limit optimization
  - temporal window optimization over part poses and `q`

