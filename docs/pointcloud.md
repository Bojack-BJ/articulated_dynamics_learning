# Pointcloud Pipeline

## Fuse RGB-D Into 4D Pointcloud

Basic flow:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco path/to/object.xml \
  --category door \
  --object-id door-mj-001 \
  --output-dir outputs/recordings \
  --duration-s 4 \
  --fps 20 \
  --camera-mode triview \
  --camera-triview-spacing-deg 90 \
  --segmentation-masks

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/recordings/door-mj-001/episode.json \
  --output-dir outputs/recordings/door-mj-001/pointcloud_4d \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  outputs/recordings/door-mj-001/pointcloud_4d/fusion_manifest.json
```

Outputs:

- `pointcloud_4d/pointcloud_4d.ply`
- `pointcloud_4d/fusion_manifest.json`
- `pointcloud_4d/viewer.html`
- `pointcloud_4d/frames/*.ply`

## Add Part Segmentation

If you want per-part labels carried into the fused pointcloud:

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

This adds:

- `part_mask_path` and `part_mask_paths_by_view` in the episode
- `part_id` in the fused PLY vertex format
- `part_segmentation`, `part_ids_present`, and `part_point_counts` in the manifest
- part-aware coloring in the HTML viewer

## Estimate Per-Part 6D Poses

There are currently two supported pose estimators:

- `pca`: baseline estimator that fits a PCA frame to each per-frame part pointcloud.
- `tracks`: CoTracker-based estimator that tracks part-mask seed pixels, backprojects them to 3D, and fits a rigid transform from persistent 3D correspondences.

The PCA path only needs a fused pointcloud with `part_id`. It is simple, but thin or symmetric parts can suffer from PCA axis flips:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/fusion_manifest.json \
  --method pca
```

This writes `part_poses.json` next to the manifest by default.

You can also override the output path and the anchor part:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/fusion_manifest.json \
  --output-json outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses_custom.json \
  --anchor-part-id 1 \
  --min-points-per-part 32
```

The CoTracker path uses the original RGB-D episode and part masks. Install the optional tracking dependencies first:

```bash
python -m pip install -e ".[tracking]"
```

On Mac, use `--device auto`, `--device mps`, or `--device cpu`. Do not pass `--device cuda` unless you are on a CUDA machine. The repository already ships with `co-tracker/` as a submodule, so initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

Track part pixels and backproject visible samples into world-frame 3D tracks:

```bash
TORCH_HOME="$PWD/.cache/torch" PYTHONPATH=src python3 -m rgbd_urdf_mvp track-part-pixels \
  outputs/recordings/door-mj-partseg-001/episode.json \
  --output-json outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_tracks.json \
  --cotracker-repo ./co-tracker \
  --cotracker-checkpoint ./co-tracker/ckpt/scaled_offline.pth \
  --device auto \
  --reference-frame 0 \
  --frame-stride 4 \
  --seed-stride-px 16 \
  --max-tracks-per-part-view 128
```

For a `60 Hz` recording, `--frame-stride 4` reduces CoTracker to roughly `15 Hz` while keeping the original episode files unchanged.

Then fit per-part SE(3) trajectories from those 3D correspondences:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_tracks.json \
  --method tracks \
  --output-json outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses.json \
  --anchor-part-id 1 \
  --min-tracks-per-part 4
```

The artifact contains:

- per-part canonical reference frame
- per-frame world pose for each part
- per-frame pose relative to the chosen anchor part
- point counts and confidence values per sample
- simple motion summaries that are useful for the next joint-inference stage

## Infer Joint Type, Axis, And Pivot

Given `part_poses.json`, infer a first-pass joint model for each non-anchor part:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses.json
```

This writes `joint_inference.json` next to `part_poses.json` by default.

When `part_tracks.json`, `part_poses.json`, and `joint_inference.json` are in the same directory as the input manifest, `visualize-pointcloud` now loads them automatically and exposes:

- per-part pose overlays with the pose estimator source
- per-part visibility filters
- CoTracker point-flow overlays
- joint overlays

Optional thresholds:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses.json \
  --rotation-threshold-rad 0.2 \
  --translation-threshold-m 0.02
```

The artifact contains:

- inferred `joint_type`
- anchor-to-child `axis`
- `pivot` point on the inferred joint axis/line
- estimated `limits`
- per-frame scalar `q` samples
- simple confidence and motion metrics

## Export Inferred Articulation To URDF/MJCF

Once `part_poses.json` and `joint_inference.json` exist, build an `articulation_artifact.json` plus URDF/MJCF package:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp export-inferred-articulation \
  outputs/recordings/door-mj-partseg-001/episode.json \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses.json \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/joint_inference.json
```

This writes `inferred_articulation/` next to `joint_inference.json` by default.

You can also override the export directory:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp export-inferred-articulation \
  outputs/recordings/door-mj-partseg-001/episode.json \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/part_poses.json \
  outputs/recordings/door-mj-partseg-001/pointcloud_4d_partseg/joint_inference.json \
  --output-dir outputs/recordings/door-mj-partseg-001/inferred_articulation_custom
```

Outputs:

- `articulation_artifact.json`
- `pipeline_result.json`
- `urdf/<object_id>.urdf`
- `urdf/<object_id>.mjcf.xml`
- `urdf/articulation.json`

## Add Masks After Recording

If the episode already exists:

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

## Viewer Checks

Geometry checks:

- `Current`: inspect one time slice
- `Current + Static`: compare dynamic frame with persistent structure
- `Static Only`: inspect fused static geometry
- `Front / Side / Top`: inspect orthographic projections
- keep `ghost history` small when debugging alignment

Part checks:

- `Color: Part`: color points by `part_id`
- `Color: Height`: ignore labels and color by geometry
- legend: verify which parts are present and how many fused points they own
- `Static Only + Color: Part`: good for separating base vs moving links

Joint checks:

- `Show Joints`: draw the inferred axis line and pivot marker on top of the current frame
- `visualize-pointcloud` auto-loads sibling `part_poses.json` and `joint_inference.json` when they exist
- use `Front / Side / Top + Show Joints` to check whether the axis is actually sitting on the hinge/slider line
- if you need non-default artifact paths, pass `--part-poses-json` and `--joint-inference-json`

## Notes

- If visual meshes self-collide badly, add `--disable-target-mesh-collision`
- If helper meshes such as `*_Clear` pollute depth and masks, add `--hide-clear-meshes`
- `--no-shared-pose-fallback` is useful when you want fusion to fail rather than silently reuse a central pose
- Be careful with shell line continuations: `\` must be the last character on the line
