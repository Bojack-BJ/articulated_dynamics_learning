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

## Notes

- If visual meshes self-collide badly, add `--disable-target-mesh-collision`
- If helper meshes such as `*_Clear` pollute depth and masks, add `--hide-clear-meshes`
- `--no-shared-pose-fallback` is useful when you want fusion to fail rather than silently reuse a central pose
- Be careful with shell line continuations: `\` must be the last character on the line
