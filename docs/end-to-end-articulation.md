# End-to-End Articulation Tutorial

This tutorial runs the current simulation-first articulation pipeline:

```text
MuJoCo model
  -> RGB-D recording with object and part masks
  -> fused 4D pointcloud with part_id labels
  -> per-part 6D pose tracks
  -> joint type / axis / pivot inference
  -> inferred articulation URDF/MJCF package
```

The current part segmentation source is MuJoCo-prior mesh segmentation rendered
as masks during recording. This is an oracle/debug path for bootstrapping the
pipeline. It is not yet the final learned visual segmentation module.

## 1. Prepare A Model

If you already have an MJCF/XML model, use it directly:

```bash
MODEL_XML=examples/mujoco_models/Refrigerator031.xml
OBJECT_ID=refrigerator031_e2e
CATEGORY=refrigerator
```

If you start from a USD asset, convert it first:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp.usd_to_mjcf \
  Lightwheel/Refrigerator031/Refrigerator031.usd \
  --category refrigerator
```

Then point `MODEL_XML` at the generated MJCF file:

```bash
MODEL_XML=examples/mujoco_models/Refrigerator031.xml
```

## 2. Record RGB-D With Object And Part Masks

Record three fixed views, object masks, and part masks:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp record-mujoco "$MODEL_XML" \
  --category "$CATEGORY" \
  --object-id "$OBJECT_ID" \
  --output-dir outputs/recordings \
  --duration-s 4 \
  --fps 20 \
  --control-mode free \
  --auto-initial-qvel-from-limits \
  --auto-initial-qvel-direction-mode away-from-qpos0 \
  --auto-initial-qvel-min-abs 1.0 \
  --auto-initial-qvel-max-abs 2.0 \
  --perturbation-scale 0.0 \
  --camera-distance 1.2 \
  --camera-mode triview \
  --camera-triview-spacing-deg 90 \
  --lookat 0.0 0.0 0.0 \
  --segmentation-masks \
  --part-segmentation-masks \
  --video
```

Outputs are written under:

```text
outputs/recordings/<object-id>/
```

The key file for the next step is:

```text
outputs/recordings/<object-id>/episode.json
```

Make sure `<object-id>` matches the `--object-id` used during recording. If a
command fails with a path ending in `episode.json `, remove the trailing space
inside the quoted path.

## 3. Fuse RGB-D Into A Part-Labeled 4D Pointcloud

Fuse all frames and views into a time-indexed pointcloud:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  "outputs/recordings/$OBJECT_ID/episode.json" \
  --output-dir "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg" \
  --pixel-stride 8 \
  --voxel-size-m 0.02 \
  --no-shared-pose-fallback
```

Expected outputs:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/fusion_manifest.json
outputs/recordings/<object-id>/pointcloud_4d_partseg/pointcloud_4d.ply
outputs/recordings/<object-id>/pointcloud_4d_partseg/frames/*.ply
```

Because the recording used `--part-segmentation-masks`, the fused PLY carries a
`part_id` field for each point.

## 4. Visualize The Part-Labeled 4D Pointcloud

Build the HTML viewer:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/fusion_manifest.json"
```

Open:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/viewer.html
```

Recommended checks:

- use `Color: Part` to verify the moving part and base part are separated
- use `Current + Static` to inspect alignment over time
- reduce `ghost history` when debugging a single frame
- if the object is polluted by helper geometry, re-record with `--hide-clear-meshes`

## 5. Estimate Per-Part 6D Pose Tracks

The default lightweight path fits one PCA frame per visible part pointcloud:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/fusion_manifest.json" \
  --method pca \
  --min-points-per-part 32
```

For thin or symmetric parts, prefer the CoTracker path. It tracks part-mask
seed pixels in RGB, backprojects visible samples with depth, then fits part
poses from 3D correspondences:

```bash
TORCH_HOME="$PWD/.cache/torch" PYTHONPATH=src python3 -m rgbd_urdf_mvp track-part-pixels \
  "outputs/recordings/$OBJECT_ID/episode.json" \
  --output-json "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/part_tracks.json" \
  --cotracker-repo ./co-tracker \
  --cotracker-checkpoint ./co-tracker/ckpt/scaled_offline.pth \
  --device auto \
  --reference-frame 0 \
  --seed-stride-px 16

PYTHONPATH=src python3 -m rgbd_urdf_mvp estimate-part-poses \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/part_tracks.json" \
  --method tracks \
  --output-json "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/part_poses.json" \
  --min-tracks-per-part 4
```

This writes:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/part_poses.json
```

The artifact contains:

- canonical reference pose per part
- per-frame world pose per part
- per-frame pose relative to the selected anchor/static part
- point counts and confidence values
- simple motion summaries

## 6. Infer Joint Type, Axis, Pivot, And Limits

Infer a first-pass joint model:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp infer-joints \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/part_poses.json" \
  --mujoco-prior off \
  --rotation-threshold-rad 0.2 \
  --translation-threshold-m 0.02
```

This writes:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/joint_inference.json
```

The artifact contains:

- `joint_type`: currently `revolute`, `prismatic`, `fixed`, or low-confidence fallback
- `axis`: estimated axis direction in the anchor frame
- `pivot`: estimated point on the revolute axis or prismatic line
- `limits`: scalar motion range from the observed trajectory
- `q_samples`: per-frame generalized coordinate estimates

## 7. Visualize Joint Overlays

Rebuild the viewer after `part_poses.json` and `joint_inference.json` exist:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/fusion_manifest.json"
```

The viewer auto-loads sibling pose and joint artifacts. Enable `Show Joints` to
inspect the inferred axis and pivot on top of the current pointcloud frame.

## 8. Export Inferred Articulation

Export a URDF/MJCF package from the inferred pose and joint artifacts:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp export-inferred-articulation \
  "outputs/recordings/$OBJECT_ID/episode.json" \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/part_poses.json" \
  "outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/joint_inference.json"
```

Expected outputs:

```text
outputs/recordings/<object-id>/pointcloud_4d_partseg/inferred_articulation/
  articulation_artifact.json
  pipeline_result.json
  urdf/<object-id>.urdf
  urdf/<object-id>.mjcf.xml
  urdf/articulation.json
```

## Current Scope And Limitations

- This tutorial covers the implemented MuJoCo/simulation path.
- Part segmentation is currently rendered from known MuJoCo mesh information.
- The Hunyuan3D generation server path is documented separately in
  [remote-generation.md](remote-generation.md).
- Hunyuan3D/PARTICULATE-based mesh-to-URDF is not yet wired into this full
  pointcloud trajectory tutorial.
- Dynamics identification is not part of this tutorial; it starts after URDF,
  `q`, and `qdot` are available.
