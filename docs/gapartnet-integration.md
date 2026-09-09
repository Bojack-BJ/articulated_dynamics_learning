# GAPartNet Integration

GAPartNet provides cleaned PartNet-Mobility and AKB-48 assets with GAPart
semantics, link poses, and annotated URDF kinematic trees. It is useful for
kinematic and perception experiments, but its assets are not directly valid
MuJoCo dynamics models: the example URDFs omit inertial parameters, and some
thin collision meshes fail MuJoCo convex-hull compilation.

## Convert an asset

The adapter preserves original OBJ meshes as non-colliding visual geometry and
adds link-local bounding-box collision proxies. Mass and diagonal inertia are
initialized from those proxies and are not ground-truth dynamic parameters.

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp convert-gapartnet-mjcf \
  /path/to/GAPartNet/object_id \
  --output-mjcf outputs/gapartnet/object_id/model.xml
```

The command also writes `model.conversion.json`, which records the source URDF,
proxy count, density, and approximation warnings.

The resulting model can be passed to the existing recorder:

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp record-mujoco \
  outputs/gapartnet/object_id/model.xml \
  --category refrigerator \
  --object-id gapartnet_object_id \
  --output-dir outputs/recordings \
  --frames 30 \
  --camera-mode triview \
  --segmentation-masks \
  --part-segmentation-masks
```

## Prepare the five-object pilot

The downloaded dataset is a large ZIP archive. Do not extract all 511,000 files
for initial experiments. Selectively extract the default pilot (one microwave,
refrigerator, dishwasher, hinged cabinet, and sliding cabinet), convert each
asset to MJCF, and generate a batch manifest:

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp prepare-gapartnet-recordings \
  /root/Users/lixiaotong/datasets/GAPartNet_PhyScene/partnet_mobility_part.zip \
  --output-dir outputs/gapartnet_pilot
```

The output contains:

```text
outputs/gapartnet_pilot/
  assets/<object-id>/
  models/gapartnet_<object-id>.xml
  catalog.json
  recording_batch.tsv
```

Run the existing end-to-end batch pipeline with the compact pilot recording
preset:

```bash
PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp run-articulation-batch \
  outputs/gapartnet_pilot/recording_batch.tsv \
  --record-config configs/record_gapartnet_pilot.yaml \
  --track-config configs/track_gapartnet_pilot.yaml \
  --output-root outputs/gapartnet_recordings \
  --track-device cuda \
  --jobs 1
```

The pilot tracking preset seeds frame 10 rather than the closed first frame.
For the microwave pilot, the closed door is nearly coplanar with the cabinet
and contributes too few visible segmentation pixels at frame 0. The denser
8-pixel seed grid and middle reference frame preserve enough door tracks for
the downstream pose and joint stages. Use `--track-device mps` on Apple Silicon
or omit the override to use the YAML's automatic device selection.

For a headless Linux machine, set `MUJOCO_GL=egl` when running the batch. The
pilot MJCF models use collision proxies for loading and inertial initialization,
but recording disables gravity and target collision so overlapping box proxies
cannot eject a closed articulated link before the commanded motion starts.

Use repeated `--object-id` arguments to prepare a different subset. The
preparer intentionally requires exactly one movable URDF joint per selected
asset for the pilot.

The batch command ends after generating `viewer_pose_flow.html`. Evaluate the
simulation result and export the inferred kinematic package with:

```bash
OBJECT_DIR=outputs/gapartnet_recordings/gapartnet_microwave_7304

PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp evaluate-kinematic-model \
  "$OBJECT_DIR/pointcloud_4d_partseg/joint_inference.json" \
  --part-poses "$OBJECT_DIR/pointcloud_4d_partseg/part_poses.json"

PYTHONPATH=src ./.venv/bin/python -m rgbd_urdf_mvp export-inferred-articulation \
  "$OBJECT_DIR/episode.json" \
  "$OBJECT_DIR/pointcloud_4d_partseg/part_poses.json" \
  "$OBJECT_DIR/pointcloud_4d_partseg/joint_inference.json"
```

The validated `7304` smoke test produced one revolute joint with full GT joint
coverage, a 7.82-degree axis error, a 0.059 m revolute-axis line-distance
error, and a 0.027 rad mean joint-limit error. The exported MJCF loads in
MuJoCo. These numbers verify plumbing rather than establish a benchmark; the
offset-aligned q RMSE remains 0.378 rad and needs temporal refit work.

## Development-machine data layout

The current development-machine layout is:

```text
/root/Users/lixiaotong/datasets/
  GAPartNet_examples/
    102442/
    45780/
  GAPartNet_PhyScene/
```

The full dataset is gated under CC BY-NC 4.0. After accepting the dataset terms
and configuring a Hugging Face token on the development machine, download it
with:

```bash
huggingface-cli download \
  yangyandan/GAPartNet_PhyScene \
  --repo-type dataset \
  --local-dir /root/Users/lixiaotong/datasets/GAPartNet_PhyScene
```

## Relationship to SAPIEN PartNet-Mobility

The full SAPIEN PartNet-Mobility v0 archive is stored at:

```text
/root/Users/lixiaotong/datasets/PartNet-Mobility/partnet-mobility-v0.zip
```

Compare it without extracting either archive:

```bash
python3 scripts/compare_partnet_archives.py \
  /root/Users/lixiaotong/datasets/PartNet-Mobility/partnet-mobility-v0.zip \
  /root/Users/lixiaotong/datasets/GAPartNet_PhyScene/partnet_mobility_part.zip \
  --output-json outputs/gapartnet_pilot/sapien_overlap.json
```

The July 2026 inventory comparison found 2,347 SAPIEN objects and 1,045
GAPartNet PartNet-Mobility objects. Every GAPartNet object ID occurs in SAPIEN,
while 1,302 SAPIEN objects are outside the GAPartNet subset. Categories agree
for all shared IDs. Core files are also almost identical: all `meta.json`
files, 1,044/1,045 `mobility.urdf` files, and all 1,044 shared
`semantics.txt` files are byte-identical. GAPartNet should therefore be treated
as an annotated and remeshed subset of SAPIEN PartNet-Mobility. Prefer GAPartNet
for its enhanced annotations where available, and use the SAPIEN-only objects
to expand object/category coverage.
