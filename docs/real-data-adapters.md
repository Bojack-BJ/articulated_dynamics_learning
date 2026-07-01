# Real Data Adapters

The core pipeline consumes `episode.json` recordings. Real robot datasets often
arrive as ROS bag exports instead, so import them once and run the existing
steps on the converted episode.

## RBO / ROSbag-Like RGB-D Exports

Expected source layout:

```text
sequence_name/
  camera_rgb/*.png
  camera_depth_registered/*.txt
  camera_depth_registered_camera_info.csv
  *_joint_states.csv                 # optional
```

Depth text files are interpreted as meters and converted to 16-bit PNG depth in
millimeters. RGB frames are synchronized to the nearest registered depth frame
by the timestamp embedded in filenames such as `000124-1501686144.964.png`.
Camera intrinsics are read from `camera_depth_registered_camera_info.csv`.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp import-rbo-recording \
  real_data/microwave01_o \
  --output-dir outputs/real_recordings/microwave01_o \
  --category microwave \
  --target-fps 20 \
  --max-frames 120
```

`--target-fps` resamples the source RGB stream onto a nominal time grid before
writing `timestamp_s`. For example, `--target-fps 20` writes timestamps near
`0.00, 0.05, 0.10, ...`, which makes real recordings easier to compare against
MuJoCo recordings captured at the same frame rate. `--frame-stride` still works
as a cheap prefilter before resampling.

The output is a normal project recording:

```text
outputs/real_recordings/microwave01_o/
  episode.json
  assets/view_0/frame_0000_rgb.png
  assets/view_0/frame_0000_depth.png
```

Generate masks with SAM2/SAM3/another provider before running pointcloud fusion
or Hunyuan image preparation:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider sam2 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --sam2-prompt-mode center-box \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2.json
```

For the “segment once, verify, then propagate” workflow, run SAM2 only on the
reference frame:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider sam2 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --sam2-prompt-mode center-box \
  --mask-kind object \
  --start-frame 0 \
  --max-frames 1 \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2-first.json
```

After visual/manual verification of the first mask, propagate it through the
episode with SAM2 video:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/microwave01_o/episode.sam2-first.json \
  --backend sam2-video \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --reference-frame 0 \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2-video.json
```

SAM2 video propagation uses the reference mask as a dense conditioning mask and
tracks masklets through the video. The older `--backend cotracker-sparse`
fallback remains available for debugging, but it rasterizes sparse tracks and
is not the preferred mask path.

This object-mask workflow is not part segmentation. To obtain part-labeled
pointclouds, first write an indexed first-frame part mask with
`segment-episode-masks --mask-kind part` from a manual label, PartSAM, or an
external part-segmentation wrapper, then propagate it:

For interactive manual labels and mid-episode corrections, use the local
annotation UI in `docs/partmask-annotation.md`.

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider mask-dir \
  --mask-dir outputs/real_recordings/microwave01_o/reference_part_masks \
  --mask-kind part \
  --start-frame 0 \
  --max-frames 1 \
  --output-episode outputs/real_recordings/microwave01_o/episode.part-first.json

PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/microwave01_o/episode.part-first.json \
  --backend sam2-video \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --reference-frame 0 \
  --mask-kind part \
  --output-episode outputs/real_recordings/microwave01_o/episode.part-sam2-video.json
```

Use `episode.part-sam2-video.json` for `fuse-pointcloud` when you need
`part_id` in the pointcloud.

## Batch Import And Masking

Batch RBO conversion uses a TSV manifest. See
`configs/batch_rbo_recordings_example.tsv`:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp import-rbo-recordings-batch \
  configs/batch_rbo_recordings_example.tsv \
  --output-root outputs/real_recordings \
  --target-fps 20 \
  --jobs 4 \
  --force
```

The command writes `outputs/real_recordings/imported_rbo_episodes.tsv`, which
can be used as the mask batch manifest after renaming or copying the
`episode_path` column.

Batch mask generation uses `configs/batch_episode_masks_example.tsv`:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks-batch \
  configs/batch_episode_masks_example.tsv \
  --provider sam2 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --start-frame 0 \
  --max-frames 1 \
  --jobs 1 \
  --force
```

Keep SAM2 batch jobs at `1` on a single Mac GPU unless profiling shows the
device is underutilized. Multiple MPS workers usually contend for the same
memory bandwidth.

Then use the standard pipeline commands:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode \
  outputs/real_recordings/microwave01_o/episode.sam2.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/real_recordings/microwave01_o/episode.sam2.json \
  --output-dir outputs/real_recordings/microwave01_o/pointcloud
```

`--mask-mode depth-near` still exists as a quick diagnostic fallback, but it is
not a semantic object segmentation model and should not be used for serious
real-data evaluation. Use `segment-episode-masks` with SAM2/SAM3/manual masks
instead.

The importer uses an identity camera pose in the camera frame. That is enough
for single-camera temporal tracking, but cross-camera fusion needs calibrated
camera poses or a dataset-specific TF adapter.
