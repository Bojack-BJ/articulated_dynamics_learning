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
  --frame-stride 2 \
  --max-frames 120 \
  --mask-mode depth-near
```

The output is a normal project recording:

```text
outputs/real_recordings/microwave01_o/
  episode.json
  assets/view_0/frame_0000_rgb.png
  assets/view_0/frame_0000_depth.png
  assets/view_0/frame_0000_mask.png
```

Then use the standard pipeline commands:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode \
  outputs/real_recordings/microwave01_o/episode.json

PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/real_recordings/microwave01_o/episode.json \
  --output-dir outputs/real_recordings/microwave01_o/pointcloud
```

`--mask-mode depth-near` is a simple foreground heuristic based on near depths
in the center crop. It is useful for quick generation-image crops, but it is not
a semantic object segmentation model. For tracking evaluation, prefer replacing
the masks with manually curated or model-generated masks when possible.

The importer uses an identity camera pose in the camera frame. That is enough
for single-camera temporal tracking, but cross-camera fusion needs calibrated
camera poses or a dataset-specific TF adapter.
