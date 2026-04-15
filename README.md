# Articulated Dynamics Learning From RGB-D

This repository contains a runnable scaffold for:

- `sim RGB-D + camera poses -> canonical mesh`
- `canonical mesh -> articulation init`
- `temporal refit -> q, qdot`
- `articulation -> kinematic + collision URDF`

The current code is scaffold-first. It is meant to keep the full contract executable now, while leaving clean replacement points for real reconstruction, PARTICULATE, and dynamics-identification backends.

## Start Here

- Project overview: [docs/overview.md](docs/overview.md)
- MuJoCo recording and USD-to-MJCF conversion: [docs/mujoco-recording.md](docs/mujoco-recording.md)
- RGB-D fusion, part segmentation, and viewer workflow: [docs/pointcloud.md](docs/pointcloud.md)
- Research-stack replacement points: [docs/research-stack.md](docs/research-stack.md)

## Quick Start

Validate an episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode examples/episodes/door/episode.json
```

Run the scaffold pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo
```

Fuse and inspect a recorded MuJoCo episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud outputs/recordings/refrigerator031_door_only/episode.json --output-dir outputs/recordings/refrigerator031_door_only/pointcloud_4d_fixed3 --pixel-stride 8 --voxel-size-m 0.02 --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud outputs/recordings/refrigerator031_door_only/pointcloud_4d_fixed3/fusion_manifest.json
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## What Is Implemented

- Episode schema for RGB-D sequences with camera poses and action logs
- Generation-first reconstruction scaffold
- Single-DOF articulation init for `door` and `drawer`
- Sliding-window refit with smoothed `q` and derived `qdot`
- URDF export with collision proxies and MJCF stub
- MuJoCo recording utilities
- Multi-view 4D pointcloud fusion
- MuJoCo-prior part masks and part-aware pointcloud visualization

## What Is Still Placeholder

- Real RGB-D generative reconstruction
- Real PARTICULATE invocation
- Dense joint-aware RGB-D alignment
- Full dynamics identification and contact modeling
