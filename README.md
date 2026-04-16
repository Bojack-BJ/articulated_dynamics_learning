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
- Remote 3D generation client/server setup: [docs/remote-generation.md](docs/remote-generation.md)
- Research-stack replacement points: [docs/research-stack.md](docs/research-stack.md)

## Installation

Clone the repository with the Hunyuan3D server submodule:

```bash
git clone --recurse-submodules git@github.com:Bojack-BJ/articulated_dynamics_learning.git
cd articulated_dynamics_learning
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

Create a local Python environment for the RGB-D to URDF project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[simulation,viz]"
```

Run the smoke tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The `Hunyuan3D-2/` directory is a submodule pointing to this project's fork of
the Hunyuan3D sample server. On the remote GPU machine, install that server in
its own environment:

```bash
cd Hunyuan3D-2
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch first using the wheel that matches your CUDA version.
# See https://pytorch.org/get-started/locally/

python -m pip install -r requirements.txt
python -m pip install -e .
```

Texture generation may need the optional Hunyuan3D rasterizer extensions:

```bash
cd hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python setup.py install
cd ../../..
```

For remote mesh generation, run the Hunyuan3D API server on the GPU machine and
call it from the Mac client. The full setup is documented in
[docs/remote-generation.md](docs/remote-generation.md).

## Quick Start

Validate an episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode examples/episodes/door/episode.json
```

`validate-episode` is a lightweight schema check that catches obviously broken
inputs before the pipeline starts. It currently checks:

- whether `category` is in the supported list
- whether the episode contains at least one frame
- whether each frame `camera_pose` and `camera_poses_by_view` is `4x4`
- whether `mask_path` / `part_mask_path` / `*_paths_by_view` have the expected types
- whether `timestamp_s` is non-negative

It does not validate geometric correctness, and it does not tell you whether
depth, pointclouds, or poses are actually aligned. Its purpose is to catch
manifest-structure errors early so fusion / pose / URDF stages fail less
deeply and more clearly.

Run the scaffold pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo
```

Pipeline path switch:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo_ff --path feedforward
```

`run` currently supports two paths:

- `feedforward`
  - `reconstruction -> articulation init -> export`
  - useful for a fast shape-to-URDF forward pass
- `optimization`
  - `reconstruction -> articulation init -> temporal refit -> export`
  - the current default path and the more complete implementation in this repo

At the moment, the scaffold is primarily centered on the
`optimization-based path`, because it continues after articulation
initialization into temporal refit and explicitly produces more stable `q`
and `qdot`.

## Pipeline Flow

```text
Episode Input
  RGB-D + camera poses + actions
        |
        v
Reconstruction
  canonical mesh / primitives
        |
        v
Articulation Init
  parts + joints + limits
        |
        v
Pipeline Path
  |-- feedforward ---> URDF / MJCF Export
  |
  `-- optimization --> Temporal Refit
                          smooth q, derive qdot
                              |
                              v
                        URDF / MJCF Export
                              |
                              v
Pipeline Result
  articulation artifact + URDF package
```

Use a YAML config instead of long CLI flags:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/run_optimization.yaml
```

Example config:

```yaml
command: run
episode: examples/episodes/door/episode.json
output-dir: outputs/door_demo_yaml
path: optimization
```

The same mechanism also works for commands with many parameters, such as
`record-mujoco`:

```yaml
command: record-mujoco
args:
  model: examples/mujoco_models/Refrigerator031.xml
  category: refrigerator
  object-id: refrigerator031_yaml
  output-dir: outputs/recordings
  duration-s: 4
  fps: 20
  camera-mode: triview
  camera-triview-spacing-deg: 90
  lookat: [0.0, 0.0, 0.0]
  segmentation-masks: true
  part-segmentation-masks: true
```

Then run it directly:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/record_refrigerator.yaml
```

If you want to override one or two fields temporarily, you can still append
normal CLI flags after the config file. Later CLI arguments override values
coming from the config file.

Fuse and inspect a recorded MuJoCo episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp fuse-pointcloud outputs/recordings/refrigerator031_door_only/episode.json --output-dir outputs/recordings/refrigerator031_door_only/pointcloud_4d --pixel-stride 8 --voxel-size-m 0.02 --no-shared-pose-fallback

PYTHONPATH=src python3 -m rgbd_urdf_mvp visualize-pointcloud outputs/recordings/refrigerator031_door_only/pointcloud_4d_fixed3/fusion_manifest.json
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Package Layout

`src/rgbd_urdf_mvp` is now split into a few subpackages by responsibility:

- `core`
  - data models, category definitions, geometry helpers, serialization
- `sim`
  - MuJoCo recording, mask rerendering, USD/MJCF conversion
- `perception`
  - reconstruction, 4D pointcloud fusion, part segmentation, part pose, viewer
- `kinematics`
  - articulation init, temporal refit, joint inference, inferred articulation export
- `export`
  - URDF / MJCF export

The package root keeps a small number of top-level entry points:

- [cli.py](src/rgbd_urdf_mvp/cli.py): command-line entry point
- [pipeline.py](src/rgbd_urdf_mvp/pipeline.py): scaffold end-to-end pipeline
- [__init__.py](src/rgbd_urdf_mvp/__init__.py): stable public exports for `RGBDToURDFPipeline` and `load_episode`

## What Is Implemented

- Episode schema for RGB-D sequences with camera poses and action logs
- Generation-first reconstruction scaffold
- Single-DOF articulation init for `door` and `drawer`
- Sliding-window refit with smoothed `q` and derived `qdot`
- URDF export with collision proxies and MJCF stub
- MuJoCo recording utilities
- Multi-view 4D pointcloud fusion
- MuJoCo-prior part masks and part-aware pointcloud visualization
- Per-part 6D pose estimation from 4D pointclouds
- Joint type/axis/pivot inference from pose tracks
- Inferred-articulation export back into URDF/MJCF
- YAML/JSON driven CLI config expansion for long commands

## What Is Still Placeholder

- Real RGB-D generative reconstruction
- Real PARTICULATE invocation
- Dense joint-aware RGB-D alignment
- Full dynamics identification and contact modeling
