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

`validate-episode` 是一个轻量 schema check，用来在真正跑 pipeline 前先排掉明显坏输入。它目前会检查：

- `category` 是否在当前支持列表里
- episode 是否至少有一帧
- 每帧 `camera_pose` 和 `camera_poses_by_view` 是否是 `4x4`
- `mask_path` / `part_mask_path` / `*_paths_by_view` 的类型是否正确
- `timestamp_s` 是否非负

它不做几何正确性验证，也不会判断点云/深度是否真的对齐。它的作用是尽早发现 manifest 结构错误，避免后面的 fusion / pose / URDF 流程在更深的位置才报错。

Run the scaffold pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo
```

Pipeline path switch:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo_ff --path feedforward
```

`run` 现在支持两条 path：

- `feedforward`
  - `reconstruction -> articulation init -> export`
  - 用来快速跑通形状到 URDF 的前向链路
- `optimization`
  - `reconstruction -> articulation init -> temporal refit -> export`
  - 当前默认路径，也是目前项目里更完整、主要在走的实现

也就是说，目前这个 scaffold 的主线更偏 `optimization-based path`，因为它会在 articulation init 之后继续做时序 refit，显式产出更稳定的 `q, qdot`。

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

同一个机制也适用于参数更多的命令，比如 `record-mujoco`：

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

然后直接运行：

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/record_refrigerator.yaml
```

如果还想临时覆盖一两个字段，也可以在配置文件后面继续跟普通 CLI 参数；后写的参数会覆盖前面的配置值。

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

`src/rgbd_urdf_mvp` 现在按职责拆成了几个子包：

- `core`
  - 数据模型、类别定义、几何 helper、序列化
- `sim`
  - MuJoCo 录制、mask 重渲染、USD/MJCF 转换
- `perception`
  - reconstruction、4D pointcloud fusion、part segmentation、part pose、viewer
- `kinematics`
  - articulation init、temporal refit、joint inference、inferred articulation export
- `export`
  - URDF / MJCF 导出

根目录保留少量顶层入口：

- [cli.py](src/rgbd_urdf_mvp/cli.py): 命令行入口
- [pipeline.py](src/rgbd_urdf_mvp/pipeline.py): scaffold end-to-end pipeline
- [__init__.py](src/rgbd_urdf_mvp/__init__.py): 对外稳定导出 `RGBDToURDFPipeline` 和 `load_episode`

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
