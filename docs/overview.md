# Overview

## Purpose

This repository implements a runnable research scaffold for the current MVP:

- `sim RGB-D + camera poses -> canonical mesh`
- `canonical mesh -> category-prior articulation init`
- `temporal refit -> q, qdot`
- `articulation -> kinematic + collision URDF`

The implementation is scaffold-first. It defines the artifact contracts, keeps clean replacement boundaries for real `3D generation` and `PARTICULATE` backends, and provides default heuristic backends for `door` and `drawer` so the full pipeline is executable now.

## Implemented

- Episode schema for RGB-D sequences with camera poses and action logs
- Generation-first reconstruction scaffold with low-support cleanup
- Single-DOF articulation init with `door -> revolute` and `drawer -> prismatic`
- Sliding-window temporal refit with smoothed `q` and derived `qdot`
- URDF export with visual meshes, collision box proxies, nominal inertial placeholders, and an MJCF stub
- MuJoCo recording utilities
- Multi-view RGB-D fusion into time-indexed 4D point clouds
- MuJoCo-prior part masks and part-labeled pointcloud export
- HTML pointcloud viewer
- Unit tests for schema loading, temporal smoothing, recording/parser behavior, fusion, and visualization

## Current Placeholders

- Real RGB-D fusion / generative 3D model inference
- Real PARTICULATE invocation
- Dense RGB-D alignment for joint refit
- Real SAPIEN runtime integration

The code is structured so these backends can replace the current heuristics without changing the main pipeline contracts.

## Layout

- `src/rgbd_urdf_mvp/models.py`: public data contracts
- `src/rgbd_urdf_mvp/reconstruction.py`: generation-first reconstruction scaffold
- `src/rgbd_urdf_mvp/articulation.py`: articulation init adapter
- `src/rgbd_urdf_mvp/refit.py`: temporal smoothing and `qdot` extraction
- `src/rgbd_urdf_mvp/urdf.py`: URDF export
- `src/rgbd_urdf_mvp/pipeline.py`: end-to-end orchestration
- `src/rgbd_urdf_mvp/mujoco_recorder.py`: MuJoCo recording and mask rendering
- `src/rgbd_urdf_mvp/pointcloud_fusion.py`: RGB-D to fused 4D pointcloud
- `src/rgbd_urdf_mvp/pointcloud_viz.py`: HTML viewer generation
- `examples/episodes/*`: sample door and drawer episodes
- `examples/mujoco_models/*`: sample MJCF assets

## Core Commands

Validate an episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp validate-episode examples/episodes/door/episode.json
```

Run the scaffold pipeline:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/door/episode.json --output-dir outputs/door_demo
PYTHONPATH=src python3 -m rgbd_urdf_mvp run examples/episodes/drawer/episode.json --output-dir outputs/drawer_demo
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
