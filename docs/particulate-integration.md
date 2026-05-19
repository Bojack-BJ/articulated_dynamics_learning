# PARTICULATE Integration

This repository tracks `git@github.com:Bojack-BJ/particulate.git` as the
`Particulate/` submodule. The intended role is a mesh-first articulation
baseline that can be compared with the RGB-D tracking pipeline:

```text
Hunyuan3D or reconstruction mesh
  -> PARTICULATE mesh articulation
  -> segmented parts + predicted joints + URDF/MJCF + animated GLB

MuJoCo/RGB-D recording
  -> part tracking + pose tracks
  -> joint_inference.json
```

## Setup

Initialize submodules:

```bash
git submodule update --init --recursive Particulate
```

PARTICULATE is CUDA-oriented and should usually run in its own GPU environment.
Its upstream `infer.py` currently requires `torch.cuda.is_available()`.
Install that environment from the submodule docs/requirements on the GPU host,
then point this project at that Python:

```bash
export PARTICULATE_PYTHON=/path/to/particulate-env/bin/python
```

## Hunyuan3D To PARTICULATE

Generate a mesh with Hunyuan3D:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp hunyuan3d-generate \
  --server-url https://your-generation-endpoint.example.com \
  --image path/to/front.png path/to/left.png path/to/right.png \
  --image-views front left right \
  --output outputs/generated/microwave_candidate.glb \
  --mode async
```

Run PARTICULATE on that mesh:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp particulate-infer \
  --mesh outputs/generated/microwave_candidate.glb \
  --output-dir outputs/particulate/microwave_candidate \
  --python-bin "$PARTICULATE_PYTHON"
```

The adapter writes:

```text
outputs/particulate/<object-id>/
  particulate_result.json
  mesh_parts_with_axes_<timestamp>.glb
  animated_textured_<timestamp>.glb
  urdf_<timestamp>/model.urdf
  mjcf_<timestamp>/model.xml
  eval/pred.npz
```

Use `--dry-run` to check command construction and paths without launching the
GPU model.

There is also a starter config at:

```text
configs/particulate_hunyuan_example.yaml
```

## Reconstruction Artifact To PARTICULATE

The same command can consume this project's `reconstruction_artifact.json`:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp particulate-infer \
  --reconstruction-artifact outputs/run/reconstruction_artifact.json \
  --output-dir outputs/particulate/from_reconstruction \
  --python-bin "$PARTICULATE_PYTHON"
```

## Compare With Tracking

After the tracking pipeline has produced `joint_inference.json`, compare the
structural summaries:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp compare-articulation-backends \
  outputs/recordings/<object-id>/pointcloud_4d_partseg/joint_inference.json \
  outputs/particulate/<object-id>/particulate_result.json
```

This writes `articulation_backend_comparison.json` next to the PARTICULATE
manifest. The current comparison is intentionally conservative: it reports part
counts, hierarchy edge counts, revolute/prismatic counts, and tracking joint
axis/limit summaries. It does not yet solve cross-method part correspondence or
axis alignment in a shared frame.
