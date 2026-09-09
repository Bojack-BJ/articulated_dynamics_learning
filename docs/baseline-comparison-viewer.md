# Baseline Comparison Viewer

The baseline viewer converts GT, Hybrid, AiM, and ReArt artifacts into one
method-independent point schema. It is intended for diagnostic visualization;
GT nearest-neighbor mappings are never fed back into baseline inference.

## Build

```bash
PYTHONPATH=src python scripts/build_baseline_comparison_viewer.py \
  path/to/reference_frame.ply \
  --output-html outputs/example/viewer_baseline_comparison.html \
  --hybrid-tracks path/to/motion_part_tracks_slots.json \
  --hybrid-metrics path/to/hybrid_metrics.json \
  --aim-run path/to/aim_run \
  --aim-metrics path/to/aim_metrics.json \
  --reart-prediction path/to/reart_prediction.npz \
  --reart-metrics path/to/reart_metrics.json \
  --gt-frames-dir path/to/dense_fusion/frames \
  --timeline-steps 31 \
  --axis-remap x,z,-y \
  --max-points 20000
```

Only supplied methods are shown. Missing optional fields disable the
corresponding color modes instead of failing.

All methods are resampled onto a common normalized timeline. ReArt's four
native poses are interpolated with per-part rigid transforms and quaternion
SLERP; the UI still identifies the source as a four-frame method. GT geometry
can be loaded from dense per-frame RGB-D fusion and displayed as a translucent
overlap layer for every prediction.

The generated directory contains:

```text
viewer_baseline_comparison.html
viewer_data/
  gt.json
  hybrid.json
  aim.json
  reart.json
  metrics.json
```

The HTML embeds all point and trajectory data. Plotly is embedded when the
Python `plotly` package is available; otherwise the same CDN loader used by the
existing enhanced viewer is retained.

## AiM Layers

The AiM adapter exposes Gaussian centers as points. It supports:

- dynamic/static labels;
- accepted pre-merge RANSAC components;
- post-merge components;
- trajectories at the available exported motion times;
- deformation magnitude and component-level rigid residual;
- opacity and Gaussian scale when present in the PLY;
- GT-only nearest-neighbor mapping with unmatched points retained.

The stage selector filters the dynamic and accepted pre-merge sets. The
Storage-47648 small-part preset restricts the display to GT parts 6, 7, and 8.

## Sampling and Metrics

`--max-points` applies deterministic sampling only to rendered points and
trajectories. Overlap matrices, per-part statistics, and evaluator metrics are
computed from the full adapted artifact. The UI reports both counts.

Main IoU, ARI, RI, and coverage values are loaded from the existing evaluator
JSON. The browser does not recompute publication metrics.

## Dense AiM Trajectories

AiM motion-stage artifacts normally contain only three diagnostic snapshots
(`t=0`, `0.5`, and `1`). Export a denser trajectory from the frozen deformation
checkpoint without rerunning optimization:

```bash
python scripts/export_aim_dense_trajectory.py path/to/aim_run \
  --aim-root path/to/AiM \
  --frames 31 \
  --is-blender \
  --sh-degree 0
```

When `dense_trajectory.npz` exists in the AiM run directory, the viewer uses it
instead of the three sparse snapshots.

The ReArt adapter accepts the official `result.pkl` directly. It replays the
canonical point labels with ReArt's per-part predicted transforms rather than
connecting independently resampled point clouds by point index.

If official ReArt fails before producing `result.pkl`, pass
`--reart-sequence-manifest` and `--reart-status`. The method remains visible,
but is explicitly labeled as failed and shows only its four native input point
clouds. No segmentation labels are fabricated.

The viewer uses one fixed cubic world-space bound for all methods and all
timesteps. Point size, point opacity, color palette, and GT overlap are
adjustable without resetting the camera.

## Remote Gallery

Large self-contained viewers can stay on the development machine:

```bash
python scripts/build_baseline_viewer_index.py outputs/baseline_viewers_v2
cd outputs/baseline_viewers_v2
python -m http.server 8765 --bind 127.0.0.1
```

Forward port `8765` with VS Code's **Ports** panel and open the forwarded URL.
The gallery switches objects; each object viewer independently switches among
GT, Ours/Hybrid, AiM, and ReArt when those artifacts are available.

The current development-machine gallery is served from:

```text
/lumos-vePFS/suzhou/Users/lixiaotong/articulated_dynamics_learning/
  outputs/baseline_viewers_v2
```

Run `scripts/validate_baseline_viewer_gallery.py` after generation to verify
method completeness, timeline length, GT density, fixed cubic bounds, and UI
controls.
