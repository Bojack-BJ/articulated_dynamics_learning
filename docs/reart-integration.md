# ReArt Remote Integration

ReArt is integrated as a point-cloud baseline. The client exports fused
per-frame point clouds directly into a ReArt sequence and sends that sequence
to a CUDA server. It does not reconstruct a mesh and then resample points.

## Data Flow

```text
local recording pointcloud_4d_partseg/fusion_manifest.json
  -> export-reart-sequence
  -> vertex-only PLY frame sequence
  -> remote-reart-run
  -> dev-h serve_remote_reart.py
  -> ReArt run_real.py through reart_run_wrapper.py
  -> result.pkl, model.pth.tar, result.txt, seg.html, timing/status
  -> zipped back to the local output directory
```

The PLY sequence stores `x y z part_id` vertices. ReArt's original real-data
loader samples mesh surfaces; the wrapper patches that loader so vertex-only
PLY files are sampled directly.

## Export A Sequence

```bash
cd "/Users/lxt/Documents/New project"

PYTHONPATH=src .venv/bin/python -m rgbd_urdf_mvp export-reart-sequence \
  outputs/recordings_refrigerators_staged_dense_mps/refrigerator039/pointcloud_4d_partseg/fusion_manifest.json \
  --output-dir outputs/reart_sequences/refrigerator039 \
  --frame-stride 4 \
  --max-points-per-frame 20000
```

Useful options:

- `--frame-stride`: downsample the episode temporally.
- `--max-frames`: cap the number of exported frames.
- `--max-points-per-frame`: random subsample per frame before ReArt sampling.
- `--include-background`: keep part id `0`; by default only foreground points are exported.

## Start The Remote Server

On `dev-h`:

```bash
cd /root/Users/lixiaotong/articulated_dynamics_learning

PYTHONPATH=src /root/Users/miniconda3/envs/hunyuan3d/bin/python \
  scripts/serve_remote_reart.py \
  --host 127.0.0.1 \
  --port 8092 \
  --project-root /root/Users/lixiaotong/articulated_dynamics_learning \
  --reart-root /root/Users/lixiaotong/articulated_dynamics_learning/ReArt \
  --reart-python /root/Users/miniconda3/envs/particulate/bin/python \
  --scratch-root /tmp/remote_reart_server \
  --output-root outputs/remote_reart_server
```

From the Mac:

```bash
ssh -N -L 8892:127.0.0.1:8092 dev-h
```

Health check:

```bash
curl http://127.0.0.1:8892/health
```

## Run ReArt Remotely

```bash
PYTHONPATH=src .venv/bin/python -m rgbd_urdf_mvp remote-reart-run \
  --server-url http://127.0.0.1:8892 \
  --sequence-dir outputs/reart_sequences/refrigerator039 \
  --output-dir outputs/reart_remote/refrigerator039 \
  --sequence-name refrigerator039 \
  --stage base \
  --base-n-iter 2000 \
  --snapshot-gap 100 \
  --num-points 4096 \
  --num-parts 10 \
  --timeout-s 3600
```

Smoke-test settings used for integration:

```bash
PYTHONPATH=src .venv/bin/python -m rgbd_urdf_mvp remote-reart-run \
  --server-url http://127.0.0.1:8892 \
  --sequence-dir outputs/reart_sequences/refrigerator039_smoke \
  --output-dir outputs/reart_remote_smoke/refrigerator039 \
  --sequence-name refrigerator039_smoke \
  --stage base \
  --base-n-iter 2 \
  --snapshot-gap 1 \
  --num-points 256 \
  --num-parts 4 \
  --timeout-s 300
```

Expected local outputs:

- `remote_reart_status.json`
- `remote_reart_timing.json`
- `remote_reart_result.zip`
- `sequence.zip`
- `reart/<sequence>/result.pkl`
- `reart/<sequence>/model.pth.tar`
- `reart/<sequence>/result.txt`
- `reart/<sequence>/seg.html`

## Axis post-processing

ReArt saves per-part canonical-to-frame poses rather than a ready-to-evaluate
joint axis. The visualization exporter derives a revolute axis from those
poses. It fits one sign-invariant common axis to all parent-relative rotations
with respect to the canonical frame, then solves all hinge constraints
`(I - R) p = t` jointly for the axis line.

Do not average axes from consecutive-frame rotation increments. Small
increments make the axis numerically unstable, and those increments are
expressed in changing local frames. On `microwave011`, that older calculation
produced about 37 degrees of axis error even though the canonical segmentation
was reasonable; the canonical-relative fit reduces it to about 4 degrees.

## Runtime Notes

The current wrapper can run without compiling ReArt's optional CUDA extensions:

- `chamferdist` falls back to a torch KNN implementation with a hand-written
  squared-distance backward pass.
- `knn_cuda.KNN` falls back to `torch.topk(torch.cdist(...))`.
- PointNet++ FPS falls back to a torch implementation.
- Plotly/Kaleido visual export is replaced with lightweight placeholder files.

These fallbacks are intended to make the pipeline runnable and debuggable on
the server. For production-scale ReArt runs, compile the original ReArt CUDA
extensions and install Kaleido if full GIF/PNG visualizations are needed.

The remote server records timing and GPU memory samples in
`remote_reart_timing.json`. The most useful GPU number is
`peak_memory_delta_mib`, because shared servers can have substantial background
memory already allocated before a job starts.

## Lightwheel Microwave Batch

Run the full microwave batch locally while using the remote ReArt server:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_reart_microwave_batch.py \
  --server-url http://127.0.0.1:8892
```

The batch exports 30 frames at stride 4 and selects the canonical frame nearest
90 percent of the observed opening excursion. Existing completed objects are
skipped, so the command is resumable. Per-object runtime and GPU0 peak memory
delta are stored in each `remote_reart_timing.json`; batch progress and
canonical-frame choices are stored in `batch_manifest.json`.

Generate the optimized/feedforward/ReArt comparison with:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_reart_microwave_comparison.py \
  --reference-json \
    outputs/feedforward_articulation/lightwheel_microwaves_view2_open_upY/_evaluation/gt_axis_position_evaluation_unified_scale_best_joint.json \
  --reart-root outputs/reart/lightwheel_microwaves_open \
  --retry-root outputs/reart/lightwheel_microwaves_open_retries
```

The report includes coverage, joint-type accuracy, axis angle, normalized axis
position, runtime, GPU memory, per-object rows, and retry-selection metadata.

## Lightwheel Refrigerator Batch

Run the staged refrigerator batch through ReArt with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_reart_refrigerator_batch.py \
  --server-url http://127.0.0.1:8892
```

The refrigerator batch exports 30 frames at stride 8 to cover the full staged
episode. The canonical frame is selected from `dynamics_log.jsonl` by computing
the normalized progress of every moving joint and choosing the exported frame
nearest 90 percent average progress. This makes the canonical point cloud occur
after the outer doors have opened and the selected drawer/slide joints are
visible.

Generate the refrigerator three-path report with:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_reart_refrigerator_comparison.py \
  --reference-json \
    outputs/feedforward_articulation/lightwheel_refrigerators_view1_dooropen_upY/_evaluation/path_comparison_summary.json \
  --reart-root outputs/reart/lightwheel_refrigerators_open
```

Unlike microwaves, refrigerators are multi-joint objects. The ReArt evaluator
fits a candidate joint for every predicted graph edge, then performs one-to-one
set matching against the GT joint set using joint type, axis angle, and
normalized axis-line position. Runtime and GPU0 memory are read from
`remote_reart_timing.json` for every object.
