# Remote Hunyuan3D + PARTICULATE Server

This is the GPU-server path for turning client observations into a mesh-first
articulation baseline:

```text
Mac client RGB observations
  -> remote-articulate-generate
  -> remote wrapper server
  -> local Hunyuan3D API server
  -> PARTICULATE infer.py
  -> zipped mesh, URDF, MJCF, animated GLB, pred.npz
  -> client output directory
```

The wrapper intentionally composes the existing Hunyuan3D API server and
PARTICULATE CLI instead of modifying either project.

For the ReArt point-cloud baseline, see [ReArt Remote Integration](reart-integration.md).

## Server Environment

Use one CUDA Linux environment for the wrapper, Hunyuan3D, and PARTICULATE:

```bash
conda create -n articulated-server python=3.10 -y
conda activate articulated-server

git clone --recurse-submodules <this-project-url>
cd <this-project>

pip install -e ".[remote-server]"
pip install -e Hunyuan3D-2
pip install -r Hunyuan3D-2/requirements.txt
pip install -r Particulate/requirements.txt
```

PARTICULATE's requirements are CUDA 12.4 oriented. If your server already has a
working PARTICULATE environment, prefer reusing it and set:

```bash
export PARTICULATE_PYTHON=/path/to/particulate-env/bin/python
```

## Start Hunyuan3D

Run the Hunyuan server on localhost:

```bash
cd Hunyuan3D-2
SNAP=$(cat /root/Users/models--tencent--Hunyuan3D-2mini/refs/main)

python api_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --model_path /root/Users/models--tencent--Hunyuan3D-2mini/snapshots/$SNAP \
  --subfolder hunyuan3d-dit-v2-mini-turbo \
  --variant fp16 \
  --device cuda
```

For quick tests, use the mini/turbo model and lower client inference settings.

## Start The Wrapper

From the project root:

```bash
PYTHONPATH=src python scripts/serve_remote_articulation.py \
  --host 127.0.0.1 \
  --port 8090 \
  --api-token "lxt" \
  --hunyuan-url http://127.0.0.1:8080 \
  --particulate-root Particulate \
  --scratch-root /tmp/remote_articulation_server \
  --particulate-python /root/Users/miniconda3/envs/particulate/bin/python \
  --particulate-ckpt-path /root/Users/lixiaotong/articulated_dynamics_learning/Particulate_ckpt/model.pt \
  --output-root outputs/remote_articulation_server
```

Expose this wrapper through the same tunnel or reverse proxy pattern used for
Hunyuan3D. Protect public endpoints with a bearer token:

```bash
export REMOTE_ARTICULATION_API_TOKEN=...
PYTHONPATH=src python scripts/serve_remote_articulation.py \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN" \
  --hunyuan-url http://127.0.0.1:8080
```

The wrapper API:

- `GET /health`
- `POST /send-articulate`
- `GET /status-articulate/{uid}`

## Expose The Wrapper

Keep the Hunyuan3D API server bound to `127.0.0.1:8080`. Only expose the
wrapper server on port `8090`; it is the API boundary that accepts client
requests and calls Hunyuan3D locally.

### Option A: SSH Local Forward

Use this when the Mac can SSH into the GPU server. Start the wrapper on the
server bound to localhost:

```bash
export REMOTE_ARTICULATION_API_TOKEN='replace-with-a-long-random-token'
PYTHONPATH=src python scripts/serve_remote_articulation.py \
  --host 127.0.0.1 \
  --port 8090 \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN" \
  --hunyuan-url http://127.0.0.1:8080 \
  --particulate-root Particulate \
  --particulate-python "$PARTICULATE_PYTHON" \
  --particulate-ckpt-path /path/to/Particulate/model.pt \
  --scratch-root /tmp/remote_articulation_server \
  --output-root outputs/remote_articulation_server
```

From the Mac:

```bash
ssh -N -L 8090:127.0.0.1:8090 user@gpu-server.example.com
```

Then use the local forwarded URL:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url http://127.0.0.1:8090 \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN"
```

### Option B: Bind To A Public Interface

Use this only behind a firewall, VPN, or trusted network. Start the wrapper on
all interfaces and keep token auth enabled:

```bash
export REMOTE_ARTICULATION_API_TOKEN='replace-with-a-long-random-token'
PYTHONPATH=src python scripts/serve_remote_articulation.py \
  --host 0.0.0.0 \
  --port 8090 \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN" \
  --hunyuan-url http://127.0.0.1:8080 \
  --particulate-root Particulate \
  --particulate-python "$PARTICULATE_PYTHON" \
  --particulate-ckpt-path /path/to/Particulate/model.pt \
  --scratch-root /tmp/remote_articulation_server \
  --output-root outputs/remote_articulation_server
```

Open only the wrapper port in the firewall, then call:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url http://gpu-server-public-host:8090 \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN"
```

### Option C: HTTPS Reverse Proxy

Use this for a longer-running shared endpoint. Run the wrapper on localhost and
put nginx, Caddy, or another TLS reverse proxy in front of it:

```text
https://articulation.example.com
  -> 127.0.0.1:8090
```

Keep `--api-token` enabled even when HTTPS is configured. The client URL then
becomes:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://articulation.example.com \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN"
```

Quick server-side checks:

```bash
curl http://127.0.0.1:8090/health
curl -H "Authorization: Bearer $REMOTE_ARTICULATION_API_TOKEN" \
  http://127.0.0.1:8090/status-articulate/not-a-real-job
```

## Client Usage

From the Mac:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://your-remote-articulation-endpoint.example.com \
  --image path/to/front.png path/to/left.png path/to/right.png \
  --image-views front left right \
  --output-dir outputs/remote_articulation/object \
  --num-inference-steps 50 \
  --octree-resolution 380 \
  --face-count 20000 \
  --particulate-target-faces 30000 \
  --particulate-global-points 10000 \
  --particulate-num-points 10000 \
  --particulate-no-strict \
  --timeout-s 3600
```

For simulator recordings, send selected masked views directly from an existing
recording:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://your-remote-articulation-endpoint.example.com \
  --episode outputs/recordings/$OBJECT_ID/episode.json \
  --generation-image-output-dir outputs/recordings/$OBJECT_ID/generation_images \
  --frame-index 0 \
  --view-indices 0 1 2 \
  --image-views front left right \
  --mask-source auto \
  --background transparent \
  --output-dir outputs/remote_articulation/$OBJECT_ID \
  --timeout-s 3600
```

`--view-indices` selects the recording camera slots and `--image-views` names
those slots for Hunyuan3D. Single-view generation is the same path with one view,
for example `--view-indices 0 --image-views front`.

Alternatively, prepare clean masked Hunyuan3D inputs from the recorded prior
masks as a separate step:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp prepare-generation-images \
  outputs/recordings/$OBJECT_ID/episode.json \
  --output-dir outputs/recordings/$OBJECT_ID/generation_images \
  --frame-index 0 \
  --view-indices 0 1 2 \
  --image-views front left right \
  --mask-source auto \
  --background transparent
```

Then send the manifest to the combined remote server:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://your-remote-articulation-endpoint.example.com \
  --image-manifest outputs/recordings/$OBJECT_ID/generation_images/generation_images.json \
  --output-dir outputs/remote_articulation/$OBJECT_ID \
  --num-inference-steps 50 \
  --octree-resolution 380 \
  --face-count 20000 \
  --particulate-target-faces 30000 \
  --particulate-global-points 10000 \
  --particulate-num-points 10000 \
  --timeout-s 3600
```

For smoke tests, keep `--particulate-global-points` and
`--particulate-num-points` around `5000-10000`. Hunyuan3D meshes with more than
roughly `50k` faces can spend a long time in PARTICULATE's CPU-side sharp-edge
sampling before the GPU forward pass; use `--face-count` and
`--particulate-target-faces` to keep the mesh small enough for iteration.

When `--scratch-root` is set, the server writes Hunyuan and PARTICULATE working
files to a node-local directory first, then copies the completed job back to
`--output-root` and returns the zip to the client. This avoids slow mesh
sampling and GLB/OBJ export on network filesystems. PARTICULATE timing logs are
printed with the `[particulate-timing]` prefix.

## Debug And Performance Notes

The slow path observed in PARTICULATE is usually before the GPU forward pass.
Hunyuan3D may emit dense GLB meshes; one sofa example produced about `178k`
faces. PARTICULATE's original `prepare_inputs` then samples two point sets:

```text
sample_points(mesh, 40000, sharp_point_ratio=0.5)
sample_points(mesh, 102400, sharp_point_ratio=0.5)
```

In upstream PARTICULATE, every `sample_points` call recomputes sharp edges by
looping over all faces in Python and building an edge dictionary. That means the
same dense mesh can be scanned twice on CPU before `model.infer` runs. In
`nvidia-smi` this appears as low GPU utilization with one CPU core busy.

The project wrapper now replaces that sampling implementation at runtime:

- uses `trimesh.face_adjacency` and `face_adjacency_edges` for vectorized
  sharp-edge detection;
- caches sharp edges per transformed mesh, so global and decode sampling reuse
  the same cache;
- logs stage timings with `[particulate-timing]`;
- writes PARTICULATE stdout/stderr to `particulate_stdout.log` and
  `particulate_stderr.log`;
- fails the remote job if upstream exits successfully but produces no artifacts.

Useful timing lines:

```text
[particulate-timing] sharp_edge_cache faces=... adjacency=... sharp_edges=... elapsed_s=...
[particulate-timing] sample_global_s=...
[particulate-timing] sample_decode_s=...
[particulate-timing] load_partfield_s=...
[particulate-timing] obtain_partfield_feats_s=...
[particulate-timing] model_infer_s=...
```

If `sharp_edge_cache` or `sample_decode_s` dominates, the bottleneck is CPU mesh
sampling rather than CUDA. If `model_infer_s` dominates, inspect GPU utilization,
CUDA/PyTorch compatibility, and available GPU memory.

Optimization history:

- Moved job working files to node-local storage with `--scratch-root`; this
  avoids repeated mesh IO and export on network filesystems.
- Added `--particulate-global-points` and `--particulate-target-faces` for
  smoke tests and controlled experiments.
- Added timing logs around `prepare_inputs`, PartField feature extraction, and
  `model.infer`.
- Replaced repeated Python sharp-edge scans with one vectorized, cached
  sharp-edge pass in the project wrapper.

With auth:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://your-remote-articulation-endpoint.example.com \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN"
```

On a slow SSH tunnel, result download can dominate the wall time even when the
remote job is done. Use `--remote-skip-download` to submit, poll, and save
status/timing without pulling the zip immediately:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url http://127.0.0.1:8888 \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN" \
  --remote-skip-download
```

In this mode the local output still contains `remote_articulation_status.json`
and `remote_articulation_timing.json`. The timing file includes
`remote_result_zip_path`, which points to the zip on the server for later `scp`
or batch transfer.

The client writes:

```text
outputs/remote_articulation/object/
  remote_articulation_result.zip
  remote_articulation_timing.json
  remote_articulation_status.json
  hunyuan/generated.glb
  particulate/particulate_result.json
  particulate/animated_textured_*.glb
  particulate/mesh_parts_with_axes_*.glb
  particulate/urdf_*/model.urdf
  particulate/mjcf_*/model.xml
  particulate/eval/pred.npz
```

`remote_articulation_timing.json` separates local client time from remote server
time:

- `client_timings.prepare_generation_images_s`: local episode frame extraction,
  object masking, crop/pad, and Hunyuan input image writing.
- `client_timings.send_s`: upload/request submission to the remote articulation
  server.
- `client_timings.wait_until_completed_s`: polling time until the remote job is
  complete.
- `client_timings.download_result_s`: final result zip transfer through the
  binary `download-articulate-zip` endpoint. This should be separate from
  polling; if it is missing or near zero while `wait_until_completed_s` is much
  larger than `server_timings.total_s`, the server is probably old and still
  returning the result zip from the status endpoint.
- `client_timings.zip_write_s`: local write time for the binary zip response.
- `client_timings.download_result_base64_s` and
  `client_timings.zip_base64_decode_write_s`: compatibility fallback timing for
  old servers that only provide a base64 JSON result.
- `client_timings.unpack_zip_s`: local unzip.
- `server_timings.hunyuan3d_s`: Hunyuan3D generation and GLB decode/write.
- `server_timings.particulate_s`: PARTICULATE mesh inference and export.
- `server_timings.packaging_s`: server-side copy/zip/base64 packaging.
- `server_resource_usage.hunyuan3d`: `nvidia-smi` samples collected while
  waiting for Hunyuan3D and writing the returned GLB.
- `server_resource_usage.particulate`: `nvidia-smi` samples collected while
  PARTICULATE runs.

Each resource stage records `baseline_gpus`, `peak_gpus`, `final_gpus`,
`peak_memory_used_mib`, and `peak_memory_delta_mib`. `peak_memory_used_mib` is
the largest total `memory.used` sample seen on any GPU during the stage.
`peak_memory_delta_mib` subtracts that GPU's stage-start baseline, so it is the
more useful number when other jobs already occupy the server. Because this is
sampled externally through `nvidia-smi`, it is an approximation; short spikes
between samples may be missed, and unrelated jobs on the same GPU can affect the
total VRAM number.

For batch reports, summarize optimized and feedforward timings into one table:

```bash
PYTHONPATH=src python scripts/summarize_articulation_timing.py \
  --batch-config configs/batch_lightwheel_refrigerators_mjcf.tsv \
  --optimized-root outputs/recordings_refrigerators_staged_dense_mps \
  --feedforward-root outputs/feedforward_articulation/lightwheel_refrigerators_view1_dooropen_upY
```

This writes:

```text
outputs/feedforward_articulation/lightwheel_refrigerators_view1_dooropen_upY/_evaluation/
  timing_summary.json
  timing_summary.tsv
  timing_summary.md
```

For the optimized path, the summary reads batch `stage_timing` JSON lines and
reports `record-mujoco`, `fuse-pointcloud`, `track-part-pixels` (CoTracker),
`estimate-part-poses`, `infer-joints`, export, and viewer serialization. It
does not include segmentation model time; for simulation batches using clean GT
masks, there is no learned segmentation stage in the optimized path. Blank
fields mean the object was skipped by `--resume`, predates timing
instrumentation, or has not completed.

Then summarize optimized-vs-feedforward accuracy, timing, VRAM, and white-background
SVG plots with the shared comparison script:

```bash
PYTHONPATH=src python scripts/summarize_articulation_comparison.py \
  --batch-config configs/batch_lightwheel_refrigerators_mjcf.tsv \
  --optimized-root outputs/recordings_refrigerators_staged_dense_mps \
  --feedforward-root outputs/feedforward_articulation/lightwheel_refrigerators_view1_dooropen_upY \
  --optimized-joint-rotation-threshold-rad 0.90
```

This writes `path_comparison_summary.json`, `path_comparison_summary.md`,
`path_comparison_per_joint.tsv`, and reusable SVG plots under the same
`_evaluation/` directory. For the refrigerator run, the optimized path should use
`configs/track_refrigerator_dense.yaml`; otherwise sparse freezer/drawer tracks
can produce apparent SE(3) rotation and inflate `prismatic->revolute` errors.
For joint type selection ablations on the optimized path only, use
`scripts/summarize_joint_type_ablation.py` to compare the older pose-gated
residual decision against the default track-model-first residual decision while
reusing the same `part_poses.json` and `part_tracks.json` artifacts.
