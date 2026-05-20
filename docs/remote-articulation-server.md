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
python api_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --model_path tencent/Hunyuan3D-2mv \
  --subfolder hunyuan3d-dit-v2-mv \
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
  --hunyuan-url http://127.0.0.1:8080 \
  --particulate-root Particulate \
  --particulate-python "$PARTICULATE_PYTHON" \
  --particulate-ckpt-path /path/to/Particulate/model.pt \
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

For simulator recordings, first prepare clean masked Hunyuan3D inputs from the
recorded prior masks:

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

With auth:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp remote-articulate-generate \
  --server-url https://your-remote-articulation-endpoint.example.com \
  --image path/to/front.png \
  --output-dir outputs/remote_articulation/object \
  --api-token "$REMOTE_ARTICULATION_API_TOKEN"
```

The client writes:

```text
outputs/remote_articulation/object/
  remote_articulation_result.zip
  remote_articulation_status.json
  hunyuan/generated.glb
  particulate/particulate_result.json
  particulate/animated_textured_*.glb
  particulate/mesh_parts_with_axes_*.glb
  particulate/urdf_*/model.urdf
  particulate/mjcf_*/model.xml
  particulate/eval/pred.npz
```

Then compare with tracking:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp compare-articulation-backends \
  outputs/recordings/<object-id>/pointcloud_4d_partseg/joint_inference.json \
  outputs/remote_articulation/object/particulate/particulate_result.json
```
