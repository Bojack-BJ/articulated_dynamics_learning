# Remote 3D Generation Server

This project can call a remote Hunyuan3D API server from the Mac client and save the generated 3D asset locally.

The intended setup is:

```text
Mac client
  rgbd_urdf_mvp hunyuan3d-generate
        |
        | HTTPS
        v
Public endpoint
  tunnel / reverse proxy / API gateway
        |
        | private HTTP
        v
GPU machine
  Hunyuan3D API server
```

## Repository Dependency

`Hunyuan3D-2/` is tracked as a git submodule that points to:

```text
git@github.com:Bojack-BJ/Hunyuan3D-2.git
branch: rgbd-urdf-api-server
```

Clone this project with submodules:

```bash
git clone --recurse-submodules <this-project-url>
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

If the Hunyuan3D server patch changes later, commit and push inside `Hunyuan3D-2/` first, then update the submodule pointer in this repository.

## Hunyuan3D API Shape

The Hunyuan3D sample server exposes these useful endpoints:

- `GET /health`
- `POST /generate`
  - synchronous generation
  - returns the model file directly
- `POST /send`
  - asynchronous generation
  - returns a `uid`
- `GET /status/{uid}`
  - polling endpoint
  - returns `model_base64` when `status == "completed"`

For long jobs, prefer the async path:

```text
POST /send -> uid -> poll GET /status/{uid} -> model_base64 -> local .glb
```

The patched `Hunyuan3D-2/api_server.py` in this workspace accepts:

- single-view input as `image: "<base64>"`
- multiview input as `image: {"front": "<base64>", "left": "<base64>", "right": "<base64>"}`

This dict format matches Hunyuan3D's multiview pipeline examples. Use this project's submodule fork for deployment; a fresh upstream Hunyuan3D checkout will fail on multi-view requests because the upstream sample server only decodes a single base64 image string.

## GPU Server Usage

Single-view Hunyuan3D-2.1 server:

```bash
cd Hunyuan3D-2
python api_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --variant fp16
```

Multiview server for 1-4 named views:

```bash
cd Hunyuan3D-2
python api_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --model_path tencent/Hunyuan3D-2mv \
  --subfolder hunyuan3d-dit-v2-mv \
  --variant fp16
```

For faster debugging, use `--subfolder hunyuan3d-dit-v2-mv-turbo` and lower the client `--num-inference-steps`.

## Client Usage

Run from the Mac:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp hunyuan3d-generate \
  --server-url https://your-public-generation-endpoint.example.com \
  --image path/to/input.png \
  --output outputs/generated/object.glb \
  --mode async \
  --timeout-s 1800 \
  --poll-interval-s 5
```

For three RGB views, send named multiview images:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp hunyuan3d-generate \
  --server-url https://your-public-generation-endpoint.example.com \
  --image path/to/front.png path/to/left.png path/to/right.png \
  --image-views front left right \
  --output outputs/generated/object.glb \
  --mode async \
  --num-inference-steps 50 \
  --octree-resolution 380
```

If `--image-views` is omitted for multiple images, the client uses `front left right back` truncated to the number of provided images. Prefer explicit `--image-views` whenever the camera order is not obvious.

Generated `.glb` meshes can be passed directly to the PARTICULATE adapter:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp particulate-infer \
  --mesh outputs/generated/object.glb \
  --output-dir outputs/particulate/object \
  --python-bin "$PARTICULATE_PYTHON"
```

See [particulate-integration.md](particulate-integration.md) for the full
comparison workflow against the RGB-D tracking pipeline.

For the production path where both Hunyuan3D and PARTICULATE run on the GPU
server and the Mac only sends observations, use the combined remote server in
[remote-articulation-server.md](remote-articulation-server.md).

If your reverse proxy or tunnel enforces bearer-token auth:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp hunyuan3d-generate \
  --server-url https://your-public-generation-endpoint.example.com \
  --image path/to/input.png \
  --output outputs/generated/object.glb \
  --api-token "$HUNYUAN3D_API_TOKEN"
```

You can also use YAML config:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp configs/hunyuan3d_generate.yaml
```

Example:

```yaml
command: hunyuan3d-generate
args:
  server-url: https://your-public-generation-endpoint.example.com
  image:
    - path/to/front.png
    - path/to/left.png
    - path/to/right.png
  image-views:
    - front
    - left
    - right
  output: outputs/generated/object.glb
  mode: async
  timeout-s: 1800
  poll-interval-s: 5
```

## Public Networking Recommendation

Do not expose the raw sample FastAPI server directly to the public internet. It is a model demo server, not a hardened production API.

Use one of these patterns instead.

### Option A: Cloudflare Tunnel

Good default when the GPU machine is behind NAT or a university/company network.

On the GPU machine:

```bash
# Run the Hunyuan3D server on localhost, not directly on a public interface.
python api_server.py --host 127.0.0.1 --port 8080

# Expose it through a tunnel.
cloudflared tunnel --url http://127.0.0.1:8080
```

Then call the generated HTTPS URL from the Mac:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp hunyuan3d-generate \
  --server-url https://xxxx.trycloudflare.com \
  --image path/to/input.png \
  --output outputs/generated/object.glb
```

For longer-term use, configure a named Cloudflare Tunnel and protect it with Cloudflare Access.

### Option B: Reverse Proxy On A Public VPS

Good when you control a small public VPS and want a stable domain.

Typical layout:

```text
Mac -> https://gen.example.com -> VPS Caddy/Nginx -> SSH reverse tunnel -> GPU localhost:8080
```

On the GPU machine:

```bash
python api_server.py --host 127.0.0.1 --port 8080
ssh -N -R 127.0.0.1:18080:127.0.0.1:8080 user@your-vps
```

On the VPS, configure Caddy or Nginx to reverse proxy:

```text
gen.example.com {
    reverse_proxy 127.0.0.1:18080
}
```

Add authentication at the reverse proxy layer before using it outside a trusted network.

### Option C: Direct Public IP

Only use this if you understand the security tradeoff.

Minimum requirements:

- put Caddy/Nginx in front of the server
- enable HTTPS
- add authentication
- restrict source IPs if possible
- set large enough request/response body limits for base64 images and GLB files
- set long proxy timeouts for generation jobs

Avoid:

```bash
python api_server.py --host 0.0.0.0 --port 8080
```

unless the port is reachable only from a protected private network.

## Notes

- The client uses Python stdlib `urllib`, so no extra HTTP dependency is needed.
- The async API is safer for long generation jobs than waiting on a single blocking HTTP response.
- If generated files are large, make sure your tunnel/proxy allows large request and response bodies.
- If you add API auth in the Hunyuan3D server itself later, keep the same client interface and enforce the token on the server side.
