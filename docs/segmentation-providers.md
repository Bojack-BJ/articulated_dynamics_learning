# Segmentation Providers

Episode masks are split into two layers:

- Provider: a model-specific wrapper that produces mask PNGs from RGB frames.
- Episode writer: the stable layer that records those masks in `episode.json`.

This keeps SAM2, SAM3, PartSAM, hand labels, and future models behind the same
episode contract:

```text
mask_paths_by_view       # binary object masks
part_mask_paths_by_view  # indexed part masks, 0 = background
```

## Write Masks Into An Episode

Use `mask-dir` when a model has already written masks:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider mask-dir \
  --mask-dir outputs/sam2_masks/microwave01_o \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2.json \
  --force
```

Accepted mask layouts:

```text
frame_0000_mask.png
frame_0000_view_0_mask.png
view_0/frame_0000_mask.png
```

Use `external-command` to connect a model wrapper without importing model
dependencies into this package:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider external-command \
  --command 'python tools/run_sam2.py --image "{rgb_path}" --output "{output_mask_path}"' \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2.json \
  --force
```

Available command-template fields:

- `{rgb_path}`
- `{output_mask_path}`
- `{frame_index}`
- `{view_index}`

Use native SAM2 when the local SAM2 repo and checkpoint are installed:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider sam2 \
  --sam2-root sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --sam2-prompt-mode center-box \
  --sam2-center-box-scale 0.75 \
  --sam2-mask-selection best-iou \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2.json
```

`--sam2-device auto` prefers Apple MPS, then CUDA, then CPU. On macOS, install
SAM2 with `SAM2_BUILD_CUDA=0` because the optional CUDA post-processing
extension is not available:

```bash
cd "/Users/lxt/Documents/New project"
/Users/lxt/.local/bin/python3.11 -m venv .venvs/sam2
source .venvs/sam2/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
cd sam2
SAM2_BUILD_CUDA=0 python -m pip install -e .
```

Run the project command with `.venvs/sam2/bin/python`, not the base project
`.venv`, because native SAM2 imports PyTorch, Hydra, and the SAM2 editable
package from the dedicated environment. MPS detection may report false inside
sandboxed shells even when it works in a normal Terminal. Verify it from a
normal Terminal:

```bash
cd "/Users/lxt/Documents/New project"
.venvs/sam2/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps built", torch.backends.mps.is_built())
print("mps available", torch.backends.mps.is_available())
print(torch.ones(1, device="mps"))
PY
```

Native SAM2 currently writes object masks only. For part masks, use a provider
that emits indexed masks through `mask-dir` or `external-command`.

Prompted SAM2 normally returns multiple masks. `--sam2-mask-selection best-iou`
is the default. For cluttered scenes, compare `smallest`, `largest`, or an
explicit `--sam2-mask-selection index --sam2-mask-index N`; the manifest records
every candidate's predicted IoU and area.

## First-Frame SAM2 Plus CoTracker Propagation

For long real episodes, segmenting every frame with SAM can be slow and makes
manual review harder. A practical workflow is:

1. Run SAM2 on only the reference frame with `--max-frames 1`.
2. Inspect or manually fix that mask.
3. Propagate the verified mask through the rest of the episode.

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider sam2 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --sam2-prompt-mode box \
  --sam2-box-xyxy 245 35 515 235 \
  --start-frame 0 \
  --max-frames 1 \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2-first.json

PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/microwave01_o/episode.sam2-first.json \
  --backend sam2-video \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --reference-frame 0 \
  --mask-kind object \
  --output-episode outputs/real_recordings/microwave01_o/episode.sam2-video.json
```

`propagate-episode-masks --backend sam2-video` uses SAM2's video predictor:
it adds the reference mask as the conditioning mask on `--reference-frame` and
then calls `propagate_in_video` to produce dense masks for the episode. This is
the preferred propagation path. `--backend cotracker-sparse` remains available
as a debug fallback, but it rasterizes sparse tracks and should not be the
default for mask quality.

For part pointclouds, the reference frame must contain an indexed part mask,
not only a binary object mask. A first-frame part mask can come from manual
labeling, PartSAM, an external SAM wrapper that composes multiple part prompts,
or any tool that writes `0=background, 1=base, 2=door, ...` as a 16-bit PNG.
Import it into the episode first:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp segment-episode-masks \
  outputs/real_recordings/microwave01_o/episode.json \
  --provider mask-dir \
  --mask-dir outputs/real_recordings/microwave01_o/reference_part_masks \
  --mask-kind part \
  --start-frame 0 \
  --max-frames 1 \
  --output-episode outputs/real_recordings/microwave01_o/episode.part-first.json

PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/microwave01_o/episode.part-first.json \
  --backend sam2-video \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --reference-frame 0 \
  --mask-kind part \
  --output-episode outputs/real_recordings/microwave01_o/episode.part-sam2-video.json
```

`--mask-kind part` writes `part_mask_paths_by_view`, so `fuse-pointcloud` can
emit PLY vertices with nonzero `part_id`. If the input episode does not already
have `metadata.part_segmentation`, the propagator creates placeholder
`part_1`, `part_2`, ... metadata from the reference mask ids; edit those names
and roles when you know which id is base vs moving part.

## Batch Runs

Use `segment-episode-masks-batch` when a batch of converted real recordings is
ready:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp segment-episode-masks-batch \
  configs/batch_episode_masks_example.tsv \
  --provider sam2 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-device auto \
  --start-frame 0 \
  --max-frames 1 \
  --jobs 1 \
  --force
```

The TSV must contain `episode_path`. Optional columns such as `object_id`,
`output_episode`, `output_dir`, `sam2_prompt_mode`, `sam2_box_xyxy`,
`start_frame`, `max_frames`, and `view_indices` override the command defaults
per episode.

## SAM3 Local Environment

SAM3 is useful when you want concept-driven masks such as `microwave door`,
`microwave body`, or `handle` rather than manually supplied part ids. Keep it in
a separate environment because the dependency stack is heavier than SAM2:

```bash
cd "/Users/lxt/Documents/New project"
/Users/lxt/.local/bin/python3.11 -m venv .venvs/sam3
.venvs/sam3/bin/python -m pip install --upgrade pip setuptools wheel
.venvs/sam3/bin/python -m pip install --retries 20 --timeout 180 "ultralytics>=8.3.237"
```

The Python package provides the SAM3 interface, but the weights are gated. After
requesting access on Hugging Face, place `sam3.pt` somewhere local, for example:

```text
models/sam3/sam3.pt
```

Then a local wrapper can be connected through `--provider external-command`, or
a native `sam3-video` provider can be added once the weight path and prompt
format are fixed for the dataset. For batch real-data validation, prefer this
split:

- `sam2-video`: local dense propagation from verified first-frame masks.
- `sam3-video`: server or local concept-prompt part discovery when weights are
  available.
- `cotracker`: track/pose estimation, not the default mask propagation path.

For joint inference, generate `--mask-kind part` with indexed masks. Object
masks are sufficient for Hunyuan generation and foreground pointcloud fusion,
but `track-part-pixels` needs part ids to seed and validate tracks.

## Compare Models

Compare a predicted episode against a reference episode:

```bash
PYTHONPATH=src python3 -m rgbd_urdf_mvp evaluate-episode-masks \
  outputs/real_recordings/microwave01_o/episode.sam2.json \
  outputs/real_recordings/microwave01_o/episode.reference.json \
  --mask-kind object \
  --output-json outputs/real_recordings/microwave01_o/sam2_mask_eval.json
```

Metrics are foreground IoU, precision, and recall per frame/view plus means.
For part masks this first version evaluates foreground quality. Part-id accuracy
needs a stable part taxonomy and correspondence table.

## macOS Model Notes

SAM2 is the most practical local macOS candidate. The official repository
requires Python 3.10+, PyTorch 2.5.1+, and torchvision 0.20.1+. Its custom CUDA
extension is optional, so image prediction can run without CUDA, but the
official examples are CUDA-oriented and video propagation will be slower on MPS.

SAM3 is better suited to a GPU server today. Ultralytics exposes SAM3 via
`ultralytics>=8.3.237`, but the weights are gated on Hugging Face and the model
is designed around concept segmentation and video tracking workloads that are
heavier than SAM2.

PartSAM is not a good macOS target right now. Its published setup pins CUDA
PyTorch, torch-scatter CUDA wheels, Apex, pointops, Open3D/VTK-style geometry
dependencies, and the README still lists inference code / pretrained model
release as TODO items. Treat it as a server-side mesh/part baseline, not a Mac
RGB-D mask provider.
