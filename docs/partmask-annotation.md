# Part Mask Annotation UI

Use the local annotation UI when automatic real-data part segmentation is not
good enough. The UI writes indexed part masks (`0 = background`, `1..N = parts`)
back into an episode-compatible JSON, then the existing mask propagation command
can carry that verified mask through the rest of the recording.

## Start The UI

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_partmask_annotation.py \
  outputs/real_recordings/cardboardbox01_o/episode.json \
  --frame-index 0 \
  --view-index 0 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --propagate-python .venvs/sam2/bin/python \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Default outputs:

- `outputs/real_recordings/<object>/assets/masks/manual_partmask/view_<N>/frame_<FFFF>_mask.png`
- `outputs/real_recordings/<object>/assets/masks/manual_partmask/annotation_prompts.json`
- `outputs/real_recordings/<object>/episode.partmask-annotated.json`

## Prompt Types

Each prompt is attached to the currently selected part:

- Text prompt: records a phrase such as `microwave door`, `box lid`, or `hand`.
- Positive point: click a pixel that belongs to the current part.
- Negative point: click a pixel that should be excluded from the current part.
- Bounding box: drag a box around the current part.
- Brush/eraser: directly edits the saved indexed mask.

Use `SAM2 Current Part` after adding point/bbox prompts to turn those prompts
into pixels for the selected part. Text prompts are currently saved for
audit/future SAM3 text prompting; local SAM2 does not consume text. `Start From
Current` calls the backend propagation path directly from the UI and uses the
saved indexed mask pixels as the propagation seed. Empty masks are rejected
before propagation starts.

## Live Backend Propagation

The UI can launch backend mask propagation and poll every generated frame mask.
This is the intended interactive loop:

1. Select a part, then add a bbox and optional positive/negative points.
2. Click `SAM2 Current Part`, or paint the part directly with the brush.
3. Repeat for each part until the overlay has non-empty indexed part pixels.
4. Click `Start From Current`.
5. Keep `Live follow latest frame` enabled to watch masks as they are written.
6. If the mask drifts, click `Pause Backend`.
7. Correct the current frame, save, then click `Start From Current` again.

During SAM2 propagation, the backend writes live preview masks as soon as each
frame is produced, so the UI can follow the latest available frame. At the end
of the run it rewrites those masks with the final merged logits.

Useful correction controls:

- `Overlay` slider: reduce mask opacity when the image is hard to inspect.
- `Current only`: hide other part ids and inspect only the selected part.
- `Keep Largest`: for the selected part, remove small disconnected speckles and
  keep only the largest connected component.
- `Latest Mask`: jump to the newest frame that has a generated mask.
- `Log lines`: limit the backend log box height/content while propagation runs.
- `Undo` or `Cmd/Ctrl+Z`: restores the previous in-browser labels/prompts/parts
  state. Click `Save Mask` after undoing if the correction should persist to
  disk.

Backend options:

- `SAM2 Video`: dense mask propagation. Use this for real annotation passes.
- `CoTracker Sparse`: fast debug propagation that rasterizes sparse tracks.

SAM2 part modes:

- `SAM2 parts: independent`: default for indexed part masks. Each part id is
  propagated in its own SAM2 state and the logits are merged afterward. This is
  slower, but avoids one part suppressing another during multi-object
  competition.
- `SAM2 parts: joint`: SAM2 native multi-object propagation. This is faster, but
  can drop a static part when another moving part dominates the mask logits.

The progress bar and frame strip show which frames already have masks. Click any
frame-strip cell to inspect or correct that frame.

The server writes live propagation outputs under:

```text
assets/masks/live_<backend>_f<frame>_v<view>_<timestamp>/
episode.live_<backend>_f<frame>_v<view>_<timestamp>.json
```

When a backend run finishes successfully, the server switches its active episode
to that propagated episode, so subsequent frame loads inspect the latest masks.

## CLI Propagation

After saving a frame mask, run SAM2 video propagation:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/cardboardbox01_o/episode.partmask-annotated.json \
  --backend sam2-video \
  --mask-kind part \
  --reference-frame 0 \
  --view-indices 0 \
  --sam2-root sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-part-mode independent \
  --output-episode outputs/real_recordings/cardboardbox01_o/episode.partmask-propagated.json \
  --output-dir outputs/real_recordings/cardboardbox01_o/assets/masks/manual_partmask_sam2
```

For sparse debug propagation:

```bash
PYTHONPATH=src .venv/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/cardboardbox01_o/episode.partmask-annotated.json \
  --backend cotracker-sparse \
  --mask-kind part \
  --reference-frame 0 \
  --view-indices 0 \
  --cotracker-repo co-tracker \
  --cotracker-checkpoint co-tracker/ckpt/scaled_offline.pth \
  --output-episode outputs/real_recordings/cardboardbox01_o/episode.partmask-cotracker.json
```

## Correct Mid-Episode Drift

If propagation drifts:

1. Pause the backend in the UI.
2. Load the bad frame.
3. Fix the part mask with the same prompt/brush controls. For sparse green
   speckles, select that part and click `Keep Largest`.
4. Save the corrected indexed mask.
5. Click `Start From Current` to restart from that frame.

You can also restart the UI on a saved propagated episode:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_partmask_annotation.py \
  outputs/real_recordings/cardboardbox01_o/episode.partmask-propagated.json \
  --frame-index 60 \
  --view-index 0 \
  --output-episode outputs/real_recordings/cardboardbox01_o/episode.partmask-corrected-f60.json
```

Then propagate from frame 60:

```bash
PYTHONPATH=src .venvs/sam2/bin/python -m rgbd_urdf_mvp propagate-episode-masks \
  outputs/real_recordings/cardboardbox01_o/episode.partmask-corrected-f60.json \
  --backend sam2-video \
  --mask-kind part \
  --reference-frame 60 \
  --view-indices 0 \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --output-episode outputs/real_recordings/cardboardbox01_o/episode.partmask-propagated-f60.json
```

## Use In The Optimized Path

Once the propagated episode looks acceptable:

```bash
PYTHONPATH=src .venv/bin/python -m rgbd_urdf_mvp fuse-pointcloud \
  outputs/real_recordings/cardboardbox01_o/episode.partmask-propagated.json \
  --output-dir outputs/real_optimized/cardboardbox01_o_manual_partmask/pointcloud_4d_partseg
```

Continue with the usual `track-part-pixels`, `estimate-part-poses`,
`infer-joints`, and `export-inferred-articulation` stages.
