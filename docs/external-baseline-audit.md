# External Articulation Baseline Audit

This audit is based on the inference, data-loading, training, and evaluation
code at the pinned upstream commits below. README claims are not treated as
sufficient evidence.

## Summary

| Method | Upstream commit | Input contract in code | Oracle/protocol dependency | Status for shared 3-view PartNet |
|---|---|---|---|---|
| ReArt | `62174a207e602bc09b30317e836d41bd32f9a447` | Short sequence of independently sampled 4D point clouds; optional learned pairwise flow | Fixed part-slot cap, not GT part count; official recommended run uses flow and assignment losses | **Minor adapter required; highest priority** |
| Ditto | `f69579f49bbfaadc665fee511ad960f5243e1c1a` | Point clouds fused from posed depth observations before and after one interaction | Pretrained category distribution; implementation predicts one mobile/static decomposition and one joint | Major protocol/model-capacity mismatch |
| PARIS | `fcab60d2e9f65d67134931f45a8592bed9e21fde` | Two sets of masked multi-view RGB images at start/end states | Separate revolute/prismatic systems; two-part static/mobile representation | Major protocol mismatch for multi-part evaluation |
| DTA | `1a48b402e4bf4bb7731296e8e230f0db3d86fe4f` | Two RGB-D scans, masks, camera poses, and LoFTR correspondences | CLI requires `--num_parts`; published examples use 2 or 3 | Major adapter plus oracle part-count ablation |
| ArtGS | `7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a` | Two-state multi-view RGB, masks, camera poses; optional point-cloud initialization | Full training reads per-scene joint type list to set slots; center initialization may be manually corrected | Not suitable for direct non-oracle comparison without a separate predicted-type protocol |
| VideoArtGS | `133ccfc60950576d75a50ae28cecad8e82ba98a9` | Monocular RGB video, depth/cameras, object masks, TAPIP3D tracks | Requires a per-scene joint inventory containing count, types, and parents; README permits manual correction | Runnable as VLM-assisted or explicitly oracle-assisted baseline |
| GaussianArt | `e296670c3864554451005142be10638ec4d4d2be` | Two-state RGB-D, cameras, and semantic part initialization | Released `run.py` reads `gt/trans.json` to derive exact part count and prismatic indices | Runnable only as clearly labeled oracle-assisted baseline |

## ReArt

**Official repository:** <https://github.com/stevenlsw/reart>

The generic `run_real.py` path loads a temporal sequence, samples 4096 points
per frame, estimates correspondences with a pretrained feature extractor when
`--use_flow_loss` is enabled, optimizes a relaxed segmentation and independent
SE(3) trajectories, projects the result to a valid kinematic tree, and
re-optimizes it.

### Input

- 4D point-cloud sequence.
- No camera parameters after RGB-D back-projection.
- No object mask after foreground point extraction.
- No GT part segmentation.
- Default `num_parts=10` is a slot cap, not a required exact part count.
- The official Sapiens benchmark supplies four point-cloud frames per object,
  each in a different global coordinate frame. This is a property of that
  benchmark input, not a generic ReArt recommendation to sample every sequence
  down to four temporal frames.
- RoboArt uses 10 timesteps with 4096 independently resampled points per
  timestep. The real-data loader consumes all point-cloud/mesh files present
  in the input directory.

The upstream real-data loader samples mesh surfaces. Our adapter changes only
the loader boundary so xyz-only PLY vertices are sampled directly. It does not
change ReArt optimization, losses, merging, tree projection, or RANSAC.

### Native output

`result.pkl` contains:

- `pred_cano_part`
- `pred_pose_list`
- `joint_connection`
- `cano_pc`
- input point-cloud sequence

The saved `result.pkl` does **not** directly contain directed parent-child
edges, joint types, axes, or pivots. The kinematic projection checkpoint does
contain `edge_index`, `joint_type_list`, `axis_list`, and `moment_list`.
Therefore ReArt does optimize screw parameters, but its Sapiens runner does not
report a native GT axis metric. Our native evaluator labels axis/type results
as converted diagnostics: predicted edges are mapped through canonical
segmentation, while the reference type/axis is fitted from relative GT SE(3).
These conditional values must not be presented as an official ReArt metric.

### Evaluation mapping

| Metric | Support |
|---|---|
| Point IoU / ARI / part count | Converted with the same observed-point evaluator as AiM and Hybrid |
| Undirected topology | Convertible after cluster-to-GT matching |
| Directed Edge F1 | Unsupported natively |
| Joint type / axis | Converted diagnostic from the projection checkpoint and relative GT SE(3) |
| Pivot / directed topology | Unsupported as a native metric |
| Official metrics | Reconstruction error, flow error/accuracy, Rand Index, tree edit distance, reanimation error |

### Shared 3-view protocol

At each timestep, the three object-masked RGB-D observations are
back-projected with their recorded extrinsics, merged in the common world
frame, and voxel-downsampled. Exported ReArt PLY files contain xyz only.
Simulation GT labels remain in the reference PLY used by evaluation and are
never included in ReArt input.

The existing 20-object run selected four frames from each 120-frame, 15 Hz
recording. This is a **Sapiens-count-matched adapter**, not a native Sapiens
protocol reproduction: it matches the number of point clouds but not the
official articulation states, temporal spacing, or per-frame global coordinate
frames. In particular, choosing source frames `0/30/60/90` is our adapter
policy, not a frame schedule recommended by ReArt.

`export-reart-sequence` now records `protocol_profile`,
`temporal_sampling_provenance`, and exact `source_frame_indices`. Use explicit
indices when reproducing a declared experiment:

```bash
PYTHONPATH=src python -m rgbd_urdf_mvp export-reart-sequence \
  outputs/recordings/object/pointcloud_4d_partseg/fusion_manifest.json \
  --output-dir outputs/reart_sequences/object \
  --frame-indices 0 30 60 90 \
  --protocol-profile sapien_count_matched_4frame
```

For a native ReArt result, use the official Sapiens or RoboArt input sequences
unchanged. For our PartNet recordings, label results as an equal-input adapter
and report the selected states instead of implying that four frames are a
paper-wide temporal sampling rule. Point-cloud sequences do not have a
meaningful camera FPS after back-projection; FPS belongs to the source RGB-D
recording and must be reported separately.

### Native Sapiens reproduction

The native runner uses the official Sapiens test data unchanged: four frames
with 512 points per frame, the official MultiBodySync flow checkpoint, 2000
relaxation iterations, and 200 kinematic projection iterations.

```bash
python scripts/run_reart_native_sapiens.py \
  --reart-root ReArt \
  --dataset-root /path/to/mbs_sapien \
  --output-root outputs/reart_native_sapiens20_v1 \
  --indices 0,36,72,108,144,180,216,252,288,324,360,396,432,468,504,540,576,612,648,684 \
  --gpu-ids 0,1 \
  --jobs 2

python scripts/evaluate_reart_native_sapiens.py \
  outputs/reart_native_sapiens20_v1/projection \
  --output-dir outputs/reart_native_sapiens20_v1/evaluation
```

The upstream `KNN_CUDA` release URL is no longer available. The current
reproduction uses an API-compatible exact `torch.cdist + topk` nearest-neighbor
backend in the isolated environment. This replaces the unavailable acceleration
extension, not the ReArt objective or neighborhood definition.

The completed 20-sequence native sanity run produced 19 successful official
graph projections. Its official multi-scan RI mean is `0.779`; the unified
point-domain metrics are Point IoU `0.739`, ARI `0.574`, and RI `0.785`.
Sequence 540 failed in the released graph-projection stage and remains a
failure. In contrast, ReArt reaches RI `0.687` on the object-aligned PartNet
cross-dataset benchmark. This gap supports labeling the latter as
cross-dataset generalization rather than native Sapiens reproduction.

### AiM paper-like sanity check

The object-aligned benchmark uses a fixed 120-frame interaction for all
objects. A separate paper-like sanity manifest covers the paper-listed PartNet
objects Storage-47024, Fridge-11304, Storage-47648, and Table-31249:

```bash
PYTHONPATH=src python scripts/record_aligned_aim_style.py \
  configs/aim_paper_like_sanity_v1.tsv \
  --models-root /path/to/partnet_models \
  --output-root outputs/aim_paper_like_sanity_v1/recordings \
  --jobs 4
```

The two simpler objects use 200 interaction frames and the two complex objects
use 500 frames, all at 15 Hz. The recorder uses the verified AiM-style static
scan plus monocular interaction protocol and full-limit staggered excitation.
This experiment is called paper-like rather than an exact Table 1 reproduction:
the released AiM repository does not provide object-specific configs for these
IDs, and our common evaluator reports observed-point IoU rather than the
paper's volumetric reconstructed-part IoU.

All four runs completed. Mean observed-point metrics are Point IoU `0.186`,
ARI `0.023`, RI `0.517`, and geometry coverage at 2% of the object bounding-box
diagonal `0.686`. Storage-47648 remains under-segmented at 2 predicted parts
versus 7 GT parts. These results show that increasing the interaction from 120
frames to the paper-like 200/500-frame schedule is not sufficient to close the
cross-protocol segmentation gap. They do not constitute a numerical
reproduction of AiM's volumetric-IoU table. See
`outputs/external_baselines_v1/aim_paper_like_sanity_v1/summary.md`.

## Ditto

**Official repository:** <https://github.com/UT-Austin-RPL/Ditto>

Actual demo code consumes posed depth images before and after an interaction,
fuses them into point clouds, and runs a pretrained feed-forward
PointNet++/ConvONets model. The network has static/mobile occupancy and
segmentation outputs plus one revolute/prismatic joint decoder.

### Feasibility

- A shared-3view start/end adapter is straightforward.
- It does not consume a temporal sequence.
- The released model is category-trained rather than instance-optimized.
- The public architecture represents one moving part and one joint, so
  arbitrary multi-part PartNet objects cannot be compared fairly.

Use Ditto only on a clearly labeled single-joint subset, or cite its reported
results. Do not include it in the main arbitrary multi-part table.

The shared exporter writes `ditto/two_state_points.npz` with 8192 normalized
object points per state, matching the official depth-map demo. It uses only
object masks, depth, and calibrated cameras; no GT part labels are included.

## PARIS

**Official repository:** <https://github.com/3dlg-hcvc/paris>

The loader reads `start` and `end` multi-view RGB(A), camera transforms, and
foreground masks. Code paths are selected by separate revolute and prismatic
configs. The model reconstructs one static and one movable neural field.

### Feasibility

- Our recordings can export the required two-state multi-view images.
- Shared 3-view is much sparser than the official two-state capture.
- The two-part representation cannot express a general multi-part tree.
- Running separate revolute/prismatic systems changes type inference into
  model selection unless both are run under a predeclared rule.

PARIS is suitable for a single-joint two-part subset, not the main multi-part
quantitative comparison.

The exporter writes PARIS-native:

```text
paris/
  start/train/0000.png
  start/camera_train.json
  end/train/0000.png
  end/camera_train.json
```

The first run uses `configs/se3.yaml` so joint type is estimated rather than
provided from GT. A later specialized revolute/prismatic refinement must use a
predeclared model-selection rule and cannot select the better GT result.
The released `BaseSystem` nevertheless loads `motion_gt_path` unconditionally;
the SE(3) training loss does not consume it, but validation/export metrics do.
Therefore an aligned run needs a GT motion sidecar for evaluation and must
record that distinction instead of claiming the repository never reads GT.

## DTA

**Official repository:** <https://github.com/NVlabs/DigitalTwinArt>

The code reconstructs two RGB-D states, loads foreground masks and camera
poses, generates/loads LoFTR pixel correspondences, and optimizes part
geometry and articulation. It can represent multiple moving parts and exports
part meshes and revolute/prismatic axis meshes.

### Feasibility

- A two-state adapter from shared-3view recordings is possible.
- Official training requires `--num_parts`; this is an exact oracle in our
  setting and must be reported as such.
- The released articulation stage expects precomputed LoFTR correspondences.
  Omitting them is allowed by the loader but is not a faithful formal run.
- Three views per state may be below the intended scan coverage.
- The method does not use the temporal frames between the two states.

### Environment validation

The official Python 3.8 / PyTorch 1.11.0+cu113 stack was validated on an A100
using CUDA 11.3, Kaolin 0.14, PyTorch3D 0.7.4, and GCC 10. The released entry
point completed one reconstruction step for both states, exported canonical
meshes, loaded state SDFs, and completed one articulation step. This establishes
execution feasibility only; it is not a benchmark result.

The official LoFTR preprocessing path was also validated using repository
revision `df7ca80f917334b94cfbe32cc2901e09a80e70a8` and the outdoor DS
checkpoint. A top-2 smoke generated correspondences for all 200 views and
produced finite `init_corr` and `last_corr` losses. Formal runs retain the
released top-30 setting; the top-2 output is not a result.

The development image exposes a native cleanup bug when `mesh_to_sdf` 0.0.15
converts both meshes sequentially in one process. Each conversion succeeds in
isolation with identical outputs, so formal runs may precompute the unchanged
SDF cache in separate processes and must record that compatibility workaround.
The process can also abort during final native-library teardown after writing a
valid checkpoint. Success must therefore be determined from expected artifacts
and completed-stage logs, not from exit status alone.

DTA is the next most valuable implementation after ReArt, but it should be
reported as `GT-part-count oracle` unless an official non-oracle selection
procedure is found.

## ArtGS

**Official repository:** <https://github.com/YuLiu-LY/ArtGS>

The dataset reader consumes start/end multi-view images, alpha masks, and
camera transforms. The pipeline trains coarse Gaussians, predicts joint types,
then trains the articulated model. However, `train.py` reads a per-scene joint
type string and derives `num_slots` from it. The repository also documents
manual correction of inaccurate part centers for real examples.

### Feasibility

- Two-state rendering can be exported from our data.
- The current full-model entry point uses per-scene joint-type/slot metadata.
- Manual center correction is incompatible with an automatic held-out
  benchmark.
- Output meshes, segmentation, type, axis, and pivot are otherwise adaptable.

Treat the released full pipeline as oracle-assisted unless the official
joint-type predictor and automatic center initialization can be run end to end
without per-instance edits. It is lower priority than DTA.

## VideoArtGS

**Official repository:** <https://github.com/YuLiu-LY/VideoArtGS>

The released pipeline performs canonical Gaussian initialization, track-only
deformation initialization, and final joint optimization. It consumes
`data.npz` (video, depth, intrinsics, extrinsics, masks), `filtered.npz`
(TAPIP3D 3D tracks and visibility), and `joint_infos.json`.

The latter is a material prior: the code sets `num_slots` and joint types from
the supplied inventory. The README proposes GPT-4o/VLM prediction and permits
manual correction. Results must therefore be labeled by inventory source:

- `vlm`: reproducible VLM-assisted baseline;
- `manual`: human-assisted diagnostic, not a non-oracle benchmark;
- `gt_oracle`: count/type/topology oracle.

Our adapter can use recorded depth/cameras instead of VGGT estimates. This is
a controlled GT-camera/depth variant, not the native VGGT pipeline, and must be
reported separately.

## GaussianArt

**Official repository:** <https://github.com/shenlc19/GaussianArt>

The repository is public and runnable, but the released entry point is not
non-oracle. `run.py` reads `gt/trans.json`, derives `num_parts`, identifies
prismatic indices, then performs depth-semantic initialization and joint
optimization. Per-view semantic maps initialize Gaussian part weights.

The shared exporter prepares RGB-D and camera files but intentionally leaves
semantic initialization and `gt/trans.json` absent. A run may proceed only
after supplying either:

- predicted Art-SAM-equivalent semantic masks plus a non-GT part-count/type
  source, if an official path becomes available; or
- GT semantic/motion metadata, labeled `oracle-assisted`.

Do not present GaussianArt oracle-assisted numbers in the non-oracle Ours
comparison without a separate column.

## Deployment status

The official repositories were checked out on the development host at the
commits listed above. Modern-CUDA compatibility smoke tests use PyTorch
`2.4.1+cu121` on A100:

- GaussianArt CUDA rasterizer and `simple-knn` compile successfully; the
  official `train.py` CLI imports.
- VideoArtGS imports with `gsplat 1.5.3`; the official `init_cano.py` CLI
  imports.
- PARIS imports in an isolated compatibility environment using Lightning
  `1.7.7`, torchmetrics `0.9.3`, and nerfacc `0.3.2`. A real package reaches
  model construction, then correctly stops because the currently exported
  package lacks the required GT evaluation motion sidecar.
- Ditto model, ConvONets generator, and joint-estimation modules import in a
  PyTorch 2.4/CUDA 12.1 compatibility environment after installing a matching
  `torch-scatter` wheel. The project-side inference adapter now follows the
  released demo's `Generator3D`, dense joint decoding, and voting path.
- Both official pretrained-model Box links and the released training-data link
  return HTTP 404 as of 2026-07-31. The five applicable two-part inputs are
  exported, but inference is recorded as a checkpoint-acquisition failure
  rather than being run with random weights.

## Recommended Order

1. **VideoArtGS:** run the continuous-video controlled adapter with VLM and
   GT-oracle inventories as separate settings.
2. **GaussianArt:** smoke-test the released oracle-assisted pipeline; do not
   hide semantic/count/type inputs.
3. **PARIS / Ditto:** run only the two-part one-joint subset.
4. **DTA:** implement a two-state adapter and explicitly report GT-part-count
   oracle results.
3. **Ditto:** run only on a single-joint category-compatible subset.
4. **PARIS:** use a two-part protocol ablation or reported paper results.
5. **ArtGS:** defer direct comparison until its per-scene slot/type dependency
   can be removed by an official non-oracle path.

## ReArt Shared-3view Pilot

The first four-object pilot uses four uniformly sampled timesteps
(`0, 30, 60, 90`) from each 120-frame episode. At each timestep, all three
RGB-D views are fused into a dense world-frame foreground point cloud. ReArt
receives 4096 sampled xyz points per timestep, a fixed 10-slot capacity, and
the official flow and assignment losses. The official relaxation and
kinematic projection stages use 2000 and 200 iterations respectively.

ReArt completed the full pipeline on Drawer, Refrigerator, and Microwave.
CoffeeMachine failed in the official tree projection because the selected root
was absent from the predicted graph. The failure is retained without patching
the graph construction.

On the three common-success objects:

| Method | Point IoU | ARI | Undersegmentation rate |
|---|---:|---:|---:|
| Hybrid | 0.675 | 0.690 | 0.333 |
| AiM | 0.289 | 0.156 | 0.667 |
| ReArt | 0.428 | 0.390 | 0.333 |

ReArt under-segments the Drawer (3 predicted labels for 11 GT-domain labels)
but over-segments the Refrigerator (9/3) and Microwave (7/2). Therefore the
pilot does not support a simple "ReArt always under-segments" explanation.
The dominant issue is selecting a valid number of coherent parts from the
fixed slot capacity.

These figures are integration-pilot results, not final benchmark estimates.
See `outputs/external_baselines_v1/comparison_summary.md` for the paired and
own-success-set tables.

### Extended 20-object result

The same fixed configuration was subsequently run once on the existing
20-object AiM shared-3view subset. Failed objects were not retried or selected
by seed.

| Method | Success | Point IoU | ARI | Exact count | Undersegmentation |
|---|---:|---:|---:|---:|---:|
| Hybrid | 20/20 | 0.640 | 0.711 | 0.500 | 0.400 |
| AiM | 18/20 | 0.254 | 0.088 | 0.167 | 0.833 |
| ReArt | 14/20 | 0.457 | 0.363 | 0.071 | 0.571 |

On the 12 objects successfully completed by all three methods, Point IoU / ARI
are 0.757 / 0.824 for Hybrid, 0.326 / 0.109 for AiM, and 0.443 / 0.363 for
ReArt.

All six ReArt failures occur during official kinematic tree projection with
`networkx.exception.NodeNotFound: Source 0 not in G`. Their relaxation-stage
segmentations contain one predicted component, so no valid articulation graph
can be constructed. ReArt also has a low exact-count rate on successful
objects, frequently over-segmenting simple objects despite covering the GT
parts. The results therefore support using ReArt as a meaningful 4D-geometry
baseline, but not as a reliable native source of joint type, axis, pivot, or
directed topology metrics.

The next implementation priority remains DTA because it supports multi-part
articulation and native joint parameters. It must be labeled as a
GT-part-count oracle baseline unless an official non-oracle model-selection
path is identified. Ditto and PARIS should be restricted to a declared
single-joint/two-part subset. ArtGS should use cited reported results until an
official end-to-end path no longer requires per-scene type/slot metadata and
manual center correction.

## Comparison Rules

- GT part masks and joints are evaluation-only.
- ReArt, AiM, and Hybrid use the same source frame and observed-reference-point
  Point IoU/ARI evaluator.
- Tables distinguish native-protocol reproduction from equal-input adapters.
  Matching only frame/view count is not sufficient to claim a native protocol.
- Native, converted, unsupported, and oracle metrics are labeled separately.
- Published mesh/voxel IoU is never placed in the observed-point IoU column.
- Baseline thresholds and optimization logic are not modified per object.

## Object-Aligned Generalization Benchmark

The aligned 20-object comparison is intentionally labeled
`cross-protocol generalization`, not `paper reproduction`.

| Method | Requested | Success | Point IoU | ARI | RI |
|---|---:|---:|---:|---:|---:|
| Ours Hybrid | 20 | 20 | 0.731 | 0.846 | 0.940 |
| ReArt native-format | 20 | 19 | 0.402 | 0.324 | 0.687 |
| AiM-style | 20 | 20 | 0.280 | 0.083 | 0.552 |

Object identity, semantic GT IDs, and metric implementations are aligned.
Inputs remain method-specific. This benchmark tests whether each method
generalizes to a broader, more complex held-out PartNet distribution.

It does not reproduce the published tables:

- AiM uses selected objects, object-specific large joint excursions,
  200-frame interactions for simpler objects, 500 frames for complex objects,
  and volumetric reconstructed-part IoU.
- The aligned benchmark uses 120-frame interactions and observed-point
  segmentation IoU/ARI/RI, with many 5--11-part instances.
- ReArt's published Sapiens result uses the official Sapiens sequence
  distribution and reports RI. The aligned benchmark renders PartNet objects
  into the native four-frame input format.
- ReArt's learned correspondence network was trained for the Sapiens/RoboArt
  distributions, so the aligned PartNet result also measures domain shift.

Two separate sanity reproductions are required before making claims about
implementation fidelity:

1. Run AiM on the paper-listed PartNet objects `47024`, `11304`, `47648`, and
   `31249`, preserving the published joint ranges and 200/500-frame schedules.
2. Run released ReArt inference/evaluation directly on official Sapiens test
   sequences and report its native RI.

These checks answer whether low aligned-baseline scores come from incorrect
integration or from the harder cross-dataset evaluation distribution.
## ArtiPoint bidirectional evaluation

ArtiPoint is evaluated in two complementary domains rather than using real
`scene28` as the sole comparison case. `scene28` is retained as a real-data
failure diagnostic because Track2Art also has poor joint-axis observability on
that sequence: sparse/depth-noisy moving points make it unsuitable for
separating baseline quality from input quality.

The bidirectional evaluation uses:

- `partnet_10944` (`refrigerator`, two parts) for the PartNet-Mobility side.
  This object belongs to the aligned external-baseline manifest and has clean
  Track2Art Hybrid segmentation and kinematics. Existing AiM, ReArt, DTA,
  ArtGS, VideoArtGS, GaussianArt, PARIS, and Ditto records are reused. ArtiPoint
  requires an object-mask cue adapter because synthetic PartNet recordings do
  not contain the hand interaction assumed by its native front end. The adapter
  may use an object mask and interaction interval, but never GT part labels or
  GT joint parameters.
- Arti4D `rh078/scene_2025-04-09-10-38-38/right-drawer-1` for ArtiPoint's
  native side. The official RGB-D frames, camera trajectory, scene metadata,
  and provided interaction cue are preserved. Track2Art is evaluated using the
  same object-mask observation domain. ReArt and AiM may be evaluated when
  their 4D-point-cloud or monocular-video adapters preserve the native
  observations. Two-state multiview methods are marked unsupported unless the
  required calibrated start/end multiview observations actually exist.

The resulting comparison is reported with an explicit protocol label:
`native`, `object_mask_adapter`, `applicable_adapter`, or `unsupported_input`.
Native and adapted rows must not be averaged without retaining this label.

The first completed pilot confirms that ArtiPoint itself is healthy on its
native domain: on `rh078/right-drawer-1`, it predicts the correct prismatic
type with 2.09-degree undirected axis error. The PartNet object-mask adapter on
`partnet_10944` fails at the unchanged official visibility filter. Although 205
tracks are classified as dynamic, no track remains visible for at least 50% of
the 119-frame moving-camera sequence (median 13 visible frames, maximum 54).
Projected depth is valid for 88.1% of samples, ruling out a depth-unit failure.
This result is retained as a protocol/domain failure; the official threshold is
not relaxed.
