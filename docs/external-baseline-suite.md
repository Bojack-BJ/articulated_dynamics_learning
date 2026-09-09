# External Baseline Suite

The suite uses one object-aligned PartNet manifest and one common evaluator while
preserving each method's declared input protocol. It is not an identical-input
comparison across every method.

## Final benchmark policy

The completed `full20` manifest remains the primary requested benchmark. It
must not be reduced after observing which external methods succeed: doing so
would create survivorship bias and systematically remove difficult objects.

For expensive baselines, the suite may additionally use a frozen stratified
`core16` subset with the following target distribution:

| Complexity | Requested objects |
|---|---:|
| 2 parts | 5 |
| 3-4 parts | 5 |
| >=5 parts | 6 |

The `core16` selection must be committed before running a method and must
balance category, revolute/prismatic motion, moving-part size, and
single-/multi-joint structure. A failed object remains in the denominator and
must not be replaced by another object after execution.

Every final result should include both:

1. **Requested-set results:** success rate plus metrics over successful runs,
   with every failure retained and categorized.
2. **Common-success results:** conditional metrics on the exact object
   intersection where the compared methods succeeded.

Common-success results are diagnostic and cannot replace requested-set success
rates. PARIS and Ditto remain restricted to the frozen two-part/one-joint
subset and are not included in multi-part denominators.

### Method grouping

Methods must be separated by inference assumptions rather than placed in one
undifferentiated leaderboard:

| Group | Methods | Interpretation |
|---|---|---|
| Non-oracle part discovery | Ours, AiM, ReArt | Main arbitrary-part discovery comparison |
| Exact-count oracle | DTA, ArtGS | Uses GT part count/slot count |
| Inventory/semantic oracle | VideoArtGS, GaussianArt | Uses GT joint inventory or semantic part initialization |
| Restricted-capacity | PARIS, Ditto | Two-part/one-joint scope only |

GaussianArt's predicted part count is not a discovery metric because the
released path consumes GT semantic part initialization and motion metadata.
PARIS does not consume GT point labels, but its one-static/one-moving
representation is a structural two-part restriction.

### Current result reconciliation

The generated suite CSV files are the source of truth for imported common
metrics. The current imported state is:

| Method | Imported status | Notes |
|---|---:|---|
| Ours Hybrid | 20/20 | Non-oracle |
| AiM aligned | 20/20 | Cross-protocol generalization, not exact-paper reproduction |
| ReArt | 19/20 | One released graph-projection failure |
| DTA | 17/20 | Three complex objects lack usable final common prediction artifacts |
| ArtGS | 20/20 | Completed after retry; GT part-count oracle |
| VideoArtGS | 14/20 | GT joint inventory assisted |
| GaussianArt | 0/20 common metrics | Native oracle-assisted aggregates exist but common segmentation is not imported |
| PARIS | 0/5 common metrics | Native results exist for four objects but the common segmentation adapter is incomplete |
| Ditto | 0/5 | Official checkpoint/deployment dependency unresolved |

Do not overwrite this table with a worker-local status report without first
importing its per-object artifacts and rebuilding the suite. In particular,
older reports that describe ArtGS as incomplete are stale.

Before freezing paper tables, resolve two GT-domain discrepancies:

- `partnet_7263`: manifest count is 3, while the current observed reference
  domain exposes only 2 parts.
- `partnet_45332`: manifest count is 5, while the current observed reference
  domain exposes only 4 parts.

The final evaluator should use a multi-frame union GT domain or explicitly
mark unobserved parts. It must not silently redefine GT complexity from a
single source frame.

## Aligned object tiers

The comparison uses three explicit tiers:

| Tier | Purpose | Methods |
|---|---|---|
| `core12` | Historical smoke/pilot set; not the final paper subset | All applicable methods |
| `core16` | Frozen stratified subset for baselines whose full20 cost is prohibitive | All applicable methods |
| `full20` | Final broad held-out comparison | Multi-part-capable methods |
| `two_part` | Native one-joint scope | PARIS and Ditto |

`core12` covers coffee machine, door, drawer, microwave, oven, and refrigerator
objects, but is retained only for historical pilot reproducibility. PARIS and
Ditto are not extrapolated to multi-part objects merely to increase table
coverage.

Generate the method/object applicability matrix and metric contracts:

```bash
PYTHONPATH=src python scripts/build_external_baseline_aligned_manifest.py \
  outputs/external_baseline_suite_v1/manifest.csv \
  --output-dir outputs/external_baseline_suite_v1/aligned_v1
```

The generated `method_object_manifest.csv` declares, for every method/object
pair, the acquisition protocol, applicability, oracle inputs, predicted
metrics, conditional metrics, and unsupported scope. This matrix is the source
of truth for dispatching baseline jobs.

## Acquisition packages

| Package | Methods | Observations |
|---|---|---|
| Continuous interaction | Ours, AiM, VideoArtGS, Articulat3D | Calibrated RGB/RGB-D interaction sequence |
| Two-state multi-view | DTA, ArtGS, PARIS, GaussianArt | 100 fixed-state views at start and end; RGB, depth, object mask, cameras |
| 4D point cloud | ReArt | Native or adapted temporal point-cloud sequence |

Create a two-state package from an `aim_style_fixed_end` episode:

```bash
PYTHONPATH=src python scripts/export_two_state_baselines.py \
  outputs/recordings/partnet_object/episode.json \
  --output-dir outputs/external_baseline_suite_v1/per_object/partnet_object/acquisition/two_state \
  --object-id partnet_object \
  --category drawer \
  --gt-part-count 4
```

The PARIS and Ditto observation directories never contain GT part masks.
GaussianArt is exported as a scaffold because its released runner additionally
requires semantic part initialization and `gt/trans.json`; these are never
fabricated or silently copied into a non-oracle package. Exact GT part count is
stored separately in `oracle_requirements.json`.

## DTA contract

The adapter writes the released loader's `cam_K.txt`, `init_keyframes.yml`,
`color_segmented`, `depth_filtered`, and `mask` layout. DTA requires
`--num_parts`; suite results must therefore be labeled `GT-part-count oracle`.
The released optimization also expects LoFTR correspondence files under
`correspondence_loftr/no_filter`. A run without those files is an execution
smoke only, not a formal DTA result.
The validated matcher uses the official LoFTR repository at
`df7ca80f917334b94cfbe32cc2901e09a80e70a8` and its outdoor DS checkpoint.
The upstream DTA repository declares the submodule URL but does not pin a
gitlink commit, so the selected revision must be stored with every run.

The released environment was validated on an A100 with Python 3.8,
PyTorch 1.11.0+cu113, CUDA 11.3, Kaolin 0.14, the official PyTorch3D 0.7.2
binary for PyTorch 1.11/CUDA 11.3, and GCC 10. A formal pilot completed all
4,000 reconstruction and articulation steps and produced the released part
meshes and both motion hypotheses. The process subsequently aborted during
native teardown/export. The suite reports this as
`completed_with_exporter_failure` only when all declared step-4,000 prediction
artifacts exist; an earlier failure remains a hard failure.

## Appendix Visualization Audit

Generate the per-object result and visualization-support appendix with:

```bash
PYTHONPATH=src python3 scripts/build_external_baseline_appendix.py
```

The generated index is:

```text
outputs/external_baseline_suite_v1/appendix_visualization_v1/index.html
```

Each aligned object receives a page comparing segmentation metrics, part
counts, joint type, axis angle, axis-line error, failure class, and native
artifact availability across methods. The audit deliberately distinguishes:

- `raw_prediction_available`: native labeled geometry can be adapted into a
  3D viewer;
- `metrics_only_raw_sync_required`: metrics are local, but native prediction
  geometry must be retained or synchronized from the development machine;
- `axis_values_available`: native joint parameters are available;
- `axis_raw_adapter_required`: the method may produce kinematics, but the
  current result has not exported them into the common viewer convention;
- `unsupported_by_method`: the released method does not select or output the
  required axis quantity;
- `method_failure`: no visualization is fabricated for a failed inference.

`raw_artifact_retention.json` is the deletion/archive whitelist. Do not remove
the listed labels, trajectories, joint parameters, or prediction geometry
until the corresponding appendix viewer has been generated and checked.

The released non-GT DTA path exports both prismatic and revolute hypotheses but
does not select a joint type. Common segmentation and geometry metrics are
therefore supported, while joint-type accuracy and type-conditional axis
metrics remain unsupported. Selecting the lower-error hypothesis with GT would
be an oracle diagnostic and must not enter the primary baseline table.

The project now provides a separate no-GT converted diagnostic:

```bash
PYTHONPATH=src python scripts/select_dta_joint_hypotheses.py \
  outputs/external_baseline_suite_v1/per_object/partnet_object/dta/adapter/predictions.json \
  --target-point-cloud path/to/end_state_object_cloud.ply
```

It enforces the released transform as either pure translation or rotation
about the exported hinge line, replays the moving-part surface into the target
state, and compares robust nearest-surface residuals. It reports both
residuals, their margin, and `ambiguous` for low-motion or weak-margin cases.
This is labeled `DTA + no-GT replay model selection (project adapter)` and does
not replace the official dual-hypothesis baseline.

After running the selector, aggregate the converted results with:

```bash
PYTHONPATH=src python scripts/summarize_dta_replay_selection.py
```

The current 17-object run contains 47 hypotheses: 21 prismatic, 20 revolute,
and 6 ambiguous. Joint-type accuracy is intentionally left pending until the
predicted DTA parts are matched to common GT joints; replay selection itself
does not use GT types or axes.

The appendix viewer consequently draws both DTA hypotheses for each recovered
part: revolute as a solid line and prismatic as a dashed line. These are useful
for qualitative inspection but are not a unique axis prediction. This applies
to the 17 successful DTA runs; the remaining three runs are method failures and
receive no fabricated axes.

AiM axes in the native comparison viewer are read directly from
`motion.json`. ReArt does not natively output joint type or axis in
`result.pkl`; its dashed axes are diagnostic analytic fits to the predicted
parent-child relative SE(3) trajectories and are labeled
`converted_from_native_part_poses`. This conversion succeeds for 19/20 aligned
objects. `partnet_102055` has no sufficiently observable relative motion, so
its axis remains unavailable.

Current appendix visualization coverage is:

| Method | Part viewers | Axis visualization |
|---|---:|---:|
| Ours Hybrid | 20/20 | 20/20 native predictions |
| AiM | 20/20 | 20/20 native `motion.json` |
| ReArt | 20/20 | 19/20 converted from native poses |
| DTA | 17/20 | 17/20 dual hypotheses, no type selection |
| ArtGS | 20/20 | 20/20 native predictions |
| VideoArtGS | 14/20 | 14/20 native predictions |
| GaussianArt | 0/20 | aggregate axis metrics only; native geometry absent |
| PARIS | 4/20 | 4/20 native two-part predictions |
| Ditto | 0/20 | no successful aligned predictions |

## Runtime reporting

Generate the measured runtime audit with:

```bash
PYTHONPATH=src python scripts/summarize_external_baseline_runtime.py
```

The resulting `runtime_comparison_v1/` directory contains per-object CSV,
aggregate CSV/JSON, and a Markdown table. Timing scope is mandatory:

- Ours currently measures feedforward slot/relation inference and excludes
  tracker feature extraction;
- DTA and ArtGS measure the official end-to-end optimization command;
- ReArt uses its native inference timer;
- missing timings remain `n/a` rather than being reconstructed from file
  modification times.

These scopes must not be collapsed into a single end-to-end speedup claim until
all methods are re-profiled from prepared input to final prediction on the same
hardware.

## ArtGS contract

The adapter writes start/end RGBA, depth, and Blender/OpenGL camera transforms.
The released coarse/predict path can predict joint types, but its scripts still
read a per-scene `num_slots`. Results must be labeled `GT-part-count oracle`.
The released implementation also consumes depth during initialization/losses;
the suite reports the actual released RGB-D path rather than describing it as
RGB-only.

## Two-state runners

Create a dry-run plan before using external repositories:

```bash
PYTHONPATH=src python scripts/run_two_state_external_baseline.py dta \
  outputs/external_baseline_suite_v1/per_object/partnet_object/acquisition/two_state \
  --repo /path/to/DigitalTwinArt \
  --output-dir outputs/external_baseline_suite_v1/per_object/partnet_object/dta \
  --object-id partnet_object \
  --gt-part-count 4 \
  --config-dir config/release
```

Replace `dta` with `artgs`, `gaussianart`, `paris`, or `ditto`. Ditto also
requires `--checkpoint`. PARIS and Ditto reject any object whose GT count is
not exactly two, preventing accidental use outside their native capacity.
GaussianArt refuses to become a non-oracle result merely because RGB-D files
exist: semantic initialization, GT motion metadata, and part count remain
explicit oracle fields.

Add `--run` only after inspecting `run_plan.json`.
The runner records the exact GT part-count oracle and leaves unsupported or
not-yet-adapted metrics empty. ArtGS jobs require isolated repository workspaces
when run concurrently because the released scripts mutate shared per-scene
JSON configuration files.

The formal suite requires exactly 100 fixed-state views at both start and end.
Older recordings with only 12 end views are suitable for smoke tests only and
must not be reported as formal DTA/ArtGS results.

## VideoArtGS contract

Export a calibrated monocular interaction and existing 3D tracks:

```bash
PYTHONPATH=src python scripts/export_videoartgs_baseline.py \
  outputs/recordings/object/episode.json \
  outputs/recordings/object/pointcloud_4d_partseg/part_tracks.json \
  configs/videoartgs/object_joint_infos.json \
  --output-dir data/external_suite/aligned/object \
  --joint-inventory-source vlm
```

The package contains the official `data.npz`, `filtered.npz`, and
`joint_infos.json` inputs. Recorded depth/cameras form a controlled-input
variant; a native run should instead retain the official VGGT/Depth-Anything
camera-depth stage. Joint inventory source is mandatory because VideoArtGS
uses it to set slot count, type, and parent topology. `manual` and `gt_oracle`
runs are not grouped with non-oracle results.

## Result schema

Each result lives at:

```text
per_object/<object_id>/<method>/metrics.json
```

Unsupported outputs remain absent. In particular, a method without directed
topology does not receive a synthesized Edge F1, and a segmentation-only method
does not inherit an analytic axis from Ours.

Axis angular error uses `acos(abs(dot(pred, gt)))` and is aggregated only for
type-correct matched joints. Revolute axis-line distance is normalized by the
object bounding-box diagonal. Oracle-initialized segmentation, including the
released GaussianArt semantic initialization, is reported separately from the
non-oracle segmentation table.

Initialize or refresh suite summaries:

```bash
PYTHONPATH=src python scripts/build_external_baseline_suite.py \
  outputs/external_baselines_v1/aligned_comparison_v1/aligned_per_object.csv \
  --output-dir outputs/external_baseline_suite_v1
```
