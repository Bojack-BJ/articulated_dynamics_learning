# Real-scene annotation and evaluation

The real-data path uses the same filtering configuration for every scene:

- dense CoTracker inference (`frame_stride=1` for the final revolute runs);
- hard rejection of individual 3D segments longer than 5 cm;
- per-frame, per-predicted-part DBSCAN (`eps=3 cm`, `min_samples=5`);
- no query-connected truncation, so a track may recover after an invalid depth sample;
- downstream slot segmentation and both neural and analytic joint estimation consume the masked artifact.

Every run keeps raw tracks, the quality manifest, filtered tracks, slot predictions,
joint predictions, runtime metadata, and HTML viewers. The quality manifest is the
authoritative record of thresholds used by a result.

## Manual annotation

Use either enhanced HTML viewer for combined part and joint annotation. In
**GT part and joint annotation**, enter a GT part ID/name, click a track, then assign
that track or its entire predicted cluster. The **Manual GT part annotation** color
mode provides immediate visual QA. For a joint, add/select it, click a point on its
axis and choose **Use click as A/pivot**, then click a second point and choose
**Use click as B**. Set parent/child GT part IDs and joint type before exporting
`real_scene_gt_annotations.json`.

Part evaluation is performed in track space. Start from propagated SAM masks, then
store manual corrections in `track_labels`, mapping Track2Art `track_id` to GT part ID.
This avoids mixing dense image IoU with sparse track IoU. A combined annotation file is:

```json
{
  "schema_version": 2,
  "annotation_source": "manual",
  "track_labels": {"0": 0, "1": 1},
  "part_names": {"0": "base", "1": "drawer"},
  "joints": [
    {
      "name": "drawer_joint",
      "joint_type": "prismatic",
      "parent_part_id": 0,
      "child_part_id": 1,
      "axis": [1, 0, 0],
      "pivot": [0, 0, 0]
    }
  ]
}
```

Evaluate with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_real_scene_annotations.py \
  outputs/SCENE/slots/motion_part_tracks_slots.json \
  outputs/SCENE/kinematics/joint_inference_neural.json \
  outputs/SCENE/annotations/real_scene_annotation.json \
  --output-json outputs/SCENE/evaluation/manual_gt_metrics.json
```

Segmentation reports Hungarian track IoU, ARI, RI, part count, and annotation
coverage. Kinematics reports directed coverage, type accuracy, undirected axis-angle
error, and bbox-normalized revolute axis-line distance. Prismatic joints do not have
a pivot-position error.

## Real-scene analytic anchor selection

For unlabeled real scenes, run track-based pose estimation with
`--anchor-selection lowest-motion`. The default `metadata` mode is retained for
simulation reproducibility, but learned slot IDs do not encode which slot is the
static base. The lowest-motion mode measures the full per-cluster centroid-path
range, so pull-return and open-close trajectories do not cancel at their
endpoints.

Joint inference similarly uses each track's full path extent for parent/child
orientation. Prismatic direction fitting uses per-track centered path covariance
rather than an aggregate visible-point centroid, which avoids false directions
caused by changing visibility under occlusion.

This heuristic is not a replacement for a clean static slot. It is reliable when
one cluster is a coherent low-motion base, as in `scene29`, but can select a mixed
or spurious slot in over-segmented scenes such as `drawer_hand`. Keep the
metadata mode for reproducible simulation evaluation and inspect the reported
`anchor_motion_range_m` before accepting a real-scene result.

## Real-scene failure backlog

The following fixes can be applied without retraining:

- select the analytic base from full-path cluster motion instead of slot ID;
- fit prismatic direction from centered complete track paths, including
  pull-return trajectories;
- reject large per-timestep depth jumps and spatial track outliers;
- validate each camera's intrinsics/extrinsics and report per-view track support;
- filter the display/background point cloud separately from track DBSCAN.

The remaining failures require representation or training changes and must not
be presented as solved by post-processing:

1. **SO(3) generalization.** The relation head is not rotation equivariant and
   real-scene neural axes exhibit a strong world-axis bias. Retrain with
   consistent object/trajectory/axis SO(3) augmentation, axis-equivariance loss,
   and joint-type-balanced sampling. Report a held-out rotation audit.
2. **Fragmented visibility.** Replace uniformly sampled visible-only trajectory
   descriptors with visibility- and gap-aware segment tokens. Train with temporal
   crops, onset shifts, speed warps, reversals, and synthetic occlusion gaps. A
   shared-joint model should be allowed to group disjoint arc segments.
3. **Multi-part under-segmentation.** `scene31` contains two moving doors but the
   slot model emits only two total clusters. Add slot-existence/model-selection
   supervision and split/merge or joint-aware refinement; downstream joint fitting
   cannot recover a missing part.
4. **Ambiguous local motion.** Short revolute arcs can be indistinguishable from
   translation. Use observability-aware losses and shared hinge fitting over
   spatially separated tracks rather than selecting a type from near-equal replay
   errors.
5. **Mixed base slots.** Lowest-motion anchor selection cannot repair a cluster
   that already mixes base and moving tracks. Add static-base supervision or a
   joint-aware reassignment stage before topology inference.

For every retraining experiment, evaluate by scene and joint type, include the
type-correct axis metric, and retain the analytic full-path estimator as an
independent diagnostic baseline.
