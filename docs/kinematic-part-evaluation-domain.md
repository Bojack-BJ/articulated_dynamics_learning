# Kinematic Part Evaluation Domain

## Part Ontology

The simulation recorder defines a kinematic part as a maximal set of MuJoCo
bodies connected only by fixed transforms. A descendant body without a joint is
merged into its nearest ancestor part. A body carrying a revolute, prismatic,
ball, or free joint starts a new part. This rule preserves small movable links;
it does not use visibility, motion magnitude, track coverage, or method output.

Recorder artifacts use:

```text
ontology = maximal-fixed-joint-connected-components
version = 2
```

Each part retains `member_body_ids`, `member_body_names`, and its merged geom
list. Legacy version-1 artifacts can be converted deterministically from their
`fixed_child` and `parent_body_id` fields.

## Evaluation Domains

Two domains are reported separately:

- `all`: every kinematic part in the object topology. A missing or never
  reconstructed part remains an evaluation failure.
- `observable`: parts satisfying fixed GT-acquisition support thresholds. The
  default requires at least 32 points across the reference frames and presence
  in at least two frames.

Observable-domain selection is generated once from the fixed GT acquisition.
It must not be inferred from CoTracker/TAPIP tracks or any baseline prediction.
Motion magnitude is deliberately excluded from segmentation observability;
motion excitation is a separate requirement for kinematic-parameter metrics.

Build a domain manifest with:

```bash
PYTHONPATH=src python3 scripts/build_kinematic_evaluation_domain.py \
  path/to/episode.json \
  path/to/reference_frame_0000.ply path/to/reference_frame_0030.ply \
  --output-json path/to/kinematic_evaluation_domain.json
```

Evaluate Ours or AiM on the same remapped reference and selected domain:

```bash
PYTHONPATH=src python3 scripts/evaluate_aim_pointcloud_iou.py \
  path/to/prediction.json path/to/reference_frame_0000.ply \
  --prediction-format track-json \
  --kinematic-domain-manifest path/to/kinematic_evaluation_domain.json \
  --kinematic-domain observable \
  --output-json path/to/evaluation.json
```

Use `--kinematic-domain all` for the strict topology-complete result. External
baselines use the same manifest and reference points; only prediction transfer
is method-specific.

## Reporting

Main tables should report both domains or clearly name the selected domain.
Never describe results restricted to part IDs represented by one method's
tracks as a common benchmark. Raw MuJoCo body IDs remain diagnostic labels and
must not be treated as semantic parts after fixed-connected collapse.
