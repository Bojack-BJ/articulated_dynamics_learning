# Real-scene physical-instance split audit

Status: visual audit completed; training/validation/test assignment NOT frozen.

Compared midpoint images from all three cameras using
`scripts/preview_real_instance_overlap.py`. These are visual identity hypotheses,
not serial-number verification. Conservatively keep suspected duplicates together.

| Provisional physical-instance group | Recordings | Visual evidence |
| --- | --- | --- |
| microwave_shared | scene33, microwave_hand_close_2, microwave_hand_open_2 | Matching white housing, diagonal blue top tape, door covering, control panel |
| oven_shared | scene32, ofen_freefall, ofen_freefall_2, ofen_hand_close, ofen_hand_open | Matching black housing, silver handle, right controls, paper-covered door |
| drawer_shared | scene29_case, drawer_hand, drawer_hand2 | Matching grey fabric drawers, pale curved handles, wood top and frame |
| small_box | scene30_small_box | Product carton, distinct from the large carton |
| miniature_double_door | scene31_small_fridge | Small double-door object |
| large_carton | scene34 | Large printed shipping carton |

## Protocol constraints

- Never split frames, camera views, opening/closing clips, or recordings of a
  suspected shared physical instance across instance-disjoint train/val/test.
- The six new clips appear to add observations of two existing objects, not six
  independent objects. In particular, scene33 must be considered even if omitted
  from an earlier seven-scene evaluation list.
- If the new oven clips are used for adaptation, scene32 must leave the unseen-
  instance test. It may be reported separately as same-instance/new-recording.
- If new microwave clips are used for adaptation, apply the same rule to scene33.
- Preserve existing evaluation outputs and GT annotations. Do not silently change
  their protocol labels or overwrite historical results.
- The old real scenes have already guided model debugging: describe their results
  as development-set results rather than a pristine held-out final test.

## Next decision

Use object-group-held-out adaptation experiments on this small collection, with
model selection on a separate validation partition. Report same-instance transfer
separately. A clean final real-world test needs physical objects not used for
adaptation or iterative debugging. Axis annotation can proceed before deciding
the split; storing GT does not authorize using test GT for training.
