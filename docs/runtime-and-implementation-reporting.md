# Runtime and Implementation Reporting

Runtime must be reported with explicit stage boundaries and hardware. A single
aggregate number is not comparable when one method includes feature extraction
and another reports only its prediction head.

## Ours

Report both cold single-object latency and batch throughput.

1. **Feature generation**
   - CoTracker: model load, RGB loading, accelerator forward, RGB-D lifting,
     and feature serialization.
   - TAPIP3D: input load, model/tensor setup, accelerator forward, temporal
     pooling, and serialization.
2. **Learned feedforward backend**
   - The relation inferencer includes the slot backbone and predicts topology,
     joint type, axis, and pivot. Do not add standalone slot latency to it.
3. **Analytic backend**
   - Track-based SE(3) part-pose fitting and analytic joint fitting are timed
     separately from the learned relation head.
4. **Excluded reporting work**
   - HTML viewer generation and GT metric evaluation are not inference.

Each run records hostname, CPU count, accelerator model/memory, PyTorch/CUDA
versions, thread environment, inter-object worker count, frame/view count, and
track count. Accelerator operations are synchronized at timing boundaries.

## Parallelism

- CoTracker batches seed queries within each view. Views are processed
  sequentially for a single object.
- TAPIP3D uses one persistent process per GPU and distributes object-view jobs
  across GPUs. CPU preparation/import use a thread pool.
- Slot and relation inference use one subprocess per object. `--workers N`
  improves dataset throughput but is not an `N`-way speedup of one object.
- Analytic pose/joint fitting is single-object CPU work; independent objects
  can run concurrently.

## Required Method Parameters

The paper and supplementary tables should include:

- input frames, effective FPS, views, image resolution, and point/track count;
- tracker checkpoint, frame stride, query batch size, and TAPIP iterations;
- slot count, hidden width, encoder/decoder depth, and attention heads;
- relation trajectory samples, maximum geometry tracks, and edge threshold;
- optimization iterations for iterative baselines;
- point count, voxel resolution, TSDF settings, and mesh face cap when used;
- oracle inputs such as known part count;
- hardware, worker count, precision, and whether model loading is included.

Mesh face count and iterative optimization steps are not applicable to the
learned Ours segmentation/kinematics path and must be reported as such rather
than as zero.

Generate the current report with:

```bash
PYTHONPATH=src python scripts/build_ours_paper_runtime_report.py \
  --backend-profile-root path/to/profiled_slot_benchmark \
  --tracker-profile-root path/to/profiled_recordings
```

The report intentionally leaves unmeasured stages as `n/a`. Legacy
`benchmark_metrics.json` files contain slot-only latency and must not be
described as end-to-end or slot-plus-relation latency.
