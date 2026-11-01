# Capacity allocation controls

The matched-compute comparison uses the two UAVid segmentation checkpoints
and their complete seventy-image validation split. It compares three stage
capacity policies at k=14 and k=16:

- `structural`: the structure-aware capacity rule used by the method.
- `uniform`: equal continuous KV capacity across stages.
- `no_density`: continuous capacity inversely proportional to query stride,
  omitting the KV-reduction density term.

The two controls match the structural policy's QK-plus-AV FLOPs at a 2048 by
2048 input. Their scale search uses network dimensions and integer grouping
counts. It selects group counts with at most one percent relative compute
difference; the search uses neither images nor task metrics. Each capacity
corresponds to the actual balanced grouping and has the same dispersed
32-region layout and execution backend.

Inspect a policy and its actual interactions on a chosen input without a GPU:

```bash
python scripts/inspect_allocation.py --case segformer_uavid --k 14 --policy uniform --output outputs/allocation_plan.json
```

`--height` and `--width` change the shape used for the final interaction
count. The policy itself remains the one matched at 2048 by 2048. Counts use
each stage's actual convolution output, KV grid and nearest-grid query
assignment, including nondivisible dimensions.

## Evaluation

```bash
python scripts/evaluate_allocation.py --case segformer_uavid --output outputs/segformer_allocations.json
python scripts/evaluate_allocation.py --case twins_uavid --output outputs/twins_allocations.json
```

These commands evaluate dense attention followed by the six sparse
allocations on the full split. The 2048-pixel sliding window and 1024-pixel
stride use pixel-space Gaussian weighting with sigma equal to one quarter
of the window side for SegFormer, and uniform overlap averaging for Twins.
SegFormer logits are normalized by the positive sum of covering weights. Image normalization and all trained
parameters follow the task evaluation entries. The runs use batch one, FP32,
disabled matrix-multiplication TF32, enabled cuDNN TF32 and the recorded
per-model cuDNN determinism settings.

`--phase diagnostic` instead uses the seven fixed independent-sequence
windows declared by the mechanism protocol. `--limit N` restricts either
scope to its first N inputs and records the scope; `--k 14` or `--k 16`
selects one resource level. Accuracy evaluation is separate from timing.

## Frozen evidence and statistics

`results/04_budget/allocation/` supplies the twelve exact plans and the
complete per-image confusion counts for full and diagnostic evaluation.
Recompute the full-split comparisons:

```bash
python scripts/analyze_allocation.py --task allocation --case segformer_uavid --output outputs/segformer_allocation_statistics.json
python scripts/analyze_allocation.py --task allocation --case twins_uavid --output outputs/twins_allocation_statistics.json
```

Use `--input PATH` to analyze newly generated evaluation JSON. Each
resampling draw pools integer confusion matrices within the drawn video
sequences before calculating mIoU. The 95 percent paired intervals use
10000 multinomial draws and seed 20261022.

The separate `results/04_budget/grouped_curves/` records contain budget
responses for the complete 500-image Cityscapes validation split grouped
into 235 acquisition sequences, five fixed SegFormer/UAVid windows and
seven fixed Twins/UAVid windows. Their manifests define the statistical
units, configurations and scope. Recompute a response and its intervals:

```bash
python scripts/analyze_allocation.py --task curves --case segformer_cityscapes --output outputs/cityscapes_budget_statistics.json
```

The arrays preserve confusion counts before metric computation. The analysis
reports each source group's metric, pooled metrics, successive budget gains
and paired intervals. Full-split curves and fixed-window diagnostics retain
their respective scopes.
