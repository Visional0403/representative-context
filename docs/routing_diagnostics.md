# Routing costs and spatial coverage

The routing diagnostics use SegFormer/UAVid at 2048 by 2048 and
Twins/Cityscapes at 1024 by 1024, with k=14 and seed 42. They share the
trained model's Q/K/V and output projections and spatial reduction.
All inputs and global attention blocks are listed in
`results/01_context/routing/CASE/coverage.json`.

## Selection and execution

The methods are fixed dispersed sampling (`point32`), eight-dimensional
pre-match top-k (`proxy8`), exact-score top-k (`exact`) and pooled-region
routing (`region`). `cover1` and `cover16` supply fixed spatial controls in
the coverage diagnostic. A method retains the maximum balanced-group
cardinality at the same capacity, subject to region-size rounding.

SegFormer uses Torch top-k and chunks bounded by 4,194,304 scoring elements.
Twins uses FP32 Triton top-k and 16,777,216 scoring elements. These limits
refer to element counts, equivalent to 16 and 64 MiB of FP32 scores.
Both execute the chosen token interactions with the fused selected-attention
kernel. The region path pools queries and keys, selects regions by pooled
dot products, gathers candidate KV, applies efficient SDPA and restores
query order. Its routes are shared across heads. `region_ref` uses the same
chosen interactions with the explicit reference aggregation.

## Independent measurement scopes

```bash
python scripts/probe_routing.py --case segformer_uavid --phase whole --method proxy8 --output outputs/routing_model.json
python scripts/probe_routing.py --case twins_cityscapes --phase cost --output outputs/routing_components.json
python scripts/probe_routing.py --case twins_cityscapes --phase coverage --output outputs/routing_coverage.json
```

Whole-model measurement uses the first image's top-left GPU window through
final logits. It includes selection and all model modules. Five warmup calls
precede twenty CUDA-event observations; a separate three-warmup, ten-call
measurement follows. Dense bookends, sparse methods and the region reference
are separately recorded in `whole.json`.

Component measurement captures each global block's Q/K/V on the same dense
trajectory. Each method has three warmup calls, ten uninstrumented timings
and five instrumented forwards. Selection, candidate organization and
aggregation have explicit regions. The records exclude the common QKV/SR
and output projections. Component medians and uninstrumented block medians
are reported separately. Their sums retain their distinct measurement scopes.

Coverage uses three fixed images: UAVid validation indices 0, 20 and 40,
or the first sorted image from each Cityscapes validation city. Each block
uses 64 fixed queries on an 8-by-8 sampling grid and all attention heads.
The arrays preserve actual selected indices, including selector tie handling.
Distances use normalized KV-grid coordinates; macroregion coverage uses a
4-by-8 partition. The adjacency statistic counts undirected horizontal and
vertical candidate pairs per retained key. Each record also contains local
3-by-3 retention, bounding-box coverage, dense probability mass, top-one
retention and aggregation error. Dense QKV and capacity are paired across
all six methods.

Coverage averages weight each block by heads times its full query count.
Aggregation relative L2 pools error and reference energies with the same
query expansion factor. Rank correlations rank the six layouts within each
fixed query/head; they describe paired measurements rather than independent
scene-level observations.

```bash
python scripts/analyze_routing.py --output outputs/routing_statistics.json
```

The CPU analyzer reads all preserved timings and selected-context arrays,
recomputes coverage and cost summaries, and checks full controlled-layout
validation counts. Use `--limit` and `--max-blocks` on the measurement entry
for restricted checks; its metadata records the resulting scope.

## Controlled clustering at fixed context

The controlled experiment holds each query's total capacity, shifted local
3-by-3 neighborhood, eight global anchors, guarded neighborhood and quadrant
quotas constant. Only remote cluster side length changes: 4, 2 or 1 KV cells.
Each KV-grid cell has its own candidate list, inherited by corresponding
queries. These overlapping per-query contexts are a diagnostic control.

```bash
python scripts/prepare_context_controls.py --case segformer_uavid --output-dir outputs/contexts
python scripts/evaluate_controlled_context.py --prepared outputs/contexts --cluster 1 --output outputs/controlled_validation.json
```

Preparation audits every query on CPU and asserts identical local, anchor
and quota arrays across cluster sizes. Evaluation uses the prepared indices
and checks all global-block calls. UAVid evaluates all seventy images with
2K sliding windows at stride 1024; Cityscapes evaluates all five hundred
images with 1K windows at stride 768. The supplied per-image confusion
matrices and final results are in `results/01_context/controlled/`.
The context kernel preserves projection weights and computes exact
softmax-weighted values over each declared candidate list.

## Spatial-support extent at fixed capacity

The extent controls progressively expand each group's allowed support while
retaining its capacity. `extent0` is a contiguous Morton domain. Level L
merges min(2**L,G) adjacent domains and distributes their tokens among that
many groups. Group cardinalities stay fixed. These controls use divisible
power-of-two grids; the deployment algorithm continues to support the
rectangular and nondivisible grids described in the algorithm documentation.

SegFormer/UAVid has levels zero through six plus dispersed 32-region sampling,
at 2K input windows with 64-by-64 KV grids. Twins/Cityscapes has levels zero
through four, dispersed 32-region sampling and a dense control, at 1K windows
with 32-by-32 KV grids. Both use k=14; the four capacity limits are
64, 128, 256 and 512. Parameters and indexed execution are shared.

```bash
python scripts/measure_extent.py --case segformer_uavid --phase geometry --output outputs/extent_geometry.json
python scripts/measure_extent.py --case twins_cityscapes --phase evaluate --pattern extent4 --output outputs/extent_validation.json
python scripts/measure_extent.py --case twins_cityscapes --phase probe --output outputs/extent_probe.json
python scripts/measure_extent.py --case segformer_uavid --phase speed --pattern extent6 --output outputs/extent_timing.json
python scripts/analyze_extent.py --output outputs/extent_statistics.json
```

The preserved confusion matrices cover every declared layout and all seventy
UAVid or five hundred Cityscapes images. Validation uses native labels and
2K/1024-stride or 1K/768-stride sliding windows, respectively. The separate
timing entry uses the first image's top-left GPU window, five warmup calls
and twenty complete-model CUDA-event observations.

The same-input diagnostic uses five preselected windows per model. Their
image IDs and crop coordinates are explicit in
`results/01_context/extent/CASE/measurements.json`. All global blocks and
heads are evaluated at 256 fixed queries from a 16-by-16 grid. Each layout
receives the same dense-trajectory Q/K/V. Its records preserve aggregation
error and reference energies, relative L2 and retained dense probability
mass. Stage summaries pool squared energies across the declared inputs;
probability masses average block/window measurements equally.

Geometry measures each full KV grid's distance to retained keys with exact
per-group membership. Complete-model coverage averages blocks by retained
head/query/key interaction counts. `--limit N` restricts validation images
or the declared diagnostic windows for a check. The output records its scope.
