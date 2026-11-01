# Reproduction map

The repository provides task inference and independent statistical analysis
of paired measurements. New outputs go to `outputs/`. The supplied `results/`
files contain the numerical evidence underlying the paper's figures and
supplementary tables.

For model-based reproduction, first obtain the three study-trained
segmentation checkpoints using [the weight preparation instructions](weights.md#study-trained-segmentation-checkpoints)
and prepare the corresponding datasets. The release files contain complete
trained backbones and prediction heads; dense and sparse paths share each task
state. The entries below use their configured `weights/` paths. Statistical
recomputation of the supplied numerical records uses `results/` directly.

| Scientific comparison | Numerical evidence | Analysis and model instructions |
| --- | --- | --- |
| Selection costs and spatial layouts | `results/01_layout/`, `results/01_context/` | [Selection controls](selection_controls.md), [routing diagnostics](routing_diagnostics.md), `scripts/measure_extent.py` |
| Context content and spatial correlation | `results/02_mechanisms/` | [Mechanisms](mechanisms.md), context-restoration and cross-task entries |
| Spatial capacity and task tolerance | `results/03_theory/` | [Mechanisms](mechanisms.md), spatial-capacity, classification and regression analysis |
| Structure-aware capacity allocation | `results/04_budget/` | [Allocation](allocation.md), [complete segmentation records](segmentation_measurements.md) |
| Attention and model execution | `results/05_execution/` | [Execution costs](execution.md), identical-layout numerical checks and timing |
| Cross-task performance and application readouts | `results/06_generalization/` | [Task evaluation](prediction.md), [complete task records](result_records.md), `scripts/analyze_applications.py` |

## Numerical evidence

The [complete UAVid suite](uavid_suite.md) provides all 63 Gaussian-fused
SegFormer validation configurations and per-image confusion counts.

## Complete curve recomputation

```bash
python scripts/analyze_segmentation_measurements.py --output outputs/segmentation_capacity_records.json
python scripts/analyze_generalization_measurements.py --output outputs/task_capacity_records.json
python scripts/analyze_applications.py --output outputs/application_statistics.json
```

The segmentation entry checks sixty paired task-and-timing points across
SegFormer, Twins, UAVid and Cityscapes. The additional-task entry checks
ninety capacity points from fifteen model-dataset combinations. The application
entry recomputes six paired readouts, source-group statistics and intervals.
Each analysis uses the evaluation unit and estimator declared in its records.

## Inference and source preparation

[assets.md](assets.md) links the task-model providers and weight specifications.
[segformer_uavid.md](segformer_uavid.md), [mmseg.md](mmseg.md) and
[prediction.md](prediction.md) describe the task commands and dataset layouts.
Accuracy and model timing run in separate processes. The sparse path reuses
the same task weights, projections and non-attention modules as its dense
comparison. Independent timing uses the precision, timer and measurement
scope in the case configuration.

## Selection and layout uncertainty

Complete Cityscapes selector confusion counts and the matched Grid and
SpargeAttention whole-model and attention-replay timings are in
`results/01_context/selection_evidence/`. Recompute them with:

```bash
python scripts/analyze_selection_evidence.py --output outputs/selection_evidence.json
```

`analyze_extent.py` also reconstructs the spatial-reach intervals from
2,000 paired image draws with seed 42. The controlled-layout entry
reconstructs 5,000-draw intervals with seed 42, resampling complete UAVid
sequences or images within Cityscapes cities. The top-k comparisons at
k=14 use the latter Cityscapes protocol and the same saved image order.
Each interval is a descriptive paired comparison.

Controlled UAVid intervals use integer-index sequence draws (5,000 draws,
seed 42, sorted sequence IDs), as specified in their measurement metadata.
Other layout comparisons retain their declared multinomial draw protocol.
