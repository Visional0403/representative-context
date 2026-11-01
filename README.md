# Representative context

This repository studies why finite attention context can preserve visual
predictions. It provides balanced spatial sampling and indexed attention
execution with a structure-aware KV capacity. The sparse path reuses the
trained attention projections and spatial-reduction layers at inference.

## Installation

The GPU implementation runs on Linux with an NVIDIA GPU, PyTorch and Triton.
Use a CUDA-enabled PyTorch installation and install the package:

```bash
python -m pip install -e '.[cuda,uavid]'
```

The exact environment used for the SegFormer/UAVid implementation is listed
in [environments/core.txt](environments/core.txt). Other model environments
are kept separately to preserve their upstream dependencies.

## Structure

- `src/representative_context/`: capacity, layouts, kernels, model adapters,
  input processing and evaluation.
- `scripts/`: executable evaluation, timing and analysis entry points.
- `configs/`: reproducible per-experiment settings.
- `datasets/` and `weights/`: task data and trained parameters.
- `results/`: numerical evidence used by the paper.
- `docs/`: implementation and reproduction instructions.
- `outputs/`: outputs from newly executed commands.

## Obtain task assets

The three study-trained segmentation checkpoints are available in
[GitHub Releases](https://github.com/Visional0403/representative-context/releases/tag/weights-v1).
They contain the trained backbones and task heads used for the paired
dense/sparse comparisons. [Task-weight instructions](docs/weights.md)
provide download commands, required paths and third-party checkpoint sources.
[Dataset instructions](docs/datasets.md) cover the complete evaluation
splits and fixed training-tile generation.

Install the Hub download dependency, then inspect or download fixed-version assets:

```bash
python -m pip install huggingface-hub==0.36.0
python scripts/prepare_hub_assets.py tcd_model oam_tcd --list
python scripts/prepare_hub_assets.py tcd_model oam_tcd
```

## SegFormer on UAVid

The task-specific decoder, retained training strategy and checkpoint
provenance are described in [docs/segformer_uavid.md](docs/segformer_uavid.md).
All paired comparisons retain the same trained dense SRA task state.

Place the official UAVid split and the pure SegFormer checkpoint in the
locations described in [datasets/README.md](datasets/README.md) and
[weights/README.md](weights/README.md). Run evaluation separately for each
execution path:

```bash
python scripts/evaluate_segformer_uavid.py --mode dense --output outputs/uavid_dense.json
python scripts/evaluate_segformer_uavid.py --mode sparse --k 16 --output outputs/uavid_sparse_k16.json
```

The validation entry evaluates the complete labeled split. `--limit N`
performs a smoke check on its first N images and records the restricted
scope. `--window-size` and `--stride` control sliding-window inference;
`--native` applies the model once to each original image.

Independent single-window timing:

```bash
python scripts/benchmark_segformer_uavid.py --mode dense --image datasets/UAVid/val/seq16/Images/000000.png --output outputs/uavid_dense_timing.json
python scripts/benchmark_segformer_uavid.py --mode sparse --image datasets/UAVid/val/seq16/Images/000000.png --output outputs/uavid_sparse_timing.json
```

Run these sequentially on an idle GPU. The timing entry uses GPU-resident
inputs, prewarms the model and layouts, and saves all CUDA-event timings.
Its scope is the complete single-window model, including projection, SR,
feed-forward and decoding layers. Input loading and whole-image stitching
belong to validation, outside this timing scope.

## Additional segmentation models and selection controls

The official SegFormer/Cityscapes model and the two Twins-SVT-S task
models use MMSegmentation 1.2.2. Its pinned source and license are in
`third_party/mmsegmentation/`; the capacity, layout and attention kernels
are independent of that framework. See [docs/mmseg.md](docs/mmseg.md) for
their evaluation and training commands.

The segmentation entries also support exact-score top-k, a fixed
eight-dimensional pre-match top-k, pooled-region routing and official
SpargeAttention controls. Their
definition, capacity matching and command options are described in
[docs/selection_controls.md](docs/selection_controls.md).

Matched routing costs, selected-context coverage and the fixed-locality
clustering controls have separate entries and complete paired records.
See [docs/routing_diagnostics.md](docs/routing_diagnostics.md) for exact
selector settings and CPU recomputation.

## Additional prediction tasks

Polyp-PVT/Kvasir-SEG, HitNet/CHAMELEON, TruFor/Columbia,
ChangeFormer/LEVIR-CD, SegFormer/OAM-TCD, GLPN/NYUv2, GMA/Sintel,
Fusformer/CAVE and QuickBird, TUR TransWeather/DIV2K and GoPro,
IML-ViT/IMD2020, AltiMap/GAMUS, GECO2/FSC147 and Pi3/TUM use a shared
evaluation entry with task-specific configurations. The code retains
their original image processing, prediction readouts and metric
aggregation. See [docs/prediction.md](docs/prediction.md) for source and
asset preparation, complete evaluation and independent model timing.

## Context mechanisms and task tolerance

The sampled classification scores, competition analysis, context-restoration
and spatial-correlation interventions are described in
[docs/mechanisms.md](docs/mechanisms.md). Complete paired records support CPU
statistical recomputation for segmentation, optical flow, spectral reconstruction
and image forensics. Independent model entries reconstruct the declared inputs.
The analysis modules also provide spatial transport, aggregation diagnostics
and signed regression-error decomposition.

## Matched-compute capacity allocation

Exact stage-allocation controls, complete paired UAVid confusion counts
and source-group budget statistics are described in
[docs/allocation.md](docs/allocation.md). CPU entries reconstruct plans and
actual interactions; model evaluation and statistical analysis use separate
commands.

## Attention and model execution costs

Pure aggregation, identical-layout execution and mutually exclusive model
cost measurements have independent entries and complete timing observations.
See [docs/execution.md](docs/execution.md) for the scopes, numerical checks
and CPU recomputation commands.

## Numerical results

The `results/` directory contains the measurements underlying the paper's
figures and supplementary tables. Paired records support independent
recomputation of task metrics, timing summaries and application statistics.
The commands in [docs/reproduction.md](docs/reproduction.md) run these analyses.

## Reproduction map

[docs/reproduction.md](docs/reproduction.md) maps the six result sections to
their numerical evidence, model commands and statistical analyses.
[docs/assets.md](docs/assets.md) links the model providers and fixed weight
metadata. Complete segmentation and cross-task capacity records can be
recomputed independently of drawing or model inference.

## Checks

```bash
python scripts/analyze_segmentation_measurements.py --output outputs/segmentation_check.json
python scripts/analyze_generalization_measurements.py --output outputs/task_check.json
```

See [docs/algorithm.md](docs/algorithm.md) for capacity, integer grouping,
spatial alignment and execution details.

Inspect exact token counts, group capacities, interaction counts and
attention-matmul FLOPs without a GPU:

```bash
python scripts/inspect_segformer_capacity.py --height 2160 --width 3840 --k 16
```

## License

Original code is licensed under the [Apache License 2.0](LICENSE).
Bundled third-party components retain their original copyright and license
notices; see [NOTICE](NOTICE) and [third_party/README.md](third_party/README.md).
Datasets and model checkpoints retain their providers' licenses and terms.

# Contact

Hongliang Ding (mail to:hongliang.ding@swjtu.edu.cn)

Zhongjie Li (mail to:531782035@my.swjtu.edu.cn)
