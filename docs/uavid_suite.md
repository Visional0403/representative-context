# Complete SegFormer/UAVid evaluation suite

The suite uses the SegFormer-B0 UAVid task checkpoint and all 70 official
validation images. Sliding-window inference fuses logits with a pixel-space
Gaussian: the center is `(size-1)/2`, the standard deviation is `size/4`,
and overlapping outputs are divided by the sum of their positive weights.
The stride is half the window size. Evaluation uses FP32, seed 42 and four
CPU threads. The supplied configuration records the window, budget, layout,
selector or stage capacities for every comparison.

The 63 configurations comprise dense inference at 1K and 2K; dispersed
32-region inference at k=12 through 17 at both window sizes; 16- and 64-region,
balanced-random, local and clustered controls; spatial-reach and fixed-local
context controls; exact and pre-match top-k; pooled-region selection;
SpargeAttention controls; Grid layouts; and matched-compute stage allocations.
Every configuration retains the same task checkpoint and evaluation split.

## Recompute supplied metrics on CPU

```bash
python scripts/analyze_uavid_suite.py --output outputs/uavid_suite_metrics.json
python scripts/analyze_segmentation_measurements.py --output outputs/segmentation_capacity_records.json
python scripts/analyze_routing.py --output outputs/routing_statistics.json
python scripts/analyze_extent.py --output outputs/extent_statistics.json
python scripts/analyze_allocation.py --task allocation --case segformer_uavid --output outputs/allocation_statistics.json
```

`results/segformer_uavid/evaluations/` contains complete dataset and per-image
confusion counts, class IoUs, pixel accuracy, and evaluation settings.
`configs/segformer_uavid/evaluation_suite.json` maps each configuration to its
measurement record. The CPU entry verifies all 4,410 image-configuration
records and recalculates their metrics. Updated curve, controlled-layout and
allocation records use these same confusion counts. Independent model timing
records are stored separately from task evaluation.

## Run model inference

Prepare the checkpoint and dataset using [weights.md](weights.md) and
[datasets.md](datasets.md), then install the [core environment](../environments/core.txt).
Selection comparisons additionally require the dependencies described in
[selection_controls.md](selection_controls.md).

```bash
python scripts/evaluate_uavid_suite.py --all --output-dir outputs/uavid_suite
```

Configurations run one at a time in separate processes. This command prepares
the fixed-local context indices on CPU before evaluation. Interrupted
configurations resume from their completed image records. New records are
written under `outputs/`, leaving the supplied measurements intact.
Progress is saved after each image. Reuse the output directory to resume
the same configuration, checkpoint and dataset. Use a new output directory
when changing the experiment settings.

For a single ordinary layout or selector:

```bash
python scripts/evaluate_uavid_suite.py --configuration dispersed32_2048_k16 --output-dir outputs/uavid_suite
```

For a fixed-local control, prepare its indices first:

```bash
python scripts/prepare_context_controls.py --case segformer_uavid --output-dir outputs/uavid_context
python scripts/evaluate_uavid_suite.py --configuration controlled_cluster1_2048_k14 --prepared outputs/uavid_context --output-dir outputs/uavid_suite
```

`--data-root` and `--checkpoint` override the default asset locations.
Whole-image native inference and fixed-window mechanism analyses use their
respective entries in [reproduction.md](reproduction.md).
