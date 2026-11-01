# Segmentation capacity-curve records

`results/04_budget/measurements/capacity_curves.json` contains the sixty
complete task-and-timing points used by the SegFormer and Twins capacity
comparisons. The models use the same task weights at every capacity.
UAVid accuracy covers all seventy original validation images; Cityscapes
accuracy covers all five hundred validation images.

The records distinguish sliding-window accuracy from single-window model
timing. Native UAVid accuracy covers both original image widths; independent
timing records retain the 3840- and 4096-wide inputs separately. Changing the
timing width does not create a new evaluation split.

```bash
python scripts/analyze_segmentation_measurements.py --output outputs/segmentation_capacity_records.json
```

The CPU entry checks complete-split sizes, class IoUs and available confusion
counts, then recomputes timing medians and speedups. Accuracy records contain
their original per-class values; the SegFormer native-image and complete
Cityscapes records also supply confusion counts. Per-image Cityscapes counts
are retained where supplied by that evaluation record. The grouped counts
used for scene intervals are in `results/04_budget/grouped_curves/`.

For the native SegFormer timing, the reported latency pools the calls from
three independent runs. The other timing records use their pooled-call
median. Timing records provide model latencies and speedups. The analysis uses
stored calls where available and the recorded summaries otherwise.
Each point's values are explicit in `summary`.

New inference and independent timing use `evaluate_segformer_uavid.py`,
`benchmark_segformer_uavid.py` and `evaluate_mmseg.py`, with the configurations
and scope described in [segformer_uavid.md](segformer_uavid.md) and
[mmseg.md](mmseg.md). Numerical recomputation and model inference are separate
operations.

The [complete UAVid suite](uavid_suite.md) supplies the pixel-space Gaussian
protocol, all 63 adopted configurations and their per-image confusion counts.

## Measurement protocols

`results/04_budget/measurement_protocols.json` identifies the timing settings
for each published model, input width and resource exponent. Twins uses five
warm-up calls and three rounds of twenty measurements. SegFormer/Cityscapes
uses ten warm-up calls and three rounds of ten measurements. SegFormer/UAVid
uses five warm-up calls and twenty measurements per round; native-image
latencies pool three independent runs. The sliding-window warm-up settings
are recorded per configuration. The thirteen summary-statistic points are
identified explicitly in the protocol file. New timing
accepts explicit warm-up, measurement and round overrides. Grid and
SpargeAttention comparisons use ten warm-up calls and three rounds of twenty.
