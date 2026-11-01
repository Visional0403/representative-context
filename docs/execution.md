# Attention execution and complete-model costs

The execution measurements use SegFormer and Twins on 1024 by 1024,
2048 by 2048 and native 2160 by 3840 UAVid inputs, with k=16.
The declared checkpoint and the first sorted image that contains the square
window or matches the native dimensions define each input. Normalization
uses the recorded FP32 CPU arithmetic before GPU upload.

## Three measurement scopes

Pure aggregation replays the actual Q/K/V tensors captured from one dense
backbone forward. Each timed call visits all eight SegFormer SRA blocks or
all nine Twins GSA blocks sequentially. It includes indexed reads and output
allocation; projections, SR, FFNs, local attention and decoding are evaluated
in the separate complete-model measurements. Single-group stages use the
same efficient dense SDPA path as the dense control.

```bash
python scripts/benchmark_execution.py --case segformer_uavid --phase pure --height 2048 --width 2048 --output outputs/segformer_pure_2k.json
```

The identical-layout execution comparison keeps the same Q/K/V and retained
interactions. It compares efficient dense SDPA, explicit group reordering
followed by batched efficient SDPA and indexed fusion. Groups are bucketed
by their actual query and KV lengths; their outputs are scattered to the
original query positions. This also covers nondivisible spatial dimensions.

```bash
python scripts/benchmark_execution.py --case twins_uavid --phase compare --height 1024 --width 1024 --output outputs/twins_execution_1k.json
```

Pure aggregation uses ten warmup calls per method; the execution comparison
uses five. Both use three rotated-order rounds of twenty CUDA-event
observations per method. Layout construction, compilation and correctness
checks precede formal timing. Every global block is checked against the
reordered result and an independent FP64 reference at sixteen fixed queries.
`--check-only` performs these checks without formal timing. Pure timing
retains its own layout buffers; comparison bucket indices are temporary
during its correctness check.

Complete-model profiling measures the forward from GPU-resident input to
task logits. Nested scopes assign each time segment to exactly one operation
category: global aggregation, projections and SR, FFNs, local attention,
remaining backbone work, decoding or remaining model work.

```bash
python scripts/benchmark_execution.py --case twins_uavid --phase profile --height 2160 --width 3840 --output outputs/twins_profile_native.json
```

This scope first records three rounds of ten unprofiled calls per method
after five warmups. It then collects twelve profiled forwards per method
after three warmups, alternating method order across three rounds, and
records five unprofiled recheck calls after three warmups. CUDA events partition
the same forward; the mean operation times sum to its mean profiled total.
Unprofiled timings supply the complete-model speed comparison. Profiling
overhead is recorded separately.

Run GPU commands sequentially in the default allocator environment. Timing
samples describe repeated calls on the stated device. Their percentile ranges
are distinct from source-group confidence intervals for task metrics.

## Complete observations and CPU analysis

`results/05_execution/measurements/` contains the six complete records.
Each includes exact per-block interaction counts, Q/K/V shapes and strides,
sixty pure and execution-comparison timing samples per method, twelve
profiled forwards per method, and the independent complete-model timings.
The numerical checks and exact input definitions accompany those observations.

```bash
python scripts/analyze_execution.py --output outputs/execution_statistics.json
```

The CPU analysis reconstructs distributions, same-layout speed ratios,
head-weighted interactions, FLOPs and full-model cost shares. It verifies
the equality of retained interactions across execution methods and that the
exclusive operation times sum to each profiled forward's duration. Use
`--input FILE ...` to select published measurement records explicitly.
