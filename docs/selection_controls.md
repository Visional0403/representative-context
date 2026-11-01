# Selection controls

The controls reuse the task model's trained Q/K/V projections, spatial
reduction and output projection. The top-k controls' retained KV counts match the maximum
balanced-group size for the requested capacity. `--keep-scale` evaluates
neighboring retained counts (0.5, 1 or 2 times that count).

`topk_exact` selects each query and head's largest full-dimensional QK
scores. The selected scaled scores feed the fused weighted-value kernel.
`topk_proxy8` uses an eight-dimensional orthogonal random projection,
fixed by the block seed, to select candidates. It computes the final
attention scores using the original head dimension. Both controls score
input-dependent routes on each forward pass; their timing includes all
selection and execution operations.

`pooled_regions` pools Q/K features in spatial regions, chooses related
regions by pooled dot products and applies full-dimensional attention
over their tokens. Routing is shared across heads. KV regions have side
four; candidate counts are rounded to the resulting region granularity.

Use either segmentation evaluation entry with a method argument:

```bash
python scripts/evaluate_segformer_uavid.py --mode sparse --k 14 --method topk_proxy8 --output outputs/uavid_proxy.json
python scripts/evaluate_mmseg.py --config configs/twins_cityscapes/validation.json --mode sparse --k 14 --method pooled_regions --output outputs/twins_regions.json
```

The generic validation entries use Torch top-k with chunks bounded by
4,194,304 scoring elements. Their selected-attention Triton kernel uses
query tile four, KV tile 64 and four warps. The standalone selection module
also exposes an explicit gather reference and exact FP32 Triton top-k.
The pooled-region validation entry uses the explicit reference aggregation.

The matched routing timing and coverage entries use the declared optimized
settings for each model: Torch top-k with 4,194,304 scoring elements for
SegFormer, Triton top-k with 16,777,216 elements for Twins, and efficient
SDPA for selected region interactions. See
[routing_diagnostics.md](routing_diagnostics.md) for independent entries,
fixed inputs, retained indices, cost decomposition and complete controlled
layout measurements.

## Official SpargeAttention control

The official implementation is pinned to
`thu-ml/SpargeAttn@ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`.
Prepare and build it with a matching CUDA toolkit:

```bash
python scripts/prepare_sources.py sparge
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation ./third_party/sources/sparge
```

The recorded build uses CUDA 12.8, PyTorch 2.10 and the Ada GPU architecture.
Use the architecture corresponding to the target GPU when building elsewhere.
The upstream source and license retain their original authorship.

`--method sparge` calls the official meansim/top-k CUDA API, including its
FP16 input conversion, INT8/FP8 quantization and selected-block execution.
The 32-dimensional heads are zero-padded to 64 dimensions for that API,
with the original softmax scale and output width preserved.
`--method sparge_fp32` retains the same official selected block map and
aggregates the original FP32 Q/K/V with its original head dimension.
`--method sparge_quant_full` applies the official quantized path at full
retention to separate quantization from sparse selection.

The target capacity follows the same structure-aware formula. Official
query/key blocks contain 128/64 tokens. Retained key-block count is
`min(ceil(K/64), ceil(min(K,B)/64))`; reported block capacity includes the
rounding. Full-retention sparse stages use the same optimized dense path.
The native official quantization API uses `simthreshd1=-0.1`,
`cdfthreshd=None`, `pvthreshd=50`, key smoothing and a disabled attention sink.
All conversion and selection work remains inside the timed model forward.

Examples for complete evaluation:

```bash
python scripts/evaluate_segformer_uavid.py --mode sparse --method sparge --k 14 --window-size 2048 --stride 1024 --output outputs/sparge_uavid_2k_k14.json
python scripts/evaluate_mmseg.py --config configs/twins_cityscapes/validation.json --mode sparse --method sparge_fp32 --k 14 --output outputs/sparge_fp32_twins_city_k14.json
```

Use the same entries' independent timing options for speed. The official
and FP32 controls share task weights, preprocessing, evaluation split and
capacity targets with the fixed-layout methods.

Grid and SpargeAttention timing comparisons use explicit `--warmup 10 --repeats 20 --rounds 3` overrides. Capacity-curve timing uses the model-specific settings in its configuration and `results/04_budget/measurement_protocols.json`.
