# Visual prediction and reconstruction experiments

The shared evaluation entry supports fifteen model-dataset combinations:

| Case | Task and split | Evaluation units | Input protocol |
| --- | --- | --- | --- |
| `polyp_kvasir` | Polyp-PVT, authors' Kvasir-SEG test split | 100 images | Native dimensions, edge padding to multiples of 32 |
| `hitnet` | HitNet, CHAMELEON | 76 images | 1536 by 1536 model input, original-size masks |
| `trufor` | TruFor, Columbia spliced-image localization | 180 images | Original image dimensions |
| `changeformer` | ChangeFormerV6, LEVIR-CD test | 128 image pairs | Original 1024 by 1024 image pairs |
| `tcd_tiled` | SegFormer-B0, OAM-TCD CC-BY test | 439 images | Nine overlapping 1K windows per original 2048 image |
| `glpn_nyu` | GLPN, NYUv2 official test split | 654 images | Official RGB extraction and depth evaluation crop |

Use the tested PyTorch 2.10.0 and Torchvision 0.25.0 CUDA 12.8 pair,
then install the prediction dependencies:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/core.txt -r environments/prediction.txt
python -c "import torch, torchvision; import representative_context.tasks.prediction"
```

Prepare the pinned architecture sources for the first four cases:

```bash
python scripts/prepare_sources.py polyp_pvt hitnet trufor changeformer
```

The Hugging Face models load their task `config.json` and
`model.safetensors` from their respective weight directories. Task weights
for the other models are pure tensor state dictionaries. The repository
loads each complete task state strictly. Model-specific asset locations
are listed in `configs/CASE/validation.json` and `weights/CASE.json`.
The original providers' conditions apply to their models and datasets.

To convert an author PyTorch checkpoint to the configured pure tensor file:

```bash
python scripts/prepare_weights.py --config configs/changeformer/validation.json --input /path/to/best_ckpt.pt
```

The converter selects the task state and retains its tensors. Model-specific
transformations include removing the DataParallel prefix and retaining the
TUR mean-prediction branch. Hugging Face task directories instead retain
their original `config.json` and `model.safetensors`.

Complete dense and sparse evaluation, run as separate processes:

```bash
python scripts/evaluate_prediction.py --config configs/polyp_kvasir/validation.json --mode dense --output outputs/polyp_dense.json
python scripts/evaluate_prediction.py --config configs/polyp_kvasir/validation.json --mode sparse --k 16 --output outputs/polyp_k16.json
```

Replace the configuration path with another case from the table. `--limit N`
selects its first N evaluation units for an explicitly restricted check.
`--data-root`, `--weights` and `--source` override asset locations. Evaluation
preserves the expected complete-split count and each sample's identity.

Independent model timing:

```bash
python scripts/evaluate_prediction.py --config configs/trufor/validation.json --mode paired --k 16 --benchmark --output outputs/trufor_k16_timing.json
```

By default, Polyp-PVT uses the median original-area image. TruFor uses
one fixed representative of every input shape and weights the per-round
median times by the dataset's shape frequencies. Other cases use the first
evaluation unit. `--sample-index N` selects a fixed individual input for
diagnosis; that output records its limited timing scope. The timing file
retains all calls, per-round medians and stratum weights. Data reading,
preprocessing and task metric calculation take place outside timing.
The canopy model's timing includes all nine windows and their probability
merge. ChangeFormer uses synchronized wall-clock timing; the other cases
use CUDA events. Timing settings and backend flags are explicit in each
configuration. Run task processes sequentially on an idle GPU.

Polyp-PVT uses its summed prediction heads, sigmoid and per-image min-max
normalization. HitNet uses its final feedback predictions with the same
normalization. TruFor reports the larger of direct and inverse fixed-
threshold F1 over its valid foreground and background masks. LEVIR-CD
and canopy scores aggregate pixel counts over the evaluation split.
NYUv2 scores average per-image metrics using the official test indices,
JPEG RGB round trip, millimetre depth quantization and Eigen crop.

## Motion, spectral fusion and image restoration

The same entry also supports:

| Case | Model and complete split | Evaluation units | Input protocol |
| --- | --- | --- | --- |
| `gma_sintel` | C+T-pretrained GMA, Sintel Final | 1041 image pairs | Native images with the model's padding to multiples of eight |
| `cave` | Fusformer, authors' CAVE test file | 11 spectral cubes | 512 by 512 RGB and 128 by 128 low-resolution spectral data |
| `qb` | HIFtool Fusformer, PanCollection QuickBird test | 20 image pairs | 256 by 256 PAN and 64 by 64 MS, normalized by 2047 |
| `div2k` | TUR TransWeather mean predictor, DIV2K validation | 100 images | Original HR images with fixed image-specific Gaussian noise |
| `gopro` | TUR TransWeather mean predictor, GoPro test | 1111 image pairs | Original aligned sharp and blurred LMDB images |

Fetch their pinned sources with:

```bash
python scripts/prepare_sources.py gma fusformer_cave fusformer_qb transweather
```

GMA uses the C+T checkpoint, twelve recurrent updates and the model's
mixed-precision configuration. Its complete evaluation uses the labeled
Sintel training split's Final rendering, without Sintel fine-tuning.
Attention probabilities are computed once and reused by recurrent value
aggregation. The sparse representation retains balanced groups and masks
padding positions for unequal group sizes. Independent timing uses the
two author sample frames, as specified by the model's timing protocol.

Fusformer retains the original interpolation, spectral fusion and output
clamping. Both dense baselines use the full-KV fused kernel. A single
sparse group uses that same dense backend. Timing uses CAVE test index
five and QuickBird test index ten. PSNR, RMSE and spectral angle use
the complete reference tensors; CAVE also reports ERGAS.

The restoration models use the published TUR mean-prediction weights.
Attention heads wider than the sparse kernel's supported range retain
optimized dense attention. Weather-query decoding is retained. Native
images receive reflected right/bottom padding to multiples of 64; outputs
are cropped and clamped to the original dimensions. DIV2K's noise has
standard deviation 25/255 and a fixed seed derived from each image ID.
PSNR and SSIM use rounded eight-bit RGB predictions. GoPro's SSIM uses
the separable form of the same 11 by 11 Gaussian kernel. Timing selects
the median image by original area and ID.

## Forensics, height, counting and multiview geometry

| Case | Model and complete split | Evaluation units | Input protocol |
| --- | --- | --- | --- |
| `iml` | IML-ViT, IMD2020 | 2010 images | Top-left 1024 crop and bottom/right zero padding before normalization |
| `gamus` | AltiMap DPT-DINOv2-L, GAMUS test | 2861 images | Original 1024 inputs, reflection padding to a multiple of 14 |
| `geco2` | GECO2 Hiera-B+, FSC147 test | 1190 images | Author's 1024 resize/padding, three exemplars and SAM2 refinement |
| `pi3` | Pi3, complete TUM RGB-D fr1/desk sequence | 10 groups, 596 matched views | Chronological nonoverlapping 64-view groups; final group has 20 views |

Prepare their architecture sources:

```bash
python scripts/prepare_sources.py iml_vit altimap dinov2 geco2 pi3
python scripts/prepare_counting_ops.py
```

The counting dependency is the pinned official Linux binary for PyTorch
2.10, CUDA 12.8, x86-64 and the CXX11 ABI. Its source revision and build
URL are explicit in the preparation entry. The other optional packages
are listed in `environments/prediction.txt`.

IML-ViT retains learned relative positions in every local and global
attention block. Global blocks use the spatial groups; the dense control
and local blocks use the same FP32 relative-position implementation.
IMD2020 F1 uses the same foreground/background boundary exclusion and
direct/inverse threshold convention as the forensics control.

The height model retains both its regression and segmentation heads.
Its reported height RMSE and MAE pool the errors over all finite test
pixels in the declared valid height range. The fixed test identities
and relative image/height paths are in `configs/gamus/samples.json`.
The global CLS token remains accessible to all spatial groups.

GECO2 replaces Hiera's three unpooled global attention blocks. Its exemplar
encoder, deformable decoder, SAM2 mask refinement, score threshold, NMS
and valid-image filtering retain their complete task computations. Create
the inference-only state from the author's task checkpoint and official
SAM2 checkpoint:

```bash
python scripts/prepare_counting_weights.py --task-checkpoint /path/to/GECO2_FSCD147.pth --sam2-checkpoint /path/to/sam2_hiera_base_plus.pt
```

The converter combines task tensors and the corresponding SAM2 prompt and
mask-decoder tensors. Loading the resulting complete model is strict.

Pi3 retains the five special tokens per view globally, original rotary
positions, point/confidence/camera heads and BF16 autocast configuration.
Linear/convolution weights inside autocast use BF16; normalization,
position/register tokens, layer scales and prediction heads retain FP32.
The original 255,000-pixel per-view cap and multiples-of-14 Lanczos resize
give 434 by 574 inputs for this sequence. Nearest depth and pose timestamps
must both be within 20 ms of each RGB timestamp. The fixed matches and
groups are in `configs/pi3/samples.json`.

Point-map evaluation fits one similarity transform per input group from
valid points on a fixed 64 by 64 grid. It scores every valid reference
point after alignment. The aggregate averages the 596 frame metrics and
586 adjacent-pair rotation metrics, accounting for the shorter final
group. Camera translation-direction error is reported separately.
Relative depth relief is `(P90 - P10) / P50` over valid local-camera
depths and is evaluated for dense predictions and k=17.

All four cases use the shared evaluation and independent timing commands.
Timing uses the first IMD2020 image, the median-index GAMUS image, FSC147
image `4935.jpg` and TUM group `fr1_desk_0320`, respectively. Each round and
execution mode receives its configured warmup. Pi3 clears unused layouts
and idle cache before warmup, outside the measured interval. No extra
memory occupier or allocator cap is applied.


## Complete records and declared summaries

Every published budget point is linked to complete task records and original
timing observations in [result_records.md](result_records.md). The configuration
selects the original latency and speedup estimators; `latency_ms` and `speedup`
are the primary fields in newly collected timing outputs. The CPU recomputation
entry checks the supplied curves independently of model inference.

## Collect new curves and application readouts

Evaluation reports save application quantities alongside the original task
metrics. CAVE uses 400--700 nm bands and the common GT spectral-sum mask;
Sintel uses mean displacement magnitude; NYUv2 uses mean depth on the
evaluation crop and valid-depth mask. Canopy/change fractions use binary
counts. Pi3 relief uses local-depth quantiles, expressed in percent.
These quantities are computed after inference and are outside timing.

After collecting complete dense/sparse evaluations and paired timing reports:

```bash
python scripts/collect_prediction_records.py --dense outputs/dense.json --sparse outputs/k14.json outputs/k16.json --timing outputs/timing_k14.json outputs/timing_k16.json --output-dir outputs/collected
python scripts/analyze_generalization_measurements.py --measurements outputs/collected --curves outputs/collected/budget_curves.json --output outputs/recomputed_curves.json
python scripts/analyze_applications.py --input outputs/collected/application_readouts.json --output outputs/recomputed_applications.json
```

Use the case's declared application exponent in `configs/applications/readouts.json`
to generate its application file. That profile maps the complete evaluation
IDs to the source-image, indoor-scene, video-scene or multiview groups used
for paired resampling. Pairing checks checkpoint, dataset, settings and
ordered sample identities before combining reports. Run collection into a
new directory; task curves remain separate from the published measurements.
