# SegFormer/UAVid evaluation and training strategy

The evaluated task model uses dense MiT-B0 spatial-reduction attention.
Its backbone was initialized from ImageNet-pretrained MiT-B0 and trained
with the UAVid task decoder described below.
Its four stages have widths (32, 64, 160, 256), two blocks per stage,
heads (1, 2, 5, 8), and SR ratios (8, 4, 2, 1).
The complete trained backbone and decoder are supplied as a pure task
checkpoint through [the weight preparation instructions](weights.md#study-trained-segmentation-checkpoints).
The same released file supports the evaluation suite, sparse-layout controls,
mechanistic analyses and independent timing.

## Prediction head

Each feature level passes through a bias-free 1x1 convolution to 256
channels, BatchNorm and ReLU. Bilinear interpolation aligns the four
branches to the first-stage grid. Their concatenation passes through
a bias-free fusion convolution, BatchNorm and ReLU, followed by
Dropout2d(0.1) and the eight-class classifier. The decoder has five
BatchNorm layers. Interpolation uses `align_corners=False`.

## Training strategy

The training configuration specifies forty epochs over 5,600 fixed
1024-by-1024-pixel crops extracted from the 200 official training images
with a 512-pixel stride.

The training implementation specifies:

| Setting | Value |
| --- | --- |
| Actual microbatch / gradient accumulation | 1 image / 4 steps |
| BatchNorm updates | Once per one-image training forward |
| Optimizer | AdamW, backbone and head learning rate 1e-4 |
| Weight decay / betas | 0.01 / (0.9, 0.999) |
| Names excluded from decay | Names containing `bias` or `LayerNorm.weight` |
| Scheduler | 125-update warmup, linear decay, learning-rate floor 1e-6 |
| Training loss | 0.5 weighted cross-entropy + 0.5 Dice |
| Class weights | `[1.5, 0.8, 2, 2.5, 2, 14, 8, 18]` |
| Class order | clutter, building, road, tree, low vegetation, static car, moving car, human |
| Rotation / horizontal flip | +/-15 degrees / flip, each with probability 0.5 |
| Photometric transforms | Independently enabled with probability 0.5 |
| Brightness / contrast / saturation factors | 0.875-1.125 / 0.7-1.3 / 0.8-1.2 |
| Hue shift | +/-9 on the OpenCV hue scale |
| Precision | FP32 |
| Normalization | RGB mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Checkpoint selection | Maximum validation mIoU; four validations per epoch |
| Selection collection | All 70 native validation images, 1024 windows, stride 512 |

The training configuration specifies seed 42. All reported paired
comparisons used the same task checkpoint and prediction head. Final
evaluation uses the Gaussian fusion protocol below.
The [training configuration](../configs/segformer_uavid/training_provenance.json)
records the architecture, crop preparation, optimizer, loss, augmentation
and validation settings. The repository provides the checkpoint-compatible
model, crop preparation, evaluation and analysis entries.

## Final paired evaluation

Dense and sparse entries use the same trained task state and decoder.
The evaluation reader uses OpenCV RGB conversion and Albumentations
ImageNet normalization. Complete-image prediction accumulates window
logits before argmax. Gaussian weights use pixel coordinates centered at
(size - 1) / 2, sigma equal to one quarter of the window size, and unit sum.
Accumulated logits are divided by the positive sum of covering weights.
A native forward bypasses window merging. Validation labels remain on
their original image grid.

Dataset metrics accumulate a global confusion matrix. Per-class IoU
uses `TP / (TP + FP + FN + 1e-8)` and mIoU averages all eight classes.
Per-image confusion matrices support paired analyses. The recorded
checkpoint-selection score and final reevaluation score are separate
fields in the weight metadata.

## Independent timing

The timing entry measures a fixed top-left window from the specified
image. Defaults are FP32, batch one, five warmups, three rounds and twenty
CUDA-event samples per round. It saves individual samples, round medians,
the overall median, mean and population standard deviation. Compilation
and layout initialization occur during warmup. Both modes use the same
input, model weights and non-attention modules. Run accuracy and speed
commands separately.

The [complete UAVid suite](uavid_suite.md) supplies all 63 adopted
pixel-space Gaussian configurations and their per-image confusion counts.
