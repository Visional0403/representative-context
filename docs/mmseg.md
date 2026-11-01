# SegFormer and Twins segmentation experiments

The SegFormer/Cityscapes baseline uses the official MMSegmentation
MiT-B0 checkpoint. The two Twins-SVT-S baselines use independently
trained task weights and an FPN segmentation head. Their stage depths
are (2, 2, 10, 4), with nine global and nine local attention blocks.
The replacement acts on global attention blocks.

The bundled MMSegmentation subset contains the components used by these
experiments, including both Twins training pipelines. The complete Twins
task checkpoints are available through [the weight preparation instructions](weights.md#study-trained-segmentation-checkpoints);
download them before running the paired evaluation commands below. Install
`environments/mmseg.txt` after the core environment. Complete evaluation:

```bash
python scripts/evaluate_mmseg.py --config configs/segformer_cityscapes/validation.json --mode dense --output outputs/segformer_city_dense.json
python scripts/evaluate_mmseg.py --config configs/segformer_cityscapes/validation.json --mode sparse --k 16 --output outputs/segformer_city_k16.json
python scripts/evaluate_mmseg.py --config configs/twins_cityscapes/validation.json --mode sparse --k 16 --output outputs/twins_city_k16.json
python scripts/evaluate_mmseg.py --config configs/twins_uavid/validation.json --mode sparse --k 16 --protocol native --output outputs/twins_uavid_native_k16.json
```

`--protocol` selects 1K or 2K sliding-window prediction, or native-image
prediction. Cityscapes uses stride 768 with a 1K window; UAVid uses half
the window size. Evaluation preserves original image dimensions and
merges logits using the task model's original MMSegmentation inference.
Unknown UAVid RGB labels are ignored with value 255. Full evaluation
requires 500 Cityscapes or 70 UAVid validation images. `--limit` selects
the first images for an explicit smoke check.

Use `--benchmark --height H --width W` to time a GPU-resident window in
a separate run. Both execution modes use identical normalization and
model components. Timings save all samples and their per-round medians.

Twins training configurations retain 80,000 iterations, batch one,
seed 42, AdamW, AMP, 1,000 warmup iterations and validation every 2,500
iterations. Use the same ImageNet backbone for independent task runs:

```bash
python scripts/train_twins.py --config configs/twins_cityscapes/training.json --work-dir outputs/train_twins_city
python scripts/train_twins.py --config configs/twins_uavid/training.json --work-dir outputs/train_twins_uavid
```

Training selects checkpoints by validation mIoU. Evaluation loads the released
complete task states strictly and shares them between dense and sparse paths.
Training retains the upstream runner's checkpoint and sampler-resume behavior.

## Label preparation

Official Cityscapes labelIds are converted on reading when train-ID files are
absent. To materialize the same mapping for upstream training, run:

```bash
python scripts/prepare_cityscapes_labels.py --data-root datasets/Cityscapes
```

Original files remain intact. Both relative and absolute data-root paths are
resolved before sample enumeration and identity recording. UAVid evaluation
uses one shared eight-color mapping; unknown colors have ignore ID 255.
