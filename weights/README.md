# Task weights

The three study-trained segmentation checkpoints are provided in
[GitHub Releases](https://github.com/Visional0403/representative-context/releases/tag/weights-v1).
Download them using [the preparation commands](../docs/weights.md#study-trained-segmentation-checkpoints)
and place them at the paths expected by the task configurations:

```text
weights/segformer/segformer_b0_uavid.pth
weights/twins/twins_svt_s_cityscapes.pth
weights/twins/twins_svt_s_uavid.pth
```

Dense and sparse inference use the same task checkpoint. Selection scores,
training settings and the available provenance are recorded in the adjacent
`segformer_b0_uavid.json`, `twins_cityscapes.json` and `twins_uavid.json` files.

[docs/weights.md](../docs/weights.md) lists third-party download sources,
task variants, provider versions, conversion commands and checkpoint paths.
The Hub preparation entry can obtain the
specified safetensors directories, HitNet and reference task files:

```bash
python scripts/prepare_hub_assets.py hitnet tcd_model glpn_nyu pi3
```

The SegFormer/UAVid training configuration specifies seed 42 and a forty-epoch
budget on 5,600 fixed crops. Its architecture, loss, augmentation and optimizer
settings are described in [the training strategy](../docs/segformer_uavid.md#training-strategy).
The Twins task checkpoints use the independent training configurations in
`configs/`. Evaluation settings are defined separately in each task configuration.

Large task binaries stay outside Git history. Third-party weights are obtained
from their providers with their accompanying terms and source specifications.
