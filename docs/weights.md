# Downloading and preparing task weights

All commands below run from the repository root. Install the core and
task dependencies from `environments/` before converting a PyTorch checkpoint.
Hub downloads use `huggingface-hub==0.36.0`, included in `environments/prediction.txt`.
Final files go
under `weights/`; downloaded archives and author checkpoints can be kept
under `outputs/downloads/`.

## Study-trained segmentation checkpoints

Download the three complete task checkpoints from
[GitHub Releases](https://github.com/Visional0403/representative-context/releases/tag/weights-v1). Dense and sparse comparisons load the same task checkpoint.
The inference entries use the paths below:

| Model and task | Required destination | Training metadata |
| --- | --- | --- |
| SegFormer / UAVid | `weights/segformer/segformer_b0_uavid.pth` | [Metadata](../weights/segformer_b0_uavid.json) |
| Twins / Cityscapes | `weights/twins/twins_svt_s_cityscapes.pth` | [Metadata](../weights/twins_cityscapes.json) |
| Twins / UAVid | `weights/twins/twins_svt_s_uavid.pth` | [Metadata](../weights/twins_uavid.json) |

From the repository root, download the files directly to their configured paths:

```bash
mkdir -p weights/segformer weights/twins
curl --fail --location https://github.com/Visional0403/representative-context/releases/download/weights-v1/segformer_b0_uavid.pth --output weights/segformer/segformer_b0_uavid.pth
curl --fail --location https://github.com/Visional0403/representative-context/releases/download/weights-v1/twins_svt_s_cityscapes.pth --output weights/twins/twins_svt_s_cityscapes.pth
curl --fail --location https://github.com/Visional0403/representative-context/releases/download/weights-v1/twins_svt_s_uavid.pth --output weights/twins/twins_svt_s_uavid.pth
```

Each metadata file above records the release asset and its content checksum.
The files load with `torch.load(path, map_location="cpu", weights_only=True)`
and match the complete task architectures. The Twins training and evaluation
entries are in [mmseg.md](mmseg.md); the SegFormer/UAVid entry is in
[segformer_uavid.md](segformer_uavid.md). The
[reproduction map](reproduction.md) also links the mechanistic analyses,
budget comparisons and independent timing commands using these task states.

SegFormer/UAVid was trained with a manually tuned, class-weighted
cross-entropy term. For class IDs `0` to `7`, the class order is
`clutter`, `building`, `road`, `tree`, `low_vegetation`, `static_car`,
`moving_car`, `human`, with weights
`[1.5, 0.8, 2.0, 2.5, 2.0, 14.0, 8.0, 18.0]`, respectively.
The dense and sparse evaluations use the same task checkpoint.
The task head has four convolution-BatchNorm-ReLU branches and a fifth
BatchNorm in the fusion block. The retained source combines cross-entropy
and Dice with equal coefficients. The forty-epoch protocol budget, optimizer,
augmentation, label-canvas convention and seed-42 training configuration
are described in the [training strategy](segformer_uavid.md#training-strategy)
and [configuration metadata](../configs/segformer_uavid/training_provenance.json).

## Third-party task checkpoints

Use the complete task checkpoints listed below. The architecture sources
are fixed separately in [third_party/sources.json](../third_party/sources.json).
ImageNet initialization alone does not supply the segmentation, regression
or reconstruction task heads used in these evaluations.

| Case | Provider and exact task file | Final destination |
| --- | --- | --- |
| SegFormer / Cityscapes | [Official MiT-B0 Cityscapes checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth) | `weights/segformer/segformer_b0_cityscapes.pth` |
| Polyp-PVT | [Authors' trained checkpoints](https://drive.google.com/drive/folders/1xC5Opwu5Afz4xiK5O9v4NnQOZY0A9-2j), `PolypPVT.pth` | `weights/polyp_pvt/polyp_pvt.pth` |
| HitNet | [Author-linked mirror: Net_epoch_best.pth](https://huggingface.co/stablediffusionuser/hitnet/resolve/b9adfd179eddbdd580924a1c1a252de89adf2f82/Net_epoch_best.pth) | `weights/hitnet/hitnet.pth` |
| TruFor | [Authors' weight archive](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip), `trufor.pth.tar` | `weights/trufor/trufor.pth` |
| ChangeFormerV6 | [Authors' LEVIR-CD checkpoint folder](https://www.dropbox.com/sh/xoswrsu9n4ypr5h/AAAmUOE-DTq59fX-tE2TwPzKa?dl=0), LEVIR `best_ckpt.pt` | `weights/changeformer/changeformer_v6.pth` |
| SegFormer / OAM-TCD | [restor/tcd-segformer-mit-b0](https://huggingface.co/restor/tcd-segformer-mit-b0/tree/904e2fb0696abb4bae63f43344bd99ec1959d092) | `weights/oam_tcd/` |
| GLPN / NYUv2 | [vinvino02/glpn-nyu](https://huggingface.co/vinvino02/glpn-nyu/tree/cb8ce03174424af9a3315da175ead88667f0578c) | `weights/glpn_nyu/` |
| GMA | [gma-things.pth](https://raw.githubusercontent.com/zacjiang/GMA/2f1fd29468a86a354d44dd25d107930b3f175043/checkpoints/gma-things.pth), Chairs+Things training | `weights/gma/gma_things.pth` |
| Fusformer / CAVE | [Authors' Trained_model/1000.pth](https://raw.githubusercontent.com/J-FHu/Fusformer/b075443ec51b958c6ae0f3cfc01eaf3a15ea7257/Trained_model/1000.pth) | `weights/fusformer/cave.pth` |
| Fusformer / QuickBird | [HIFtool PretrainModel/QB/Fusformer/400.pth](https://raw.githubusercontent.com/Caoxuheng/HIFtool/d364ecb60aa944af17223c050bad24199899ae00/PretrainModel/QB/Fusformer/400.pth) | `weights/fusformer/quickbird.pth` |
| TUR TransWeather / DIV2K and GoPro | [Transweather_TUR_5task.pth](https://huggingface.co/GWu/TUR/resolve/0caa4ef69202d1841d98c18f2c0fe1fc2ae23aa8/Transweather_TUR_5task.pth), mean predictor | `weights/transweather/transweather_tur.pth` |
| IML-ViT | [Authors' TruFor-protocol checkpoint](https://drive.google.com/file/d/1jlXw97GkyBbY4u5-e_liuhahKSQWCAFu/view), `iml-vit_checkpoint_trufor_20231104.pth` | `weights/iml/iml_vit.pth` |
| AltiMap / GAMUS | [altimap-height best.pth](https://huggingface.co/Dilavesh/altimap-height/resolve/d9c715f97a817ab8feb00dd2fbef463f2d99dd01/best.pth) | `weights/gamus/altimap_gamus.pth` |
| GECO2 | [GECO2_FSCD147.pth](https://drive.google.com/file/d/1OG6-tTP2Egvx-O5oK6lxPzvQO1LWgHbb/view) and [SAM2 Hiera-B+](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt) | `weights/geco2/geco2_fsc147.pth` |
| Pi3 | [Pi3 model.safetensors](https://huggingface.co/yyfz233/Pi3/resolve/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors) | `weights/pi3/model.safetensors` |

The Hub preparation entry downloads the declared versions and writes their
configured destinations. It leaves existing files in place:

```bash
python scripts/prepare_hub_assets.py hitnet tcd_model glpn_nyu pi3
python scripts/prepare_hub_assets.py tur gamus_weights
```

TUR and AltiMap download the author checkpoint to `outputs/downloads/`;
convert them as shown below. To inspect any specification without downloading:

```bash
python scripts/prepare_hub_assets.py tcd_model glpn_nyu --list
```

Google Drive and Dropbox assets can be downloaded through the linked
provider pages. Select the task variant in the table. The GMA checkpoint
is the Chairs+Things model; IML-ViT uses the authors' TruFor-protocol release.
The Pi3 entry uses the specified Pi3 weights and architecture source.

## Preparing pure PyTorch task states

After downloading the author files into `outputs/downloads/`, use the case
configuration to select the task state and standard output path:

```bash
python scripts/prepare_weights.py --config configs/segformer_cityscapes/validation.json --input outputs/downloads/segformer_cityscapes_author.pth
python scripts/prepare_weights.py --config configs/polyp_kvasir/validation.json --input outputs/downloads/PolypPVT.pth
python scripts/prepare_weights.py --config configs/trufor/validation.json --input outputs/downloads/trufor.pth.tar
python scripts/prepare_weights.py --config configs/changeformer/validation.json --input outputs/downloads/best_ckpt.pt
python scripts/prepare_weights.py --config configs/gma_sintel/validation.json --input outputs/downloads/gma-things.pth
python scripts/prepare_weights.py --config configs/cave/validation.json --input outputs/downloads/fusformer_cave_author.pth
python scripts/prepare_weights.py --config configs/qb/validation.json --input outputs/downloads/fusformer_qb_author.pth
python scripts/prepare_weights.py --config configs/div2k/validation.json --input outputs/downloads/Transweather_TUR_5task.pth
python scripts/prepare_weights.py --config configs/iml/validation.json --input outputs/downloads/iml-vit_checkpoint_trufor_20231104.pth
python scripts/prepare_weights.py --config configs/gamus/validation.json --input outputs/downloads/altimap_gamus_author.pth
```

The CAVE and QuickBird author files are saved under the descriptive local
names used above. The converter extracts `state_dict`, `model`, `net`,
`model_G_state_dict` or `params` as required by the selected task. It removes
the GMA/CAVE DataParallel prefix and retains the TUR mean-prediction branch.
DIV2K and GoPro share the resulting TUR task file. HitNet's downloaded file
already contains its named task tensors. Hugging Face task directories retain
their configuration and safetensors files directly.

GECO2 combines the counting task state with the SAM2 prompt and mask heads:

```bash
python scripts/prepare_counting_weights.py --task-checkpoint outputs/downloads/GECO2_FSCD147.pth --sam2-checkpoint outputs/downloads/sam2_hiera_base_plus.pt
```

These preparation steps preserve inference tensors and create the format
expected by the strict model loaders. Existing destination files are left
unchanged; `--output` selects a separate destination for conversion.

## Twins training initialization

The two Twins baselines use the same
[ImageNet-pretrained Twins-SVT-S initialization](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_small_20220308-7e1c3695.pth).
Place this checkpoint at `weights/twins/twins_svt_s_imagenet.pth` when
running the training configurations. The released task checkpoints already
contain the complete trained backbone and prediction head.

Each provider retains its own model and data terms. Third-party checkpoint files are obtained from the sources above.
The three study-trained checkpoints are distributed as release assets, separate from the source archive.
