# Downloading and preparing datasets

Evaluation uses the complete task splits listed below. Store assets under
`datasets/` with the paths in [datasets/README.md](../datasets/README.md).
The evaluation entries perform the task-specific decoding, normalization,
padding and metric calculation described in [prediction.md](prediction.md).

## Sources and evaluation units

| Dataset | Download source | Required split and files | Repository location |
| --- | --- | --- | --- |
| UAVid | [Dataset authors](https://uavid.nl/) | Original 200-image training and 70-image validation splits; RGB images and color labels | `datasets/UAVid/train/` and `datasets/UAVid/val/` |
| Cityscapes | [Official downloads](https://www.cityscapes-dataset.com/downloads/) | `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`; all 500 validation images | `datasets/Cityscapes/leftImg8bit/` and `datasets/Cityscapes/gtFine/` |
| Kvasir-SEG | [Polyp-PVT authors' data package](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view) | `TestDataset/Kvasir`, 100 image-mask pairs | `datasets/kvasir_seg/{images,masks}/` |
| CHAMELEON | [SINet benchmark distribution](https://github.com/DengPingFan/SINet#22-usage) | `TestDataset/CHAMELEON`, all 76 image-mask pairs | `datasets/chameleon/{Imgs,GT}/` |
| Columbia | [Dataset and terms](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) and [author-linked download](https://www.dropbox.com/sh/786qv3yhvc7s9ki/AACbEEzGPrD3_y38bpWHzgdqa?dl=0) | All 180 spliced images and their boundary/region masks | `datasets/columbia/4cam_splc/` |
| LEVIR-CD | [Dataset authors](https://justchenhao.github.io/LEVIR/) | Original `test.zip`, all 128 original 1024-pixel image pairs | `datasets/levir_cd/test.zip` |
| OAM-TCD | [Authors' CC-BY test shard](https://huggingface.co/datasets/restor/tcd/tree/d97d4da0ebbb6e249ae95ac5e19656babd972eb2) | `data/test-00000-of-00001.parquet`, all 439 images | `datasets/oam_tcd/test.parquet` |
| NYUv2 | [Labeled dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [GLPDepth split file](https://github.com/vinvino02/GLPDepth/blob/main/datasets/splits.mat) | `nyu_depth_v2_labeled.mat` and `splits.mat`; 654 official test indices | `datasets/nyuv2/` |
| MPI-Sintel | [Official dataset](http://sintel.is.tue.mpg.de/downloads) | Training-image archive and training-extras archive; all 1,041 labeled Final pairs | `datasets/sintel/` |
| CAVE | [Fusformer-prepared data](https://drive.google.com/file/d/16F172sSs4ZwkEUu1hZ7CpjEqVP3WBhm4/view) and [authors' data instructions](https://github.com/J-FHu/Fusformer/tree/b075443ec51b958c6ae0f3cfc01eaf3a15ea7257) | `Test_CAVE.h5` from `CAVEh5.zip`, all eleven test cubes | `datasets/cave/Test_CAVE.h5` |
| QuickBird | [PanCollection reference file](https://huggingface.co/datasets/elsting/PanCollection/tree/b39e5daf0a322105977a7d4ce53ca0c4806defed) and [PanCollection project](https://github.com/liangjiandeng/PanCollection) | `test_data/test_qb_multiExm1.h5`, all twenty reference examples | `datasets/quickbird/test_qb_multiExm1.h5` |
| DIV2K | [Official validation HR archive](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) | All 100 original validation HR images | `datasets/div2k_hr/*.png` |
| GoPro | [NAFNet evaluation-data instructions](https://github.com/megvii-research/NAFNet/blob/main/docs/GoPro.md) and [test LMDB archive](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view) | All 1,111 test pairs in `input.lmdb` and `target.lmdb` | `datasets/gopro_test/GoPro/test/` |
| IMD2020 | [Dataset authors](https://staff.utia.cas.cz/novozada/db/) | Real-life manipulated-image collection, all 2,010 manipulated images and masks | `datasets/imd2020/` |
| GAMUS | [Fixed dataset version](https://huggingface.co/datasets/earthflow/GAMUS/tree/a3c0e2511f06d909612406f436cf8abb4da805f5) | All 2,861 test RGB/height pairs in the supplied sample list | `datasets/gamus/{images,heights}/test/` |
| FSC147 | [Dataset project](https://github.com/cvlab-stonybrook/LearningToCountEverything) and [fixed reference copy](https://huggingface.co/datasets/isentropic/FSC147/tree/3e420cb6537e803dd6d4516623ce82a79c0317b8) | Official test list of 1,190 images, count annotations and exemplar boxes | `datasets/fsc147/` |
| TUM RGB-D | [Official sequence downloads](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) | Complete `rgbd_dataset_freiburg1_desk` sequence: RGB, depth and pose timestamps | `datasets/tum/rgbd_dataset_freiburg1_desk/` |

Cityscapes requires an account with the data provider. Downloads remain at
their original providers and retain the associated access and use terms.

## Fixed-version Hub preparation

The preparation entry downloads the declared files into their evaluation
locations. It uses the versions in
[configs/assets/hub_assets.json](../configs/assets/hub_assets.json):

```bash
python scripts/prepare_hub_assets.py oam_tcd quickbird
python scripts/prepare_hub_assets.py fsc147
python scripts/prepare_hub_assets.py gamus
```

FSC147 preparation reads the official `test` list, downloads its 1,190 images
and places the two annotation files under `annotations/`. GAMUS preparation
reads [configs/gamus/samples.json](../configs/gamus/samples.json) and downloads
the RGB and height file for every listed test ID. These commands preserve the
complete evaluation splits and their source filenames.

To print the source versions and destination specifications without a download:

```bash
python scripts/prepare_hub_assets.py oam_tcd quickbird fsc147 gamus --list
```

Hub downloads require `huggingface-hub==0.36.0`, included in
`environments/prediction.txt`. The `--list` option uses only the Python
standard library and performs no network requests.

## UAVid and fixed training tiles

Arrange the original RGB images and color labels as:

```text
datasets/UAVid/train/seq*/Images/*.png
datasets/UAVid/train/seq*/Labels/*.png
datasets/UAVid/val/seq*/Images/*.png
datasets/UAVid/val/seq*/Labels/*.png
```

Generate the 5,600 training image-label tile pairs used by the segmentation
baselines with:

```bash
python scripts/prepare_uavid_crops.py
```

[training_crops.json](../configs/segformer_uavid/training_crops.json) fixes
the sequence, original frame, top coordinate and left coordinate for every
tile. Tiles have side 1024 and stride-512 origins over the full source image,
with right/bottom extension to multiples of 512. Original image content
is copied at its original scale. Outside-image RGB pixels are zero;
color-label extension is RGB `[0, 0, 255]`. The SegFormer training reader
assigns this color to clutter (class 0), whereas the Twins training reader
maps it to ignore index 255. These are model-specific label decoders for
the same color-valued crop collection. See the
[SegFormer training strategy](segformer_uavid.md#training-strategy).
Outputs are:

```text
datasets/UAVidCrops/train/seq*/Images/FRAME_TOP_LEFT.png
datasets/UAVidCrops/train/seq*/Labels/FRAME_TOP_LEFT.png
```

Validation uses original UAVid images. Its sliding-window or native-image
inference is selected in the evaluation command.

## Manual archive preparation

- Extract Cityscapes' `leftImg8bit` and `gtFine` directories directly under
  `datasets/Cityscapes/`, preserving city and frame names. Evaluation accepts
  official `*_labelIds.png` files and converts pixel values to the 19-class
  train-ID mapping. Existing `*_labelTrainIds.png` files are read directly.
  To prepare train-ID files for both evaluation and upstream training, run
  `python scripts/prepare_cityscapes_labels.py`. This creates mapped pixel
  values and preserves the originals; renaming a file is not conversion.
  The mapping follows the [official Cityscapes label definitions](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).
- From the Polyp-PVT data package, place `TestDataset/Kvasir/images` and
  `masks` under `datasets/kvasir_seg/`. For CHAMELEON, retain `Imgs` and `GT`
  under `datasets/chameleon/`.
- For Columbia, preserve the original `4cam_splc` TIFF images and the
  `edgemask/*_edgemask.jpg` files. The evaluation reader derives foreground,
  background and excluded boundary pixels from these masks.
- Keep the original LEVIR-CD `test.zip` intact. Its `A`, `B` and `label`
  directories are read directly from the archive, using all 128 original pairs.
- Place the NYUv2 labeled MATLAB file and split file under `datasets/nyuv2/`.
  RGB extraction, JPEG round trip, depth quantization and evaluation crop are
  performed by the task reader.
- Keep `MPI-Sintel-training_images.zip` and `MPI-Sintel-training_extras.zip`
  under `datasets/sintel/`. The reader pairs Final images with their labeled
  flow and occlusion entries; the test benchmark's unlabeled split is separate.
- Extract only `Test_CAVE.h5` from the Fusformer data package to
  `datasets/cave/`. It contains eleven test cubes; the single-image demo is a
  separate file.
- Extract the original DIV2K validation HR PNGs to `datasets/div2k_hr/`.
  The task reader creates the fixed image-specific noise used for denoising.
- Extract the GoPro evaluation LMDB package and retain `GoPro/test/input.lmdb`
  and `GoPro/test/target.lmdb` under `datasets/gopro_test/`. Keep each LMDB's
  `data.mdb` and accompanying key metadata together.
- Extract the IMD2020 real-life manipulated collection under `datasets/imd2020/`,
  preserving its source directories, images and `*_mask.png` names. The reader
  finds all 2,010 manipulated-image/mask pairs.
- Extract the complete TUM `freiburg1_desk` archive under `datasets/tum/`.
  Keep its `rgb`, `depth`, `rgb.txt`, `depth.txt` and `groundtruth.txt` files.
  [configs/pi3/samples.json](../configs/pi3/samples.json) supplies the fixed
  timestamp matches and ten nonoverlapping input groups, with 596 valid views.

Task weights and architecture preparation are described in
[weights.md](weights.md). Evaluation commands and metric aggregation are in
[prediction.md](prediction.md), [segformer_uavid.md](segformer_uavid.md) and
[mmseg.md](mmseg.md).

GAMUS evaluation uses all 2,861 official test pairs: 361 Washington DC,
1,000 New York and 1,500 Philadelphia tiles. The fixed manifest accepts
both upstream RGB-image naming conventions (`_RGB.h5` and `_IMG.h5`).
