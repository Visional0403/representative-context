# MMSegmentation components

This directory contains the MMSegmentation 1.2.2 components used by the
SegFormer/Cityscapes and Twins-SVT-S experiments. The selected modules
support MiT and Twins backbones, SegFormer and FPN prediction heads,
Cityscapes and UAVid data processing, evaluation, and Twins training.

Upstream: https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2

The Apache License 2.0 and upstream copyright notices are retained.
Package exports are restricted to the selected components. The MiT
attention class uses its current forward method with MMCV-lite 2.1.0.
The model computations and checkpoint parameter names are preserved.

Task configurations are in `configs/`. Global attention replacement is
implemented in `representative_context/adapters/mmseg.py`.
