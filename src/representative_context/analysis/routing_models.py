"""Declared segmentation inputs for matched routing diagnostics."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from representative_context.analysis.execution_models import load_model
from representative_context.budget import plan_capacity
from representative_context.data.uavid import normalize_image, read_rgb
from representative_context.evaluation.mmseg import image_pairs
from representative_context.runtime import resource_path


def load(root, case, data_root=None, checkpoint=None):
    model, blocks, config = load_model(root, case, k=14, checkpoint=checkpoint)
    data_root = Path(data_root or resource_path(config["data_root"], root))
    native = case == "segformer_uavid"
    if native:
        from representative_context.data.uavid import evaluation_pairs

        pairs = evaluation_pairs(data_root)
    else:
        pairs = image_pairs(data_root, "Cityscapes")
    images = [pair[0] for pair in pairs]
    assert len(images) == (70 if native else 500)
    side = 2048 if native else 1024
    specs = []
    for stage, block, core in blocks:
        qside = side // core.query_stride
        kside = qside // core.sr_ratio
        core.diagnostic_capacity = plan_capacity(
            core.query_stride, core.sr_ratio, kside**2, 14
        ).capacity_limit
        specs.append((stage - 1, block - 1, core, qside, kside))
    if native:
        indices = (0, 20, 40)
    else:
        cities = sorted({path.stem.split("_")[0] for path in images})
        indices = tuple(
            next(
                index
                for index, path in enumerate(images)
                if path.stem.startswith(city + "_")
            )
            for city in cities
        )

    def get_window(index):
        if native:
            return normalize_image(read_rgb(images[index]))[:, :side, :side][
                None
            ].cuda()
        rgb = np.asarray(Image.open(images[index]).convert("RGB"))[:side, :side]
        window = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
            .cuda()
            .float()
        )
        return (
            window - window.new_tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
        ) / window.new_tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)

    def call(window):
        return (
            model(window)
            if native
            else model.encode_decode(
                window, [{"img_shape": (side, side), "ori_shape": (side, side)}]
            )
        )

    return model, specs, images, indices, get_window, call, data_root
