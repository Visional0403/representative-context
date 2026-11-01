"""Official-layout labels, validation and GPU-resident timing inputs."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from representative_context.data.labels import cityscapes_train_ids, uavid_label_ids


def image_pairs(root, dataset):
    root = Path(root).resolve()
    if dataset == "Cityscapes":
        images = sorted((root / "leftImg8bit/val").glob("*/*_leftImg8bit.png"))
        pairs = []
        for image in images:
            label = root / "gtFine/val" / image.parent.name / image.name.replace(
                "_leftImg8bit.png", "_gtFine_labelTrainIds.png")
            if not label.is_file():
                label = label.with_name(label.name.replace("_labelTrainIds.png", "_labelIds.png"))
            if not label.is_file():
                raise FileNotFoundError(label)
            pairs.append((image, label))
        return pairs
    return [
        (image, image.parent.parent / "Labels" / image.name)
        for image in sorted((root / "val").glob("seq*/Images/*.png"))
    ]


def read_label(path, dataset):
    with Image.open(path) as image:
        if dataset == "Cityscapes":
            values = np.asarray(image, dtype=np.uint8)
            return cityscapes_train_ids(values) if Path(path).name.endswith("_labelIds.png") else values
        rgb = np.asarray(image.convert("RGB"))
    return uavid_label_ids(rgb)


def metrics(hist):
    intersection = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    iou = np.divide(
        intersection, union, out=np.full(len(hist), np.nan), where=union > 0
    )
    return {
        "miou_percent": float(np.nanmean(iou) * 100),
        "class_iou_percent": [None if np.isnan(x) else float(x * 100) for x in iou],
        "pixel_accuracy_percent": float(intersection.sum() / hist.sum() * 100),
    }


def timing_input(image, height, width, device="cuda"):
    with Image.open(image) as source:
        rgb = np.asarray(source.convert("RGB"))
    if height > rgb.shape[0] or width > rgb.shape[1]:
        raise ValueError("The requested timing window must fit in the original image.")
    image = torch.from_numpy(
        np.ascontiguousarray(rgb[:height, :width].transpose(2, 0, 1))
    )
    image = image.unsqueeze(0).to(device, torch.float32)
    mean = image.new_tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
    std = image.new_tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)
    return (image - mean) / std
