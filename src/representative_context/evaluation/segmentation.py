"""Gaussian sliding-window prediction and dataset-level confusion metrics."""

import numpy as np
import torch


def positions(length: int, size: int, stride: int) -> list[int]:
    if size > length or min(size, stride) < 1 or stride > size:
        raise ValueError(
            "Use a positive stride at most the window size, and a window within the image."
        )
    values = list(range(0, length - size + 1, stride))
    if length - size not in values:
        values.append(length - size)
    return values


def gaussian_weight(size: int, device) -> torch.Tensor:
    """Return unit-sum Gaussian weights with pixel-space sigma equal to size/4."""
    if size < 1:
        raise ValueError("The Gaussian window size must be positive.")
    axis = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    sigma = size / 4.0
    gaussian = torch.exp(-(x.square() + y.square()) / (2 * sigma**2))
    return (gaussian / gaussian.sum()).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_image(
    model, image: torch.Tensor, classes=8, window_size=1024, stride=512, native=False
):
    """Predict an original image; loading and normalization occur before this call."""
    if native:
        return model(image)
    batch, _, height, width = image.shape
    result = torch.zeros(
        (batch, classes, height, width), device=image.device, dtype=torch.float32
    )
    weights = torch.zeros(
        (batch, 1, height, width), device=image.device, dtype=torch.float32
    )
    gaussian = gaussian_weight(window_size, image.device)
    for top in positions(height, window_size, stride):
        for left in positions(width, window_size, stride):
            region = (
                ...,
                slice(top, top + window_size),
                slice(left, left + window_size),
            )
            result[region] += model(image[region]) * gaussian
            weights[region] += gaussian
    return result / weights


def confusion_matrix(prediction, target, classes=8, ignore=255):
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = target != ignore
    prediction, target = prediction[valid], target[valid]
    if np.any(target < 0) or np.any(target >= classes):
        raise ValueError(
            "Target labels must belong to the declared classes or ignore value."
        )
    return np.bincount(
        classes * target + prediction, minlength=classes * classes
    ).reshape(classes, classes)


def segmentation_metrics(hist):
    hist = np.asarray(hist, dtype=np.float64)
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / (union + 1e-8)
    return {
        "miou_percent": float(np.nanmean(iou) * 100),
        "class_iou_percent": (iou * 100).tolist(),
        "pixel_accuracy_percent": float(intersection.sum() / (hist.sum() + 1e-8) * 100),
    }
