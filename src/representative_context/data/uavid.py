"""Official UAVid RGB labels and validation normalization."""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch

from .labels import CLASS_NAMES, COLORS, uavid_label_ids


def evaluation_pairs(root: Path, split: str = "val") -> list[tuple[Path, Path]]:
    """Return the complete split in lexicographic image-path order."""
    pairs = []
    for image in sorted((root / split).glob("**/Images/*.png")):
        label = image.parent.parent / "Labels" / image.name
        if not label.is_file():
            raise FileNotFoundError(label)
        pairs.append((image, label))
    if not pairs:
        raise FileNotFoundError(f"No labeled images in {root / split}")
    return pairs


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_image(image: np.ndarray) -> torch.Tensor:
    transform = A.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
    )
    return torch.from_numpy(transform(image=image)["image"]).permute(2, 0, 1).float()


def read_pair(image_path: Path, label_path: Path):
    image, mask = read_rgb(image_path), read_rgb(label_path)
    if image.shape != mask.shape:
        raise ValueError("Image and label dimensions must agree.")
    label = uavid_label_ids(mask)
    return normalize_image(image), torch.from_numpy(label).long()
