"""Shared evaluation-label mappings for UAVid and Cityscapes."""

import numpy as np

CLASS_NAMES = (
    "clutter", "building", "road", "tree", "low_vegetation",
    "static_car", "moving_car", "human",
)
COLORS = (
    (0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0),
    (128, 128, 0), (192, 0, 192), (64, 0, 128), (64, 64, 0),
)
CITYSCAPES_CLASS_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 31, 32, 33)


def uavid_label_ids(rgb):
    """Decode the eight official RGB colors; all other colors are ignored."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("UAVid labels require an RGB image.")
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(COLORS):
        result[np.all(rgb == color, axis=-1)] = index
    return result


def cityscapes_train_ids(label_ids):
    """Map official labelIds to 19 train IDs, with void classes mapped to 255."""
    values = np.asarray(label_ids)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Cityscapes labelIds must be a two-dimensional integer array.")
    result = np.full(values.shape, 255, dtype=np.uint8)
    for train_id, label_id in enumerate(CITYSCAPES_CLASS_IDS):
        result[values == label_id] = train_id
    return result
