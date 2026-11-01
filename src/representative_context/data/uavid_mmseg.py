"""UAVid sequence and crop datasets with RGB annotation loading."""

from pathlib import Path
import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS

from .uavid import CLASS_NAMES, COLORS


@DATASETS.register_module()
class UAVidSequenceDataset(BaseSegDataset):
    METAINFO = dict(classes=CLASS_NAMES, palette=[list(color) for color in COLORS])

    def __init__(self, split, **kwargs):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown UAVid split: {split}")
        super().__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            data_prefix=dict(img_path=split, seg_map_path=split),
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self):
        root = Path(self.data_prefix["img_path"])
        images = sorted(root.glob("seq*/Images/*.png"))
        if not images:
            raise FileNotFoundError(root)
        records = []
        for image in images:
            label = image.parent.parent / "Labels" / image.name
            if not label.is_file():
                raise FileNotFoundError(label)
            records.append(
                dict(
                    img_path=str(image),
                    seg_map_path=str(label),
                    label_map=self.label_map,
                    reduce_zero_label=False,
                    seg_fields=[],
                )
            )
        return records


@TRANSFORMS.register_module()
class LoadUAVidColorAnnotations(BaseTransform):
    def transform(self, results):
        with Image.open(results["seg_map_path"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        label = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for index, color in enumerate(COLORS):
            label[np.all(rgb == color, axis=2)] = index
        results["gt_seg_map"] = label
        results["seg_fields"].append("gt_seg_map")
        return results
