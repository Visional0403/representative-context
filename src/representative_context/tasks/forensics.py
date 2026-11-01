"""The complete IMD2020 evaluation protocol for IML-ViT."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter
import torch
from torch.nn import functional as F


class ForensicsTask:
    def __init__(self, data_root):
        root = Path(data_root)
        self.rows = []
        for mask in sorted(root.rglob("*_mask.png")):
            image = next(
                (
                    mask.with_name(mask.stem[:-5] + suffix)
                    for suffix in (".jpg", ".png", ".JPG")
                    if mask.with_name(mask.stem[:-5] + suffix).exists()
                ),
                None,
            )
            if image is None:
                raise FileNotFoundError(f"Missing image for mask: {mask}")
            self.rows.append(
                dict(id=image.relative_to(root).as_posix(), image=image, mask=mask)
            )
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        return [(self.rows[0], 1.0)]

    def prepare(self, row):
        image = np.asarray(Image.open(row["image"]).convert("RGB"))[:1024, :1024]
        target = (
            np.array(Image.open(row["mask"]).convert("L"), dtype=np.float32)[
                :1024, :1024
            ]
            / 255
            > 0.5
        )
        inputs = (
            torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None].cuda() / 255
        )
        inputs = F.pad(inputs, (0, 1024 - inputs.shape[-1], 0, 1024 - inputs.shape[-2]))
        mean = inputs.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = inputs.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (inputs - mean) / std, dict(
            target=target, fg=minimum_filter(target, 15), bg=~maximum_filter(target, 11)
        )

    def forward(self, model, inputs):
        features = model.featurePyramid_net(model.encoder_net(inputs))
        logits = model.predict_head(list(features.values()))
        return F.interpolate(
            logits, (1024, 1024), mode="bilinear", align_corners=False
        ).sigmoid()

    def score(self, output, target, inputs):
        height, width = target["target"].shape
        probability = output[0, 0, :height, :width].cpu().numpy()
        prediction = probability > 0.5
        fg, bg = target["fg"], target["bg"]
        tp, fn = int((prediction & fg).sum()), int((~prediction & fg).sum())
        fp, tn = int((prediction & bg).sum()), int((~prediction & bg).sum())
        direct = 2 * tp / max(1, 2 * tp + fp + fn)
        inverse = 2 * fn / max(1, 2 * fn + tn + tp)
        return dict(F1=max(direct, inverse), direct_F1=direct)

    def aggregate(self, records):
        return {
            name: float(np.mean([row["metrics"][name] for row in records]))
            for name in records[0]["metrics"]
        }
