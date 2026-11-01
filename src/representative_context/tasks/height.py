"""Pixel-level height prediction on the fixed complete GAMUS test split."""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F


class HeightTask:
    def __init__(self, data_root, samples, timing_sample_id=None):
        self.root = Path(data_root)
        self.rows = json.loads(Path(samples).read_text(encoding="utf-8"))
        self.count = len(self.rows)
        self.timing_sample_id = timing_sample_id or self.rows[len(self.rows) // 2]["id"]

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        return [(next(row for row in self.rows if row["id"] == self.timing_sample_id), 1.0)]

    def prepare(self, row):
        arrays = {}
        for field in ("image", "height"):
            with h5py.File(self.root / row[field], "r") as source:
                arrays[field] = source["image"][:]
        rgb = arrays["image"].astype(np.float32)
        values = rgb - np.array([123.675, 116.28, 103.53], np.float32)
        values /= np.array([58.395, 57.12, 57.375], np.float32)
        return (
            torch.from_numpy(values.transpose(2, 0, 1).copy())[None].cuda(),
            arrays["height"],
        )

    def forward(self, model, inputs):
        height, width = inputs.shape[-2:]
        bottom, right = (-height) % 14, (-width) % 14
        padded = (
            F.pad(inputs, (0, right, 0, bottom), mode="reflect")
            if bottom or right
            else inputs
        )
        output = model(padded)
        return {
            "regression": output["regression"][..., :height, :width].clamp_min(0),
            "segmentation": output["segmentation"][..., :height, :width],
        }

    def score(self, output, target, inputs):
        prediction = output["regression"][0, 0].float().cpu().numpy().astype(np.float64)
        valid = np.isfinite(target) & (target > -5) & (target < 1000)
        residual = prediction[valid] - target[valid]
        return dict(
            valid_pixels=int(valid.sum()),
            squared_error_sum=float(np.dot(residual, residual)),
            absolute_error_sum=float(np.abs(residual).sum()),
            rmse_m=float(np.sqrt(np.mean(residual**2))),
            mae_m=float(np.abs(residual).mean()),
        )

    def aggregate(self, records):
        total = sum(row["metrics"]["valid_pixels"] for row in records)
        squared = sum(row["metrics"]["squared_error_sum"] for row in records)
        absolute = sum(row["metrics"]["absolute_error_sum"] for row in records)
        return dict(
            rmse_m=math.sqrt(squared / total),
            mae_m=absolute / total,
            valid_pixels=total,
        )
