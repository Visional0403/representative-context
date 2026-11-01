"""Complete reduced-resolution spectral-fusion evaluation splits."""

import math
from pathlib import Path

import h5py
import numpy as np
import torch


class FusionTask:
    def __init__(self, case, data_root):
        self.case = case
        filename = "Test_CAVE.h5" if case == "cave" else "test_qb_multiExm1.h5"
        self.data = h5py.File(Path(data_root) / filename, "r")
        key = "GT" if case == "cave" else "gt"
        self.count = self.data[key].shape[0]
        self.rows = [
            {"id": f"{case}_test_{index:02d}", "index": index}
            for index in range(self.count)
        ]

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        return [(self.rows[self.count // 2], 1.0)]

    def prepare(self, row):
        index = row["index"]
        if self.case == "cave":
            first = self.data["LRHSI"][index].astype(np.float32)
            second = self.data["RGB"][index].astype(np.float32)
            target = self.data["GT"][index].astype(np.float64)
        else:
            first = self.data["ms"][index].astype(np.float32) / 2047
            second = self.data["pan"][index].astype(np.float32) / 2047
            target = self.data["gt"][index] / 2047
        return (
            torch.from_numpy(first)[None].cuda(),
            torch.from_numpy(second)[None].cuda(),
        ), target

    def forward(self, model, inputs):
        output = model(*inputs)
        return output[0] if self.case == "cave" else output

    def score(self, output, target, inputs):
        prediction = output[0].cpu().numpy().astype(np.float64)
        error = prediction - target
        mse = float(np.mean(error**2))
        norm = np.linalg.norm(prediction, axis=0) * np.linalg.norm(target, axis=0)
        valid = norm > 1e-12
        angle = np.rad2deg(
            np.arccos(
                np.clip(np.sum(prediction * target, axis=0)[valid] / norm[valid], -1, 1)
            )
        )
        result = {
            "psnr_db": float(-10 * math.log10(max(mse, 1e-16))),
            "rmse": math.sqrt(mse),
            "sam_degrees": float(angle.mean()),
        }
        if self.case == "cave":
            means = target.mean(axis=(1, 2))
            result["ergas"] = float(
                100
                / 4
                * np.sqrt(
                    np.mean(
                        np.mean(error**2, axis=(1, 2)) / np.maximum(means**2, 1e-12)
                    )
                )
            )
        return result

    def aggregate(self, records):
        return {
            name: float(np.mean([row["metrics"][name] for row in records]))
            for name in records[0]["metrics"]
        }
