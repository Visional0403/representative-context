"""Native-resolution image restoration with fixed target pairing and noise."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .div2k_seeds import NOISE_SEEDS


class RestorationTask:
    def __init__(self, case, data_root):
        self.case, self.root = case, Path(data_root)
        if case == "div2k":
            self.rows = [
                {"id": path.stem, "target": path}
                for path in sorted(self.root.rglob("*.png"))
            ]
        else:
            import lmdb

            self.environments = []
            for kind in ("input", "target"):
                paths = [
                    path
                    for path in self.root.rglob(kind + ".lmdb")
                    if "__MACOSX" not in path.parts and (path / "data.mdb").is_file()
                ]
                if len(paths) != 1:
                    raise ValueError(f"Expected one GoPro {kind} LMDB.")
                self.environments.append(
                    lmdb.open(
                        str(paths[0]),
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                    )
                )
            indices = []
            for environment in self.environments:
                with environment.begin() as transaction:
                    indices.append(
                        sorted(
                            key.decode()
                            for key in transaction.cursor().iternext(
                                keys=True, values=False
                            )
                        )
                    )
            if indices[0] != indices[1]:
                raise ValueError("GoPro input and target keys do not match.")
            self.rows = [{"id": key} for key in indices[0]]
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        def size(row):
            return (
                int(np.prod(Image.open(row["target"]).size))
                if self.case == "div2k"
                else 1280 * 720
            )

        ordered = sorted(self.rows, key=lambda row: (size(row), row["id"]))
        return [(ordered[len(ordered) // 2], 1.0)]

    def prepare(self, row):
        if self.case == "div2k":
            target = np.asarray(Image.open(row["target"]).convert("RGB")).copy()
            seed = NOISE_SEEDS[row["id"]]
            image = target.astype(np.float32) / 255 + np.random.default_rng(
                seed
            ).normal(0, 25 / 255, target.shape).astype(np.float32)
        else:
            pair = []
            for environment in self.environments:
                with environment.begin() as transaction:
                    payload = transaction.get(row["id"].encode())
                    pair.append(
                        cv2.imdecode(
                            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                        )[:, :, ::-1].copy()
                    )
            image, target = pair
        tensor = (
            torch.from_numpy(image.transpose(2, 0, 1).copy())
            .float()
            .cuda()
            .unsqueeze(0)
        )
        if image.dtype == np.uint8:
            tensor = tensor / 255
        return tensor, target

    def forward(self, model, inputs):
        height, width = inputs.shape[-2:]
        bottom, right = (-height) % 64, (-width) % 64
        padded = (
            F.pad(inputs, (0, right, 0, bottom), mode="reflect")
            if bottom or right
            else inputs
        )
        return model(padded)[..., :height, :width].clamp(0, 1)

    def score(self, output, target, inputs):
        raw = output[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        prediction, truth = np.rint(raw * 255).astype(np.float64), target.astype(
            np.float64
        )
        mse = np.mean((prediction - truth) ** 2)
        window = cv2.getGaussianKernel(11, 1.5)

        def filtered(value):
            if self.case == "gopro":
                return cv2.sepFilter2D(value, -1, window, window)[5:-5, 5:-5]
            return cv2.filter2D(value, -1, window @ window.T)[5:-5, 5:-5]

        mx, my = filtered(prediction), filtered(truth)
        vx, vy = filtered(prediction**2) - mx * mx, filtered(truth**2) - my * my
        cross = filtered(prediction * truth) - mx * my
        ssim = ((2 * mx * my + 6.5025) * (2 * cross + 58.5225)) / (
            (mx * mx + my * my + 6.5025) * (vx + vy + 58.5225)
        )
        return {
            "psnr_db": float(10 * math.log10(255**2 / max(mse, 1e-16))),
            "ssim": float(ssim.mean()),
            "unquantized_mse": float(np.mean((raw - truth / 255) ** 2)),
        }

    def aggregate(self, records):
        return {
            name: float(np.mean([row["metrics"][name] for row in records]))
            for name in records[0]["metrics"]
        }
