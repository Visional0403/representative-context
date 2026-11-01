"""Sintel Final optical flow with the released C+T GMA model."""

from collections import defaultdict
import io
from pathlib import Path, PurePosixPath
import struct
import zipfile

import numpy as np
from PIL import Image
import torch


def archive_index(archive, target):
    entries = {}
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if "training" not in parts:
            continue
        index = parts.index("training")
        if len(parts) > index + 3 and parts[index + 1] == target:
            entries[(parts[index + 2], parts[index + 3])] = name
    return entries


class MotionTask:
    def __init__(self, data_root, source):
        root = Path(data_root)
        self.source = Path(source)
        self.images = zipfile.ZipFile(root / "MPI-Sintel-training_images.zip")
        self.extras = zipfile.ZipFile(root / "MPI-Sintel-training_extras.zip")
        images, flows = archive_index(self.images, "final"), archive_index(
            self.extras, "flow"
        )
        scenes = defaultdict(list)
        for (scene, filename), path in images.items():
            scenes[scene].append((filename, path))
        self.rows = []
        for scene, frames in sorted(scenes.items()):
            frames.sort()
            for (filename, first), (_, second) in zip(frames[:-1], frames[1:]):
                key = scene, Path(filename).with_suffix(".flo").name
                if key in flows:
                    self.rows.append(
                        {
                            "id": f"{scene}/{filename}",
                            "scene": scene,
                            "first": first,
                            "second": second,
                            "flow": flows[key],
                        }
                    )
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        images = sorted((self.source / "imgs").glob("frame_*.png"))
        return [
            (
                {
                    "id": "author_sample_0047_0048",
                    "first": images[0],
                    "second": images[1],
                    "sample_files": True,
                },
                1.0,
            )
        ]

    def prepare(self, row):
        from utils.utils import InputPadder

        tensors = []
        for field in ("first", "second"):
            payload = (
                row[field].read_bytes()
                if row.get("sample_files")
                else self.images.read(row[field])
            )
            rgb = np.asarray(
                Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8
            )
            tensors.append(
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].cuda()
            )
        padder = InputPadder(tensors[0].shape)
        if row.get("sample_files"):
            target = None
        else:
            payload = self.extras.read(row["flow"])
            if payload[:4] != b"PIEH":
                raise ValueError("Invalid Sintel flow header.")
            width, height = struct.unpack_from("<ii", payload, 4)
            flow = np.frombuffer(payload, dtype="<f4", offset=12).reshape(
                height, width, 2
            )
            target = {"flow": flow, "padder": padder}
        return tuple(padder.pad(*tensors)), target

    def forward(self, model, inputs):
        return model(*inputs, iters=12, test_mode=True)[1]

    def score(self, output, target, inputs):
        prediction = (
            target["padder"].unpad(output[0]).float().permute(1, 2, 0).cpu().numpy()
        )
        error = np.linalg.norm(prediction - target["flow"], axis=-1)
        return {
            "epe": float(error.mean()),
            "epe_sum": float(error.sum(dtype=np.float64)),
            "pixels": int(error.size),
        }

    def aggregate(self, records):
        return {
            "epe": sum(row["metrics"]["epe_sum"] for row in records)
            / sum(row["metrics"]["pixels"] for row in records)
        }
