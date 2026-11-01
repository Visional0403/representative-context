"""Dataset protocols for six spatial-reduction dense-prediction experiments."""

import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .metrics import (
    binary_counts,
    binary_metrics,
    normalized_probability,
    nyu_depth_metrics,
)


class PredictionTask:
    """Read the complete evaluation split and retain its original prediction protocol."""

    def __init__(self, case, data_root):
        self.case, self.root = case, Path(data_root)
        if case == "polyp_kvasir":
            self.rows = [
                {
                    "id": path.name,
                    "image": path,
                    "mask": self.root / "masks" / (path.stem + ".jpg"),
                }
                for path in sorted((self.root / "images").iterdir())
                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ]
            masks = {
                path.stem: path
                for path in (self.root / "masks").iterdir()
                if path.suffix.lower() in (".png", ".tif", ".tiff")
            }
            if len(masks) != len(self.rows) or set(masks) != {
                r["image"].stem for r in self.rows
            }:
                raise ValueError("Image and mask identities do not match.")
            for row in self.rows:
                row["mask"] = masks[row["image"].stem]
        elif case == "hitnet":
            directory = self.root / "Imgs"
            if not directory.exists():
                directory = self.root / "Image"
            self.rows = [
                {
                    "id": path.name,
                    "image": path,
                    "mask": self.root / "GT" / f"{path.stem}.png",
                }
                for path in sorted(directory.iterdir())
                if path.suffix.lower() in (".jpg", ".png")
            ]
        elif case == "trufor":
            self.rows = [
                {
                    "id": path.name,
                    "image": path,
                    "mask": self.root / "edgemask" / f"{path.stem}_edgemask.jpg",
                }
                for path in sorted(self.root.glob("*.tif"))
            ]
        elif case == "changeformer":
            self.archive = zipfile.ZipFile(self.root / "test.zip")
            self.rows = [{"id": f"test_{index}.png"} for index in range(1, 129)]
            names = set(self.archive.namelist())
            if any(
                f"{part}/{row['id']}" not in names
                for row in self.rows
                for part in ("A", "B", "label")
            ):
                raise ValueError(
                    "The LEVIR-CD archive does not contain all 128 labeled image pairs."
                )
        elif case == "tcd_tiled":
            import pyarrow.parquet as pq

            self.parquet = pq.ParquetFile(self.root / "test.parquet")
            self.rows = [
                {"id": str(index), "row_index": index}
                for index in range(self.parquet.metadata.num_rows)
            ]
        elif case == "glpn_nyu":
            import h5py
            from scipy.io import loadmat

            self.h5 = h5py.File(self.root / "nyu_depth_v2_labeled.mat", "r")
            indices = (
                loadmat(self.root / "splits.mat")["testNdxs"].ravel().astype(int) - 1
            )
            self.rows = [
                {"id": str(index + 1), "index": int(index)} for index in indices
            ]
        else:
            raise ValueError(f"Unsupported evaluation task: {case}")
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        rows = self.rows if limit is None else self.rows[:limit]
        if self.case == "tcd_tiled":
            batches = self.parquet.iter_batches(batch_size=1)
            for row, batch in zip(rows, batches):
                yield {**row, "record": batch.to_pylist()[0]}
        else:
            yield from rows

    def timing_representatives(self):
        if self.case == "polyp_kvasir":
            ordered = sorted(
                self.rows,
                key=lambda row: (np.prod(Image.open(row["image"]).size), row["id"]),
            )
            return [(ordered[int(0.5 * (len(ordered) - 1))], 1.0)]
        if self.case == "trufor":
            shapes = {}
            for row in self.rows:
                shape = Image.open(row["image"]).size[::-1]
                shapes.setdefault(shape, []).append(row)
            return [
                (rows[0], len(rows) / self.count)
                for shape, rows in sorted(shapes.items())
            ]
        return [(next(self.iter_records(1)), 1.0)]

    def prepare(self, row):
        case = self.case
        if case == "glpn_nyu":
            import cv2

            index = row["index"]
            rgb = self.h5["images"][index].T
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[7:474, 7:632] = rgb[7:474, 7:632]
            encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1]
            image = cv2.cvtColor(
                cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            depth = (self.h5["rawDepths"][index].T * 1000.0).astype(np.uint16).astype(
                np.float32
            ) / 1000.0
            x = (
                torch.from_numpy(image.copy())
                .cuda()
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            )
            return x, torch.from_numpy(depth).cuda()
        if case == "changeformer":
            images = []
            for part in ("A", "B", "label"):
                with self.archive.open(f"{part}/{row['id']}") as stream:
                    images.append(
                        Image.open(io.BytesIO(stream.read())).convert(
                            "L" if part == "label" else "RGB"
                        )
                    )
            if any(image.size != (1024, 1024) for image in images):
                raise ValueError(
                    "LEVIR-CD evaluation uses original 1024 by 1024 image pairs."
                )
            inputs = tuple(
                TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)[None].cuda()
                for image in images[:2]
            )
            return inputs, np.asarray(images[2]) != 0
        if case == "tcd_tiled":
            record = row["record"]
            rgb = np.asarray(
                Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
            ).copy()
            mask = np.asarray(Image.open(io.BytesIO(record["annotation"]["bytes"])))
            if mask.ndim == 3 and mask.shape[2] == 3:
                mask = np.any(mask != 0, axis=2)
            if rgb.shape[:2] != mask.shape:
                raise ValueError("Canopy annotation dimensions do not match the image.")
            x = (
                torch.from_numpy(rgb.transpose(2, 0, 1).copy())
                .float()
                .div_(255)
                .unsqueeze(0)
            )
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            return ((x - mean) / std).cuda(), mask > 0
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        if case == "trufor":
            from scipy.ndimage import minimum_filter, maximum_filter

            mask = np.asarray(Image.open(row["mask"]).convert("RGB"))
            target = mask[:, :, 1] > mask[:, :, 0]
            x = (
                torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy())
                .float()[None]
                .cuda()
                / 256
            )
            return x, {
                "target": target,
                "fg": minimum_filter(target, 15),
                "bg": ~maximum_filter(target, 11),
            }
        mask = np.array(Image.open(row["mask"]).convert("L"), dtype=np.float32)
        if case == "hitnet":
            image = TF.resize(
                image, [1536, 1536], interpolation=TF.InterpolationMode.BILINEAR
            )
            target = mask / 255
        else:
            target = mask / (mask.max() + 1e-8) > 0.5
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError("Polyp image and mask dimensions differ.")
        x = TF.normalize(
            TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )[None]
        if case == "polyp_kvasir":
            height, width = mask.shape
            x = F.pad(x, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        return x.cuda(), target

    def forward(self, model, inputs):
        if self.case == "changeformer":
            return model(*inputs)[-1]
        if self.case == "tcd_tiled":
            if tuple(inputs.shape[-2:]) != (2048, 2048):
                raise ValueError(
                    "Canopy evaluation expects native 2048 by 2048 images."
                )
            result = inputs.new_zeros((1, 2, 2048, 2048))
            counts = inputs.new_zeros((1, 1, 2048, 2048))
            for top in (0, 768, 1024):
                for left in (0, 768, 1024):
                    tile = inputs[
                        ..., top : top + 1024, left : left + 1024
                    ].contiguous()
                    output = F.interpolate(
                        model(tile).logits,
                        (1024, 1024),
                        mode="bilinear",
                        align_corners=False,
                    ).softmax(1)
                    result[..., top : top + 1024, left : left + 1024] += output
                    counts[..., top : top + 1024, left : left + 1024] += 1
            return result / counts
        if self.case == "glpn_nyu":
            return model(pixel_values=inputs).predicted_depth[0]
        return model(inputs)

    def score(self, output, target, inputs):
        case = self.case
        if case == "glpn_nyu":
            if output.shape != target.shape:
                output = F.interpolate(
                    output[None, None],
                    size=target.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            return nyu_depth_metrics(output, target)
        if case in ("changeformer", "tcd_tiled"):
            prediction = output.argmax(1)[0].byte().cpu().numpy() != 0
            return {"counts": binary_counts(prediction, target)}
        if case == "polyp_kvasir":
            output = F.interpolate(
                output[0] + output[1],
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            output = output[..., : target.shape[0], : target.shape[1]]
            probability = normalized_probability(output)
            prediction = probability > 0.5
            intersection = np.count_nonzero(prediction & target)
            return {
                "Dice": (2 * intersection + 1e-8)
                / (np.count_nonzero(prediction) + np.count_nonzero(target) + 1e-8),
                "IoU": (intersection + 1e-8)
                / (np.count_nonzero(prediction | target) + 1e-8),
            }
        if case == "hitnet":
            output = F.interpolate(
                output[0][-1] + output[1],
                target.shape,
                mode="bilinear",
                align_corners=False,
            )
            probability = normalized_probability(output)
            counts = binary_counts(probability > 0.5, target > 0.5)
            tp, fp, fn = (counts[key] for key in ("TP", "FP", "FN"))
            return {
                "Dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "IoU": tp / max(1, tp + fp + fn),
                "MAE": float(np.abs(probability - target).mean()),
            }
        probability = output[0].softmax(1)[0, 1].cpu().numpy()
        prediction, fg, bg = probability > 0.5, target["fg"], target["bg"]
        tp, fn = int((prediction & fg).sum()), int((~prediction & fg).sum())
        fp, tn = int((prediction & bg).sum()), int((~prediction & bg).sum())
        direct = 2 * tp / max(1, 2 * tp + fp + fn)
        inverse = 2 * fn / max(1, 2 * fn + tn + tp)
        return {"F1": max(direct, inverse), "direct_F1": direct}

    def aggregate(self, records):
        if self.case in ("changeformer", "tcd_tiled"):
            total = {
                key: sum(row["metrics"]["counts"][key] for row in records)
                for key in ("TP", "FP", "FN", "TN")
            }
            values = binary_metrics(total)
            if self.case == "tcd_tiled":
                tp, fp, fn, tn = (total[key] for key in ("TP", "FP", "FN", "TN"))
                values["miou_percent"] = 50 * (
                    tp / max(1, tp + fp + fn) + tn / max(1, tn + fp + fn)
                )
            return {**values, "counts": total}
        return {
            key: float(np.mean([row["metrics"][key] for row in records]))
            for key in records[0]["metrics"]
        }
