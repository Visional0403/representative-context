"""FSC147 test counting with the author's three exemplars and refinement."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms


class CountingTask:
    def __init__(self, data_root):
        self.root = Path(data_root)
        split = json.loads(
            (self.root / "annotations/Train_Test_Val_FSC_147.json").read_text()
        )["test"]
        self.annotations = json.loads(
            (self.root / "annotations/annotation_FSC147_384.json").read_text()
        )
        self.rows = [
            dict(id=name, target_count=len(self.annotations[name]["points"]))
            for name in split
        ]
        self.count = len(self.rows)

    def iter_records(self, limit=None):
        yield from (self.rows if limit is None else self.rows[:limit])

    def timing_representatives(self):
        return [(next(row for row in self.rows if row["id"] == "4935.jpg"), 1.0)]

    def prepare(self, row):
        from utils.data import resize_and_pad

        image = np.asarray(
            Image.open(self.root / "images_384_VarV2" / row["id"]).convert("RGB")
        ).copy()
        height, width = image.shape[:2]
        boxes = torch.tensor(
            self.annotations[row["id"]]["box_examples_coordinates"], dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
        tensor, boxes, scale = resize_and_pad(
            tensor, boxes, size=1024, zero_shot=False, train=False
        )
        valid = torch.tensor(
            [math.floor(width * scale), math.floor(height * scale)],
            device="cuda",
            dtype=torch.float32,
        )
        tensor = (
            tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor[None].cuda(), boxes[None].cuda(), valid), row["target_count"]

    def forward(self, model, inputs):
        image, examples, valid = inputs
        predictions, *_ = model(image, examples)
        output = predictions[0]
        if output["pred_boxes"].numel() == 0:
            return torch.empty((0, 4), device=image.device)
        values = output["box_v"]
        mask = values > values.max() * 0.11
        boxes, scores = output["pred_boxes"][mask], values[mask]
        boxes = boxes[nms(boxes, scores, 0.5)].clamp(0, 1)
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5 * 1024
        return boxes[(centers[:, 0] < valid[0]) & (centers[:, 1] < valid[1])]

    def score(self, output, target, inputs):
        count = len(output)
        return dict(
            predicted_count=count,
            target_count=target,
            absolute_error=abs(count - target),
            signed_error=count - target,
        )

    def aggregate(self, records):
        errors = np.asarray(
            [row["metrics"]["signed_error"] for row in records], dtype=np.float64
        )
        return dict(
            mae=float(np.abs(errors).mean()), rmse=float(np.sqrt(np.mean(errors**2)))
        )
