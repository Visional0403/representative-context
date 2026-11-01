"""Fixed-window segmentation models for paired attention interventions."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from representative_context.analysis.context import factors
from representative_context.runtime import read_config, resource_path


class SegmentationAdapter:
    def __init__(self, case, root, data_root=None, checkpoint=None):
        root = Path(root)
        cfg = read_config(root / "configs" / case / "validation.json")
        self.case = case
        self.native = case == "segformer_uavid"
        self.city = case == "twins_cityscapes"
        self.side = 1024 if self.city else 2048
        self.grid = self.side // 32
        self.classes = 19 if self.city else 8
        self.root = data_root or resource_path(cfg["data_root"], root)
        self.checkpoint = checkpoint or resource_path(cfg["checkpoint"], root)
        if self.native:
            from representative_context.models.segformer import load_segformer
            from representative_context.adapters.sra import install_sra
            from representative_context.data.uavid import evaluation_pairs

            self.model = load_segformer(self.checkpoint)
            install_sra(self.model, k=14, pattern="dispersed32", dense=True)
            self.specs = [
                (stage, index, block, block.attn)
                for stage, layer in enumerate(self.model.backbone.layers)
                for index, block in enumerate(layer[1])
            ]
            self.pairs = evaluation_pairs(self.root)
        else:
            from representative_context.models.mmseg import load_mmseg
            from representative_context.adapters.mmseg import install_mmseg
            from representative_context.evaluation.mmseg import image_pairs

            protocol = "slide1024" if self.city else "slide2048"
            self.model = load_mmseg(
                cfg,
                self.checkpoint,
                root / "third_party/mmsegmentation",
                protocol=protocol,
            )
            install_mmseg(
                self.model, cfg["architecture"], k=14, pattern="dispersed32", dense=True
            )
            self.specs = [
                (stage, index, block, block.attn.core)
                for stage, layer in enumerate(self.model.backbone.layers)
                for index, block in enumerate(layer)
                if hasattr(block.attn, "core")
            ]
            self.pairs = image_pairs(self.root, cfg["dataset"])
        self.images = [pair[0] for pair in self.pairs]
        expected = cfg.get("expected_images", cfg.get("expected_split_size"))
        if len(self.images) != expected:
            raise ValueError(
                f"Expected {expected} validation images, found {len(self.images)}."
            )
        if self.city:
            self.indices = []
            for city in sorted({path.parent.name for path in self.images}):
                selected = [
                    index
                    for index, path in enumerate(self.images)
                    if path.parent.name == city
                ]
                self.indices.append(selected[len(selected) // 2])
        else:
            self.indices = [5, 15, 25, 35, 45, 55, 65]
        self.targets = [item for item in self.specs if item[0] == 3]
        if len(self.targets) != 2:
            raise ValueError(
                "Restoration expects the two final-stage global attention blocks."
            )

    def window(self, index):
        from representative_context.data.uavid import normalize_image, read_rgb
        from representative_context.evaluation.mmseg import read_label

        path, label_path = self.pairs[index]
        with Image.open(path) as image:
            width, height = image.size

            def grid(length):
                stride = 768 if self.city else 1024
                starts = list(range(0, length - self.side + 1, stride))
                if starts[-1] != length - self.side:
                    starts.append(length - self.side)
                return starts

            ys, xs = grid(height), grid(width)
            top, left = ys[len(ys) // 2], xs[len(xs) // 2]
            rgb = np.array(
                image.convert("RGB").crop(
                    (left, top, left + self.side, top + self.side)
                )
            )
        if self.native:
            image = normalize_image(read_rgb(path))
            window = (
                image[:, top : top + self.side, left : left + self.side]
                .unsqueeze(0)
                .cuda()
            )
        else:
            window = (
                torch.from_numpy(rgb.transpose(2, 0, 1).copy())
                .unsqueeze(0)
                .cuda()
                .float()
            )
            window = (
                window - self.model.data_preprocessor.mean
            ) / self.model.data_preprocessor.std
        gt = read_label(label_path, "Cityscapes" if self.city else "UAVid")[
            top : top + self.side, left : left + self.side
        ].copy()
        metadata = dict(
            index=index,
            scene=path.parent.name if self.city else path.parent.parent.name,
            image=path.relative_to(self.root).as_posix(),
            top_left=[top, left],
        )
        return window, gt, metadata

    def mode(self, pattern):
        for _, _, _, core in self.specs:
            core.dense = pattern is None
            if pattern is not None:
                core.pattern = pattern

    def decode(self, features):
        if self.native:
            captured = {}

            def capture(module, inputs, output):
                captured["native"] = output

            handle = self.model.decode_head.conv_seg.register_forward_hook(capture)
            try:
                logits = self.model.decode_head(features)
            finally:
                handle.remove()
            return captured["native"], logits
        native = self.model.decode_head(self.model.neck(tuple(features)))
        return native, self.model.decode_head.predict_by_feat(
            native, [{"img_shape": (self.side, self.side)}]
        )

    def tail(self, features, captured, pattern, arm, labels):
        value = captured[self.targets[0][1]]["block"]
        details = []
        for _, index, block, core in self.targets:
            if arm == "identity":
                branch = captured[index]["branch"]
            else:
                candidates, detail = factors(
                    core,
                    block.norm1(value),
                    labels,
                    self.grid,
                    self.classes,
                    (pattern,),
                )
                branch = candidates[pattern][arm]
                details.append(detail)
            value = value + block.drop_path(branch)
            value = value + block.drop_path(
                block.ffn(block.norm2(value), self.grid, self.grid)
                if self.native
                else block.ffn(block.norm2(value))
            )
            if not self.native and index == self.targets[0][1]:
                value = self.model.backbone.layers[3][index + 1](
                    value, (self.grid, self.grid)
                )
        norm = (
            self.model.backbone.layers[3][2]
            if self.native
            else self.model.backbone.norm_list[3]
        )
        value = (
            norm(value)
            .reshape(1, self.grid, self.grid, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        native, logits = self.decode([*features[:3], value])
        return native, logits, details
