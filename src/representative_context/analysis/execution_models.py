"""Segmentation inputs and QKV capture for independent execution measurements."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from representative_context.adapters.selection import extract_qkv
from representative_context.analysis.execution import (
    edge_record,
    grouped_sdpa,
    make_plan,
)
from representative_context.budget import plan_capacity
from representative_context.kernels.indexed import build_layout, indexed_attention
from representative_context.layouts import make_ranks
from representative_context.runtime import read_config, resource_path


def load_model(root, case, k=16, checkpoint=None):
    root = Path(root)
    config = read_config(root / f"configs/{case}/validation.json")
    weight = checkpoint or resource_path(config["checkpoint"], root)
    native = case == "segformer_uavid"
    if native:
        from representative_context.models.segformer import load_segformer
        from representative_context.adapters.sra import install_sra

        model = load_segformer(weight)
        install_sra(model, k=k, dense=True)
        blocks = [
            (stage + 1, index + 1, block.attn)
            for stage, layer in enumerate(model.backbone.layers)
            for index, block in enumerate(layer[1])
        ]
    else:
        from representative_context.models.mmseg import load_mmseg
        from representative_context.adapters.mmseg import install_mmseg

        model = load_mmseg(
            config,
            weight,
            root / "third_party/mmsegmentation",
            protocol="slide1024" if case == "twins_cityscapes" else "slide2048",
        )
        install_mmseg(model, config["architecture"], k=k, dense=True)
        blocks = [
            (stage + 1, index + 1, block.attn.core)
            for stage, layer in enumerate(model.backbone.layers)
            for index, block in enumerate(layer)
            if hasattr(block.attn, "core")
        ]
    assert len(blocks) == (8 if native else 9)
    return model, blocks, config


def select_image(data_root, height, width, image=None):
    for path in (
        [image]
        if image is not None
        else sorted((Path(data_root) / "val").glob("seq*/Images/*.png"))
    ):
        with Image.open(path) as source:
            w, h = source.size
        if (h, w) == (height, width) if height != width else h >= height and w >= width:
            return path
    raise ValueError(
        "No image contains the requested window or matches the native shape."
    )


def load_input(path, height, width, native):
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    rgb = rgb[:height, :width]
    mean = (0.485, 0.456, 0.406) if native else (123.675, 116.28, 103.53)
    std = (0.229, 0.224, 0.225) if native else (58.395, 57.12, 57.375)
    if native:
        rgb = rgb / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)) / np.asarray(
        std, dtype=np.float32
    ).reshape(1, 1, 3)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].cuda()


@torch.inference_mode()
def capture_entries(model, blocks, inputs, k=16, with_buckets=True):
    captures, handles = {}, []
    for index, (_, _, core) in enumerate(blocks):

        def capture(module, values, current=index):
            captures[current] = (
                values[0].detach().cpu(),
                int(values[1]),
                int(values[2]),
            )

        handles.append(core.register_forward_pre_hook(capture))
    try:
        features = model.backbone(inputs)
        del features
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    entries, records = [], []
    for index, (stage, block, core) in enumerate(blocks):
        x, height, width = captures.pop(index)
        q, key, value, kh, kw = extract_qkv(core, x.cuda(), height, width)
        plan = plan_capacity(core.query_stride, core.sr_ratio, kh * kw, k)
        layout = None
        if plan.groups > 1:
            ranks = torch.tensor(
                make_ranks(kh, kw, plan.capacity_limit, seed=core.seed),
                device="cuda",
                dtype=torch.int64,
            )
            layout = build_layout(ranks, (height, width), (kh, kw), plan.capacity_limit)
        buckets, bucket_record = make_plan(layout) if with_buckets else (None, None)
        entries.append((q, key, value, layout, buckets))
        records.append(
            dict(
                stage=stage,
                block=block,
                seed=core.seed,
                **edge_record(
                    (height, width),
                    (kh, kw),
                    layout,
                    q.shape[1],
                    q.shape[-1],
                    plan.capacity_limit,
                ),
                bucket_plan=bucket_record,
            )
        )
    return entries, records


@torch.inference_mode()
def check_entries(entries, records):
    for (q, key, value, layout, plans), record in zip(entries, records):
        if layout is not None and plans is None:
            plans, _ = make_plan(layout)
        reference = grouped_sdpa(q, key, value, plans)
        candidate = (
            reference.clone()
            if layout is None
            else indexed_attention(q, key, value, layout)
        )
        delta = candidate - reference
        relative = float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        )
        absolute = float(delta.abs().max())
        assert relative <= 1e-4 and absolute <= 1e-3, (relative, absolute)
        nq, nk = q.shape[2], key.shape[2]
        qs = np.unique(np.linspace(0, nq - 1, min(16, nq), dtype=int))
        if layout is not None:
            rank = np.empty(nk, dtype=np.int64)
            start = 0
            for group, count in enumerate(record["kv_counts"]):
                rank[layout.kv_order.cpu().numpy()[start : start + count]] = group
                start += count
        sampled = []
        for qi in qs:
            if layout is None:
                ki = np.arange(nk)
            else:
                qh, qw = layout.query_shape
                kh, kw = layout.kv_shape
                y, x = divmod(int(qi), qw)
                group = rank[(y * kh // qh) * kw + x * kw // qw]
                ki = np.flatnonzero(rank == group)
            ki = torch.as_tensor(ki, device="cuda")
            score = (
                torch.matmul(
                    q[:, :, qi : qi + 1].double(),
                    key.index_select(2, ki).double().transpose(-1, -2),
                )
                / q.shape[-1] ** 0.5
            )
            oracle = torch.matmul(
                score.softmax(-1), value.index_select(2, ki).double()
            ).float()
            sampled.append(
                [
                    float((oracle - reference[:, :, qi : qi + 1]).abs().max()),
                    float((oracle - candidate[:, :, qi : qi + 1]).abs().max()),
                ]
            )
        record["numerical_check"] = dict(
            full_relative_l2=relative,
            full_max_abs_error=absolute,
            fp64_queries=len(qs),
            fp64_reordered_max_abs_error=max(row[0] for row in sampled),
            fp64_indexed_max_abs_error=max(row[1] for row in sampled),
        )
        assert (
            max(
                record["numerical_check"]["fp64_reordered_max_abs_error"],
                record["numerical_check"]["fp64_indexed_max_abs_error"],
            )
            <= 1e-3
        )
