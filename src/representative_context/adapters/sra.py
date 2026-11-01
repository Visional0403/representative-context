"""Shared projection and spatial-reduction path for dense and sparse SRA."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from representative_context.budget import plan_capacity
from representative_context.layouts import make_ranks

from torch.nn.attention import SDPBackend, sdpa_kernel


class SpatialReductionAttention(nn.Module):
    """Preserve learned projections and SR; select an execution path at inference.

    Input is [1, H*W, C]. Layouts are cached per shape and device. A single
    group uses the same optimized dense SDPA path as the dense control.
    """

    def __init__(
        self,
        original: nn.Module,
        query_stride: int,
        k: int = 16,
        pattern: str = "dispersed32",
        seed: int = 42,
        dense: bool = False,
    ):
        super().__init__()
        self.in_proj_weight = original.attn.in_proj_weight
        self.in_proj_bias = original.attn.in_proj_bias
        self.out_proj = original.attn.out_proj
        self.sr = original.sr if original.sr_ratio > 1 else None
        self.norm = original.norm if original.sr_ratio > 1 else None
        self.sr_ratio = original.sr_ratio
        self.num_heads = original.num_heads
        self.query_stride = query_stride
        self.k = k
        self.pattern = pattern
        self.seed = seed
        self.dense = dense
        self.cache = {}
        self.last_plan = None

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, queries, channels = x.shape
        if batch != 1 or self.training:
            raise ValueError("The indexed SRA adapter supports batch-one inference.")
        dim = channels // self.num_heads
        bias = self.in_proj_bias
        q = F.linear(
            x,
            self.in_proj_weight[:channels],
            bias[:channels] if bias is not None else None,
        )
        q = q.reshape(batch, queries, self.num_heads, dim).permute(0, 2, 1, 3)
        reduced = x
        kh, kw = height, width
        if self.sr is not None:
            reduced_map = self.sr(
                x.transpose(1, 2).reshape(batch, channels, height, width)
            )
            kh, kw = reduced_map.shape[-2:]
            reduced = self.norm(reduced_map.flatten(2).transpose(1, 2))
        kv = F.linear(
            reduced,
            self.in_proj_weight[channels:],
            bias[channels:] if bias is not None else None,
        )
        kv = kv.reshape(batch, kh * kw, 2, self.num_heads, dim).permute(2, 0, 3, 1, 4)
        key, value = kv.unbind(0)
        plan = plan_capacity(self.query_stride, self.sr_ratio, kh * kw, self.k)
        self.last_plan = plan
        if self.dense or plan.groups == 1:
            backend = (
                SDPBackend.MATH
                if q.device.type != "cuda"
                else (
                    SDPBackend.EFFICIENT_ATTENTION
                    if q.dtype == torch.float32
                    else SDPBackend.FLASH_ATTENTION
                )
            )
            with sdpa_kernel(backend):
                output = F.scaled_dot_product_attention(q, key, value)
        else:
            from representative_context.kernels.indexed import (
                build_layout,
                indexed_attention,
            )

            shape = (height, width, kh, kw, x.device, self.pattern, self.k, self.seed)
            if shape not in self.cache:
                if len(self.cache) >= 7:
                    self.cache.pop(next(iter(self.cache)))
                ranks = torch.tensor(
                    make_ranks(kh, kw, plan.capacity_limit, self.pattern, self.seed),
                    dtype=torch.int64,
                    device=x.device,
                )
                self.cache[shape] = build_layout(
                    ranks, (height, width), (kh, kw), plan.capacity_limit
                )
            output = indexed_attention(q, key, value, self.cache[shape])
        return self.out_proj(output.transpose(1, 2).reshape(batch, queries, channels))


def install_sra(
    model: nn.Module,
    k: int = 16,
    pattern: str = "dispersed32",
    seed: int = 42,
    dense: bool = False,
    method=None,
    keep_scale=1.0,
) -> list[SpatialReductionAttention]:
    """Replace global SRA blocks, keeping the original parameter objects."""
    wrappers = []
    for stage, layer in enumerate(model.backbone.layers):
        for block_index, block in enumerate(layer[1]):
            if method is not None and not dense:
                from .selection import SelectionSRA

                block.attn = SelectionSRA(
                    block.attn,
                    4 * 2**stage,
                    k,
                    method,
                    seed + stage * 100 + block_index,
                    keep_scale,
                ).eval()
            else:
                block.attn = SpatialReductionAttention(
                    block.attn,
                    4 * 2**stage,
                    k,
                    pattern,
                    seed + stage * 100 + block_index,
                    dense,
                ).eval()
            wrappers.append(block.attn)
    return wrappers
