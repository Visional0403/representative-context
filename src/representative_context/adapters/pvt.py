"""Spatial-reduction attention for PVT-family dense prediction models."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from representative_context.budget import plan_capacity
from representative_context.layouts import make_ranks
from representative_context.kernels.indexed import build_layout, indexed_attention


class PVTSpatialAttention(nn.Module):
    """Share the original PVT projections, SR, normalization and output layers."""

    def __init__(self, original, query_stride, k=16, seed=42, dense=False):
        super().__init__()
        self.q, self.kv, self.proj = original.q, original.kv, original.proj
        self.proj_drop = original.proj_drop
        self.attn_drop = original.attn_drop
        self.sr_ratio, self.num_heads, self.scale = (
            original.sr_ratio,
            original.num_heads,
            original.scale,
        )
        self.sr = original.sr if self.sr_ratio > 1 else None
        self.norm = original.norm if self.sr_ratio > 1 else None
        self.query_stride, self.k, self.seed, self.dense = query_stride, k, seed, dense
        self.cache, self.last_plan = {}, None

    def forward(self, x, height, width):
        batch, queries, channels = x.shape
        if self.training or batch != 1 or queries != height * width:
            raise ValueError(
                "PVT indexed attention requires batch-one inference on a full spatial grid."
            )
        dim = channels // self.num_heads
        if self.scale != dim**-0.5:
            raise ValueError(
                "The indexed kernel uses inverse-square-root head scaling."
            )
        q = self.q(x).reshape(batch, queries, self.num_heads, dim).permute(0, 2, 1, 3)
        reduced, kh, kw = x, height, width
        if self.sr is not None:
            reduced = self.sr(x.transpose(1, 2).reshape(batch, channels, height, width))
            kh, kw = reduced.shape[-2:]
            reduced = self.norm(reduced.flatten(2).transpose(1, 2))
        key, value = (
            self.kv(reduced)
            .reshape(batch, kh * kw, 2, self.num_heads, dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        plan = plan_capacity(self.query_stride, self.sr_ratio, kh * kw, self.k)
        self.last_plan = plan
        if self.dense or plan.groups == 1:
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if q.dtype == torch.float32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                output = F.scaled_dot_product_attention(
                    q, key, value, dropout_p=0.0, scale=self.scale
                )
        else:
            shape = (height, width, kh, kw, plan.capacity_limit, x.device)
            if shape not in self.cache:
                ranks = torch.tensor(
                    make_ranks(kh, kw, plan.capacity_limit, seed=self.seed),
                    dtype=torch.int64,
                    device=x.device,
                )
                self.cache[shape] = build_layout(
                    ranks, (height, width), (kh, kw), plan.capacity_limit
                )
            output = indexed_attention(
                q, key, value, self.cache[shape], num_stages=1 if dim > 64 else 2
            )
        return self.proj_drop(
            self.proj(output.transpose(1, 2).reshape(batch, queries, channels))
        )


def install_pvt(backbone, k=16, seed=42, dense=False, seed_scheme="stage"):
    wrappers = []
    for stage, stride in enumerate((4, 8, 16, 32), start=1):
        for index, block in enumerate(getattr(backbone, f"block{stage}")):
            block_seed = seed + (
                100 * (stage - 1) + index if seed_scheme == "stage" else len(wrappers)
            )
            block.attn = PVTSpatialAttention(
                block.attn, stride, k, block_seed, dense
            ).eval()
            wrappers.append(block.attn)
    return wrappers
