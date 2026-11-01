"""Spatial context control with global CLS exchange."""

from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl
from representative_context.kernels.indexed import (
    GroupLayout,
    build_layout,
    indexed_attention,
)
from representative_context.layouts.spatial import point_coverage_ranks
from representative_context.budget import plan_capacity
from representative_context.kernels.prefix import (
    dense_attention,
    indexed_space_cls_attention,
)


class DinoPoint32Controller:

    def __init__(self, attention, stride, index, special=1):
        self.original = attention
        self.reference = attention.forward
        self.stride = stride
        self.index = index
        self.special = special
        self.grid = None
        self.mode = "dense_sdpa"
        self.k = 14
        self.layouts = {}
        self.last = None

    def forward(self, x, attn_bias=None):
        if self.mode == "dense_original":
            return (
                self.reference(x)
                if attn_bias is None
                else self.reference(x, attn_bias=attn_bias)
            )
        if attn_bias is not None:
            raise ValueError(
                "This spatial adapter requires attention without a nested bias."
            )
        b, n, c = x.shape
        a = self.original
        qkv = (
            a.qkv(x)
            .reshape(b, n, 3, a.num_heads, c // a.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, key, val = qkv.unbind(0)
        h, w = self.grid
        if n != h * w + self.special:
            raise ValueError(
                f"Special-token count does not match the spatial grid: {n}, {h}, {w}"
            )
        budget = plan_capacity(self.stride, 1, h * w, self.k).capacity_limit
        groups = math.ceil(h * w / budget)
        self.last = {
            "spatial_query_tokens": h * w,
            "spatial_key_tokens": h * w,
            "special_tokens": self.special,
            "query_stride": self.stride,
            "k": self.k,
            "capacity_limit": budget,
            "groups": groups,
            "description": "Each spatial group also reads the global CLS token.",
        }
        if self.mode == "dense_sdpa" or groups == 1:
            out = dense_attention(q, key, val, a.scale)
        elif self.mode == "sparse_point32":
            cache_key = (h, w, budget, x.device)
            if cache_key not in self.layouts:
                ranks = torch.tensor(
                    point_coverage_ranks(h, w, groups, 32, 42 + self.index),
                    device=x.device,
                    dtype=torch.int64,
                )
                self.layouts[cache_key] = build_layout(ranks, (h, w), (h, w), budget)
            layout = self.layouts[cache_key]
            if self.special:
                out = torch.cat(
                    [
                        indexed_space_cls_attention(
                            q[i : i + 1], key[i : i + 1], val[i : i + 1], layout
                        )
                        for i in range(b)
                    ],
                    dim=0,
                )
            else:
                out = indexed_attention(
                    q, key, val, layout, num_stages=1 if c // a.num_heads > 64 else 2
                )
        else:
            raise ValueError(self.mode)
        out = a.proj(out.transpose(1, 2).reshape(b, n, c))
        return a.proj_drop(out) if hasattr(a, "proj_drop") else out

    @property
    def dense(self):
        return self.mode != "sparse_point32"

    @dense.setter
    def dense(self, value):
        self.mode = "dense_sdpa" if value else "sparse_point32"


def install_dino_point32(encoder, special=1):
    before = {n: v.data_ptr() for n, v in encoder.state_dict().items()}
    controllers = []
    stride = encoder.patch_embed.proj.stride
    if stride[0] != stride[1]:
        raise ValueError("The spatial layout requires a square patch stride.")
    for i, block in enumerate(encoder.blocks):
        if not hasattr(block, "attn"):
            raise ValueError(
                "Use an encoder with individually accessible attention blocks."
            )
        a = block.attn
        if not math.isclose(a.scale, (a.qkv.in_features // a.num_heads) ** (-0.5)):
            raise ValueError("Attention scale does not match the head dimension.")
        control = DinoPoint32Controller(a, stride[0], i, special=special)
        a.forward = control.forward
        controllers.append(control)

    def set_grid(module, args):
        h, w = args[0].shape[-2:]
        for control in controllers:
            control.grid = (h // stride[0], w // stride[1])

    encoder.patch_embed.register_forward_pre_hook(set_grid)
    if before != {n: v.data_ptr() for n, v in encoder.state_dict().items()}:
        raise AssertionError(
            "Attention replacement changed model parameter identities."
        )
    return controllers
