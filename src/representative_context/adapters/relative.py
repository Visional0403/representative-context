"""Spatial context control preserving learned relative positions."""

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
from representative_context.kernels.relative import (
    fused_sam_attention,
    sdpa_sam_attention,
)


class RelativeController:

    def __init__(self, attention, is_global, index, get_rel_pos):
        self.a, self.is_global, self.index, self.get_rel_pos = (
            attention,
            is_global,
            index,
            get_rel_pos,
        )
        self.original = attention.forward
        self.mode, self.k = ("dense_fused", 14)
        self.local_backend = "fused"
        self.cache = {}
        self.last = None

    def forward(self, x):
        if self.mode == "dense_original":
            return self.original(x)
        b, h, w, c = x.shape
        a = self.a
        n = h * w
        q, key, val = (
            a.qkv(x).reshape(b, n, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )
        spatial = q.reshape(b * a.num_heads, h, w, -1)
        rh = self.get_rel_pos(h, h, a.rel_pos_h)
        rw = self.get_rel_pos(w, w, a.rel_pos_w)
        rh = (
            torch.einsum("bhwc,hkc->bhwk", spatial, rh)
            .reshape(b * a.num_heads, n, h)
            .contiguous()
        )
        rw = (
            torch.einsum("bhwc,wkc->bhwk", spatial, rw)
            .reshape(b * a.num_heads, n, w)
            .contiguous()
        )
        budget = plan_capacity(16, 1, h * w, self.k).capacity_limit
        groups = (
            math.ceil(n / budget)
            if self.is_global and self.mode == "sparse_point32"
            else 1
        )
        self.last = {
            "Q": n,
            "KV": n,
            "query_stride": 16,
            "capacity_limit": budget,
            "groups": groups,
            "global_attention": self.is_global,
        }
        if self.mode == "dense_sdpa" or (
            self.mode == "sparse_point32"
            and (not self.is_global)
            and (self.local_backend == "sdpa")
        ):
            y = sdpa_sam_attention(q, key, val, rh, rw, h, w)
        else:
            order = q
            if groups > 1:
                ck = (h, w, budget, x.device)
                if ck not in self.cache:
                    ranks = torch.tensor(
                        point_coverage_ranks(h, w, groups, 32, 42 + self.index),
                        dtype=torch.int64,
                        device=x.device,
                    )
                    self.cache[ck] = build_layout(ranks, (h, w), (h, w), budget)
                order = self.cache[ck].kv_order
            y = fused_sam_attention(q, key, val, rh, rw, order, groups, h, w)
        return a.proj(y.transpose(1, 2).reshape(b, h, w, c))

    @property
    def dense(self):
        return self.mode != "sparse_point32"

    @dense.setter
    def dense(self, value):
        self.mode = "dense_fused" if value else "sparse_point32"


def install_relative(encoder, get_rel_pos):
    before = {n: p.data_ptr() for n, p in encoder.state_dict().items()}
    controls = []
    for i, block in enumerate(encoder.blocks):
        control = RelativeController(block.attn, block.window_size == 0, i, get_rel_pos)
        block.attn.forward = control.forward
        controls.append(control)
    if before != {n: p.data_ptr() for n, p in encoder.state_dict().items()}:
        raise AssertionError(
            "Attention replacement changed model parameter identities."
        )
    return controls
