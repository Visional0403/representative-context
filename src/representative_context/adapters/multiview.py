"""Spatial groups with global per-view tokens and unchanged rotary positions."""

import math

import torch

from representative_context.budget import plan_capacity
from representative_context.kernels.indexed import build_layout
from representative_context.kernels.prefix import (
    dense_attention,
    indexed_space_prefix_attention,
)
from representative_context.layouts.spatial import point_coverage_ranks


class MultiViewAttention:
    def __init__(self, attention, index, k, dense):
        self.original, self.index, self.k, self.dense = attention, index, k, dense
        self.grid = None
        self.cache = {}

    def forward(self, x, attn_bias=None, xpos=None):
        if attn_bias is not None:
            raise ValueError(
                "The multiview spatial adapter requires attention without an additional bias."
            )
        batch, count, width = x.shape
        height, grid_width = self.grid
        spatial = height * grid_width
        if count % (spatial + 5):
            raise ValueError("Token count does not match five global tokens per view.")
        views = count // (spatial + 5)
        special = 5 * views
        a = self.original
        dim = width // a.num_heads
        q, key, value = (
            a.qkv(x)
            .reshape(batch, count, 3, a.num_heads, dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if hasattr(a, "q_norm"):
            q, key = a.q_norm(q).to(value.dtype), a.k_norm(key).to(value.dtype)
        if getattr(a, "rope", None) is not None:
            q, key = a.rope(q, xpos), a.rope(key, xpos)
        budget = plan_capacity(14, 1, views * spatial, self.k).capacity_limit
        groups = math.ceil(views * spatial / budget)
        if self.dense or groups == 1:
            output = dense_attention(q, key, value, a.scale)
        else:
            identity = height, grid_width, views, budget, x.device
            if identity not in self.cache:
                indices = torch.arange(count, device=x.device).reshape(
                    views, spatial + 5
                )
                order = torch.cat(
                    (indices[:, :5].reshape(-1), indices[:, 5:].reshape(-1))
                )
                inverse = torch.argsort(order)
                ranks = torch.tensor(
                    point_coverage_ranks(
                        views * height, grid_width, groups, 32, 42 + self.index
                    ),
                    device=x.device,
                    dtype=torch.int64,
                )
                layout = build_layout(
                    ranks,
                    (views * height, grid_width),
                    (views * height, grid_width),
                    budget,
                )
                self.cache[identity] = order, inverse, layout
            order, inverse, layout = self.cache[identity]
            if batch > 1:

                def fold(tensor):
                    return tensor.reshape(1, batch * a.num_heads, count, dim)

                output = indexed_space_prefix_attention(
                    fold(q[:, :, order]),
                    fold(key[:, :, order]),
                    fold(value[:, :, order]),
                    layout,
                    special,
                ).reshape(batch, a.num_heads, count, dim)[:, :, inverse]
            else:
                output = torch.cat(
                    [
                        indexed_space_prefix_attention(
                            q[b : b + 1, :, order],
                            key[b : b + 1, :, order],
                            value[b : b + 1, :, order],
                            layout,
                            special,
                        )
                        for b in range(batch)
                    ],
                    0,
                )[:, :, inverse]
        return a.proj_drop(a.proj(output.transpose(1, 2).reshape(batch, count, width)))


def install_multiview_attention(model, k=16, dense=False):
    controls = []
    for index, (_, attention) in enumerate(model.named_modules()):
        if all(
            hasattr(attention, name) for name in ("qkv", "proj", "num_heads", "scale")
        ):
            control = MultiViewAttention(attention, index, k, dense)
            attention.forward = control.forward
            controls.append(control)

    def set_grid(module, args):
        grid = args[0].shape[-2] // 14, args[0].shape[-1] // 14
        for control in controls:
            control.grid = grid

    model.register_forward_pre_hook(set_grid)
    return controls
