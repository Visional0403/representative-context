"""Global spatial attention for spectral and multispectral fusion models."""

import types

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from representative_context.budget import plan_capacity
from representative_context.kernels.indexed import build_layout, indexed_attention
from representative_context.layouts import make_ranks


class FusionAttention:
    """Share QKV and output projections while controlling spatial context."""

    def __init__(self, attention, index, k, dense=False, dense_backend="sdpa"):
        self.original, self.index, self.k = attention, index, k
        self.dense, self.dense_backend, self.grid, self.cache = (
            dense,
            dense_backend,
            None,
            {},
        )

    def forward(self, x, mask=None):
        if mask is not None:
            raise ValueError("Spatial fusion evaluation expects unmasked input grids.")
        a = self.original
        batch, count, channels = x.shape
        height, width = self.grid
        if count != height * width:
            raise ValueError("Token count does not match the fusion grid.")
        q, key, value = [
            part.reshape(batch, count, a.heads, -1).permute(0, 2, 1, 3)
            for part in a.to_qkv(x).chunk(3, dim=-1)
        ]
        plan = plan_capacity(1, 1, count, self.k)
        if (self.dense or plan.groups == 1) and self.dense_backend == "indexed":
            identity = height, width, "dense"
            if identity not in self.cache:
                ranks = torch.arange(count, device=x.device, dtype=torch.int64)
                self.cache[identity] = build_layout(
                    ranks, (height, width), (height, width), count
                )
            output = indexed_attention(q, key, value, self.cache[identity])
        elif self.dense or plan.groups == 1:
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if q.dtype == torch.float32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                output = F.scaled_dot_product_attention(
                    q, key, value, dropout_p=0.0, scale=a.scale
                )
        else:
            identity = height, width, plan.capacity_limit
            if identity not in self.cache:
                ranks = torch.tensor(
                    make_ranks(
                        height, width, plan.capacity_limit, seed=42 + self.index
                    ),
                    device=x.device,
                    dtype=torch.int64,
                )
                self.cache[identity] = build_layout(
                    ranks, (height, width), (height, width), plan.capacity_limit
                )
            output = indexed_attention(q, key, value, self.cache[identity])
        return a.to_out(output.transpose(1, 2).reshape(batch, count, channels))


def install_fusion_attention(
    model, attention_type, k=16, dense=False, dense_backend="sdpa"
):
    controls = []
    for module in model.modules():
        if isinstance(module, attention_type):
            control = FusionAttention(module, len(controls), k, dense, dense_backend)
            module.forward = control.forward
            controls.append(control)

    def set_grid(_module, arguments):
        for control in controls:
            control.grid = tuple(arguments[1].shape[-2:])

    model.register_forward_pre_hook(set_grid)
    return controls
