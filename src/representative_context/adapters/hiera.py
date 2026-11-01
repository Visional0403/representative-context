"""Global spatial attention in the fixed Hiera counting backbone."""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from representative_context.budget import plan_capacity
from representative_context.kernels.indexed import build_layout, indexed_attention
from representative_context.layouts import make_ranks


class HieraGlobalAttention:
    def __init__(self, attention, index, k=16, dense=False):
        self.original, self.index, self.k, self.dense = attention, index, k, dense
        self.cache = {}

    def forward(self, x):
        batch, height, width, channels = x.shape
        a = self.original
        q, key, value = (
            a.qkv(x).reshape(batch, height * width, 3, a.num_heads, -1).unbind(2)
        )
        q, key, value = [part.transpose(1, 2) for part in (q, key, value)]
        stride = 1024 // height
        plan = plan_capacity(stride, 1, height * width, self.k)
        if self.dense or plan.groups == 1:
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if q.dtype == torch.float32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                output = F.scaled_dot_product_attention(
                    q, key, value, dropout_p=0.0, scale=q.shape[-1] ** -0.5
                )
        else:
            identity = height, width, plan.capacity_limit, x.device
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
        return a.proj(output.transpose(1, 2).reshape(batch, height, width, -1))


def install_hiera_attention(model, k=16, dense=False):
    controls = []
    for index, block in enumerate(model.backbone.trunk.blocks):
        if block.window_size != 0:
            continue
        if block.attn.q_pool is not None:
            raise ValueError(
                "The counting adapter expects unpooled global attention blocks."
            )
        control = HieraGlobalAttention(block.attn, index, k, dense)
        block.attn.forward = control.forward
        controls.append(control)
    return controls
