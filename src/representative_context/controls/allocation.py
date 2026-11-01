"""Stage-capacity intervention with the same projections and indexed execution."""

import types
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from representative_context.adapters.selection import extract_qkv
from representative_context.budget import CapacityPlan, plan_capacity
from representative_context.kernels.indexed import build_layout, indexed_attention
from representative_context.layouts import make_ranks


def forward_allocated(self, x, height, width):
    q, key, value, kh, kw = extract_qkv(self, x, height, width)
    capacity = self.allocated_capacity
    if capacity is None:
        plan = plan_capacity(self.query_stride, self.sr_ratio, kh * kw, self.k)
    else:
        groups = (kh * kw + capacity - 1) // capacity
        plan = CapacityPlan(
            self.query_stride,
            self.sr_ratio,
            kh * kw,
            self.k,
            capacity,
            groups,
            kh * kw // groups,
            (kh * kw + groups - 1) // groups,
        )
    self.last_plan = plan
    if self.dense or plan.groups == 1:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = F.scaled_dot_product_attention(q, key, value)
    else:
        identity = (height, width, kh, kw, x.device, plan.capacity_limit, self.seed)
        if identity not in self.cache:
            ranks = torch.tensor(
                make_ranks(kh, kw, plan.capacity_limit, seed=self.seed),
                device=x.device,
                dtype=torch.int64,
            )
            self.cache[identity] = build_layout(
                ranks, (height, width), (kh, kw), plan.capacity_limit
            )
        output = indexed_attention(q, key, value, self.cache[identity])
    return self.out_proj(output.transpose(1, 2).reshape(x.shape))


def set_stage_capacities(specs, capacities):
    for stage, _, _, core in specs:
        if not hasattr(core, "allocated_capacity"):
            core.forward = types.MethodType(forward_allocated, core)
        core.allocated_capacity = None if capacities is None else capacities[stage]
        core.dense = capacities is None
        core.cache.clear()
