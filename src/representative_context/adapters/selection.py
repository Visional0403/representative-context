"""Weight-preserving SRA controls for token or pooled-region selection."""

import math
import torch
from torch.nn import functional as F

from representative_context.adapters.sra import SpatialReductionAttention
from representative_context.budget import plan_capacity


def extract_qkv(core, x, height, width):
    batch, queries, channels = x.shape
    dim, bias = channels // core.num_heads, core.in_proj_bias
    q = F.linear(
        x, core.in_proj_weight[:channels], None if bias is None else bias[:channels]
    )
    q = q.reshape(batch, queries, core.num_heads, dim).permute(0, 2, 1, 3)
    reduced, kh, kw = x, height, width
    if core.sr is not None:
        reduced_map = core.sr(x.transpose(1, 2).reshape(batch, channels, height, width))
        kh, kw = reduced_map.shape[-2:]
        reduced = core.norm(reduced_map.flatten(2).transpose(1, 2))
    kv = F.linear(
        reduced,
        core.in_proj_weight[channels:],
        None if bias is None else bias[channels:],
    )
    kv = kv.reshape(batch, kh * kw, 2, core.num_heads, dim).permute(2, 0, 3, 1, 4)
    return q, *kv.unbind(0), kh, kw


class SelectionSRA(SpatialReductionAttention):
    """Match selection capacity to the maximum balanced-group KV count."""

    def __init__(
        self,
        original,
        stride,
        k,
        method,
        seed=42,
        keep_scale=1.0,
        selector="torch",
        score_elements=4 * 1024 * 1024,
    ):
        super().__init__(original, stride, k, seed=seed)
        self.method, self.keep_scale = method, keep_scale
        self.selector, self.score_elements = selector, score_elements
        if method not in (
            "topk_exact",
            "topk_proxy8",
            "pooled_regions",
            "sparge",
            "sparge_fp32",
            "sparge_quant_full",
        ):
            raise ValueError(method)
        if keep_scale <= 0:
            raise ValueError("The retained-capacity multiplier must be positive.")
        if method == "topk_proxy8":
            dim = self.in_proj_weight.shape[1] // self.num_heads
            generator = torch.Generator().manual_seed(seed)
            projection = torch.linalg.qr(
                torch.randn(dim, min(8, dim), generator=generator)
            ).Q
            projection *= math.sqrt(dim / projection.shape[-1])
            self.register_buffer(
                "routing_projection",
                projection.to(self.in_proj_weight),
                persistent=False,
            )
        else:
            self.register_buffer("routing_projection", None, persistent=False)

    def forward(self, x, height, width):
        if x.shape[0] != 1 or self.training:
            raise ValueError("Selection controls support batch-one inference.")
        q, key, value, kh, kw = extract_qkv(self, x, height, width)
        plan = plan_capacity(self.query_stride, self.sr_ratio, kh * kw, self.k)
        self.last_plan = plan
        keep = min(kh * kw, max(1, round(plan.max_group_size * self.keep_scale)))
        if self.method.startswith("sparge"):
            from representative_context.controls.sparge import attention

            if self.keep_scale != 1.0:
                raise ValueError(
                    "Use k to set Sparge capacity; the top-k keep multiplier must equal one."
                )
            mode = "quant_full" if self.method == "sparge_quant_full" else self.method
            output = attention(q, key, value, plan.capacity_limit, mode)
        elif self.method == "pooled_regions":
            from representative_context.controls.regions import bilevel_region_attention

            output = bilevel_region_attention(
                q, key, value, (height, width), (kh, kw), keep
            )
        else:
            from representative_context.controls.topk import topk_attention_optimized

            route_q = route_k = None
            if self.routing_projection is not None and keep < kh * kw:
                route_q, route_k = (
                    q @ self.routing_projection,
                    key @ self.routing_projection,
                )
            output = topk_attention_optimized(
                q,
                key,
                value,
                keep,
                route_q,
                route_k,
                score_elements=self.score_elements,
                selector=self.selector,
            )
        output = output.transpose(1, 2).reshape(x.shape)
        return self.out_proj(output)
