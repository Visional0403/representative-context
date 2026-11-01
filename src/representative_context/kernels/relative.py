"""Dense and grouped attention with native decomposed relative positions."""

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


@triton.jit
def _sam_rel_attention(
    Q,
    K,
    V,
    O,
    ORDER,
    RH,
    RW,
    sqb,
    sqh,
    sqn,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sob,
    soh,
    son,
    sod,
    HEADS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = (batch_head // HEADS, batch_head % HEADS)
    start = group * (TOKENS // GROUPS) + tl.minimum(group, TOKENS % GROUPS)
    length = TOKENS // GROUPS + (group < TOKENS % GROUPS).to(tl.int32)
    qp = tl.program_id(2) * BQ + tl.arange(0, BQ)
    if tl.program_id(2) * BQ >= length:
        return
    if GROUPS == 1:
        qi = qp
    else:
        qi = tl.load(ORDER + start + qp, qp < length, other=0)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        Q + batch * sqb + head * sqh + qi[:, None] * sqn + d[None, :] * sqd,
        (qp < length)[:, None] & (d < HEAD_DIM)[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    accumulator = tl.zeros((BQ, BLOCK_D), tl.float32)
    for block in range(tl.cdiv(length, BK)):
        kp = block * BK + tl.arange(0, BK)
        if GROUPS == 1:
            ki = kp
        else:
            ki = tl.load(ORDER + start + kp, kp < length, other=0)
        k = tl.load(
            K + batch * skb + head * skh + ki[:, None] * skn + d[None, :] * skd,
            (kp < length)[:, None] & (d < HEAD_DIM)[None, :],
            other=0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="tf32x3") * HEAD_DIM ** (-0.5)
        valid = (qp < length)[:, None] & (kp < length)[None, :]
        rh = tl.load(
            RH + (batch_head * TOKENS + qi[:, None]) * HEIGHT + ki[None, :] // WIDTH,
            valid,
            other=0,
        )
        rw = tl.load(
            RW + (batch_head * TOKENS + qi[:, None]) * WIDTH + ki[None, :] % WIDTH,
            valid,
            other=0,
        )
        scores = tl.where((kp < length)[None, :], scores + rh + rw, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        v = tl.load(
            V + batch * svb + head * svh + ki[:, None] * svn + d[None, :] * svd,
            (kp < length)[:, None] & (d < HEAD_DIM)[None, :],
            other=0,
        )
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(v.dtype), v, input_precision="tf32x3")
        maximum = next_max
    tl.store(
        O + batch * sob + head * soh + qi[:, None] * son + d[None, :] * sod,
        accumulator / denominator[:, None],
        (qp < length)[:, None] & (d < HEAD_DIM)[None, :],
    )


def fused_sam_attention(q, k, v, rh, rw, order, groups, height, width):
    """Evaluate dense or grouped attention with learned relative positions."""
    batch, heads, tokens, dim = q.shape
    if (
        tokens != height * width
        or k.shape != q.shape
        or v.shape != q.shape
        or (not 1 <= dim <= 128)
    ):
        raise ValueError("Attention tensor shapes do not match the spatial grid.")
    if rh.shape != (batch * heads, tokens, height) or rw.shape != (
        batch * heads,
        tokens,
        width,
    ):
        raise ValueError(
            "Decomposed relative-position shapes do not match the spatial grid."
        )
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _sam_rel_attention[
        groups, batch * heads, triton.cdiv(triton.cdiv(tokens, groups), 32)
    ](
        q,
        k,
        v,
        output,
        order,
        rh.contiguous(),
        rw.contiguous(),
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        heads,
        height,
        width,
        tokens,
        groups,
        dim,
        max(16, triton.next_power_of_2(dim)),
        32,
        32,
        num_warps=4,
        num_stages=1,
    )
    return output


def sdpa_sam_attention(q, k, v, rh, rw, height, width):
    """Dense SDPA reference with the same decomposed relative-position terms."""
    batch, heads, tokens, _dim = q.shape
    bias = (rh[:, :, :, None] + rw[:, :, None, :]).reshape(batch, heads, tokens, tokens)
    if tokens % 8:
        bias = F.pad(bias, (0, 8 - tokens % 8))[..., :tokens]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
