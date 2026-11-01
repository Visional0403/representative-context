"""Indexed spatial attention with globally retained special tokens."""

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
def _space_attention_with_cls(
    Q,
    K,
    V,
    O,
    ORDER,
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
    SPACE: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = (batch_head // HEADS, batch_head % HEADS)
    start = group * (SPACE // GROUPS) + tl.minimum(group, SPACE % GROUPS)
    length = SPACE // GROUPS + (group < SPACE % GROUPS).to(tl.int32)
    qp = tl.program_id(2) * BQ + tl.arange(0, BQ)
    if tl.program_id(2) * BQ >= length:
        return
    qi = tl.load(ORDER + start + qp, qp < length, other=0) + 1
    channel = tl.arange(0, BLOCK_D)
    q = tl.load(
        Q + batch * sqb + head * sqh + qi[:, None] * sqn + channel[None, :] * sqd,
        (qp < length)[:, None] & (channel < DIM)[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BLOCK_D), tl.float32)
    for block in range(tl.cdiv(length + 1, BK)):
        kp = block * BK + tl.arange(0, BK)
        spatial_index = tl.load(ORDER + start + kp, kp < length, other=0) + 1
        ki = tl.where(kp == length, 0, spatial_index)
        valid = kp < length + 1
        k = tl.load(
            K + batch * skb + head * skh + ki[:, None] * skn + channel[None, :] * skd,
            valid[:, None] & (channel < DIM)[None, :],
            other=0,
        )
        score = tl.dot(q, tl.trans(k), input_precision="tf32x3") * DIM ** (-0.5)
        score = tl.where(valid[None, :], score, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(score, 1))
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(score - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        v = tl.load(
            V + batch * svb + head * svh + ki[:, None] * svn + channel[None, :] * svd,
            valid[:, None] & (channel < DIM)[None, :],
            other=0,
        )
        acc = acc * correction[:, None] + tl.dot(
            probability.to(v.dtype), v, input_precision="tf32x3"
        )
        maximum = new_maximum
    tl.store(
        O + batch * sob + head * soh + qi[:, None] * son + channel[None, :] * sod,
        acc / denominator[:, None],
        (qp < length)[:, None] & (channel < DIM)[None, :],
    )


def dense_attention(q, k, v, scale):
    backend = (
        SDPBackend.EFFICIENT_ATTENTION
        if q.dtype == torch.float32
        else SDPBackend.FLASH_ATTENTION
    )
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)


def indexed_space_cls_attention(q, k, v, layout: GroupLayout):
    """Retain global CLS exchange alongside balanced spatial groups."""
    batch, heads, count, dim = q.shape
    space = math.prod(layout.query_shape)
    if (
        batch != 1
        or count != space + 1
        or k.shape != q.shape
        or (v.shape != q.shape)
        or (not 1 <= dim <= 128)
        or (layout.query_shape != layout.kv_shape)
    ):
        raise ValueError(
            "Expected batch one, one CLS token and a complete spatial grid."
        )
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _space_attention_with_cls[
        layout.groups, heads, triton.cdiv(triton.cdiv(space, layout.groups), 64)
    ](
        q,
        k,
        v,
        output,
        layout.kv_order,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        heads,
        space,
        layout.groups,
        dim,
        max(16, triton.next_power_of_2(dim)),
        64,
        64,
        num_warps=4,
        num_stages=2,
    )
    output[:, :, :1] = dense_attention(q[:, :, :1], k, v, dim ** (-0.5))
    return output


@triton.jit
def _space_attention_with_prefix(
    Q,
    K,
    V,
    O,
    ORDER,
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
    SPACE: tl.constexpr,
    GROUPS: tl.constexpr,
    SPECIAL: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    group, batch_head = (tl.program_id(0), tl.program_id(1))
    batch, head = (batch_head // HEADS, batch_head % HEADS)
    start = group * (SPACE // GROUPS) + tl.minimum(group, SPACE % GROUPS)
    length = SPACE // GROUPS + (group < SPACE % GROUPS).to(tl.int32)
    qp = tl.program_id(2) * BQ + tl.arange(0, BQ)
    if tl.program_id(2) * BQ >= length:
        return
    qi = tl.load(ORDER + start + qp, qp < length, other=0) + SPECIAL
    channel = tl.arange(0, BLOCK_D)
    q = tl.load(
        Q + batch * sqb + head * sqh + qi[:, None] * sqn + channel[None, :] * sqd,
        (qp < length)[:, None] & (channel < DIM)[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BLOCK_D), tl.float32)
    for block in range(tl.cdiv(length + SPECIAL, BK)):
        kp = block * BK + tl.arange(0, BK)
        spatial = tl.load(ORDER + start + kp, kp < length, other=0) + SPECIAL
        ki = tl.where(kp < length, spatial, kp - length)
        valid = kp < length + SPECIAL
        k = tl.load(
            K + batch * skb + head * skh + ki[:, None] * skn + channel[None, :] * skd,
            valid[:, None] & (channel < DIM)[None, :],
            other=0,
        )
        score = tl.dot(q, tl.trans(k), input_precision="tf32x3") * DIM ** (-0.5)
        score = tl.where(valid[None, :], score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        correction = tl.exp(maximum - new_max)
        probability = tl.exp(score - new_max[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        v = tl.load(
            V + batch * svb + head * svh + ki[:, None] * svn + channel[None, :] * svd,
            valid[:, None] & (channel < DIM)[None, :],
            other=0,
        )
        acc = acc * correction[:, None] + tl.dot(
            probability.to(v.dtype), v, input_precision="tf32x3"
        )
        maximum = new_max
    tl.store(
        O + batch * sob + head * soh + qi[:, None] * son + channel[None, :] * sod,
        acc / denominator[:, None],
        (qp < length)[:, None] & (channel < DIM)[None, :],
    )


def indexed_space_prefix_attention(q, k, v, layout: GroupLayout, special: int):
    """Retain all prefix-token exchanges alongside balanced spatial groups."""
    batch, heads, count, dim = q.shape
    space = math.prod(layout.query_shape)
    if (
        batch != 1
        or special < 1
        or count != space + special
        or (k.shape != q.shape)
        or (v.shape != q.shape)
        or (not 1 <= dim <= 128)
        or (layout.query_shape != layout.kv_shape)
    ):
        raise ValueError(
            "Expected batch one, matching spatial QKV grids and a positive prefix length."
        )
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _space_attention_with_prefix[
        layout.groups, heads, triton.cdiv(triton.cdiv(space, layout.groups), 64)
    ](
        q,
        k,
        v,
        output,
        layout.kv_order,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        heads,
        space,
        layout.groups,
        special,
        dim,
        max(16, triton.next_power_of_2(dim)),
        64,
        64,
        num_warps=4,
        num_stages=2,
    )
    output[:, :, :special] = dense_attention(q[:, :, :special], k, v, dim ** (-0.5))
    return output
