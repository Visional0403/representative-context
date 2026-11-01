"""Per-query controlled contexts with fixed local neighbors and global anchors."""

from __future__ import annotations
from types import MethodType
import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _context_attention(
    Q,
    K,
    V,
    O,
    IDS,
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
    QH: tl.constexpr,
    QW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BUDGET: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    cell, batch_head = (tl.program_id(0), tl.program_id(1))
    batch, head = (batch_head // HEADS, batch_head % HEADS)
    rh: tl.constexpr = QH // KH
    rw: tl.constexpr = QW // KW
    copies: tl.constexpr = rh * rw
    pos, channel = (tl.arange(0, BQ), tl.arange(0, BD))
    qi = (cell // KW * rh + pos // rw) * QW + cell % KW * rw + pos % rw
    q = tl.load(
        Q + batch * sqb + head * sqh + qi[:, None] * sqn + channel[None, :] * sqd,
        (pos < copies)[:, None] & (channel < D)[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BD), tl.float32)
    for start in range(tl.cdiv(BUDGET, BK)):
        kp = start * BK + tl.arange(0, BK)
        ki = tl.load(IDS + cell * BUDGET + kp, kp < BUDGET, other=0)
        k = tl.load(
            K + batch * skb + head * skh + ki[:, None] * skn + channel[None, :] * skd,
            (kp < BUDGET)[:, None] & (channel < D)[None, :],
            other=0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="tf32x3") * D ** (-0.5)
        scores = tl.where((kp < BUDGET)[None, :], scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        v = tl.load(
            V + batch * svb + head * svh + ki[:, None] * svn + channel[None, :] * svd,
            (kp < BUDGET)[:, None] & (channel < D)[None, :],
            other=0,
        )
        acc = acc * correction[:, None] + tl.dot(
            probabilities.to(v.dtype), v, input_precision="tf32x3"
        )
        maximum = new_max
    tl.store(
        O + batch * sob + head * soh + qi[:, None] * son + channel[None, :] * sod,
        acc / denominator[:, None],
        (pos < copies)[:, None] & (channel < D)[None, :],
    )


def context_attention(q, k, v, indices, qside, side):
    assert q.shape[2] == qside**2 and k.shape[2] == side**2 and (qside % side == 0)
    assert indices.shape[0] == side**2 and indices.dtype == torch.int32
    copies = (qside // side) ** 2
    bq = max(16, triton.next_power_of_2(copies))
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _context_attention[side**2, q.shape[0] * q.shape[1]](
        q,
        k,
        v,
        output,
        indices,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        q.shape[1],
        qside,
        qside,
        side,
        side,
        indices.shape[1],
        q.shape[-1],
        max(16, triton.next_power_of_2(q.shape[-1])),
        bq,
        64,
        num_warps=4,
        num_stages=2,
    )
    return output


def forward(self, x, height, width):
    batch, queries, channels = x.shape
    assert (
        batch == 1
        and (not self.training)
        and (height == width == self.controlled_qside)
    )
    dim, bias = (channels // self.num_heads, self.in_proj_bias)
    q = F.linear(
        x, self.in_proj_weight[:channels], None if bias is None else bias[:channels]
    )
    q = q.reshape(batch, queries, self.num_heads, dim).permute(0, 2, 1, 3)
    reduced, kh, kw = (x, height, width)
    if self.sr is not None:
        reduced_map = self.sr(x.transpose(1, 2).reshape(batch, channels, height, width))
        kh, kw = reduced_map.shape[-2:]
        reduced = self.norm(reduced_map.flatten(2).transpose(1, 2))
    assert kh == kw == self.controlled_side
    assert self.controlled_indices.shape == (kh * kw, self.controlled_capacity)
    kv = F.linear(
        reduced,
        self.in_proj_weight[channels:],
        None if bias is None else bias[channels:],
    )
    kv = kv.reshape(batch, kh * kw, 2, self.num_heads, dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    output = context_attention(q, k, v, self.controlled_indices, height, kh)
    self.call_count += 1
    self.last_contexts = kh * kw
    return self.out_proj(output.transpose(1, 2).reshape(batch, queries, channels))


def install(core, indices, side, qside):
    core.controlled_capacity = indices.shape[1]
    core.controlled_indices = torch.as_tensor(indices, dtype=torch.int32, device="cuda")
    core.controlled_side = side
    core.controlled_qside = qside
    core.forward = MethodType(forward, core)
    core.call_count = 0
    core.cache.clear()
