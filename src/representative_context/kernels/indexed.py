"""Indexed grouped attention with online softmax and masked head channels."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _pack_kv(ranks, order, K: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.load(ranks + i, i < K, other=0)
    group = rank % G
    start = group * (K // G) + tl.minimum(group, K % G)
    tl.store(order + start + rank // G, i, i < K)


@triton.jit
def _cell_counts(
    order,
    counts,
    K: tl.constexpr,
    QH: tl.constexpr,
    QW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(order + i, i < K, other=0)
    y, x = (key // KW, key % KW)
    qh = tl.cdiv((y + 1) * QH, KH) - tl.cdiv(y * QH, KH)
    qw = tl.cdiv((x + 1) * QW, KW) - tl.cdiv(x * QW, KW)
    tl.store(counts + i, qh * qw, i < K)


@triton.jit
def _pack_irregular_q(
    order,
    prefix,
    q_order,
    QH: tl.constexpr,
    QW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cell = tl.program_id(0)
    key = tl.load(order + cell)
    y, x = (key // KW, key % KW)
    y0, x0 = (tl.cdiv(y * QH, KH), tl.cdiv(x * QW, KW))
    y1, x1 = (tl.cdiv((y + 1) * QH, KH), tl.cdiv((x + 1) * QW, KW))
    width = x1 - x0
    i = tl.arange(0, BLOCK)
    original = (y0 + i // width) * QW + x0 + i % width
    tl.store(q_order + tl.load(prefix + cell) + i, original, i < (y1 - y0) * width)


@triton.jit
def _indexed_attention(
    Q,
    K,
    V,
    O,
    ORDER,
    QORDER,
    PREFIX,
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
    KEYS: tl.constexpr,
    GROUPS: tl.constexpr,
    QH: tl.constexpr,
    QW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    REGULAR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    group = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = (batch_head // HEADS, batch_head % HEADS)
    k_start = group * (KEYS // GROUPS) + tl.minimum(group, KEYS % GROUPS)
    k_len = KEYS // GROUPS + (group < KEYS % GROUPS).to(tl.int32)
    q_pos = tl.program_id(2) * BQ + tl.arange(0, BQ)
    # Query membership follows nearest-neighbor alignment with the KV grid.
    if REGULAR:
        rh: tl.constexpr = QH // KH
        rw: tl.constexpr = QW // KW
        copies: tl.constexpr = rh * rw
        q_len = k_len * copies
        key_for_q = tl.load(ORDER + k_start + q_pos // copies, q_pos < q_len, other=0)
        local = q_pos % copies
        q_index = (key_for_q // KW * rh + local // rw) * QW
        q_index += key_for_q % KW * rw + local % rw
    else:
        q_start = tl.load(PREFIX + k_start)
        q_len = tl.load(PREFIX + k_start + k_len) - q_start
        q_index = tl.load(QORDER + q_start + q_pos, q_pos < q_len, other=0)
    if tl.program_id(2) * BQ >= q_len:
        return
    channel = tl.arange(0, BLOCK_D)
    valid_channel = channel < HEAD_DIM
    q = tl.load(
        Q + batch * sqb + head * sqh + q_index[:, None] * sqn + channel[None, :] * sqd,
        (q_pos < q_len)[:, None] & valid_channel[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BLOCK_D), tl.float32)
    # Online softmax rescales the accumulated output at each retained KV tile.
    for start in range(tl.cdiv(k_len, BK)):
        kp = start * BK + tl.arange(0, BK)
        ki = tl.load(ORDER + k_start + kp, kp < k_len, other=0)
        k = tl.load(
            K + batch * skb + head * skh + ki[:, None] * skn + channel[None, :] * skd,
            (kp < k_len)[:, None] & valid_channel[None, :],
            other=0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision=PRECISION) * HEAD_DIM ** (-0.5)
        scores = tl.where((kp < k_len)[None, :], scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        v = tl.load(
            V + batch * svb + head * svh + ki[:, None] * svn + channel[None, :] * svd,
            (kp < k_len)[:, None] & valid_channel[None, :],
            other=0,
        )
        acc = acc * correction[:, None]
        acc += tl.dot(probabilities.to(v.dtype), v, input_precision=PRECISION)
        maximum = new_max
    tl.store(
        O + batch * sob + head * soh + q_index[:, None] * son + channel[None, :] * sod,
        acc / denominator[:, None],
        (q_pos < q_len)[:, None] & valid_channel[None, :],
    )


@dataclass
class GroupLayout:
    query_shape: tuple[int, int]
    kv_shape: tuple[int, int]
    groups: int
    kv_order: torch.Tensor
    q_order: torch.Tensor | None = None
    q_prefix: torch.Tensor | None = None

    @property
    def index_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.kv_order, self.q_order, self.q_prefix)
            if t is not None
        )


def build_layout(
    ranks: torch.Tensor,
    query_shape: tuple[int, int],
    kv_shape: tuple[int, int],
    kv_budget: int,
) -> GroupLayout:
    """Pack KV indices by group; construct query indices for non-divisible grids."""
    qh, qw = query_shape
    kh, kw = kv_shape
    keys = kh * kw
    if ranks.ndim != 1 or ranks.numel() != keys or kv_budget < 1:
        raise ValueError(
            "Ranks must cover every KV token and capacity must be positive."
        )
    if qh < kh or qw < kw:
        raise ValueError(
            "The query grid must be at least as large as the KV grid on each axis."
        )
    groups = triton.cdiv(keys, kv_budget)
    order = torch.empty(keys, dtype=torch.int32, device=ranks.device)
    _pack_kv[triton.cdiv(keys, 256),](ranks, order, keys, groups, 256)
    layout = GroupLayout(query_shape, kv_shape, groups, order)
    if qh % kh or qw % kw:
        counts = torch.empty(keys, dtype=torch.int32, device=ranks.device)
        _cell_counts[triton.cdiv(keys, 256),](order, counts, keys, qh, qw, kh, kw, 256)
        prefix = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
        q_order = torch.empty(qh * qw, dtype=torch.int32, device=ranks.device)
        block = triton.next_power_of_2(triton.cdiv(qh, kh) * triton.cdiv(qw, kw))
        _pack_irregular_q[keys,](order, prefix, q_order, qh, qw, kh, kw, block)
        layout.q_prefix, layout.q_order = (prefix, q_order)
    return layout


def indexed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: GroupLayout,
    precision: str = "tf32x3",
    num_stages: int = 2,
) -> torch.Tensor:
    """Read original Q/K/V addresses and write outputs in original query order."""
    batch, heads, queries, dim = q.shape
    qh, qw = layout.query_shape
    kh, kw = layout.kv_shape
    if queries != qh * qw or k.shape[2] != kh * kw or (not 1 <= dim <= 128):
        raise ValueError(
            "Tensor shapes must match the layout and head dimensions must lie in 1 through 128."
        )
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    max_q_group = (
        triton.cdiv(kh * kw, layout.groups) * triton.cdiv(qh, kh) * triton.cdiv(qw, kw)
    )
    _indexed_attention[layout.groups, batch * heads, triton.cdiv(max_q_group, 64)](
        q,
        k,
        v,
        output,
        layout.kv_order,
        layout.q_order if layout.q_order is not None else layout.kv_order,
        layout.q_prefix if layout.q_prefix is not None else layout.kv_order,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        heads,
        kh * kw,
        layout.groups,
        qh,
        qw,
        kh,
        kw,
        layout.q_order is None,
        dim,
        max(16, triton.next_power_of_2(dim)),
        64,
        64,
        precision,
        num_warps=4,
        num_stages=num_stages,
    )
    return output


def reference_group_ids(
    ranks: torch.Tensor,
    query_shape: tuple[int, int],
    kv_shape: tuple[int, int],
    groups: int,
):
    """Return query and KV membership using nearest-neighbor grid alignment."""
    ids = ranks.remainder(groups)
    q_ids = (
        F.interpolate(
            ids.view(1, 1, *kv_shape).float(), size=query_shape, mode="nearest"
        )
        .flatten(1)
        .long()
    )
    return (q_ids, ids.view(1, -1))
