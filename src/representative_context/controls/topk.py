"""Optimized exact-score and fixed-projection top-k attention."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl


@triton.jit
def _topk_rows(
    S,
    T,
    I,
    KEYS: tl.constexpr,
    KEEP: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
    BLOCK_KEEP: tl.constexpr,
):
    row = tl.program_id(0)
    key = tl.arange(0, BLOCK_KEYS)
    score = tl.load(S + row * KEYS + key, key < KEYS, other=-float("inf"))
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 2147483648 != 0, ~bits, bits ^ 2147483648)
    packed = ordered.to(tl.uint64) << 32 | key.to(tl.uint64)
    selected = tl.topk(packed, BLOCK_KEEP)
    selected_order = (selected >> 32).to(tl.uint32)
    selected_bits = tl.where(
        selected_order & 2147483648 != 0, selected_order ^ 2147483648, ~selected_order
    )
    values = selected_bits.to(tl.float32, bitcast=True)
    indices = (selected & 4294967295).to(tl.int32)
    pos = tl.arange(0, BLOCK_KEEP)
    tl.store(T + row * KEEP + pos, values, pos < KEEP)
    tl.store(I + row * KEEP + pos, indices, pos < KEEP)


def triton_topk(scores, keep):
    """Select FP32 scores with index-based tie ordering."""
    if scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError("Sorting requires contiguous FP32 scores.")
    keys = scores.shape[-1]
    shape = (*scores.shape[:-1], keep)
    values = torch.empty(shape, device=scores.device, dtype=scores.dtype)
    indices = torch.empty(shape, device=scores.device, dtype=torch.int64)
    _topk_rows[scores.numel() // keys,](
        scores,
        values,
        indices,
        KEYS=keys,
        KEEP=keep,
        BLOCK_KEYS=triton.next_power_of_2(keys),
        BLOCK_KEEP=triton.next_power_of_2(keep),
        num_warps=8 if keys > 2048 else 4,
    )
    return (values, indices)


@triton.jit
def _selected_attention(
    Q,
    K,
    V,
    S,
    I,
    O,
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
    ssb,
    ssh,
    ssn,
    ssk,
    sib,
    sih,
    sin,
    sik,
    sob,
    soh,
    son,
    sod,
    start,
    HEADS: tl.constexpr,
    COUNT: tl.constexpr,
    KEEP: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    EXACT: tl.constexpr,
):
    row = tl.program_id(0) * BQ + tl.arange(0, BQ)
    bh = tl.program_id(1)
    batch, head = (bh // HEADS, bh % HEADS)
    channel = tl.arange(0, BD)
    qvalid = row < COUNT
    q = tl.load(
        Q
        + batch * sqb
        + head * sqh
        + (row + start)[:, None] * sqn
        + channel[None, :] * sqd,
        qvalid[:, None] & (channel < DIM)[None, :],
        other=0,
    )
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BD), tl.float32)
    for block in range(tl.cdiv(KEEP, BK)):
        pos = block * BK + tl.arange(0, BK)
        mask = qvalid[:, None] & (pos < KEEP)[None, :]
        key = tl.load(
            I + batch * sib + head * sih + row[:, None] * sin + pos[None, :] * sik,
            mask,
            other=0,
        ).to(tl.int32)
        if EXACT:
            scores = tl.load(
                S + batch * ssb + head * ssh + row[:, None] * ssn + pos[None, :] * ssk,
                mask,
                other=0,
            ).to(tl.float32)
        else:
            selected_k = tl.load(
                K
                + batch * skb
                + head * skh
                + key[:, :, None] * skn
                + channel[None, None, :] * skd,
                mask[:, :, None] & (channel < DIM)[None, None, :],
                other=0,
            )
            scores = tl.sum(selected_k.to(tl.float32) * q[:, None, :], 2)
            scores *= DIM ** (-0.5)
        scores = tl.where((pos < KEEP)[None, :], scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        selected_v = tl.load(
            V
            + batch * svb
            + head * svh
            + key[:, :, None] * svn
            + channel[None, None, :] * svd,
            mask[:, :, None] & (channel < DIM)[None, None, :],
            other=0,
        )
        acc = acc * correction[:, None]
        acc += tl.sum(probabilities[:, :, None] * selected_v.to(tl.float32), 1)
        maximum = new_max
    tl.store(
        O
        + batch * sob
        + head * soh
        + (row + start)[:, None] * son
        + channel[None, :] * sod,
        acc / denominator[:, None],
        qvalid[:, None] & (channel < DIM)[None, :],
    )


def selected_attention(
    q,
    k,
    v,
    indices,
    scores=None,
    start=0,
    output=None,
    block_q=4,
    block_k=64,
    num_warps=4,
):
    """Execute attention on selected indices, optionally reusing exact scaled scores."""
    batch, heads, _, dim = q.shape
    count, keep = indices.shape[-2:]
    if output is None:
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
    score_tensor = indices if scores is None else scores
    _selected_attention[triton.cdiv(count, block_q), batch * heads](
        q,
        k,
        v,
        score_tensor,
        indices,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *score_tensor.stride(),
        *indices.stride(),
        *output.stride(),
        start,
        HEADS=heads,
        COUNT=count,
        KEEP=keep,
        DIM=dim,
        BD=triton.next_power_of_2(dim),
        BQ=block_q,
        BK=block_k,
        EXACT=scores is not None,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return output


def topk_attention_optimized(
    q,
    k,
    v,
    keep,
    route_q=None,
    route_k=None,
    max_gather_elements=4 * 1024 * 1024,
    score_elements=16 * 1024 * 1024,
    block_q=4,
    block_k=64,
    num_warps=4,
    selector="torch",
):
    """Chunk QK scoring and fuse selected KV reads with softmax and AV."""
    if q.device.type != "cuda" or torch.is_grad_enabled():
        raise ValueError(
            "Selected attention requires CUDA inference with gradients disabled."
        )
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Q/K/V must be matching four-dimensional tensors.")
    if q.shape[-1] != k.shape[-1] or not 1 <= keep <= k.shape[-2]:
        raise ValueError("Invalid head dimension or retained edge count.")
    if (route_q is None) != (route_k is None):
        raise ValueError("Provide both proxy Q and proxy K.")
    if keep == k.shape[-2]:
        backend = (
            SDPBackend.EFFICIENT_ATTENTION
            if q.dtype == torch.float32
            else SDPBackend.FLASH_ATTENTION
        )
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v)
    batch, heads, queries, dim = q.shape
    chunk = min(8192, max(1, score_elements // (batch * heads * k.shape[-2])))
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    exact = route_q is None
    scoring_q, scoring_k = (q, k) if exact else (route_q, route_k)
    kt = scoring_k.transpose(-2, -1)
    for start in range(0, queries, chunk):
        end = min(start + chunk, queries)
        scores = torch.matmul(scoring_q[:, :, start:end], kt)
        if exact:
            scores.mul_(dim ** (-0.5))
        if selector == "triton":
            top_scores, indices = triton_topk(scores, keep)
        elif selector == "torch":
            top_scores, indices = torch.topk(scores, keep, dim=-1, sorted=False)
        else:
            raise ValueError(selector)
        selected_attention(
            q,
            k,
            v,
            indices,
            top_scores if exact else None,
            start,
            output,
            block_q,
            block_k,
            num_warps,
        )
    return output
