"""Official SpargeAttention and its matched-pattern FP32 aggregation control."""

from __future__ import annotations
import math
from pathlib import Path
import sys


def block_budget(keys: int, capacity: int, block_size: int = 64):
    """Round target capacity to the official key-block granularity."""
    if keys < 1 or capacity < 1 or block_size < 1:
        raise ValueError("KV count, capacity and block size must be positive.")
    total = math.ceil(keys / block_size)
    selected = min(total, math.ceil(min(keys, capacity) / block_size))
    return {
        "key_tokens": keys,
        "target_capacity": min(keys, capacity),
        "key_block_size": block_size,
        "key_blocks": total,
        "selected_blocks": selected,
        "topk_ratio": selected / total,
        "block_capacity": min(keys, selected * block_size),
        "full_retention": capacity >= keys,
    }


def decode_deltas(deltas):
    """Decode the official difference-encoded block lookup table."""
    result, current = ([], 0)
    for delta in deltas:
        current += delta
        result.append(current)
    return result


_OFFICIAL = None


def official():
    global _OFFICIAL
    if _OFFICIAL is None:
        import importlib.util

        if importlib.util.find_spec("spas_sage_attn") is None:
            source = Path(__file__).resolve().parents[3] / "third_party/sources/sparge"
            if not source.is_dir():
                raise ImportError(
                    "Install the pinned SpargeAttention package or prepare its source cache."
                )
            sys.path.insert(0, str(source))
        import spas_sage_attn.core as core

        _OFFICIAL = core
    return _OFFICIAL


def prepare_qkv(q, k, v):
    import torch
    import torch.nn.functional as F

    dim = q.shape[-1]
    if dim not in (32, 64, 128):
        raise ValueError("Supported head dimensions are 32, 64 and 128.")
    target = max(64, dim)
    q16, k16, v16 = (x.to(torch.float16) for x in (q, k, v))
    if target != dim:
        q16, k16, v16 = (F.pad(x, (0, target - dim)) for x in (q16, k16, v16))
    return (q16.contiguous(), k16.contiguous(), v16.contiguous())


def select(q16, k16, ratio):
    return official().get_block_map_meansim_fuse_quant(
        q16,
        k16,
        k16.mean(dim=-2, keepdim=True),
        is_causal=False,
        BLKQ=128,
        BLKK=64,
        simthreshd1=-0.1,
        cdfthreshd=None,
        topk=ratio,
        return_lut=True,
        attention_sink=False,
    )


def selected_edges(lut, counts, queries, keys):
    """Count valid token interactions, excluding padded boundary positions."""
    import torch

    nq, nk = (counts.shape[-1], lut.shape[-1])
    qweights = (queries - torch.arange(nq, device=lut.device) * 128).clamp(0, 128)
    kweights = (keys - torch.arange(nk, device=lut.device) * 64).clamp(0, 64)
    valid = torch.arange(nk, device=lut.device) < counts[..., None]
    decoded = lut.cumsum(-1)
    safe_indices = torch.where(valid, decoded, 0).long().clamp(0, nk - 1)
    per_row = (kweights[safe_indices] * valid).sum(-1)
    return (per_row * qweights).sum(dtype=torch.float64)


def fp32_attention(q, k, v, lut, counts):
    import torch
    import triton
    from representative_context.kernels.sparge import sparse_block_attention

    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    sparse_block_attention[triton.cdiv(q.shape[-2], 32), q.shape[1], q.shape[0]](
        q,
        k,
        v,
        lut,
        counts,
        out,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        q.shape[-2],
        k.shape[-2],
        q.shape[1],
        lut.shape[-2],
        lut.shape[-1],
        DIM=q.shape[-1],
        SCALE=q.shape[-1] ** (-0.5),
        BM=32,
        BN=64,
        num_warps=4,
        num_stages=2,
    )
    return out


def attention(q, k, v, capacity, mode="sparge", record=None):
    """Execute the official quantized API or its selected blocks with FP32 aggregation."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    spec = block_budget(k.shape[-2], capacity)
    full_quant = mode == "quant_full"
    if spec["full_retention"] and (not full_quant):
        if record is not None:
            record(None, None, q.shape, k.shape, True)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    ratio = 1.0 if full_quant else spec["topk_ratio"]
    q16, k16, v16 = prepare_qkv(q, k, v)
    core = official()
    if mode == "sparge_fp32":
        lut, counts, *_ = select(q16, k16, ratio)
        if record is not None:
            record(lut, counts, q.shape, k.shape, False)
        return fp32_attention(q, k, v, lut, counts)
    if mode not in ("sparge", "quant_full"):
        raise ValueError("Unknown Sparge execution mode.")
    original = core.get_block_map_meansim_fuse_quant
    if record is not None:

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            record(result[0], result[1], q.shape, k.shape, False)
            return result

        core.get_block_map_meansim_fuse_quant = observed
    try:
        out = core.spas_sage2_attn_meansim_topk_cuda(
            q16,
            k16,
            v16,
            topk=ratio,
            is_causal=False,
            scale=q.shape[-1] ** (-0.5),
            simthreshd1=-0.1,
            cdfthreshd=None,
            pvthreshd=50,
            smooth_k=True,
            attention_sink=False,
        )
    finally:
        core.get_block_map_meansim_fuse_quant = original
    return out[..., : q.shape[-1]].to(q.dtype)
