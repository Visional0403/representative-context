"""Paired routing cost and spatial coverage diagnostics."""

from __future__ import annotations
import math
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from representative_context.controls.regions import _regionize, _unregionize
from representative_context.controls.topk import selected_attention, triton_topk


class StageTimer:
    """Keep CUDA stage timings separate from uninstrumented forward measurements."""

    def __init__(self, enabled=False):
        self.enabled, self.events = (enabled, [])

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        yield
        b.record()
        self.events.append((name, a, b))

    def result(self):
        torch.cuda.synchronize()
        result = {}
        for name, a, b in self.events:
            result[name] = result.get(name, 0.0) + a.elapsed_time(b)
        return result


def dense_attention(q, k, v):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


def qkv(core, x, height, width):
    """Reuse attention projections and spatial reduction for routing diagnostics."""
    batch, queries, channels = x.shape
    dim, bias = (channels // core.num_heads, core.in_proj_bias)
    q = F.linear(
        x, core.in_proj_weight[:channels], None if bias is None else bias[:channels]
    )
    q = q.reshape(batch, queries, core.num_heads, dim).permute(0, 2, 1, 3)
    reduced, kh, kw = (x, height, width)
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
    return (q, *kv.unbind(0), kh, kw)


def projection(dim, seed, device):
    gen = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dim, min(8, dim), generator=gen)
    return (torch.linalg.qr(matrix).Q * math.sqrt(dim / min(8, dim))).to(device)


def routed_topk(
    q, k, v, keep, proxy=None, score_elements=4 * 1024**2, selector="torch", timer=None
):
    """Measure the same chunking, selector and fused selected-attention operations."""
    timer = timer or StageTimer()
    with timer.stage("proxy_projection"):
        rq, rk = (q, k) if proxy is None else (q @ proxy, k @ proxy)
    batch, heads, queries, dim = q.shape
    chunk = min(8192, max(1, score_elements // (batch * heads * k.shape[-2])))
    with timer.stage("output_allocation"):
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
    for start in range(0, queries, chunk):
        end = min(start + chunk, queries)
        with timer.stage("pairing_scores"):
            scores = rq[:, :, start:end] @ rk.transpose(-2, -1)
            if proxy is None:
                scores.mul_(dim ** (-0.5))
        with timer.stage("topk_and_indices"):
            values, indices = (
                triton_topk(scores, keep)
                if selector == "triton"
                else torch.topk(scores, keep, dim=-1, sorted=False)
            )
        with timer.stage("selected_attention"):
            selected_attention(
                q,
                k,
                v,
                indices,
                values if proxy is None else None,
                start,
                output,
                4,
                64,
                4,
            )
    return output


def region_prepare(q, k, v, q_hw, k_hw):
    rows, cols = [math.ceil(side / 4) for side in k_hw]
    qr, qvalid, qth, qtw = _regionize(q, *q_hw, rows, cols)
    kr, kvalid, kth, ktw = _regionize(k, *k_hw, rows, cols)
    vr, _, _, _ = _regionize(v, *k_hw, rows, cols)
    assert q_hw[0] % rows == q_hw[1] % cols == k_hw[0] % rows == k_hw[1] % cols == 0
    batch, heads, count, _, dim = qr.shape
    qm = (qr * qvalid[None, None, :, :, None]).sum(3)
    qm = qm / qvalid.sum(-1)[None, None, :, None]
    km = (kr * kvalid[None, None, :, :, None]).sum(3)
    km = km / kvalid.sum(-1)[None, None, :, None]
    qm = qm.permute(0, 2, 1, 3).reshape(batch, count, heads * dim)
    km = km.permute(0, 1, 3, 2).reshape(batch, heads * dim, count)
    return (qr, kr, vr, qm, km, (rows, cols, qth, qtw, kth, ktw))


def region_attention(q, k, v, q_hw, k_hw, keep, optimized=True, timer=None):
    """Select pooled regions and execute their chosen token interactions with SDPA."""
    timer = timer or StageTimer()
    with timer.stage("region_organization_and_pooling"):
        qr, kr, vr, qm, km, geometry = region_prepare(q, k, v, q_hw, k_hw)
    rows, cols, qth, qtw, kth, ktw = geometry
    batch, heads, count, qt, dim = qr.shape
    kt = kth * ktw
    selected = min(count, math.ceil(keep / kt))
    assert selected * kt == keep
    with timer.stage("pairing_scores"):
        scores = qm @ km
    with timer.stage("topk_and_indices"):
        routes = scores.topk(selected, dim=-1, sorted=False).indices
    with timer.stage("candidate_kv_copy"):
        idx = (
            routes[:, None]
            .expand(batch, heads, count, selected)
            .reshape(batch * heads, count, selected)
        )
        bh = torch.arange(batch * heads, device=q.device)[:, None, None]
        sk = kr.reshape(batch * heads, count, kt, dim)[bh, idx].reshape(
            batch, heads, count, keep, dim
        )
        sv = vr.reshape(batch * heads, count, kt, dim)[bh, idx].reshape(
            batch, heads, count, keep, dim
        )
    with timer.stage("selected_attention"):
        if optimized:
            out = dense_attention(
                qr.reshape(batch * heads * count, 1, qt, dim),
                sk.reshape(batch * heads * count, 1, keep, dim),
                sv.reshape(batch * heads * count, 1, keep, dim),
            )
            out = out.reshape(batch, heads, count, qt, dim)
        else:
            logits = qr @ sk.transpose(-2, -1) * dim ** (-0.5)
            out = logits.softmax(-1) @ sv
    with timer.stage("restore_query_order"):
        return _unregionize(out, *q_hw, rows, cols, qth, qtw)
