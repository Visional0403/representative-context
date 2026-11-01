"""Explicit top-k reference for paired numerical checks."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl


def topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: int,
    route_q: torch.Tensor | None = None,
    route_k: torch.Tensor | None = None,
    max_gather_elements: int = 4 * 1024 * 1024,
) -> torch.Tensor:
    """Explicit gather reference for exact or low-dimensional proxy top-k selection."""
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Q/K/V must have matching batch, head and channel dimensions.")
    if q.shape[-1] != k.shape[-1] or not 1 <= keep <= k.shape[-2]:
        raise ValueError("Invalid head dimension or top-k count.")
    if (route_q is None) != (route_k is None):
        raise ValueError("Provide both proxy Q and proxy K.")
    if route_q is not None and (
        route_q.shape[:-1] != q.shape[:-1]
        or route_k.shape[:-1] != k.shape[:-1]
        or route_q.shape[-1] != route_k.shape[-1]
    ):
        raise ValueError("Proxy Q/K dimensions do not match the attention inputs.")
    if keep == k.shape[-2]:
        if q.device.type == "cuda":
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if q.dtype == torch.float32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v)
        return F.scaled_dot_product_attention(q, k, v)
    batch, heads, queries, dim = q.shape
    chunk = min(1024, max(16, max_gather_elements // (batch * heads * keep * dim)))
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    exact = route_q is None
    scoring_q = q if exact else route_q
    scoring_k = k if exact else route_k
    scoring_kt = scoring_k.transpose(-2, -1)
    scale = dim ** (-0.5)
    for start in range(0, queries, chunk):
        end = min(start + chunk, queries)
        scores = torch.matmul(scoring_q[:, :, start:end], scoring_kt)
        if exact:
            scores = scores * scale
        top_scores, indices = torch.topk(scores, keep, dim=-1, sorted=False)
        if exact:
            attention_scores = top_scores
        else:
            selected_k = torch.gather(
                k.unsqueeze(2).expand(-1, -1, end - start, -1, -1),
                3,
                indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim),
            )
            attention_scores = (
                torch.matmul(selected_k, q[:, :, start:end].unsqueeze(-1)).squeeze(-1)
                * scale
            )
        selected_v = torch.gather(
            v.unsqueeze(2).expand(-1, -1, end - start, -1, -1),
            3,
            indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim),
        )
        probabilities = torch.softmax(attention_scores, dim=-1)
        output[:, :, start:end] = (probabilities.unsqueeze(-1) * selected_v).sum(dim=-2)
    return output


def exact_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: int,
    max_gather_elements: int = 4 * 1024 * 1024,
) -> torch.Tensor:
    return topk_attention(q, k, v, keep, max_gather_elements=max_gather_elements)
