"""Pooled-region top-k selection control."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl


def _regionize(
    x: torch.Tensor, height: int, width: int, region_rows: int, region_cols: int
):
    """Partition spatial tokens and return a mask for padded positions."""
    batch, heads, tokens, dim = x.shape
    if tokens != height * width:
        raise ValueError("Token count does not match the spatial shape.")
    tile_h = math.ceil(height / region_rows)
    tile_w = math.ceil(width / region_cols)
    padded_h, padded_w = (region_rows * tile_h, region_cols * tile_w)
    image = x.transpose(-2, -1).reshape(batch, heads, dim, height, width)
    image = F.pad(image, (0, padded_w - width, 0, padded_h - height))
    regions = (
        image.reshape(batch, heads, dim, region_rows, tile_h, region_cols, tile_w)
        .permute(0, 1, 3, 5, 4, 6, 2)
        .reshape(batch, heads, region_rows * region_cols, tile_h * tile_w, dim)
    )
    valid = torch.ones((1, 1, 1, height, width), device=x.device, dtype=torch.bool)
    valid = F.pad(valid, (0, padded_w - width, 0, padded_h - height))
    valid = (
        valid.reshape(1, 1, 1, region_rows, tile_h, region_cols, tile_w)
        .permute(0, 1, 3, 5, 4, 6, 2)
        .reshape(region_rows * region_cols, tile_h * tile_w)
    )
    return (regions, valid, tile_h, tile_w)


def _unregionize(
    x: torch.Tensor,
    height: int,
    width: int,
    region_rows: int,
    region_cols: int,
    tile_h: int,
    tile_w: int,
) -> torch.Tensor:
    """Restore tokens to their original spatial order."""
    batch, heads, _, _, dim = x.shape
    image = (
        x.reshape(batch, heads, region_rows, region_cols, tile_h, tile_w, dim)
        .permute(0, 1, 6, 2, 4, 3, 5)
        .reshape(batch, heads, dim, region_rows * tile_h, region_cols * tile_w)
    )
    return image[:, :, :, :height, :width].flatten(3).transpose(-2, -1)


def bilevel_region_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    kv_budget: int,
    kv_region_side: int = 4,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple:
    """Route pooled regions with shared-head top-k, then attend to their KV tokens."""
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Q/K/V must have shape [batch, heads, tokens, channels].")
    if q.shape[-1] != k.shape[-1] or q.shape[-2] != math.prod(q_hw):
        raise ValueError("Query shape does not match its spatial grid.")
    if k.shape[-2] != math.prod(k_hw) or kv_budget < 1 or kv_region_side < 1:
        raise ValueError("KV shape or region parameters are invalid.")
    if kv_budget >= k.shape[-2]:
        if q.device.type == "cuda":
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if q.dtype == torch.float32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v)
        return F.scaled_dot_product_attention(q, k, v)
    region_rows = math.ceil(k_hw[0] / kv_region_side)
    region_cols = math.ceil(k_hw[1] / kv_region_side)
    q_regions, q_valid, q_tile_h, q_tile_w = _regionize(
        q, *q_hw, region_rows, region_cols
    )
    k_regions, k_valid, k_tile_h, k_tile_w = _regionize(
        k, *k_hw, region_rows, region_cols
    )
    v_regions, _, _, _ = _regionize(v, *k_hw, region_rows, region_cols)
    batch, heads, region_count, q_tile_tokens, dim = q_regions.shape
    kv_tile_tokens = k_tile_h * k_tile_w
    selected_region_count = min(
        region_count, max(1, math.ceil(kv_budget / kv_tile_tokens))
    )
    q_mean = (q_regions * q_valid[None, None, :, :, None]).sum(3)
    q_mean = q_mean / q_valid.sum(-1)[None, None, :, None]
    k_mean = (k_regions * k_valid[None, None, :, :, None]).sum(3)
    k_mean = k_mean / k_valid.sum(-1)[None, None, :, None]
    q_mean = q_mean.permute(0, 2, 1, 3).reshape(batch, region_count, heads * dim)
    k_mean = k_mean.permute(0, 1, 3, 2).reshape(batch, heads * dim, region_count)
    routes = (q_mean @ k_mean).topk(selected_region_count, dim=-1, sorted=False).indices
    head_routes = routes[:, None].expand(
        batch, heads, region_count, selected_region_count
    )
    bh = batch * heads
    flat_routes = head_routes.reshape(bh, region_count, selected_region_count)
    batch_index = torch.arange(bh, device=q.device)[:, None, None]
    flat_k = k_regions.reshape(bh, region_count, kv_tile_tokens, dim)
    flat_v = v_regions.reshape(bh, region_count, kv_tile_tokens, dim)
    selected_k = flat_k[batch_index, flat_routes].reshape(
        batch, heads, region_count, selected_region_count * kv_tile_tokens, dim
    )
    selected_v = flat_v[batch_index, flat_routes].reshape(
        batch, heads, region_count, selected_region_count * kv_tile_tokens, dim
    )
    selected_valid = k_valid[routes].reshape(
        batch, region_count, selected_region_count * kv_tile_tokens
    )
    logits = torch.matmul(q_regions, selected_k.transpose(-2, -1)) * dim ** (-0.5)
    logits = logits.masked_fill(
        ~selected_valid[:, None, :, None, :], torch.finfo(logits.dtype).min
    )
    weights = torch.softmax(logits, dim=-1)
    output_regions = weights @ selected_v
    output = _unregionize(
        output_regions, *q_hw, region_rows, region_cols, q_tile_h, q_tile_w
    )
    if return_valid_counts:
        return (output, selected_valid.sum(-1), q_valid.sum(-1))
    return output
