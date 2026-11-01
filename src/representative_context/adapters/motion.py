"""Reusable motion-attention weights and balanced grouped value aggregation."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from einops import rearrange
from representative_context.layouts.spatial import point_coverage_ranks


@dataclass
class SparseAttentionMap:
    weights: torch.Tensor
    order: torch.Tensor
    groups: int


class MotionAttention(nn.Module):

    def __init__(self, original: nn.Module, mode: str, budget: int, seed: int):
        super().__init__()
        if mode not in {"dense", "sparse"} or budget < 1:
            raise ValueError(
                "Choose dense or sparse attention with a positive capacity."
            )
        if original.args.position_only or original.args.position_and_content:
            raise ValueError(
                "This adapter requires the model's content-based attention configuration."
            )
        self.original = original
        self.mode = mode
        self.budget = budget
        self.seed = seed
        self.cache: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self.last_shape: tuple[int, int] | None = None
        self.last_groups: int | None = None

    def _order(
        self, height: int, width: int, groups: int, device: torch.device
    ) -> torch.Tensor:
        cache_key = (height, width, groups, str(device))
        if cache_key not in self.cache:
            count = height * width
            ranks = point_coverage_ranks(height, width, groups, 32, self.seed)
            order = sorted(
                range(count),
                key=lambda index: (ranks[index] % groups, ranks[index] // groups),
            )
            if count % groups:
                per_group = (count + groups - 1) // groups
                sizes = [
                    count // groups + (group < count % groups)
                    for group in range(groups)
                ]
                padded = []
                offset = 0
                for size in sizes:
                    padded.extend(order[offset : offset + size])
                    padded.extend([count] * (per_group - size))
                    offset += size
                order = padded
            self.cache[cache_key] = torch.tensor(order, device=device, dtype=torch.long)
        return self.cache[cache_key]

    def forward(self, fmap: torch.Tensor) -> torch.Tensor | SparseAttentionMap:
        original = self.original
        batch, _, height, width = fmap.shape
        count = height * width
        heads = original.heads
        query, key = original.to_qk(fmap).chunk(2, dim=1)
        query, key = [
            rearrange(value, "b (h d) y x -> b h (y x) d", h=heads)
            for value in (query, key)
        ]
        query = query * original.scale
        if self.mode == "dense":
            self.last_shape, self.last_groups = ((height, width), 1)
            return torch.matmul(query, key.transpose(-1, -2)).softmax(dim=-1)
        groups = (count + self.budget - 1) // self.budget
        if groups == 1:
            self.last_shape, self.last_groups = ((height, width), 1)
            return torch.matmul(query, key.transpose(-1, -2)).softmax(dim=-1)
        order = self._order(height, width, groups, fmap.device)
        per_group = (count + groups - 1) // groups
        if order.numel() != count:
            query = torch.cat(
                (query, query.new_zeros(batch, heads, 1, query.shape[-1])), dim=2
            )
            key = torch.cat((key, key.new_zeros(batch, heads, 1, key.shape[-1])), dim=2)
        query = query.index_select(2, order).reshape(
            batch, heads, groups, per_group, -1
        )
        key = key.index_select(2, order).reshape(batch, heads, groups, per_group, -1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if order.numel() != count:
            valid_keys = (order < count).reshape(groups, per_group)
            scores = scores.masked_fill(~valid_keys[None, None, :, None, :], -torch.inf)
        weights = scores.softmax(dim=-1)
        self.last_shape, self.last_groups = ((height, width), groups)
        return SparseAttentionMap(weights, order, groups)

    @property
    def dense(self):
        return self.mode == "dense"

    @dense.setter
    def dense(self, value):
        self.mode = "dense" if value else "sparse"


class MotionAggregation(nn.Module):

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original

    def forward(
        self, attention: torch.Tensor | SparseAttentionMap, fmap: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(attention, torch.Tensor):
            return self.original(attention, fmap)
        original = self.original
        batch, _, height, width = fmap.shape
        heads = original.heads
        count = height * width
        per_group = attention.weights.shape[-1]
        value = original.to_v(fmap)
        value = rearrange(value, "b (h d) y x -> b h (y x) d", h=heads)
        if attention.order.numel() != count:
            value = torch.cat(
                (value, value.new_zeros(batch, heads, 1, value.shape[-1])), dim=2
            )
        value = value.index_select(2, attention.order).reshape(
            batch, heads, attention.groups, per_group, -1
        )
        ordered_output = torch.matmul(attention.weights, value)
        ordered_output = ordered_output.reshape(
            batch, heads, attention.order.numel(), -1
        )
        output = ordered_output.new_empty(batch, heads, count, ordered_output.shape[-1])
        if attention.order.numel() == count:
            output.index_copy_(2, attention.order, ordered_output)
        else:
            valid_queries = attention.order < count
            output.index_copy_(
                2,
                attention.order[valid_queries],
                ordered_output[:, :, valid_queries, :],
            )
        output = rearrange(output, "b h (y x) d -> b (h d) y x", y=height, x=width)
        if original.project is not None:
            output = original.project(output)
        return fmap + original.gamma * output


def install_motion_attention(
    model: nn.Module, mode: str = "dense", budget: int = 2048, seed: int = 42
) -> MotionAttention:
    if isinstance(model.att, MotionAttention):
        raise ValueError("Motion attention is already installed.")
    probe = MotionAttention(model.att, mode, budget, seed)
    model.att = probe
    model.update_block.aggregator = MotionAggregation(model.update_block.aggregator)
    return probe
