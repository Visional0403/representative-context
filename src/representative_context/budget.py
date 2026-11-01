"""Exact integer capacity and balanced group counts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityPlan:
    query_stride: int
    sr_ratio: int
    kv_tokens: int
    k: int
    capacity_limit: int
    groups: int
    min_group_size: int
    max_group_size: int


def plan_capacity(
    query_stride: int, sr_ratio: int, kv_tokens: int, k: int = 16
) -> CapacityPlan:
    """Set B = max(1, ceil(2**k / (s*r*r))) and G = ceil(K/B).

    B bounds the number of KV tokens in a group. Actual balanced groups
    contain floor(K/G) or ceil(K/G) tokens. No resolution rounding is
    applied before counting K.
    """
    for name, value in (
        ("query_stride", query_stride),
        ("sr_ratio", sr_ratio),
        ("kv_tokens", kv_tokens),
        ("k", k),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < (0 if name == "k" else 1):
            raise ValueError(f"{name} is out of range.")
    denominator = query_stride * sr_ratio * sr_ratio
    capacity = max(1, ((1 << k) + denominator - 1) // denominator)
    groups = (kv_tokens + capacity - 1) // capacity
    return CapacityPlan(
        query_stride,
        sr_ratio,
        kv_tokens,
        k,
        capacity,
        groups,
        kv_tokens // groups,
        (kv_tokens + groups - 1) // groups,
    )
