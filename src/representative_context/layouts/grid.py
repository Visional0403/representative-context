"""Equal-offset Grid layouts for matched-capacity comparisons."""

from __future__ import annotations
import math
from functools import lru_cache


@lru_cache(maxsize=128)
def grid_spec(height: int, width: int, budget: int) -> dict:
    """Choose a divisible rectangular Grid with a near-square sampling grid."""
    if min(height, width, budget) < 1:
        raise ValueError("Grid dimensions and capacity must be positive integers.")
    keys = height * width
    groups = math.ceil(keys / budget)
    if keys % groups:
        raise ValueError(
            "Grid layouts require equal group capacities and divisible grid dimensions."
        )
    capacity = keys // groups
    candidates = [
        (rows, capacity // rows)
        for rows in range(1, height + 1)
        if capacity % rows == 0
        and height % rows == 0
        and (width % (capacity // rows) == 0)
    ]
    if not candidates:
        raise ValueError(
            f"Cannot construct a rectangular Grid layout: {height}×{width}, capacity {capacity}"
        )
    rows, cols = min(
        candidates,
        key=lambda shape: (
            abs(math.log(shape[0] / shape[1])),
            shape[0] > shape[1],
            shape[0],
        ),
    )
    return {
        "kv_shape": [height, width],
        "capacity_limit": budget,
        "group_capacity": capacity,
        "groups": groups,
        "region_grid": [rows, cols],
        "region_shape": [height // rows, width // cols],
        "square_grid": rows == cols,
    }


def grid_ranks(height: int, width: int, budget: int) -> list[int]:
    """Encode equal-offset connections across a rectangular spatial grid."""
    spec = grid_spec(height, width, budget)
    rows, cols = spec["region_grid"]
    local_h, local_w = spec["region_shape"]
    groups = spec["groups"]
    ranks = [-1] * (height * width)
    for y in range(height):
        for x in range(width):
            group = y % local_h * local_w + x % local_w
            position = y // local_h * cols + x // local_w
            ranks[y * width + x] = group + groups * position
    if sorted(ranks) != list(range(height * width)):
        raise AssertionError("Grid must generate a complete permutation.")
    return ranks


def official_grid_groups(height: int, width: int, side: int) -> list[list[int]]:
    """Expand the square Grid permutation as an independent CPU reference."""
    if height % side or width % side:
        raise ValueError(
            "Square Grid requires both spatial dimensions to be divisible by the grid side."
        )
    h, w = (height // side, width // side)
    return [
        [(a * h + u) * width + b * w + v for a in range(side) for b in range(side)]
        for u in range(h)
        for v in range(w)
    ]
