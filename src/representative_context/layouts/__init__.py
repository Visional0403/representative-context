"""Deterministic layouts and fixed balanced random controls."""

import random

from .grid import grid_ranks
from .spatial import _balanced_ranks, _morton, coverage_ranks, point_coverage_ranks


def dispersed_ranks(
    height: int, width: int, groups: int, regions: int = 32, seed: int = 42
) -> list[int]:
    """Disperse balanced groups over the image, including low-capacity grids."""
    if not 1 <= groups <= height * width:
        raise ValueError("Groups must lie between one and the KV token count.")
    try:
        return point_coverage_ranks(height, width, groups, regions, seed)
    except ValueError as error:
        if "Insufficient" not in str(error) or regions != 32:
            raise
    members = [[] for _ in range(groups)]
    for ry in range(4):
        for rx in range(8):
            region = ry * 8 + rx
            cells = sorted(
                (
                    (y, x)
                    for y in range(ry * height // 4, (ry + 1) * height // 4)
                    for x in range(rx * width // 8, (rx + 1) * width // 8)
                ),
                key=lambda cell: _morton(
                    cell[0] - ry * height // 4, cell[1] - rx * width // 8
                ),
            )
            for index, (y, x) in enumerate(cells):
                members[(seed + 17 * region + index) % groups].append(y * width + x)
    return _balanced_ranks(members, height * width)


def make_ranks(
    height: int, width: int, capacity: int, pattern: str = "dispersed32", seed: int = 42
) -> list[int]:
    """Return one rank per KV token; ranks modulo G encode group membership."""
    if min(height, width, capacity) < 1:
        raise ValueError("Grid dimensions and capacity must be positive.")
    keys = height * width
    groups = (keys + capacity - 1) // capacity
    if pattern.startswith("extent"):
        from .extent import extent_ranks

        return extent_ranks(height, width, groups, int(pattern[6:]), seed)
    if pattern == "balanced_random":
        ranks = list(range(keys))
        random.Random(seed).shuffle(ranks)
        return ranks
    if pattern == "grid":
        return grid_ranks(height, width, capacity)
    if pattern in ("dispersed16", "dispersed32", "dispersed64"):
        return dispersed_ranks(
            height, width, groups, int(pattern.removeprefix("dispersed")), seed
        )
    if pattern in ("clustered1", "clustered4", "clustered16"):
        return coverage_ranks(
            height, width, groups, int(pattern.removeprefix("clustered")), seed
        )
    raise ValueError(f"Unknown layout: {pattern}")
