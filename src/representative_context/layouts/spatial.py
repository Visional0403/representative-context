"""Balanced spatial sampling layouts for grouped attention."""

from __future__ import annotations
import math

POINT_REGION_GRIDS = {16: (4, 4), 32: (4, 8), 64: (8, 8)}


def _morton(row: int, col: int) -> int:
    """Interleave column and row bits to order spatial positions along a Z curve."""
    code = 0
    bit = 0
    while row or col:
        code |= (col & 1) << (2 * bit)
        code |= (row & 1) << (2 * bit + 1)
        row >>= 1
        col >>= 1
        bit += 1
    return code


def _balanced_ranks(members: list[list[int]], keys: int) -> list[int]:
    """Encode balanced membership as ranks: group = rank % groups."""
    groups = len(members)
    quotas = [keys // groups + (group < keys % groups) for group in range(groups)]
    overflow = []
    for group, values in enumerate(members):
        while len(values) > quotas[group]:
            overflow.append(values.pop())
    for group, values in enumerate(members):
        while len(values) < quotas[group]:
            values.append(overflow.pop())
    if overflow:
        raise RuntimeError("Failed to balance group capacities.")
    ranks = [-1] * keys
    for group, values in enumerate(members):
        for position, token in enumerate(values):
            ranks[token] = group + groups * position
    if sorted(ranks) != list(range(keys)):
        raise RuntimeError("The rank permutation must cover every KV token.")
    return ranks


def coverage_ranks(
    kv_height: int, kv_width: int, groups: int, regions: int, seed: int
) -> list[int]:
    """Assign spatially contiguous 2 by 2 tiles across the specified region grid."""
    side = {1: 1, 4: 2, 16: 4}.get(regions)
    if side is None or groups < 1 or kv_height < 1 or (kv_width < 1):
        raise ValueError(
            "Regions must be 1, 4, or 16; grid dimensions and groups must be positive."
        )
    if kv_height % (2 * side) or kv_width % (2 * side):
        raise ValueError(
            "Clustered layouts require equal 2 by 2 tiles and equal spatial regions."
        )
    tile_height, tile_width = (kv_height // 2, kv_width // 2)
    region_height, region_width = (tile_height // side, tile_width // side)
    partial_coverage = region_height * region_width < groups
    members: list[list[int]] = [[] for _ in range(groups)]
    for region_row in range(side):
        for region_col in range(side):
            region = region_row * side + region_col
            tiles = sorted(
                (
                    (row, col)
                    for row in range(region_height)
                    for col in range(region_width)
                ),
                key=lambda tile: _morton(*tile),
            )
            if partial_coverage:
                for position, (local_row, local_col) in enumerate(tiles):
                    target = (seed + region * len(tiles) + position) % groups
                    row = 2 * (region_row * region_height + local_row)
                    col = 2 * (region_col * region_width + local_col)
                    members[target].extend(
                        (
                            row * kv_width + col,
                            row * kv_width + col + 1,
                            (row + 1) * kv_width + col,
                            (row + 1) * kv_width + col + 1,
                        )
                    )
                continue
            common, remainder = divmod(len(tiles), groups)
            rotation = region * (seed % groups + 1) % groups
            cursor = 0
            for segment in range(groups):
                count = common + (segment < remainder)
                target = (segment + rotation) % groups
                for local_row, local_col in tiles[cursor : cursor + count]:
                    row = 2 * (region_row * region_height + local_row)
                    col = 2 * (region_col * region_width + local_col)
                    members[target].extend(
                        (
                            row * kv_width + col,
                            row * kv_width + col + 1,
                            (row + 1) * kv_width + col,
                            (row + 1) * kv_width + col + 1,
                        )
                    )
                cursor += count
    return _balanced_ranks(members, kv_height * kv_width)


def point_coverage_ranks(
    kv_height: int, kv_width: int, groups: int, regions: int, seed: int
) -> list[int]:
    """Assign individual KV positions across 16, 32, or 64 sampling regions."""
    grid = POINT_REGION_GRIDS.get(regions)
    if grid is None or groups < 1 or kv_height < 1 or (kv_width < 1):
        raise ValueError(
            "Point sampling requires 16, 32, or 64 regions and positive grid dimensions and groups."
        )
    region_rows, region_cols = grid
    if kv_height % region_rows or kv_width % region_cols:
        return _point_coverage_ranks_uneven(
            kv_height, kv_width, groups, region_rows, region_cols, seed
        )
    region_height = kv_height // region_rows
    region_width = kv_width // region_cols
    cells_per_region = region_height * region_width
    if cells_per_region < groups:
        members: list[list[int]] = [[] for _ in range(groups)]
        for region_row in range(region_rows):
            for region_col in range(region_cols):
                region = region_row * region_cols + region_col
                cells = sorted(
                    (
                        (row, col)
                        for row in range(region_height)
                        for col in range(region_width)
                    ),
                    key=lambda cell: _morton(*cell),
                )
                for position, (local_row, local_col) in enumerate(cells):
                    target = (seed + region * len(cells) + position) % groups
                    row = region_row * region_height + local_row
                    col = region_col * region_width + local_col
                    members[target].append(row * kv_width + col)
        return _balanced_ranks(members, kv_height * kv_width)
    if cells_per_region % groups:
        return _point_coverage_ranks_uneven(
            kv_height, kv_width, groups, region_rows, region_cols, seed
        )
    members: list[list[int]] = [[] for _ in range(groups)]
    for region_row in range(region_rows):
        for region_col in range(region_cols):
            region = region_row * region_cols + region_col
            cells = sorted(
                (
                    (row, col)
                    for row in range(region_height)
                    for col in range(region_width)
                ),
                key=lambda cell: _morton(*cell),
            )
            step = (2 * (seed + 11 * region) + 1) % cells_per_region
            while math.gcd(step, cells_per_region) != 1:
                step = (step + 2) % cells_per_region
            shift = (17 * seed + 31 * region) % cells_per_region
            for position, (local_row, local_col) in enumerate(cells):
                group = (step * position + shift) % groups
                row = region_row * region_height + local_row
                col = region_col * region_width + local_col
                members[group].append(row * kv_width + col)
    return _balanced_ranks(members, kv_height * kv_width)


def _point_coverage_ranks_uneven(
    kv_height: int,
    kv_width: int,
    groups: int,
    region_rows: int,
    region_cols: int,
    seed: int,
) -> list[int]:
    """Distribute KV positions on non-divisible grids under exact global quotas."""
    keys = kv_height * kv_width
    quotas = [keys // groups + (group < keys % groups) for group in range(groups)]
    if min(quotas) < region_rows * region_cols:
        raise ValueError("Insufficient group capacity to cover every sampling region.")
    regions: list[list[int]] = []
    for region_row in range(region_rows):
        y0 = region_row * kv_height // region_rows
        y1 = (region_row + 1) * kv_height // region_rows
        for region_col in range(region_cols):
            x0 = region_col * kv_width // region_cols
            x1 = (region_col + 1) * kv_width // region_cols
            cells = sorted(
                ((row, col) for row in range(y0, y1) for col in range(x0, x1)),
                key=lambda cell: _morton(cell[0] - y0, cell[1] - x0),
            )
            if len(cells) < groups:
                raise ValueError(
                    "Insufficient positions in a sampling region to assign one KV token to every group."
                )
            regions.append([row * kv_width + col for row, col in cells])
    members: list[list[int]] = [[] for _ in range(groups)]
    for region, cells in enumerate(regions):
        rotation = (seed + 17 * region) % groups
        for position in range(groups):
            members[(rotation + position) % groups].append(cells[position])
    for region, cells in enumerate(regions):
        rotation = (seed + 17 * region + groups // 2) % groups
        for offset, token in enumerate(cells[groups:]):
            candidate = (rotation + offset) % groups
            while len(members[candidate]) >= quotas[candidate]:
                candidate = (candidate + 1) % groups
            members[candidate].append(token)
    return _balanced_ranks(members, keys)
