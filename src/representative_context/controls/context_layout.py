"""Controlled contexts with identical local neighbors, anchors and capacity."""

import numpy as np
from scipy.spatial import cKDTree
from representative_context.layouts.spatial import _morton

PATTERNS = (4, 2, 1)
GUARD = 3


def anchor_indices(side):
    return np.array(
        sorted(
            {
                0,
                side - 1,
                (side - 1) * side,
                side * side - 1,
                side // 2,
                side // 2 * side,
                (side - 1) * side + side // 2,
                side // 2 * side + side - 1,
            }
        ),
        dtype=np.int32,
    )


def reverse_bits(value, bits):
    return sum(((value >> bit & 1) << bits - 1 - bit for bit in range(bits)))


def quadrant_orders(side, cluster, seed):
    """Traverse remote tiles evenly and retain consecutive positions within each tile."""
    half, orders = (side // 2, [])
    tiles = half // cluster
    bits = 2 * (tiles.bit_length() - 1)
    tile_list = sorted(
        ((y, x) for y in range(tiles) for x in range(tiles)),
        key=lambda p: reverse_bits(_morton(*p), bits) ^ seed % tiles**2,
    )
    inner = sorted(
        ((y, x) for y in range(cluster) for x in range(cluster)),
        key=lambda p: _morton(*p) ^ seed % cluster**2,
    )
    for quadrant in range(4):
        y0, x0 = (quadrant // 2 * half, quadrant % 2 * half)
        orders.append(
            np.array(
                [
                    (y0 + ty * cluster + y) * side + x0 + tx * cluster + x
                    for ty, tx in tile_list
                    for y, x in inner
                ],
                dtype=np.int32,
            )
        )
    return orders


def construct(side, budget, cluster, seed):
    """Construct one query context per KV cell with fixed local and anchor subsets."""
    if (
        side < 16
        or side & side - 1
        or budget < 24
        or (budget >= side**2)
        or (cluster not in PATTERNS)
    ):
        raise ValueError(
            "Use a dyadic square KV grid, valid capacity and cluster size 4, 2 or 1."
        )
    keys = side**2
    y, x = np.divmod(np.arange(keys), side)
    anchors, orders = (anchor_indices(side), quadrant_orders(side, cluster, seed))
    anchor_mask = np.zeros(keys, dtype=bool)
    anchor_mask[anchors] = True
    indices = np.empty((keys, budget), dtype=np.int32)
    local = np.empty((keys, 9), dtype=np.int32)
    quotas = np.empty((keys, 4), dtype=np.int32)
    base_counts = np.empty(keys, dtype=np.int32)
    for cell in range(keys):
        top, left = (
            np.clip(y[cell] - 1, 0, side - 3),
            np.clip(x[cell] - 1, 0, side - 3),
        )
        loc = np.array(
            [(top + dy) * side + left + dx for dy in range(3) for dx in range(3)],
            dtype=np.int32,
        )
        base = np.union1d(loc, anchors).astype(np.int32)
        forbidden = (np.abs(y - y[cell]) <= GUARD) & (
            np.abs(x - x[cell]) <= GUARD
        ) | anchor_mask
        remaining = budget - len(base)
        quota = np.full(4, remaining // 4, dtype=np.int32)
        quota[(np.arange(remaining % 4) + cell) % 4] += 1
        remote = []
        for quad, order in enumerate(orders):
            valid = order[~forbidden[order]]
            if len(valid) < quota[quad]:
                raise ValueError(
                    "Insufficient remote positions for the fixed exclusion area and capacity."
                )
            remote.extend(valid[: quota[quad]])
        indices[cell] = np.concatenate((base, remote))
        local[cell], quotas[cell], base_counts[cell] = (loc, quota, len(base))
    return (indices, local, anchors, quotas, base_counts)


def audit(indices, local, anchors, quotas, base_counts, side, qside):
    """Check every context and the complete mapping from queries to contexts."""
    keys, budget = indices.shape
    assert keys == side**2 and qside % side == 0
    assert indices.min() >= 0 and indices.max() < keys
    assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0)
    y, x = np.divmod(np.arange(keys), side)
    near_rows, remote_mask = ([], np.zeros((keys, keys), dtype=bool))
    for cell in range(keys):
        row = indices[cell]
        assert np.isin(local[cell], row).all() and np.isin(anchors, row).all()
        assert cell in local[cell] and cell in row
        expected_near = np.union1d(local[cell], anchors)
        expected_near = expected_near[
            (np.abs(expected_near // side - y[cell]) <= GUARD)
            & (np.abs(expected_near % side - x[cell]) <= GUARD)
        ]
        actual_near = row[
            (np.abs(row // side - y[cell]) <= GUARD)
            & (np.abs(row % side - x[cell]) <= GUARD)
        ]
        assert np.array_equal(np.sort(actual_near), expected_near)
        near_rows.extend([len(actual_near), *np.sort(actual_near).tolist()])
        remote = row[base_counts[cell] :]
        assert np.all(
            (np.abs(remote // side - y[cell]) > GUARD)
            | (np.abs(remote % side - x[cell]) > GUARD)
        )
        quad = (remote // side >= side // 2).astype(int) * 2 + (
            remote % side >= side // 2
        )
        assert np.array_equal(np.bincount(quad, minlength=4), quotas[cell])
        remote_mask[cell, remote] = True
    copies = qside // side
    qy, qx = np.divmod(np.arange(qside**2), qside)
    q_to_cell = qy // copies * side + qx // copies
    assert np.array_equal(
        np.bincount(q_to_cell, minlength=keys), np.full(keys, copies**2)
    )
    cell = np.arange(keys)[:, None]
    offset = np.arange(copies**2)[None, :]
    q_order = (
        (cell // side * copies + offset // copies) * qside
        + cell % side * copies
        + offset % copies
    ).ravel()
    assert np.array_equal(np.sort(q_order), np.arange(qside**2))
    assert np.array_equal(q_to_cell[q_order], np.repeat(np.arange(keys), copies**2))
    rem = remote_mask.reshape(keys, side, side)
    adjacent = int(
        (rem[:, :, 1:] & rem[:, :, :-1]).sum() + (rem[:, 1:] & rem[:, :-1]).sum()
    )
    probe = sorted(
        {
            int((iy + 0.5) * side / 8) * side + int((ix + 0.5) * side / 8)
            for iy in range(8)
            for ix in range(8)
        }
    )
    grid = np.stack((y, x), axis=1)
    gaps = []
    for cell in probe:
        selected = np.stack(np.divmod(indices[cell], side), axis=1)
        gaps.append(
            float(
                cKDTree(selected).query(grid, workers=1)[0].mean()
                / ((side - 1) * 2**0.5)
            )
        )
    return {
        "passed": True,
        "verified_contexts": keys,
        "verified_queries": qside**2,
        "local_keys": 9,
        "anchor_keys": len(anchors),
        "capacity": budget,
        "self_key_retention": 1.0,
        "remote_neighbors_per_key": adjacent / int(remote_mask.sum()),
        "geometry_query_cells": probe,
        "geometry_queries": len(probe),
        "geometry_scope": "Fixed 8 by 8 query centers; nearest distances use the complete KV grid; invariants cover every query.",
        "coverage_score": 1 - float(np.mean(gaps)),
        "remote_exclusion_radius": GUARD,
        "bounding_box": [0, 0, side - 1, side - 1],
    }
