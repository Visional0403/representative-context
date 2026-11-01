"""Balanced layouts that progressively expand each group's spatial support."""

from .spatial import _balanced_ranks, _morton


def extent_ranks(height, width, groups, level, seed):
    """Mix adjacent Morton domains at fixed capacity on divisible dyadic grids."""
    keys = height * width
    if any(value < 1 or value & (value - 1) for value in (height, width, groups)):
        raise ValueError(
            "Extent layouts require positive power-of-two dimensions and groups."
        )
    if keys % groups or groups > keys or not 0 <= level <= 6:
        raise ValueError("Invalid capacity or extent level.")
    capacity, mixing = keys // groups, min(2**level, groups)
    order = sorted(
        range(keys), key=lambda index: _morton(index // width, index % width)
    )
    members = [[] for _ in range(groups)]
    for region in range(groups // mixing):
        start = region * capacity * mixing
        for position, token in enumerate(order[start : start + capacity * mixing]):
            cell, offset = divmod(position, mixing)
            phase = ((2 * seed + 1) * cell + 17 * seed) % mixing
            group = region * mixing + (offset + phase) % mixing
            members[group].append(token)
    assert all(len(values) == capacity for values in members)
    return _balanced_ranks(members, keys)


def spatial_statistics(ranks, height, width, groups):
    """Measure every grid position's distance to each group's nearest retained key."""
    import math
    import numpy as np
    from scipy.spatial import cKDTree

    ranks = np.asarray(ranks, dtype=np.int64)
    keys, capacity = height * width, height * width // groups
    assert np.array_equal(np.sort(ranks), np.arange(keys))
    ids = ranks % groups
    assert np.all(np.bincount(ids, minlength=groups) == capacity)
    positions = np.stack(np.unravel_index(np.arange(keys), (height, width)), axis=1)
    diagonal = math.hypot(height - 1, width - 1)
    rows = []
    for group in range(groups):
        members = positions[ids == group]
        distances = cKDTree(members).query(positions, k=1, workers=1)[0]
        rows.append(
            dict(
                group=group,
                keys=len(members),
                mean_nearest_distance=float(distances.mean() / diagonal),
                largest_gap=float(distances.max() / diagonal),
                bounding_box_fraction=float(
                    np.prod(np.ptp(members, axis=0) + 1) / keys
                ),
            )
        )
    gap = float(np.mean([row["mean_nearest_distance"] for row in rows]))
    query_cell = (height // 2 - 1) * width + width // 2 - 1
    return dict(
        groups=groups,
        capacity=capacity,
        mean_nearest_distance=gap,
        coverage_score=1 - gap,
        mean_largest_gap=float(np.mean([row["largest_gap"] for row in rows])),
        mean_bounding_box_fraction=float(
            np.mean([row["bounding_box_fraction"] for row in rows])
        ),
        groups_detail=rows,
        illustrative_query_cell=[query_cell // width, query_cell % width],
        illustrative_candidate_indices=np.flatnonzero(ids == ids[query_cell]).tolist(),
    )
