import math
import numpy as np


def protected_permutation(side, cell, fraction, seed):
    """Reassign remote K/V within quadrants while fixing local context and anchors."""
    from representative_context.controls.context_layout import anchor_indices

    ids = np.arange(side * side)
    y, x = np.divmod(ids, side)
    cy, cx = divmod(cell, side)
    fixed = (abs(y - cy) <= 3) & (abs(x - cx) <= 3) | np.isin(ids, anchor_indices(side))
    quadrant = (y >= side // 2).astype(int) * 2 + (x >= side // 2)
    rng = np.random.default_rng(seed + cell)
    perm = ids.copy()
    for quad in range(4):
        eligible = ids[(quadrant == quad) & ~fixed]
        order = rng.permutation(eligible)
        count = math.floor(len(order) * fraction)
        if count >= 2:
            chosen = order[:count]
            perm[chosen] = np.roll(chosen, 1)
    assert np.array_equal(np.sort(perm), ids)
    assert np.array_equal(perm[fixed], ids[fixed])
    assert np.array_equal(quadrant[perm], quadrant)
    return (perm, fixed)
