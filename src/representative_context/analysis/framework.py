"""Classification competition, spatial representation and regression diagnostics."""

from __future__ import annotations
from functools import lru_cache
import numpy as np


def competition(z, y, gt):
    z, y = (np.asarray(z, np.float64), np.asarray(y, np.float64))
    gt = np.asarray(gt)
    n, classes = z.shape
    assert y.shape == z.shape and gt.shape == (n,)
    assert np.isfinite(z).all() and np.isfinite(y).all()
    winner, current = (z.argmax(1), y.argmax(1))
    rows = np.arange(n)
    rival = np.arange(classes)[None] != winner[:, None]
    margin = z[rows, winner, None] - z
    unique = ((margin > 0) | ~rival).all(1)
    v, w = (z - z.mean(1, keepdims=True), y - y.mean(1, keepdims=True))
    norm = np.einsum("ij,ij->i", v, v)
    lam = np.divide(np.einsum("ij,ij->i", v, w), norm, out=np.zeros(n), where=norm > 0)
    tangent = w - lam[:, None] * v
    good = unique & (norm > 0) & (lam > 0)
    advance = tangent - tangent[rows, winner, None]
    exposure = np.divide(
        np.maximum(advance, 0),
        lam[:, None] * margin,
        out=np.zeros_like(z),
        where=rival & good[:, None],
    )
    bound = np.where(good, np.minimum(1.0, exposure.max(1) ** 2), 1.0)
    flip = winner != current
    new = (winner == gt) & (current != gt)
    fix = (winner != gt) & (current == gt)
    switch = flip & ~new & ~fix
    assert np.all(flip <= bound + 1e-12)
    assert np.array_equal(flip, new | fix | switch)
    accuracy = (current == gt).astype(int) - (winner == gt).astype(int)
    assert np.array_equal(accuracy, fix.astype(int) - new.astype(int))
    return (
        {
            "pixels": n,
            "label_changes": int(flip.sum()),
            "new_errors": int(new.sum()),
            "corrections": int(fix.sum()),
            "wrong_class_switches": int(switch.sum()),
            "truncated_competition_sum": float(bound.sum()),
            "unit_bound_pixels": int((~good).sum()),
            "centered_error_squared_sum": float(np.square(w - v).sum()),
        },
        bound,
    )


def pool_classification(rows):
    keys = (
        "pixels",
        "label_changes",
        "new_errors",
        "corrections",
        "wrong_class_switches",
        "truncated_competition_sum",
        "unit_bound_pixels",
        "centered_error_squared_sum",
    )
    out = {k: sum((r[k] for r in rows)) for k in keys}
    n = out["pixels"]
    out.update(
        {
            "label_change_percent": 100 * out["label_changes"] / n,
            "competition_bound_percent": 100 * out["truncated_competition_sum"] / n,
            "new_error_percent": 100 * out["new_errors"] / n,
            "correction_percent": 100 * out["corrections"] / n,
            "accuracy_change_pp": 100 * (out["corrections"] - out["new_errors"]) / n,
            "centered_vector_rms": np.sqrt(out["centered_error_squared_sum"] / n),
        }
    )
    return out


def morton_order(side):
    yy, xx = np.indices((side, side))
    codes = np.zeros((side, side), np.int64)
    for bit in range(side.bit_length()):
        codes |= (xx >> bit & 1) << 2 * bit
        codes |= (yy >> bit & 1) << 2 * bit + 1
    return codes.ravel().argsort()


@lru_cache(maxsize=256)
def layout_geometry(side, b, pattern, seed):
    from representative_context.layouts.spatial import (
        coverage_ranks,
        point_coverage_ranks,
    )

    keys, groups = (side**2, side**2 // b)
    assert keys % b == 0 and b % side == 0
    order = morton_order(side)
    inverse = np.empty(keys, np.int64)
    inverse[order] = np.arange(keys)
    yy, xx = np.divmod(np.arange(keys), side)
    coords = (np.stack((yy, xx), 1) + 0.5) / side
    if pattern == "strip":
        chosen = np.flatnonzero(xx < b // side)[None]
        qgroups = np.zeros(keys, np.int64)
    else:
        fn = point_coverage_ranks if pattern == "point32" else coverage_ranks
        regions = 32 if pattern == "point32" else 16
        group = np.asarray(fn(side, side, groups, regions, seed)) % groups
        assert np.all(np.bincount(group, minlength=groups) == b)
        chosen = np.stack([np.flatnonzero(group == g) for g in range(groups)])
        qgroups = group
    distances = []
    microchecks = []
    for selected in chosen:
        assert len(np.unique(selected)) == b
        selected = selected[np.argsort(inverse[selected])]
        if pattern == "strip":
            target = yy * side + xx // groups
            cost = np.linalg.norm(coords - coords[target], axis=1).mean()
            assert np.all(np.bincount(target, minlength=keys)[chosen[0]] == groups)
            assert np.isclose(cost, (1 - b / keys) / 2, atol=1e-14)
        else:
            target = np.repeat(selected, groups)
            cost = np.linalg.norm(coords[order] - coords[target], axis=1).mean()
            if pattern == "point32":
                occupied = inverse[selected] // groups
                one_per_cell = np.array_equal(occupied, np.arange(b))
                assert one_per_cell
                assert cost <= np.sqrt(2.5 / b) + 1e-12
                microchecks.append(one_per_cell)
        distances.append(float(cost))
    return {
        "selected_sets": chosen,
        "query_groups": qgroups,
        "transport_distances": np.array(distances),
        "microcell_checks": microchecks,
    }


def prepare_attention(q, k, v, side):
    q, k, v = [np.asarray(a, np.float64) for a in (q, k, v)]
    heads, queries, dim = q.shape
    keys = side**2
    scores = q @ k.transpose(0, 2, 1) * dim ** (-0.5)
    weights = np.exp(scores - scores.max(2, keepdims=True))
    weights /= weights.sum(2, keepdims=True)
    dense = weights @ v
    local_lipschitz = np.zeros((heads, queries))
    for h in range(heads):
        for i in range(queries):
            g = (keys * weights[h, i, :, None] * (v[h] - dense[h, i])).reshape(
                side, side, dim
            )
            edge = max(
                np.linalg.norm(np.diff(g, axis=0), axis=2).max(),
                np.linalg.norm(np.diff(g, axis=1), axis=2).max(),
            )
            local_lipschitz[h, i] = side * np.sqrt(2) * edge
    return (weights, dense, v, local_lipschitz)


def attention_diagnostics(q, k, v, qids, qside, side, b, pattern, seed, prepared=None):
    weights, dense, v, local_lipschitz = (
        prepare_attention(q, k, v, side) if prepared is None else prepared
    )
    heads, queries, keys = weights.shape
    geo = layout_geometry(side, b, pattern, seed)
    kv_cells = qids // qside * side // qside * side + qids % qside * side // qside
    groups = geo["query_groups"][kv_cells]
    selection = geo["selected_sets"][groups]
    mask = np.zeros((queries, keys), bool)
    mask[np.arange(queries)[:, None], selection] = True
    assert np.all(mask.sum(1) == b)
    mass = (weights * mask[None]).sum(2)
    ds = keys / b * mass
    retained = weights * mask[None] @ v
    sparse = retained / mass[:, :, None]
    omitted = weights @ v - retained
    identity = (1 - mass[:, :, None]) * (
        sparse
        - np.divide(
            omitted,
            1 - mass[:, :, None],
            out=np.zeros_like(dense),
            where=1 - mass[:, :, None] > 1e-15,
        )
    )
    identity_error = float(np.max(np.abs(sparse - dense - identity)))
    assert identity_error < 1e-09
    error = np.linalg.norm(sparse - dense, axis=2)
    dense_norm = np.linalg.norm(dense, axis=2)
    transport = geo["transport_distances"][groups]
    bound = local_lipschitz * transport[None] / ds
    assert np.all(error <= bound + 1e-09)
    return {
        "aggregation_l2_error": error,
        "dense_aggregation_l2": dense_norm,
        "D_S": ds,
        "retained_probability_mass": mass,
        "transport_bound": np.broadcast_to(transport, error.shape),
        "weighted_residual_lipschitz_bound": local_lipschitz,
        "aggregation_error_bound": bound,
        "identity_max_abs_error": identity_error,
    }


def regression_decomposition(reference, current, gt):
    valid = np.isfinite(gt).all(-1) & (np.abs(gt) < 1000000000.0).all(-1)
    d, s, t = [np.asarray(a, np.float64)[valid] for a in (reference, current, gt)]
    r, e = (d - t, s - d)
    cross = 2 * np.einsum("ij,ij->i", r, e)
    energy = np.einsum("ij,ij->i", e, e)
    base = np.einsum("ij,ij->i", r, r)
    changed = np.square(s - t).sum(1) - base
    residual = changed - cross - energy
    assert np.max(np.abs(residual)) < 1e-08 * max(1.0, np.max(base))
    epe_d, epe_s = (np.linalg.norm(r, axis=1), np.linalg.norm(s - t, axis=1))
    out = {
        "valid_pixels": int(valid.sum()),
        "dense_mse": float(base.mean()),
        "sparse_mse": float(np.square(s - t).sum(1).mean()),
        "residual_cross_term": float(cross.mean()),
        "perturbation_energy": float(energy.mean()),
        "mse_change": float(changed.mean()),
        "identity_max_abs_residual": float(np.max(np.abs(residual))),
        "mse_improvement_fraction": float((changed < 0).mean()),
        "mse_degradation_fraction": float((changed > 0).mean()),
        "mse_improvement_contribution": float(np.minimum(changed, 0).mean()),
        "mse_degradation_contribution": float(np.maximum(changed, 0).mean()),
        "dense_epe": float(epe_d.mean()),
        "sparse_epe": float(epe_s.mean()),
        "epe_change": float((epe_s - epe_d).mean()),
    }
    for key in (
        "residual_cross_term",
        "perturbation_energy",
        "mse_change",
        "mse_improvement_contribution",
        "mse_degradation_contribution",
    ):
        out[key + "_percent_of_dense_mse"] = 100 * out[key] / out["dense_mse"]
    arrays = {
        "valid_mask": valid,
        "residual_cross_term": cross,
        "perturbation_energy": energy,
        "mse_change": changed,
    }
    return (out, arrays)
