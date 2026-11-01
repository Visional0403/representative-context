"""Cross-task context diagnostics with controlled rectangular grids."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import torch
import torch.nn.functional as F

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (20261025, 20261026)


def anchors(height, width):
    return np.array(
        sorted(
            {
                0,
                width - 1,
                (height - 1) * width,
                height * width - 1,
                width // 2,
                height // 2 * width,
                (height - 1) * width + width // 2,
                height // 2 * width + width - 1,
            }
        ),
        dtype=np.int64,
    )


def query_ids(height, width, valid_h, valid_w):
    ys = np.unique(
        np.minimum(valid_h - 1, ((np.arange(4) + 0.5) * valid_h / 4).astype(int))
    )
    xs = np.unique(
        np.minimum(valid_w - 1, ((np.arange(4) + 0.5) * valid_w / 4).astype(int))
    )
    return (ys[:, None] * width + xs[None, :]).ravel().astype(np.int64)


def morton_array(y, x, bits):
    code = np.zeros_like(y, dtype=np.int64)
    for bit in range(bits):
        code |= (x >> bit & 1) << 2 * bit
        code |= (y >> bit & 1) << 2 * bit + 1
    return code


@lru_cache(maxsize=32)
def quadrant_order(height, width, cluster, quadrant):
    y0, y1 = (0, height // 2) if quadrant // 2 == 0 else (height // 2, height)
    x0, x1 = (0, width // 2) if quadrant % 2 == 0 else (width // 2, width)
    th, tw = (math.ceil((y1 - y0) / cluster), math.ceil((x1 - x0) / cluster))
    ty, tx = np.indices((th, tw))
    bits = max((th - 1).bit_length(), (tw - 1).bit_length())
    code = morton_array(ty.ravel(), tx.ravel(), bits)
    reverse = np.zeros_like(code)
    for bit in range(2 * bits):
        reverse |= (code >> bit & 1) << 2 * bits - 1 - bit
    tile_order = np.argsort(reverse ^ 42, kind="stable")
    inner_y, inner_x = np.indices((cluster, cluster))
    inner = np.argsort(
        morton_array(inner_y.ravel(), inner_x.ravel(), max(1, cluster.bit_length() - 1))
        ^ 42 % cluster**2
    )
    yy = y0 + ty.ravel()[tile_order, None] * cluster + inner_y.ravel()[inner]
    xx = x0 + tx.ravel()[tile_order, None] * cluster + inner_x.ravel()[inner]
    valid = (yy < y1) & (xx < x1)
    return (yy * width + xx)[valid].astype(np.int64)


@lru_cache(maxsize=64)
def geometry(height, width, valid_h, valid_w, budget):
    if not (
        3 <= valid_h <= height
        and 3 <= valid_w <= width
        and (24 <= budget < height * width)
    ):
        raise ValueError(
            "Controlled contexts require a valid grid and a partial retained capacity."
        )
    count = height * width
    ids = np.arange(count, dtype=np.int64)
    y, x = np.divmod(ids, width)
    valid = (y < valid_h) & (x < valid_w)
    quadrants = (y >= height // 2).astype(int) * 2 + (x >= width // 2)
    fixed_anchors = anchors(height, width)
    qs = query_ids(height, width, valid_h, valid_w)
    all_pairs = np.concatenate(
        (
            np.stack(
                (
                    ids.reshape(height, width)[:, :-1].ravel(),
                    ids.reshape(height, width)[:, 1:].ravel(),
                ),
                1,
            ),
            np.stack(
                (
                    ids.reshape(height, width)[:-1].ravel(),
                    ids.reshape(height, width)[1:].ravel(),
                ),
                1,
            ),
        )
    )
    all_pairs = all_pairs[
        valid[all_pairs].all(1)
        & (quadrants[all_pairs[:, 0]] == quadrants[all_pairs[:, 1]])
    ]
    contexts = []
    for qid in qs:
        qy, qx = divmod(int(qid), width)
        top, left = (
            int(np.clip(qy - 1, 0, valid_h - 3)),
            int(np.clip(qx - 1, 0, valid_w - 3)),
        )
        local = np.array(
            [(top + dy) * width + left + dx for dy in range(3) for dx in range(3)],
            dtype=np.int64,
        )
        base = np.union1d(local, fixed_anchors)
        near = (abs(y - qy) <= 3) & (abs(x - qx) <= 3)
        protected = near | np.isin(ids, fixed_anchors) | ~valid
        excluded = near | np.isin(ids, fixed_anchors)
        remaining = budget - len(base)
        quotas = np.full(4, remaining // 4, dtype=np.int64)
        quotas[(np.arange(remaining % 4) + qid) % 4] += 1
        rows = {1: [base], 4: [base]}
        eligible = []
        for quadrant in range(4):
            order = quadrant_order(height, width, 1, quadrant)
            available = order[~excluded[order]]
            real = available[valid[available]]
            padding = available[~valid[available]]
            quota = int(quotas[quadrant])
            nreal = min(
                len(real),
                max(quota - len(padding), round(quota * len(real) / len(available))),
            )
            npad = quota - nreal
            if npad > len(padding) or nreal > len(real):
                raise RuntimeError(
                    "Quadrant quotas cannot satisfy the declared neighborhood and capacity."
                )
            common_padding = padding[:npad]
            for cluster in (1, 4):
                order = quadrant_order(height, width, cluster, quadrant)
                real_order = order[valid[order] & ~excluded[order]]
                rows[cluster].extend((common_padding, real_order[:nreal]))
            eligible.append(ids[(quadrants == quadrant) & ~protected])
        rows = {
            cluster: np.concatenate(parts).astype(np.int64)
            for cluster, parts in rows.items()
        }
        neighbors = all_pairs[(~protected[all_pairs]).all(1)]
        if len(neighbors) == 0:
            raise RuntimeError(
                "The declared query has no eligible remote neighbor pair."
            )
        for row in rows.values():
            assert len(row) == budget and len(np.unique(row)) == budget
            assert np.isin(local, row).all() and np.isin(fixed_anchors, row).all()
            assert np.array_equal(
                np.sort(row[near[row]]), np.intersect1d(base, ids[near])
            )
            remote = row[len(base) :]
            assert np.array_equal(np.bincount(quadrants[remote], minlength=4), quotas)
        assert np.array_equal(
            np.sort(rows[1][~valid[rows[1]]]), np.sort(rows[4][~valid[rows[4]]])
        )
        contexts.append(
            {
                "queries": int(qid),
                "candidates": rows,
                "protected": protected,
                "local": local,
                "anchors": fixed_anchors,
                "quadrant_quotas": quotas,
                "eligible": eligible,
                "neighbor_pairs": neighbors,
            }
        )
    return {
        "height": height,
        "width": width,
        "valid_height": valid_h,
        "valid_width": valid_w,
        "capacity": budget,
        "queries": qs,
        "valid_positions": np.flatnonzero(valid),
        "contexts": contexts,
    }


@lru_cache(maxsize=640)
def permutation(height, width, valid_h, valid_w, budget, qi, fraction, seed):
    geom = geometry(height, width, valid_h, valid_w, budget)
    context = geom["contexts"][qi]
    result = np.arange(height * width, dtype=np.int64)
    if fraction:
        rng = np.random.default_rng(seed + context["queries"])
        for eligible in context["eligible"]:
            order = rng.permutation(eligible)
            n = math.floor(len(order) * fraction)
            if n >= 2:
                chosen = order[:n]
                result[chosen] = np.roll(chosen, 1)
    return result


def audit_geometry(geom):
    h, w, vh, vw, b = (
        geom[k] for k in ("height", "width", "valid_height", "valid_width", "capacity")
    )
    ids = np.arange(h * w)
    y, x = np.divmod(ids, w)
    quad = (y >= h // 2).astype(int) * 2 + (x >= w // 2)
    checked = 0
    for qi, context in enumerate(geom["contexts"]):
        for f in FRACTIONS:
            for seed in SEEDS if f else SEEDS[:1]:
                p = permutation(h, w, vh, vw, b, qi, f, seed)
                assert np.array_equal(np.sort(p), ids)
                assert np.array_equal(
                    p[context["protected"]], ids[context["protected"]]
                )
                assert np.array_equal(quad[p], quad)
                checked += 1
    return {
        "passed": True,
        "grid": [h, w],
        "valid_grid": [vh, vw],
        "retained_keys": b,
        "sampled_queries": len(geom["queries"]),
        "query_indices": geom["queries"].tolist(),
        "verified_permutations": checked,
        "identical_local_context": True,
        "identical_anchors": True,
        "identical_quadrant_quotas": True,
        "identical_padding_candidates": True,
    }


def relative_difference(actual, reference):
    error = actual.double() - reference.double()
    absolute = float(error.abs().max())
    relative = float(error.norm() / reference.double().norm().clamp_min(1e-12))
    if relative > 0.0001 and absolute > 2e-05:
        raise RuntimeError(
            f"Dense aggregation invariance failed: absolute difference {absolute}, relative difference {relative}"
        )
    return (absolute, relative)


class Diagnostic:

    def __init__(self):
        self.device_cache = {}

    def device_geometry(self, geom, device):
        key = tuple(
            (
                geom[k]
                for k in ("height", "width", "valid_height", "valid_width", "capacity")
            )
        )
        if key not in self.device_cache:
            self.device_cache[key] = [
                {
                    "candidates": {
                        c: torch.as_tensor(ids, device=device)
                        for c, ids in row["candidates"].items()
                    },
                    "neighbor_pairs": torch.as_tensor(
                        row["neighbor_pairs"], device=device
                    ),
                }
                for row in geom["contexts"]
            ]
        return (key, self.device_cache[key])

    def __call__(self, scores, value, geom, label, cpu_spot=False):
        """Reassign complete score columns and values while preserving the dense trajectory."""
        key, device_rows = self.device_geometry(geom, value.device)
        assert scores.shape[:2] == (value.shape[0], len(geom["queries"]))
        assert scores.shape[-1] == value.shape[1] == geom["height"] * geom["width"]
        dense = scores.softmax(-1) @ value
        valid_ids = torch.as_tensor(geom["valid_positions"], device=value.device)
        centered = value - value[:, valid_ids].mean(1, keepdim=True)
        normalized = F.normalize(centered, dim=-1)
        arrays = {label + "_dense": dense.cpu().numpy().copy()}
        rows = []
        for fraction in FRACTIONS:
            for seed in SEEDS if fraction else SEEDS[:1]:
                outputs = {1: [], 4: []}
                correlations, inner_sums, inner_counts, checks = ([], [], [], [])
                for qi, context in enumerate(device_rows):
                    p = permutation(*key, qi, fraction, seed)
                    perm = torch.as_tensor(p, device=value.device)
                    ps = scores[:, qi].index_select(-1, perm)
                    pv = value.index_select(1, perm)
                    current_dense = torch.bmm(ps.softmax(-1).unsqueeze(1), pv).squeeze(
                        1
                    )
                    checks.append(relative_difference(current_dense, dense[:, qi]))
                    pairs = context["neighbor_pairs"]
                    nv = normalized.index_select(1, perm)
                    dots = (nv[:, pairs[:, 0]] * nv[:, pairs[:, 1]]).sum(-1)
                    total, count = (float(dots.double().sum()), dots.numel())
                    inner_sums.append(total)
                    inner_counts.append(count)
                    correlations.append(total / count)
                    for cluster in (1, 4):
                        ids = context["candidates"][cluster]
                        output = torch.bmm(
                            ps[:, ids].softmax(-1).unsqueeze(1), pv[:, ids]
                        ).squeeze(1)
                        outputs[cluster].append(output)
                arrays_key = f"{label}_f{fraction:.2f}_s{seed}"
                out = {c: torch.stack(parts, 1) for c, parts in outputs.items()}
                reference = float(dense.double().square().sum())
                errors = {
                    str(c): float(
                        (tensor.double() - dense.double()).square().sum()
                        / max(reference, 1e-24)
                    )
                    ** 0.5
                    for c, tensor in out.items()
                }
                for c, tensor in out.items():
                    arrays[arrays_key + f"_c{c}"] = tensor.cpu().numpy().copy()
                arrays[arrays_key + "_correlation_sums"] = np.array(inner_sums)
                arrays[arrays_key + "_correlation_counts"] = np.array(inner_counts)
                row = {
                    "reassignment_fraction": fraction,
                    "seed": seed,
                    "array_prefix": arrays_key,
                    "neighbor_content_correlation": float(np.mean(correlations)),
                    "aggregation_relative_l2": errors,
                    "excess_error": errors["4"] - errors["1"],
                    "dense_max_abs_change": max((v[0] for v in checks)),
                    "dense_max_relative_change": max((v[1] for v in checks)),
                }
                rows.append(row)
        if cpu_spot:
            cpu_value = value.double().cpu().numpy()
            cpu_scores = scores[:, 0].double().cpu().numpy()

            def aggregate(s, v):
                p = np.exp(s - s.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                return np.einsum("hk,hkd->hd", p, v)

            reference = aggregate(cpu_scores, cpu_value)
            cpu_checks = []
            for fraction in (0.0, 1.0):
                p = permutation(*key, 0, fraction, SEEDS[0])
                pd = aggregate(cpu_scores[:, p], cpu_value[:, p])
                assert np.allclose(pd, reference, rtol=1e-11, atol=1e-11)
                for c in (1, 4):
                    ids = geom["contexts"][0]["candidates"][c]
                    output = aggregate(cpu_scores[:, p[ids]], cpu_value[:, p[ids]])
                    actual = arrays[f"{label}_f{fraction:.2f}_s{SEEDS[0]}_c{c}"][:, 0]
                    assert np.allclose(output, actual, rtol=0.0002, atol=2e-05)
                    cpu_checks.append(float(np.max(np.abs(output - actual))))
        else:
            cpu_checks = None
        return (
            {
                "block": label,
                "grid": [geom["height"], geom["width"]],
                "valid_grid": [geom["valid_height"], geom["valid_width"]],
                "capacity": geom["capacity"],
                "query_indices": geom["queries"].tolist(),
                "heads": value.shape[0],
                "head_dimension": value.shape[-1],
                "interventions": rows,
                "cpu_reconstruction_max_abs_error": (
                    max(cpu_checks) if cpu_checks else None
                ),
            },
            arrays,
            dense,
        )
