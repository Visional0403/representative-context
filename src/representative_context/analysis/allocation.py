"""Matched-compute capacities, exact interaction counts and paired statistics."""

from __future__ import annotations
import itertools
import math
import numpy as np

STRIDES = (4, 8, 16, 32)
SR = (8, 4, 2, 1)
MODEL_SPEC = {
    "segformer_uavid": {"channels": (32, 64, 160, 256), "blocks": (2, 2, 2, 2)},
    "twins_uavid": {"channels": (64, 128, 256, 512), "blocks": (1, 1, 5, 2)},
}


def sizes(keys, groups):
    q, t = divmod(keys, groups)
    return np.r_[
        np.full(t, q + 1, dtype=np.int64), np.full(groups - t, q, dtype=np.int64)
    ]


def cost(model, groups, side=2048):
    keys = (side // 32) ** 2
    spec = MODEL_SPEC[model]
    layers = []
    for s, r, c, depth, g in zip(STRIDES, SR, spec["channels"], spec["blocks"], groups):
        count = sizes(keys, g)
        edges = int(np.dot(count, count)) * r * r
        flops = 4 * c * edges * depth
        layers.append(
            {
                "stage": len(layers) + 1,
                "Q": (side // s) ** 2,
                "K": keys,
                "channels": c,
                "blocks": depth,
                "groups": g,
                "min_retained_keys": int(count.min()),
                "max_retained_keys": int(count.max()),
                "effective_retained_keys": edges / (side // s) ** 2,
                "edges_per_block": edges,
                "stage_flops": flops,
            }
        )
    return {
        "stages": layers,
        "attention_flops": sum((t["stage_flops"] for t in layers)),
        "total_edges": sum((t["edges_per_block"] * t["blocks"] for t in layers)),
    }


def allocation(model, k, policy):
    keys = 4096
    baseline = [max(1, math.ceil(2**k / (s * r * r))) for s, r in zip(STRIDES, SR)]
    basegroups = [math.ceil(keys / b) for b in baseline]
    target = cost(model, basegroups)["attention_flops"]
    if policy == "structural":
        groups, continuous, error = (basegroups, baseline, 0.0)
    else:
        shape = np.ones(4) if policy == "uniform" else 1 / np.asarray(STRIDES)
        spec = MODEL_SPEC[model]
        weights = np.asarray(
            [
                4 * c * (2048 // s) ** 2 * n
                for c, s, n in zip(spec["channels"], STRIDES, spec["blocks"])
            ]
        )
        scale = target / float(np.dot(weights, shape))
        best = None
        for factor in np.linspace(0.95, 1.05, 101):
            continuous = np.minimum(keys, scale * factor * shape)
            group_float = keys / continuous
            choices = []
            for b, gf in zip(continuous, group_float):
                bs = {max(1, math.floor(b)), max(1, math.ceil(b))}
                for g in (max(1, math.floor(gf)), min(keys, math.ceil(gf))):
                    upper = math.ceil(keys / g)
                    bs.add(upper)
                    if math.ceil(keys / upper) != g:
                        bs.add(max(1, upper - 1))
                choices.append(sorted({math.ceil(keys / value) for value in bs}))
            for groups in itertools.product(*choices):
                actual = cost(model, groups)
                error = abs(actual["attention_flops"] / target - 1)
                deviation = float(
                    np.mean(
                        np.log(
                            np.array(
                                [x["effective_retained_keys"] for x in actual["stages"]]
                            )
                            / continuous
                        )
                        ** 2
                    )
                )
                key = (error, deviation, tuple(groups))
                if best is None or key < best[0]:
                    best = (key, list(groups), continuous.tolist())
        (error, _, _), groups, continuous = best
        assert error <= 0.01, (model, k, policy, error)
    budgets = [math.ceil(keys / g) for g in groups]
    assert [math.ceil(keys / b) for b in budgets] == list(groups)
    actual = cost(model, groups)
    return {
        "case": model,
        "k": k,
        "policy": policy,
        "capacities": budgets,
        "groups": list(groups),
        "continuous_capacities": list(continuous),
        "target_attention_flops": target,
        "relative_compute_difference": actual["attention_flops"] / target - 1,
        **actual,
    }


def miou(hist):
    h = np.asarray(hist, dtype=np.float64)
    diag = np.diagonal(h, axis1=-2, axis2=-1)
    union = h.sum(-2) + h.sum(-1) - diag
    ratio = np.divide(diag, union, out=np.full_like(union, np.nan), where=union > 0)
    return 100 * np.nanmean(ratio, axis=-1)


def paired_bootstrap(histograms, draws=10000):
    """Pool confusion counts before computing each source-group bootstrap metric."""
    hist = np.asarray(histograms, dtype=np.int64)
    rng = np.random.default_rng(20261022)
    counts = rng.multinomial(len(hist), np.full(len(hist), 1 / len(hist)), draws)
    assert float(hist.max()) * len(hist) < 2**53
    pooled = counts.astype(float) @ hist.reshape(len(hist), -1).astype(float)
    pooled = pooled.reshape((draws,) + hist.shape[1:])
    metrics = miou(pooled)
    return (miou(hist.sum(0)), metrics)


from representative_context.layouts import make_ranks


def layout_cost(model, height, width, budgets):
    """Count actual layouts and nearest-grid query assignments, including irregular grids."""
    spec = MODEL_SPEC[model]
    h, w = (height, width)
    result = []
    for stage, (r, c, depth, budget) in enumerate(
        zip(SR, spec["channels"], spec["blocks"], budgets)
    ):
        stride = 4 if stage == 0 else 2
        h, w = (math.ceil(h / stride), math.ceil(w / stride))
        kh, kw = (h // r, w // r)
        keys = kh * kw
        groups = math.ceil(keys / budget)
        details = []
        for i in range(depth):
            block = i if model == "segformer_uavid" else 2 * i + 1
            ranks = np.asarray(
                make_ranks(kh, kw, budget, seed=42 + 100 * stage + block)
            )
            assert np.array_equal(np.sort(ranks), np.arange(keys))
            ids = ranks % groups
            kvn = np.bincount(ids, minlength=groups)
            yy = np.arange(h) * kh // h
            xx = np.arange(w) * kw // w
            qids = ids.reshape(kh, kw)[yy[:, None], xx]
            qn = np.bincount(qids.ravel(), minlength=groups)
            assert int(qn.sum()) == h * w and int(kvn.sum()) == keys
            assert kvn.max() <= budget and kvn.max() - kvn.min() <= 1
            edges = int(np.dot(qn, kvn))
            details.append({"block": block, "edges": edges, "FLOPs": 4 * c * edges})
        result.append(
            {
                "stage": stage + 1,
                "query_grid": [h, w],
                "kv_grid": [kh, kw],
                "capacities": budget,
                "groups": groups,
                "min_retained_keys": int(kvn.min()),
                "max_retained_keys": int(kvn.max()),
                "dense_single_group": groups == 1,
                "edges": sum((x["edges"] for x in details)),
                "FLOPs": sum((x["FLOPs"] for x in details)),
                "dense_edges": h * w * keys * depth,
                "dense_flops": 4 * c * h * w * keys * depth,
                "per_block": details,
            }
        )
    return {
        "stages": result,
        **{
            key: sum((x[key] for x in result))
            for key in ("edges", "FLOPs", "dense_edges", "dense_flops")
        },
    }


def summarize(hist, names, ids):
    hist = np.asarray(hist, dtype=np.int64)
    point, boot = paired_bootstrap(hist)
    loss = 100 * (point[0] - point[1:]) / point[0]
    bloss = 100 * (boot[:, :1] - boot[:, 1:]) / boot[:, :1]
    gains = np.diff(point[1:])
    bgain = np.diff(boot[:, 1:], axis=1)
    per = miou(hist)
    return {
        "configurations": names,
        "source_groups": ids,
        "source_group_count": len(ids),
        "per_group_confusion": hist.tolist(),
        "pooled_miou_percent": point.tolist(),
        "relative_loss_percent": loss.tolist(),
        "relative_loss_interval": np.quantile(bloss, [0.025, 0.975], axis=0).T.tolist(),
        "adjacent_miou_recovery_pp": gains.tolist(),
        "adjacent_recovery_interval": np.quantile(
            bgain, [0.025, 0.975], axis=0
        ).T.tolist(),
        "adjacent_loss_recovery_interval": np.quantile(
            -np.diff(bloss, axis=1), [0.025, 0.975], axis=0
        ).T.tolist(),
        "per_group_miou_percent": per.tolist(),
        "groups_with_smaller_last_absolute_change": int(
            np.sum(
                np.abs(np.diff(per[:, 1:], axis=1)[:, -1])
                < np.abs(np.diff(per[:, 1:], axis=1)[:, 0])
            )
        ),
        "median_first_last_absolute_change": np.median(
            np.abs(np.diff(per[:, 1:], axis=1)[:, [0, -1]]), axis=0
        ).tolist(),
        "groups_with_smaller_last_recovery": int(
            np.sum(
                np.diff(per[:, 1:], axis=1)[:, -1] < np.diff(per[:, 1:], axis=1)[:, 0]
            )
        ),
    }
