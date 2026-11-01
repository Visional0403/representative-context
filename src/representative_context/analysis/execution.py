"""Exact execution groups, paired CUDA timings and complete-forward cost regions."""

from collections import defaultdict
from contextlib import contextmanager
import statistics
import numpy as np


def distribution(values):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all() and (x >= 0).all()
    return {
        "count": len(x),
        "median_ms": float(np.median(x)),
        "mean_ms": float(x.mean()),
        "percentiles_10_90_ms": np.percentile(x, [10, 90]).tolist(),
        "percentiles_25_75_ms": np.percentile(x, [25, 75]).tolist(),
        "min_max_ms": [float(x.min()), float(x.max())],
    }


def grouped_orders(kv_order, query_shape, kv_shape):
    """Construct original query indices from the complete nearest-grid preimage."""
    qh, qw = query_shape
    kh, kw = kv_shape
    order = np.asarray(kv_order, dtype=np.int64)
    assert np.array_equal(np.sort(order), np.arange(kh * kw))
    parts, counts = ([], [])
    for key in order:
        y, x = divmod(int(key), kw)
        y0, y1 = ((y * qh + kh - 1) // kh, ((y + 1) * qh + kh - 1) // kh)
        x0, x1 = ((x * qw + kw - 1) // kw, ((x + 1) * qw + kw - 1) // kw)
        indexes = (np.arange(y0, y1)[:, None] * qw + np.arange(x0, x1)).ravel()
        parts.append(indexes)
        counts.append(len(indexes))
    qorder = np.concatenate(parts)
    assert np.array_equal(np.sort(qorder), np.arange(qh * qw))
    return (qorder, np.r_[0, np.cumsum(counts)])


def bucket_plan(kv_order, query_shape, kv_shape, groups):
    order = np.asarray(kv_order, dtype=np.int64)
    qorder, prefix = grouped_orders(order, query_shape, kv_shape)
    keys, buckets = (len(order), defaultdict(list))
    qcounts, kcounts = ([], [])
    for g in range(groups):
        start = g * (keys // groups) + min(g, keys % groups)
        length = keys // groups + (g < keys % groups)
        qstart, qend = (prefix[start], prefix[start + length])
        qi, ki = (qorder[qstart:qend], order[start : start + length])
        qcounts.append(len(qi))
        kcounts.append(length)
        buckets[len(qi), length].append((qi, ki))
    result = [
        {
            "query_length": nq,
            "kv_length": nk,
            "groups": len(rows),
            "query_indices": np.concatenate([x[0] for x in rows]),
            "kv_indices": np.concatenate([x[1] for x in rows]),
        }
        for (nq, nk), rows in sorted(buckets.items())
    ]
    qi = np.concatenate([b["query_indices"] for b in result])
    ki = np.concatenate([b["kv_indices"] for b in result])
    assert np.array_equal(np.sort(qi), np.arange(np.prod(query_shape)))
    assert np.array_equal(np.sort(ki), np.arange(keys))
    return (result, qcounts, kcounts)


def edge_record(q_shape, kv_shape, layout, heads, dim, budget):
    qh, qw = q_shape
    kh, kw = kv_shape
    queries, keys = (qh * qw, kh * kw)
    if layout is None:
        groups, key_counts, query_counts = (1, [keys], [queries])
        sparse = queries * keys
        index_bytes = 0
    else:
        groups = layout.groups
        key_counts = [
            keys // groups + (index < keys % groups) for index in range(groups)
        ]
        if layout.q_prefix is None:
            query_counts = [count * (qh // kh) * (qw // kw) for count in key_counts]
        else:
            prefix = layout.q_prefix.cpu().tolist()
            query_counts = []
            start = 0
            for count in key_counts:
                query_counts.append(prefix[start + count] - prefix[start])
                start += count
        sparse = sum(
            (q_count * k_count for q_count, k_count in zip(query_counts, key_counts))
        )
        index_bytes = layout.index_bytes
    if (
        sum(query_counts) != queries
        or sum(key_counts) != keys
        or max(key_counts) > budget
    ):
        raise RuntimeError(
            "Actual group capacity or interaction counts are inconsistent."
        )
    dense = queries * keys
    return {
        "query_grid": list(q_shape),
        "kv_grid": list(kv_shape),
        "Q": queries,
        "KV": keys,
        "heads": heads,
        "head_dimension": dim,
        "capacity_limit": budget,
        "groups": groups,
        "query_counts": query_counts,
        "kv_counts": key_counts,
        "dense_edges_per_head": dense,
        "sparse_edges_per_head": sparse,
        "retention": sparse / dense,
        "dense_flops": 4 * heads * dim * dense,
        "sparse_flops": 4 * heads * dim * sparse,
        "index_bytes": index_bytes,
        "dense_single_group": groups == 1,
    }


def paired_measure(methods, warmup=5, rounds=3, repeats=20):
    import torch

    for call in methods.values():
        for _ in range(warmup):
            out = call()
            del out
    torch.cuda.synchronize()
    values = {name: [] for name in methods}
    orders = []
    names = list(methods)
    for r in range(rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        current = {}
        for name in order:
            times = []
            for _ in range(repeats):
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                out = methods[name]()
                b.record()
                b.synchronize()
                times.append(float(a.elapsed_time(b)))
                del out
            current[name] = times
            values[name].extend(times)
        orders.append({"round": r + 1, "order": order, "timing_ms": current})
    return {
        "methods": {
            name: {**distribution(x), "timing_ms": x} for name, x in values.items()
        },
        "rounds": orders,
    }


def make_plan(layout):
    import torch

    if layout is None:
        return (None, None)
    buckets, nq, nk = bucket_plan(
        layout.kv_order.cpu().numpy(),
        layout.query_shape,
        layout.kv_shape,
        layout.groups,
    )
    if layout.q_order is not None:
        joined = np.concatenate([np.asarray(x) for x in [layout.q_order.cpu().numpy()]])
        from representative_context.analysis.execution import grouped_orders

        expected, _ = grouped_orders(
            layout.kv_order.cpu().numpy(), layout.query_shape, layout.kv_shape
        )
        assert np.array_equal(joined, expected)
    plans = []
    for b in buckets:
        plans.append(
            {
                "nq": b["query_length"],
                "nk": b["kv_length"],
                "ng": b["groups"],
                "qi": torch.as_tensor(b["query_indices"], device="cuda"),
                "ki": torch.as_tensor(b["kv_indices"], device="cuda"),
            }
        )
    return (
        plans,
        {
            "buckets": len(plans),
            "query_counts": nq,
            "kv_counts": nk,
            "bucket_shapes": [
                {"query_length": p["nq"], "kv_length": p["nk"], "groups": p["ng"]}
                for p in plans
            ],
            "edges_per_head": sum((a * b for a, b in zip(nq, nk))),
        },
    )


def grouped_sdpa(q, k, v, plans):
    import torch
    from torch.nn import functional as F
    from representative_context.adapters.sra import sdpa_kernel, SDPBackend

    if plans is None:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    batch, heads, queries, dim = q.shape
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    for p in plans:
        qg = q.index_select(2, p["qi"]).view(batch, heads * p["ng"], p["nq"], dim)
        kg = k.index_select(2, p["ki"]).view(batch, heads * p["ng"], p["nk"], dim)
        vg = v.index_select(2, p["ki"]).view(batch, heads * p["ng"], p["nk"], dim)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            result = F.scaled_dot_product_attention(qg, kg, vg)
        output.index_copy_(2, p["qi"], result.reshape(batch, heads, -1, dim))
    return output


class Timeline:
    """Mutually exclusive GPU time regions within the complete model forward."""

    def __init__(self):
        import torch

        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(2048)]
        self.enabled = False
        self.global_depth = 0
        self.restore = []

    def point(self, label):
        event = self.events[len(self.points)]
        event.record()
        self.points.append((event, label))

    @contextmanager
    def scope(self, label):
        if not self.enabled:
            yield
            return
        previous = self.current
        self.point(label)
        self.current = label
        try:
            yield
        finally:
            self.point(previous)
            self.current = previous

    def patch(self, obj, attr, make):
        original = getattr(obj, attr)
        setattr(obj, attr, make(original))
        self.restore.append((obj, attr, original))

    def wrap(self, module, label, global_core=False):

        def make(original):

            def call(*args, **kwargs):
                if global_core:
                    self.global_depth += 1
                try:
                    with self.scope(label):
                        return original(*args, **kwargs)
                finally:
                    if global_core:
                        self.global_depth -= 1

            return call

        self.patch(module, "forward", make)

    def run(self, call):
        self.current = "other_model"
        self.points = []
        self.enabled = True
        self.point(self.current)
        try:
            result = call()
            self.point("end")
        finally:
            self.enabled = False
        self.points[-1][0].synchronize()
        parts = {}
        for (a, label), (b, _) in zip(self.points, self.points[1:]):
            parts[label] = parts.get(label, 0) + float(a.elapsed_time(b))
        total = float(self.points[0][0].elapsed_time(self.points[-1][0]))
        assert abs(sum(parts.values()) - total) < 0.03, (parts, total)
        return (
            result,
            {"total_ms": total, "parts_ms": parts, "events": len(self.points)},
        )

    def close(self):
        for obj, attr, original in reversed(self.restore):
            setattr(obj, attr, original)


def profile(args, model, blocks, inputs, report):
    import torch
    from torch.nn import functional as F
    from representative_context.kernels import indexed as module

    if args.model == "segformer":
        forward = lambda: model(inputs)
        transformer_blocks = [b for layer in model.backbone.layers for b in layer[1]]
    else:
        forward = lambda: model.encode_decode(
            inputs,
            [
                {
                    "img_shape": (args.height, args.width),
                    "ori_shape": (args.height, args.width),
                }
            ],
        )
        transformer_blocks = [b for layer in model.backbone.layers for b in layer]

    def call(mode):
        for _, _, core in blocks:
            core.dense = mode == "dense"
        return forward()

    calls = {name: lambda n=name: call(n) for name in ("dense", "indexed")}
    with torch.inference_mode():
        report["unprofiled_timing"] = paired_measure(calls, repeats=10)
        timeline = Timeline()
        timeline.wrap(model.backbone, "other_backbone")
        timeline.wrap(model.decode_head, "decoding")
        for block in transformer_blocks:
            if hasattr(block, "ffn"):
                timeline.wrap(block.ffn, "FFN")
        cores = {id(core) for _, _, core in blocks}
        for _, _, core in blocks:
            timeline.wrap(core, "other_backbone", global_core=True)
            for name in ("sr", "norm", "out_proj"):
                part = getattr(core, name, None)
                if part is not None:
                    timeline.wrap(part, "projection_and_sr")
        local_count = 0
        if args.model == "twins":
            for block in transformer_blocks:
                if id(getattr(block.attn, "core", None)) not in cores:
                    timeline.wrap(block.attn, "local_attention")
                    local_count += 1
            assert local_count == 9

        def linear_make(original):

            def linear(*a, **kw):
                if timeline.global_depth:
                    with timeline.scope("projection_and_sr"):
                        return original(*a, **kw)
                return original(*a, **kw)

            return linear

        def attention_make(original):

            def attention(*a, **kw):
                if timeline.global_depth:
                    with timeline.scope("global_aggregation"):
                        return original(*a, **kw)
                return original(*a, **kw)

            return attention

        timeline.patch(F, "linear", linear_make)
        timeline.patch(F, "scaled_dot_product_attention", attention_make)
        timeline.patch(module, "indexed_attention", attention_make)
        parts = {name: [] for name in calls}
        try:
            for mode in calls:
                for _ in range(3):
                    z = call(mode)
                    del z
            for r in range(3):
                order = ["dense", "indexed"] if r % 2 == 0 else ["indexed", "dense"]
                for mode in order:
                    for repeat in range(4):
                        z, row = timeline.run(lambda: call(mode))
                        row.update({"round": r + 1, "repeat": repeat + 1})
                        parts[mode].append(row)
                        del z
        finally:
            timeline.close()
        report["profile_observations"] = parts
        report["local_attention_modules"] = local_count
        report["unprofiled_recheck"] = paired_measure(
            calls, warmup=3, rounds=1, repeats=5
        )
        report["profile_summary"] = {}
        for mode, rows in parts.items():
            labels = sorted({key for row in rows for key in row["parts_ms"]})
            mean_parts = {
                key: statistics.mean((r["parts_ms"].get(key, 0) for r in rows))
                for key in labels
            }
            total_mean = statistics.mean((r["total_ms"] for r in rows))
            report["profile_summary"][mode] = {
                "count": len(rows),
                "mean_total_ms": total_mean,
                "mean_parts_ms": mean_parts,
                "profiled_total_distribution": distribution(
                    [r["total_ms"] for r in rows]
                ),
                "profiled_vs_unprofiled_percent": 100
                * (
                    statistics.median((r["total_ms"] for r in rows))
                    / report["unprofiled_timing"]["methods"][mode]["median_ms"]
                    - 1
                ),
            }
            assert abs(sum(mean_parts.values()) - total_mean) < 0.03
